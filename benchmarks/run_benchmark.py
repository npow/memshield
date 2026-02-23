"""Benchmark runner — measures MemShield detection rates against tiered datasets.

Tiers:
  1 — Existing prompt injection datasets (broad baseline)
  2 — Memory-specific entries (hand-crafted + MemoryGraft-style)
  3 — Adversarial paired entries (subtle, hard to detect)

Usage:
    # Dry run (print dataset, no LLM calls):
    python benchmarks/run_benchmark.py --dry-run --tier 2
    python benchmarks/run_benchmark.py --dry-run --tier 1 2 3

    # Run against cloud LLM:
    OPENAI_API_KEY=sk-... python benchmarks/run_benchmark.py --tier 2 3

    # Run against local Ollama:
    python benchmarks/run_benchmark.py --tier 2 3 \
        --local-url http://localhost:11434/v1 --local-model llama3.1:8b

    # Run via claude-relay (any OpenAI-compatible endpoint works):
    python benchmarks/run_benchmark.py --tier 2 3 \
        --local-url http://localhost:8082/v1 --local-model sonnet

    # Run all tiers with both local and cloud (full comparison):
    python benchmarks/run_benchmark.py --tier 1 2 3 \
        --local-url http://localhost:11434/v1 --local-model llama3.1:8b \
        --cloud-model gpt-4o --compare

    # Compare claude-relay vs direct OpenAI:
    python benchmarks/run_benchmark.py --tier 2 3 --compare \
        --local-url http://localhost:8082/v1 --local-model sonnet \
        --cloud-model gpt-4o

    # Limit tier 1 size (it can be large):
    python benchmarks/run_benchmark.py --tier 1 2 3 --max-tier1 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add src to path so we can import memshield
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add benchmarks dir for local imports
sys.path.insert(0, str(Path(__file__).parent))

from memshield._types import Verdict
from memshield.validation import build_validation_prompt, parse_validation_response
from dataset import BenchmarkEntry
from loaders import load_all_tiers, get_tier_stats


@dataclass
class BenchmarkResult:
    """Aggregated results from a benchmark run."""
    model_name: str = ""
    tier_label: str = ""
    total: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    ambiguous: int = 0
    errors: int = 0
    per_category: dict[str, dict[str, int]] = field(default_factory=dict)
    per_source: dict[str, dict[str, int]] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "tier": self.tier_label,
            "total": self.total,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "ambiguous": self.ambiguous,
            "errors": self.errors,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "per_category": self.per_category,
            "per_source": self.per_source,
        }


def run_benchmark(
    provider: object,
    entries: list[BenchmarkEntry],
    model_name: str = "",
    tier_label: str = "",
    quiet: bool = False,
) -> BenchmarkResult:
    """Run benchmark against a labeled dataset."""
    result = BenchmarkResult(model_name=model_name, tier_label=tier_label)

    for i, entry in enumerate(entries):
        result.total += 1
        prompt = build_validation_prompt(entry.content)

        start = time.monotonic()
        try:
            response = provider.generate(prompt, temperature=0.0)  # type: ignore[union-attr]
        except Exception as e:
            if not quiet:
                print(f"  [{i+1:3d}/{len(entries)}] ERROR: {e}")
            result.errors += 1
            continue
        elapsed_ms = (time.monotonic() - start) * 1000
        result.latencies_ms.append(elapsed_ms)

        validation = parse_validation_response(response)

        # Classify
        if entry.is_poisoned:
            if validation.verdict == Verdict.POISONED:
                label = "TP"
                result.true_positives += 1
            elif validation.verdict == Verdict.AMBIGUOUS:
                label = "AMB"
                result.ambiguous += 1
            else:
                label = "FN"
                result.false_negatives += 1
        else:
            if validation.verdict == Verdict.CLEAN:
                label = "TN"
                result.true_negatives += 1
            elif validation.verdict == Verdict.AMBIGUOUS:
                label = "AMB"
                result.ambiguous += 1
            else:
                label = "FP"
                result.false_positives += 1

        # Per-category tracking
        cat = entry.category
        if cat not in result.per_category:
            result.per_category[cat] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "amb": 0}
        result.per_category[cat][label.lower()] = result.per_category[cat].get(label.lower(), 0) + 1

        # Per-source tracking
        src = entry.source
        if src not in result.per_source:
            result.per_source[src] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "amb": 0}
        result.per_source[src][label.lower()] = result.per_source[src].get(label.lower(), 0) + 1

        if not quiet:
            status = "POISONED" if entry.is_poisoned else "CLEAN"
            print(
                f"  [{i+1:3d}/{len(entries)}] {label:3s} | "
                f"truth={status:8s} | verdict={validation.verdict.value:9s} | "
                f"conf={validation.confidence:.2f} | {elapsed_ms:7.0f}ms | "
                f"{entry.content[:55]}..."
            )

    return result


def print_results(result: BenchmarkResult) -> None:
    """Print formatted results."""
    header = f"RESULTS: {result.model_name}"
    if result.tier_label:
        header += f" ({result.tier_label})"

    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)
    print(f"Total:              {result.total}")
    print(f"True positives:     {result.true_positives}")
    print(f"True negatives:     {result.true_negatives}")
    print(f"False positives:    {result.false_positives}")
    print(f"False negatives:    {result.false_negatives}")
    print(f"Ambiguous:          {result.ambiguous}")
    print(f"Errors:             {result.errors}")
    print()
    print(f"Precision:          {result.precision:.1%}")
    print(f"Recall:             {result.recall:.1%}")
    print(f"F1 Score:           {result.f1:.1%}")
    print(f"False Positive Rate:{result.false_positive_rate:.1%}")
    print(f"Mean Latency:       {result.mean_latency_ms:.0f}ms")

    if result.per_category:
        print()
        print("Per-category:")
        print(f"  {'Category':<30s} {'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s} {'AMB':>4s}")
        print(f"  {'-'*30} {'--':>4s} {'--':>4s} {'--':>4s} {'--':>4s} {'---':>4s}")
        for cat, c in sorted(result.per_category.items()):
            print(
                f"  {cat:<30s} {c.get('tp',0):4d} {c.get('fp',0):4d} "
                f"{c.get('tn',0):4d} {c.get('fn',0):4d} {c.get('amb',0):4d}"
            )

    if result.per_source:
        print()
        print("Per-source:")
        print(f"  {'Source':<35s} {'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s} {'AMB':>4s}")
        print(f"  {'-'*35} {'--':>4s} {'--':>4s} {'--':>4s} {'--':>4s} {'---':>4s}")
        for src, c in sorted(result.per_source.items()):
            print(
                f"  {src:<35s} {c.get('tp',0):4d} {c.get('fp',0):4d} "
                f"{c.get('tn',0):4d} {c.get('fn',0):4d} {c.get('amb',0):4d}"
            )


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print side-by-side comparison of multiple model runs."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"  {'Model':<25s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'FPR':>7s} {'Latency':>9s}")
    print(f"  {'-'*25} {'-----':>7s} {'------':>7s} {'--':>7s} {'---':>7s} {'-------':>9s}")
    for r in results:
        label = r.model_name
        if r.tier_label:
            label += f" [{r.tier_label}]"
        print(
            f"  {label:<25s} {r.precision:6.1%} {r.recall:6.1%} "
            f"{r.f1:6.1%} {r.false_positive_rate:6.1%} {r.mean_latency_ms:7.0f}ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MemShield benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/run_benchmark.py --dry-run --tier 2
  python benchmarks/run_benchmark.py --tier 2 3 --cloud-model gpt-4o
  python benchmarks/run_benchmark.py --tier 1 2 3 --local-url http://localhost:11434/v1
  python benchmarks/run_benchmark.py --tier 2 --compare --local-url http://localhost:11434/v1 --cloud-model gpt-4o
        """,
    )
    parser.add_argument("--tier", nargs="+", type=int, default=[2, 3],
                        help="Which tiers to run (default: 2 3)")
    parser.add_argument("--max-tier1", type=int, default=500,
                        help="Max entries per dataset in tier 1 (default: 500)")
    parser.add_argument("--local-url",
                        help="Local LLM base URL (e.g., http://localhost:11434/v1)")
    parser.add_argument("--local-model", default="llama3.1:8b",
                        help="Local model name (default: llama3.1:8b)")
    parser.add_argument("--cloud-model", default="gpt-4o",
                        help="Cloud model name (default: gpt-4o)")
    parser.add_argument("--cloud-only", action="store_true",
                        help="Use cloud model only")
    parser.add_argument("--compare", action="store_true",
                        help="Run both local and cloud, print comparison")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print dataset without running LLM")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print summary, not per-entry results")
    args = parser.parse_args()

    # Load datasets
    print(f"Tiers: {args.tier}")
    print()
    entries = load_all_tiers(tiers=args.tier, max_tier1_per_dataset=args.max_tier1)
    stats = get_tier_stats(entries)

    print()
    print(f"Total: {stats['total']} entries ({stats['clean']} clean, {stats['poisoned']} poisoned)")
    print(f"Sources:")
    for src, counts in sorted(stats["by_source"].items()):
        print(f"  {src:<35s} {counts['clean']:4d} clean, {counts['poisoned']:4d} poisoned")
    print()

    if args.dry_run:
        for i, entry in enumerate(entries):
            status = "POISONED" if entry.is_poisoned else "CLEAN   "
            print(f"  [{i+1:3d}] {status} | {entry.category:<30s} | {entry.source:<30s} | {entry.content[:50]}...")
        print(f"\nTotal: {len(entries)} entries")
        return

    from memshield.adapters.openai_provider import OpenAIProvider

    tier_label = f"tier {'+'.join(str(t) for t in args.tier)}"
    all_results: list[BenchmarkResult] = []

    if args.compare and args.local_url:
        # Run both local and cloud
        print(f"--- Local model: {args.local_model} at {args.local_url} ---")
        local_provider = OpenAIProvider(
            model=args.local_model,
            base_url=args.local_url,
            api_key="not-needed",
        )
        local_result = run_benchmark(
            local_provider, entries,
            model_name=args.local_model,
            tier_label=tier_label,
            quiet=args.quiet,
        )
        print_results(local_result)
        all_results.append(local_result)

        print(f"\n--- Cloud model: {args.cloud_model} ---")
        cloud_provider = OpenAIProvider(model=args.cloud_model)
        cloud_result = run_benchmark(
            cloud_provider, entries,
            model_name=args.cloud_model,
            tier_label=tier_label,
            quiet=args.quiet,
        )
        print_results(cloud_result)
        all_results.append(cloud_result)

        print_comparison(all_results)
    else:
        # Single model run
        if args.local_url and not args.cloud_only:
            model_name = args.local_model
            provider = OpenAIProvider(
                model=args.local_model,
                base_url=args.local_url,
                api_key="not-needed",
            )
        else:
            model_name = args.cloud_model
            provider = OpenAIProvider(model=args.cloud_model)

        print(f"Model: {model_name}")
        print()
        result = run_benchmark(
            provider, entries,
            model_name=model_name,
            tier_label=tier_label,
            quiet=args.quiet,
        )
        print_results(result)
        all_results.append(result)

    # Save results
    output = [r.to_dict() for r in all_results]
    output_path = Path(__file__).parent / "results.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
