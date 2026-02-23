# memshield

[![CI](https://github.com/npow/memshield/actions/workflows/ci.yml/badge.svg)](https://github.com/npow/memshield/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/memshield)](https://pypi.org/project/memshield/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Stop memory poisoning attacks on your AI agents.

## The problem

When an attacker poisons your agent's persistent memory — its RAG knowledge base, vector store, or conversation history — every future decision is compromised. Unlike prompt injection, which requires the attacker to be present each session, memory poisoning is a one-time attack with permanent effect. The agent keeps making bad decisions long after the initial compromise, across all users and all sessions.

Existing defenses are either red-teaming tools that find the vulnerability but don't fix it, or academic prototypes tied to specific agent architectures. Nothing you can `pip install` today.

## Quick start

```bash
pip install memshield
```

```python
from memshield import MemShield
from memshield.adapters.openai_provider import OpenAIProvider

shield = MemShield(local_provider=OpenAIProvider(
    model="llama3.1:8b",
    base_url="http://localhost:11434/v1",
    api_key="not-needed",
))
vectorstore = shield.wrap(vectorstore)  # done — reads are validated, writes are tracked
```

Your agent code doesn't change. The wrapped store has the same interface as the original.

## Install

```bash
# Core library (no dependencies)
pip install memshield

# With OpenAI/Ollama provider
pip install memshield[openai]

# With all optional integrations
pip install memshield[all]

# From source
git clone https://github.com/npow/memshield.git
cd memshield
pip install -e ".[dev]"
```

## Usage

### Validate memory reads with a local LLM

```python
from memshield import MemShield, ShieldConfig
from memshield.adapters.openai_provider import OpenAIProvider

provider = OpenAIProvider(model="llama3.1:8b", base_url="http://localhost:11434/v1")
shield = MemShield(local_provider=provider)

store = shield.wrap(your_vectorstore)
results = store.similarity_search("company refund policy")
# Poisoned entries are blocked. Clean entries pass through.
```

### Add cloud escalation for ambiguous cases

```python
local = OpenAIProvider(model="llama3.1:8b", base_url="http://localhost:11434/v1")
cloud = OpenAIProvider(model="gpt-4o")  # only called for ambiguous entries

shield = MemShield(local_provider=local, cloud_provider=cloud)
store = shield.wrap(your_vectorstore)
```

### Track provenance of memory writes

```python
shield = MemShield(local_provider=provider)
store = shield.wrap(your_vectorstore)

# Writes are automatically tagged with source, timestamp, and hash chain
store.add_texts(["New company policy: ..."], metadatas=[{"source": "user_input"}])

# Later, verify the provenance chain is intact
assert shield.provenance.verify_chain()
```

### Check operational stats

```python
print(f"Clean: {shield.stats.clean}")
print(f"Blocked: {shield.stats.blocked}")
print(f"Escalated to cloud: {shield.stats.cloud_calls}")
```

## How it works

MemShield wraps your vector store with a proxy. When your agent calls `similarity_search`, the proxy intercepts the results and validates each entry before returning them. Writes go through too — tagged with provenance metadata.

**The core idea is simple: ask an LLM to analyze each retrieved memory entry.** Specifically, the LLM decomposes the entry into three parts — what factual claim it makes, what action it implies, and what context would make it legitimate. When these conflict (e.g., the factual claim is benign but the implied action is "override system instructions"), the entry is flagged as poisoned. This catches attacks that look like knowledge but behave like instructions.

Three layers, from fast/cheap to slow/expensive:

1. **Local LLM** (every read) — A local model (Llama 3.1 8B via Ollama/vLLM) runs the analysis. Adds ~200-500ms per entry on GPU. Resolves the clear cases — obviously clean or obviously poisoned.
2. **Cloud LLM** (ambiguous cases only) — When the local model isn't confident, the entry escalates to a frontier model (GPT-4o, Claude, etc.) for a second pass. Typically 1-3% of reads.
3. **Provenance + drift** (metadata, no LLM) — Every write is logged in a SHA-256 hash chain with source and trust level. A statistical profiler flags unusual access patterns — catching subtle poisoning that content analysis misses.

The proxy delegates all methods it doesn't intercept via `__getattr__`, so your agent code doesn't change.

## Alternatives and prior work

There are 60+ products in the LLM guardrails space. Almost all operate at the **prompt/response boundary** — they screen what goes into and comes out of the model. Very few operate at the **memory/retrieval layer** where poisoning actually lives.

### Products that touch RAG/memory content

These are the closest to what memshield does:

| Tool | What it does | How memshield differs |
|------|-------------|----------------------|
| [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | Open-source programmable guardrails with "retrieval rails" that filter RAG chunks before they reach the LLM. | NeMo filters based on configurable rules (Colang). memshield uses LLM reasoning to detect semantic manipulation — entries that follow the rules but contain disguised instructions. NeMo is broader (dialog rails, output rails); memshield is deeper on memory specifically. |
| [Daxa AI](https://www.daxa.ai) | Shift-left RAG security: scans content before vectorization, labels malicious vectors, policy-based retrieval controls. | Daxa scans at ingestion time (pre-vectorization). memshield validates at retrieval time (post-vectorization). Different interception points — Daxa prevents bad data from entering; memshield catches it on the way out. Complementary. |
| [Preamble](https://www.preamble.com) | Runtime guardrails explicitly built for RAG, LLMs, and agents. | Preamble is a commercial platform. memshield is open-source middleware you embed in your code. Different deployment models. |
| [NeuralTrust](https://www.neuraltrust.ai) | Generative Application Firewall that maintains context across sessions and detects accumulation attacks. | NeuralTrust is a full commercial firewall platform. memshield is a pip-installable library that wraps your vector store. Different scale and scope. |
| [Meta LlamaFirewall](https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall) | Open-source guardrails with AlignmentCheck that audits agent reasoning from untrusted data sources. | LlamaFirewall audits the agent's chain-of-thought after retrieval. memshield validates the memory entries themselves before they reach the agent. Different layer — LlamaFirewall trusts the data and checks the reasoning; memshield checks the data. |

### Prompt/response guardrails (don't address memory)

The bulk of the market. These screen model I/O but don't inspect what's stored in your knowledge base:

| Tool | Status |
|------|--------|
| [Lakera Guard](https://www.lakera.ai) | Acquired by Check Point. Prompt injection detection API. |
| [Guardrails AI](https://www.guardrailsai.com) | Open-source I/O validators. No memory-specific validators. |
| [LLM Guard](https://github.com/protectai/llm-guard) (Protect AI → Palo Alto) | Open-source input/output scanners. |
| [Arthur AI Shield](https://www.arthur.ai) | LLM firewall for PII, toxicity, hallucination, injection. |
| [Prompt Security](https://www.prompt.security) | Acquired by SentinelOne. Shadow AI + injection defense. |
| [CalypsoAI](https://calypsoai.com) | Acquired by F5. Inference-time security. |
| [DynamoGuard](https://dynamo.ai) (Dynamo AI) | Real-time guardrails with sub-50ms latency. |

### Cloud provider guardrails

| Provider | Feature | Handles memory? |
|----------|---------|----------------|
| AWS Bedrock | Guardrails with contextual grounding checks | Partial — checks responses against source docs, doesn't scan the knowledge base itself |
| Google Vertex AI | Model Armor for injection/jailbreak/content safety | No — prompt/response only |
| Azure AI | Content Safety + AI Threat Protection | Partial — anomaly monitoring, not knowledge base scanning |
| Cisco AI Defense | Built on Robust Intelligence acquisition | Prompt/response screening |
| Palo Alto Prisma AIRS | Built on Protect AI acquisition | Prompt/response + model scanning |

### AI security platforms (broader scope)

Funded companies focused on AI security posture, not specifically memory defense:

[Noma Security](https://noma.security) ($132M raised) · [HiddenLayer](https://hiddenlayer.com) (model-level attacks) · [Lasso Security](https://lasso.security) ($28M, MCP gateway) · [Straiker](https://straiker.ai) ($21M, agentic AI) · [WitnessAI](https://witnessai.com) ($58M, AI governance) · [Cranium AI](https://cranium.ai) ($32M, governance) · [Zenity](https://zenity.io) (agent behavior monitoring)

### Research (no commercial product)

| Paper | What it proved |
|-------|---------------|
| [A-MemGuard](https://arxiv.org/abs/2510.02373) (NTU/Oxford, 2025) | Consensus validation reduces memory poisoning attack success by >95%. memshield adapts this approach. |
| [RAGuard](https://openreview.net/forum?id=onh7sLJ1kl) (NeurIPS 2025) | Adversarial retriever training + zero-knowledge filter blocks RAG corpus poisoning. |
| [RAGPart & RAGMask](https://arxiv.org/abs/2601.05504) (U. Maryland, 2026) | Fragment-and-vote and masking approaches reduce poison impact. |
| [TrustRAG](https://arxiv.org/abs/2501.00000) (2025) | K-means clustering detects poisoned embeddings. |
| [MemoryGraft](https://arxiv.org/abs/2512.16962) (2025) | Attack paper proposing Cryptographic Provenance Attestation (unbuilt). memshield implements provenance tracking. |

### Where memshield fits

memshield is **open-source middleware** that wraps your vector store. It's not a platform, not a firewall, not a governance tool. It's a library you `pip install` and add to your agent in three lines.

It complements the products above:
- Use **Promptfoo** to test if you're vulnerable → use **memshield** to fix it at runtime
- Use **NeMo Guardrails** for broad dialog/output rails → use **memshield** for deep memory content analysis
- Use **Lakera/LLM Guard** for prompt injection on user inputs → use **memshield** for poisoning in stored knowledge
- Use **Daxa** to prevent bad data from entering your vector store → use **memshield** to catch what Daxa missed on the way out

**Honest caveats:**
- NVIDIA NeMo Guardrails already has retrieval rails. If you're in the NeMo ecosystem, check if that's sufficient before adding memshield.
- Platform vendors (Meta, Microsoft, AWS, Google) may add native memory integrity features within 6-12 months.
- This is a new tool with no production deployments yet. The benchmark results will tell us how well it actually works.

## Configuration

```python
from memshield import MemShield, ShieldConfig, FailurePolicy

config = ShieldConfig(
    confidence_threshold=0.7,        # min confidence to resolve locally
    untrusted_confidence_boost=0.15, # extra confidence required for untrusted sources
    failure_policy=FailurePolicy.ALLOW_WITH_WARNING,  # what to do with ambiguous entries
    enable_provenance=True,
    enable_drift_detection=True,
)
shield = MemShield(local_provider=provider, config=config)
```

| Option | Default | Description |
|--------|---------|-------------|
| `confidence_threshold` | 0.7 | Minimum confidence to resolve an entry locally |
| `untrusted_confidence_boost` | 0.15 | Extra confidence required for entries from untrusted sources |
| `failure_policy` | `ALLOW_WITH_WARNING` | `BLOCK`, `ALLOW_WITH_WARNING`, or `ALLOW_WITH_REVIEW` |
| `enable_provenance` | `True` | Track cryptographic provenance of memory writes |
| `enable_drift_detection` | `True` | Monitor for unusual memory access patterns |

## Development

```bash
git clone https://github.com/npow/memshield.git
cd memshield
pip install -e ".[dev]"
pytest -v
```

### Pluggable strategies

memshield ships with multiple validation strategies you can use individually or ensemble:

```python
from memshield import (
    MemShield, ConsensusStrategy, KeywordHeuristicStrategy, EnsembleStrategy
)
from memshield.adapters.openai_provider import OpenAIProvider

# Fast keyword heuristic (instant, zero cost, catches obvious attacks)
shield = MemShield(strategy=KeywordHeuristicStrategy())

# LLM consensus — A-MemGuard approach (deep, catches subtle attacks)
provider = OpenAIProvider(model="llama3.1:8b", base_url="http://localhost:11434/v1")
shield = MemShield(strategy=ConsensusStrategy(provider))

# Ensemble — run both, flag as poisoned if either detects it
shield = MemShield(strategy=EnsembleStrategy(
    [KeywordHeuristicStrategy(), ConsensusStrategy(provider)],
    mode="any_poisoned",  # or "majority" for balanced precision/recall
))
```

### Run benchmarks

The benchmark compares strategies against labeled datasets including reconstructed [AgentPoison](https://github.com/AI-secure/AgentPoison) (NeurIPS 2024) attack data — the same corpus A-MemGuard evaluated against.

```bash
# Heuristic baseline (instant, no LLM needed):
python benchmarks/run_benchmark.py --strategy heuristic --tier 2 3 4

# Compare all strategies head-to-head:
python benchmarks/run_benchmark.py --compare-strategies --tier 2 3 4 \
    --local-url http://localhost:11434/v1 --local-model llama3.1:8b \
    --cloud-model gpt-4o

# Via claude-relay:
python benchmarks/run_benchmark.py --compare-strategies --tier 2 3 4 \
    --local-url http://localhost:8082/v1 --local-model sonnet
```

Benchmark tiers:

| Tier | Source | Entries | What it tests |
|------|--------|---------|---------------|
| 2 | Hand-crafted + MemoryGraft-style | 55 | Memory-specific: experience records, schema spoofing, rubric mimicry |
| 3 | Adversarial paired entries | 18 | Subtle pairs where clean and poisoned look structurally identical |
| 4 | AgentPoison (NeurIPS 2024) | 330 | Reconstructed StrategyQA + EHR poisoned passages with published golden triggers |

## License

[Apache-2.0](LICENSE)
