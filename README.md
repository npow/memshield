# memshield

[![CI](https://github.com/npow/memshield/actions/workflows/ci.yml/badge.svg)](https://github.com/npow/memshield/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/memshield)](https://pypi.org/project/memshield/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Stop memory poisoning attacks on your AI agents.

## The problem

When an attacker poisons your agent's persistent memory — its RAG knowledge base, vector store, or conversation history — every future decision is compromised. Unlike prompt injection, which requires the attacker to be present each session, memory poisoning is a one-time attack with permanent effect. The agent keeps making bad decisions long after the initial compromise, across all users and all sessions.

The best available defense is a 30-star academic prototype that requires a PhD to deploy. No production tool exists that you can `pip install` and add to your agent in three lines.

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

MemShield wraps your vector store client with a transparent proxy. Three layers run on every memory operation:

**Layer 1 — Local LLM validation.** Every retrieved memory entry passes through consensus validation on a local LLM (Llama 3.1 8B or similar). The LLM generates three independent interpretations of the entry and flags conflicts that indicate embedded instructions.

**Layer 2 — Cloud escalation.** Entries where the local model's confidence is ambiguous escalate to a frontier LLM (GPT-4o, Claude, etc.) for a second opinion. Only 1-3% of reads need this.

**Layer 3 — Provenance and drift.** Every write is logged in a SHA-256 hash chain with source attribution and trust levels. A statistical profiler detects when memory access patterns deviate from baseline — catching subtle poisoning that per-entry validation misses.

The proxy intercepts `similarity_search` and `add_documents` while delegating everything else to the wrapped store via `__getattr__`. Your agent code never knows MemShield is there.

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

### Run benchmarks

```bash
# Dry run (see the dataset, no LLM calls)
python benchmarks/run_benchmark.py --dry-run --tier 2 3

# Against a local Ollama instance
python benchmarks/run_benchmark.py --tier 2 3 \
    --local-url http://localhost:11434/v1 --local-model llama3.1:8b

# Against OpenAI
OPENAI_API_KEY=sk-... python benchmarks/run_benchmark.py --tier 2 3

# Compare local vs cloud
python benchmarks/run_benchmark.py --tier 2 3 --compare \
    --local-url http://localhost:11434/v1 --local-model llama3.1:8b \
    --cloud-model gpt-4o
```

## License

[Apache-2.0](LICENSE)
