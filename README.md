# memshield

[![CI](https://github.com/npow/memshield/actions/workflows/ci.yml/badge.svg)](https://github.com/npow/memshield/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/memshield)](https://pypi.org/project/memshield/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Docs](https://img.shields.io/badge/docs-mintlify-18a34a?style=flat-square)](https://mintlify.com/npow/memshield)

A drop-in Python library that wraps any vector store to provide two things simultaneously:

- **Memory poisoning defense** — validates every retrieved chunk before it reaches your LLM, blocking injected instructions disguised as knowledge
- **EU AI Act Article 12 audit logging** — per-inference tamper-evident records of what your AI retrieved, with RFC 3161 timestamps, Ed25519 signatures, and GDPR crypto-shredding

One line of code. Same interface as your existing store.

```python
store = shield.wrap(your_chroma_or_pinecone_store)
docs = store.similarity_search("what medication is the patient on?")
# poisoned entries blocked; Article 12 audit record written automatically
```

## Why this exists

**Security:** Memory poisoning corrupts agent behavior permanently. Unlike prompt injection (attacker must be present each session), a poisoned vector store entry affects every future query across all users. Red-teaming tools find the vulnerability — MemShield blocks it at runtime.

**Compliance:** EU AI Act Article 12 (enforceable August 2026) requires high-risk AI systems to automatically generate tamper-evident logs of what data was retrieved at inference time, retained for at least 6 months. No RAG framework produces this natively.

Both problems share the same interception point: the retrieval layer.

## Install

```bash
pip install memshield                  # core (no deps)
pip install memshield[openai]          # OpenAI/Ollama provider
pip install memshield[audit]           # Article 12 audit log (requires cryptography + rfc3161ng)
pip install memshield[audit,openai]    # both
pip install memshield[all]             # everything
```

## Quick start

### Poisoning defense only

```python
from memshield import MemShield
from memshield.strategies import KeywordHeuristicStrategy, ConsensusStrategy, EnsembleStrategy
from memshield.adapters.openai_provider import OpenAIProvider

shield = MemShield(
    strategy=EnsembleStrategy([
        KeywordHeuristicStrategy(),
        ConsensusStrategy(OpenAIProvider(model="gpt-4o-mini")),
    ])
)

store = shield.wrap(your_vectorstore)
docs = store.similarity_search("company refund policy")
# poisoned entries are blocked; clean entries pass through unchanged
```

### With Article 12 audit logging

```python
from memshield import MemShield, AuditConfig
from memshield.strategies import EnsembleStrategy, KeywordHeuristicStrategy, ConsensusStrategy
from memshield.adapters.openai_provider import OpenAIProvider

shield = MemShield(
    strategy=EnsembleStrategy([
        KeywordHeuristicStrategy(),
        ConsensusStrategy(OpenAIProvider()),
    ]),
    audit=AuditConfig(
        store="./audit.db",
        knowledge_base_id="kb_prod_v3",
        pii_fields=["query", "content"],  # AES-256-GCM encrypted at rest
        key_store="./keys.db",
        tsa_url="https://timestamp.sectigo.com",  # RFC 3161, free, no account needed
    ),
)

store = shield.wrap(your_vectorstore)
docs = store.similarity_search(
    "what medication is the patient on?",
    user_id="u_123",
    inference_id="req_xyz",  # optional — correlate with your own request logs
)

record = shield.audit_log.last_record()
print(record.inference_id)       # req_xyz
print(record.chain_hash)         # sha256:8b3e...
print(record.timestamp_rfc3161)  # base64-encoded TSA proof from Sectigo
```

### GDPR right-to-erasure

```python
# Deletes the user's encryption key. Ciphertext becomes permanently unreadable.
# Chain hashes, timestamps, and doc IDs remain intact for audit continuity.
shield.audit_log.erase_user(user_id="u_123")
```

## Audit log schema

Every inference produces one record, aligned to ISO/IEC DIS 24970:2025:

```json
{
  "inference_id": "req_xyz",
  "timestamp_iso": "2026-03-10T09:00:00.000Z",
  "timestamp_rfc3161": "<base64 DER token from Sectigo TSA>",
  "key_id": "<sha256 fingerprint of Ed25519 signing key>",
  "user_id": "u_123",
  "query_hash": "sha256:a3f8...",
  "query_encrypted": "<aes-256-gcm ciphertext>",
  "knowledge_base_id": "kb_prod_v3",
  "retrieved": [
    {
      "doc_id": "doc_456", "chunk_index": 3,
      "content_hash": "sha256:c9d2...", "content_encrypted": "<aes-256-gcm ciphertext>",
      "score": 0.94, "verdict": "clean", "trust_level": "verified"
    }
  ],
  "blocked": [
    {
      "content_hash": "sha256:f1a9...", "content_encrypted": "<aes-256-gcm ciphertext>",
      "verdict": "poisoned", "confidence": 0.97, "attack_type": "T1_instruction_override"
    }
  ],
  "chain_hash": "sha256:8b3e...",
  "previous_chain_hash": "sha256:2d7a...",
  "signature": "ed25519:...",
  "iso24970_event_type": "retrieval",
  "iso24970_schema_version": "DIS-2025"
}
```

**Tamper evidence:** append-only SQLite (WAL mode), SHA-256 hash chain, Ed25519 signature per record, RFC 3161 timestamp token. Expired records (default 180-day retention) are replaced with tombstones that preserve chain continuity.

## CLI

```bash
memshield audit verify   --db ./audit.db
memshield audit export   --db ./audit.db --from 2026-01-01 --format jsonl
memshield audit inspect  --db ./audit.db --inference-id req_xyz
memshield audit erase-user --db ./audit.db --key-store ./keys.db \
    --user-id u_123 --knowledge-base-id kb_prod_v3
memshield keys rotate --key-file ./memshield.key
```

## MCP server

Any MCP-compatible agent (Claude Code, Cursor) can use MemShield with one config entry:

```json
{
  "mcpServers": {
    "memshield": {
      "command": "memshield",
      "args": ["mcp", "--audit-db", "./audit.db"]
    }
  }
}
```

Tools exposed: `audit_inspect(inference_id)`, `audit_verify()`, `audit_export(from_date?)`.

## Supported vector stores

| Store | How to use |
|-------|-----------|
| Chroma | `shield.wrap(chroma_collection)` |
| LangChain vectorstores | `shield.wrap(langchain_vectorstore)` |
| LlamaIndex retrievers | `LlamaIndexRetrieverAdapter(retriever)` then `shield.wrap(...)` |
| Pinecone | `PineconeStoreAdapter(index, embed_fn)` then `shield.wrap(...)` |
| pgvector | `PgVectorStoreAdapter(conn, embed_fn)` then `shield.wrap(...)` |
| Qdrant | `QdrantStoreAdapter(client, collection, embed_fn)` then `shield.wrap(...)` |

## Validation strategies

```python
from memshield.strategies import KeywordHeuristicStrategy, ConsensusStrategy, EnsembleStrategy

# Instant, zero-cost — catches obvious injection patterns
KeywordHeuristicStrategy()

# LLM-based consensus (A-MemGuard approach) — catches subtle attacks
ConsensusStrategy(OpenAIProvider(model="gpt-4o-mini"))

# Ensemble — flag if either detects poisoning (maximum recall)
EnsembleStrategy([KeywordHeuristicStrategy(), ConsensusStrategy(provider)], mode="any_poisoned")

# Ensemble — majority vote (balanced precision/recall)
EnsembleStrategy([KeywordHeuristicStrategy(), ConsensusStrategy(provider)], mode="majority")
```

## Benchmark

[memshield-bench](https://huggingface.co/datasets/npow/memshield-bench) — 1,178 labeled entries across 10 attack types, including AgentPoison (NeurIPS 2024) data:

| Strategy | Precision | Recall | F1 |
|----------|-----------|--------|----|
| Keyword heuristic | 100% | 14.5% | 25.4% |
| LLM consensus | 97.1% | 98.6% | 97.8% |
| Ensemble (majority) | 100% | 100% | 100% |
| Ensemble (any\_poisoned) | 94.5% | **100%** | 97.2% |

Keyword heuristic catches 0% of AgentPoison attacks — sophisticated attacks are disguised as reasoning traces with no obvious instruction patterns. LLM consensus is required.

## Configuration

```python
from memshield import ShieldConfig, FailurePolicy

config = ShieldConfig(
    confidence_threshold=0.7,
    failure_policy=FailurePolicy.BLOCK,  # or ALLOW_WITH_WARNING, ALLOW_WITH_REVIEW
    enable_provenance=True,
    enable_drift_detection=True,
)
shield = MemShield(strategy=..., config=config)
```

## What MemShield is not

- **Not a full EU AI Act compliance solution.** Article 12 logging covers what was retrieved. It does not produce Annex IV technical documentation or Article 9 risk management plans. For complete Article 12 coverage pair with an LLM observability tool (Langfuse, Helicone).
- **Not a managed service.** Self-hosted only.
- **Not a TypeScript library.** Python only.

## Alternatives

Most AI security products operate at the prompt/response boundary. MemShield operates at the retrieval layer — where poisoning actually happens.

Closest alternatives: [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) (retrieval rails, broader scope), [Daxa](https://www.daxa.ai) (pre-vectorization scanning, complementary).

## Development

```bash
git clone https://github.com/npow/memshield.git
cd memshield
pip install -e ".[dev]"
pytest
```

## License

[Apache-2.0](LICENSE)
