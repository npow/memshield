# MemShield PRD
**Version:** 0.2
**Date:** 2026-03-09
**Status:** Final

---

## 1. Problem

AI applications built on RAG (retrieval-augmented generation) have two related, unsolved problems:

**Problem A — Security:** Memory poisoning attacks permanently corrupt agent behavior. Unlike prompt injection, which requires real-time attacker presence, a poisoned vector store entry affects every future query across all users and all sessions. Existing defenses are either research prototypes or red-teaming tools that identify vulnerabilities without fixing them at runtime.

**Problem B — Compliance:** The EU AI Act Article 12 (enforceable August 2026) requires high-risk AI systems to automatically generate tamper-evident logs of what data was retrieved at inference time, retained for at least 6 months. No current RAG framework produces this artifact natively. Systima shipped a hash-chained LLM call logger (March 2026), but it does not log retrieval, does not handle the GDPR/Article 12 PII tension, and is TypeScript-only.

These problems are solved by the same interception point: the retrieval layer. MemShield already sits there. The opportunity is to extend it from a security tool into a **RAG trust layer** that solves both simultaneously.

---

## 2. Target User

**Primary:** Developers at AI startups building products for regulated industries — healthtech, legaltech, fintech. They are:
- Using LangChain, LlamaIndex, or raw vector store APIs (Chroma, Pinecone, pgvector, Qdrant)
- Writing Python
- Facing enterprise customers who ask "what data did your AI use?" or "how do I know it hasn't been tampered with?"
- Approaching the August 2026 EU AI Act deadline without a compliance plan

**Secondary:** Platform/infra developers building AI agent frameworks who want to offer compliance guarantees to their own users.

**Not targeted:** End users, compliance officers, lawyers. This is a developer tool.

---

## 3. Jobs to Be Done

**J1 — Security:** "When my agent retrieves from its knowledge base, I want poisoned entries blocked before they reach the LLM, so a compromised vector store doesn't corrupt my agent's behavior."

**J2 — Compliance:** "When an enterprise customer or EU regulator asks what data my AI used to make a decision, I need a per-inference tamper-evident record I can produce without rebuilding my stack."

**J3 — GDPR reconciliation:** "When a user invokes their right to erasure, I need to satisfy GDPR without breaking my Article 12 audit chain."

**J4 — Operational visibility:** "I want to know if my agent's memory access patterns are drifting — a signal that something is being injected or my knowledge base is degrading."

---

## 4. What MemShield Is

A drop-in Python library that wraps any vector store. One line of code. Zero changes to the rest of the application.

```python
from memshield import MemShield, AuditConfig
from memshield.strategies import KeywordHeuristicStrategy, ConsensusStrategy, EnsembleStrategy
from memshield.adapters.openai_provider import OpenAIProvider

shield = MemShield(
    strategy=EnsembleStrategy([
        KeywordHeuristicStrategy(),
        ConsensusStrategy(OpenAIProvider()),
    ]),
    audit=AuditConfig(
        store="./audit.db",
        key_store="./keys.db",
        pii_fields=["query", "content"],  # maps to query_encrypted, content_encrypted in schema
    ),
)

# Wrap your existing store — same interface, now shielded
store = shield.wrap(your_chroma_or_pinecone_store)

# Every query now: validates chunks + writes Article 12 audit record
docs = store.similarity_search("what medication is the patient on?")

# Audit record written automatically. Retrieve it:
record = shield.audit.last_record()
print(record.inference_id)   # inf_abc123
print(record.chain_hash)     # sha256:8b3e...
print(record.rfc3161_token)  # base64-encoded TSA proof
```

---

## 5. Decisions

All decisions are final. Nothing is deferred or "to be determined."

### Naming: MemShield
Keep MemShield. It has existing GitHub presence, a benchmark dataset, tests, and Apache-2.0 license. memclave stays an empty directory and is abandoned. One product, one name.

### ZK: No
ZK retrieval proofs are not being built. The audit trail (hash chains + Ed25519 signatures + RFC 3161 timestamps) is sufficient for EU AI Act Article 12. ZK would prove a chunk was derived from a specific document without revealing it — that's Goal B privacy, which the target user does not need. EU regulators have legal authority to see the underlying data; they don't need a zero-knowledge proof. zkml-bench remains a separate standalone project and is not integrated into MemShield.

### Crypto-shredding: Yes
Required. Not optional. Every regulated-industry RAG system processes user queries that contain PII. Without crypto-shredding, the audit log itself becomes a GDPR liability. It is load-bearing for the target user and ships in v0.2. Implementation: AES-256-GCM encryption of PII fields with per-user keys in a separate SQLite key store. On erasure request: delete the key. Chain hashes remain intact. Legal basis: GDPR Article 6(1)(c).

### Timestamp anchoring: RFC 3161, default on
RFC 3161 with Sectigo's free public TSA endpoint as the default. It is legally recognized under EU eIDAS regulation, synchronous (<1s), and free. OpenTimestamps (Bitcoin-anchored) is not included — its legal standing in EU regulatory proceedings is unproven and the ~1 hour confirmation latency makes it impractical for per-inference logging. Every audit log entry gets an RFC 3161 timestamp token by default. It can be disabled for air-gapped deployments.

### ISO/IEC 24970 alignment: Yes
The audit schema field names are explicitly mapped to the ISO/IEC DIS 24970:2025 draft standard. The standard won't be finalized until Q4 2026 but the draft is stable enough to align against, and the approach (append-only, hash-chained, per-event logging) is extremely unlikely to be invalidated. This is a free differentiator: "the only Python RAG library pre-aligned to the draft AI logging standard."

### memshield-bench on HuggingFace: Yes
Publish the benchmark dataset (403 labeled entries, 10 attack types, NeurIPS 2024 data) to HuggingFace as a standalone dataset. Zero cost, drives awareness, establishes credibility, creates a canonical benchmark others will reference. Publish before v0.2 launch.

### Pricing: Always free
Apache-2.0, always free. No paid tier, no managed cloud, no support contracts. The goal is production integrations and case studies, not revenue. If a revenue model emerges, it will be from watching what 25+ users actually ask for — not from a pre-designed pricing page.

### JavaScript/TypeScript: No
Python only. Systima already owns TypeScript. The primary target uses Python. Building JS is scope creep that dilutes focus without a clear advantage.

### Managed cloud: No
Self-hosted only. Regulated-industry developers will not send data to a startup's cloud. Self-hosted is a feature, not a limitation.

### Dashboard/GUI: No
CLI only. `memshield audit verify`, `export`, `inspect`. No UI until the CLI audit export is validated as a real workflow by real users.

### MCP server: Yes, in v0.2
Not deferred to v0.3. The MCP server is a thin wrapper (~100 lines) and it means any Claude Code / Cursor user can add MemShield with one config line. Distribution multiplier, low cost, ships with v0.2.

### claude-relay integration: No
Not building MemShield middleware into claude-relay. They serve different interception points (LLM calls vs. retrieval). Keeping them independent is cleaner. A developer who uses both can wire them together themselves.

---

## 6. Core Capabilities

### 6.1 Memory Poisoning Defense (existing, v0.1)

Already built and benchmarked:

| Strategy | Precision | Recall | Cost |
|---|---|---|---|
| KeywordHeuristic | — | 0% sophisticated attacks | free |
| Consensus (LLM) | 97.1% | 98.6% | LLM call/chunk |
| Ensemble (any_poisoned) | — | 100% | LLM call/chunk |

Attack taxonomy covered (10 types from memshield-bench):
- T1 Instruction Override, T2 Data Exfiltration, T3 Behavioral Drift
- T4 Factual Manipulation, T5 Backdoor Trigger, T6 Experience Poisoning
- T7 Recommendation Poisoning, T8 Configuration Poisoning
- T9 Security Degradation, T10 C2/Persistence

Existing infrastructure:
- `ProvenanceTracker`: SHA-256 hash chain on writes, trust level assignment
- `DriftDetector`: statistical baseline profiling, z-score anomaly detection
- `VectorStoreProxy`: transparent wrap of any store
- Adapters: OpenAI, Chroma, LangChain

### 6.2 Article 12 Audit Log (new, v0.2)

Per-inference tamper-evident record of exactly what was retrieved, from which knowledge base version, at what time. Schema fields mapped to ISO/IEC DIS 24970:2025.

**Schema (per inference):**
```json
{
  "inference_id": "req_xyz_or_uuid",
  "timestamp_iso": "2026-03-09T14:23:01.442Z",
  "timestamp_rfc3161": "<base64-encoded TSA token>",
  "key_id": "sha256 fingerprint of signing public key",
  "user_id": "u_123",
  "query_hash": "sha256:a3f8...",
  "query_encrypted": "<aes-256-gcm ciphertext, null if pii_fields not configured>",
  "knowledge_base_id": "kb_prod_v3",
  "retrieved": [
    {
      "doc_id": "doc_456",
      "chunk_index": 3,
      "content_hash": "sha256:c9d2...",
      "content_encrypted": "<aes-256-gcm ciphertext, null if pii_fields not configured>",
      "score": 0.94,
      "verdict": "clean",
      "trust_level": "verified"
    }
  ],
  "blocked": [
    {
      "content_hash": "sha256:f1a9...",
      "content_encrypted": "<aes-256-gcm ciphertext, or plaintext if pii_fields not configured>",
      "verdict": "poisoned",
      "confidence": 0.97,
      "attack_type": "T1_instruction_override"
    }
  ],
  "chain_hash": "sha256:8b3e...",
  "previous_chain_hash": "sha256:2d7a...",
  "signature": "ed25519:...",
  "iso24970_event_type": "retrieval",
  "iso24970_schema_version": "DIS-2025"
}
```

**`inference_id`:** Developer can provide their own (`store.similarity_search(query, inference_id="req_xyz")`) to correlate with their own request logs. If not provided, MemShield generates a UUID. The `inference_id` is returned on the result object.

**`knowledge_base_id`:** Required. Developer-provided via `AuditConfig(knowledge_base_id="kb_prod_v3")`. No auto-detection — fails loudly if not set.

**Blocked entries:** Store `content_encrypted` (or plaintext if `pii_fields` not configured) for blocked chunks. Regulators need to reconstruct what was blocked, not just that something was.

**Tamper evidence:** Append-only SQLite in WAL mode with a threading.Lock for concurrent writes. Hash-chained entries. Ed25519 signature per record. RFC 3161 timestamp token from Sectigo free TSA on every entry. Each record includes `key_id` (SHA-256 fingerprint of signing public key) so verification works across key rotations.

**Storage backends:** SQLite (default, zero-ops, local) and Postgres (multi-node production).

**Retention enforcement:** Default 180 days (Article 12 minimum). Configurable. Expired entries are replaced with tombstone records — `chain_hash`, `previous_chain_hash`, `inference_id`, and `timestamp` are preserved, all other fields nulled. Chain integrity holds. Tombstones appear in exports so auditors see gaps explicitly rather than a broken chain.

**CLI:**
```bash
memshield audit verify --db ./audit.db
memshield audit export --from 2026-01-01 --format jsonl
memshield audit inspect --inference-id inf_abc123
memshield audit erase-user --user-id u_123
memshield keys rotate --key-file ./memshield.key
```

### 6.3 GDPR Crypto-Shredding (new, v0.2)

Resolves the GDPR Article 17 (erasure, within one month per Article 12(3)) vs. EU AI Act Article 12 (retention, 180 days) conflict.

**Implementation:** AES-256-GCM. When `pii_fields` is configured, query content and chunk content are stored as encrypted ciphertext (`query_encrypted`, `content_encrypted`) alongside their hashes. Keys stored in a separate SQLite database (`keys.db`), keyed by `user_id`. On erasure: delete the key row for that `user_id`. Encrypted fields become unreadable. The hash chain, timestamps, doc IDs, scores, and verdicts remain intact.

**Note on `user_id`:** `user_id` is stored in plaintext in the audit log to enable the `erase-user` operation. Under GDPR Article 4(5), pseudonymous identifiers are still personal data if the developer holds a mapping back to the individual. Treat `user_id` accordingly — use an opaque internal identifier, not an email address or name.

```python
audit = AuditConfig(
    store="./audit.db",
    key_store="./keys.db",
    key_file="./memshield.key",        # auto-generated on first run if absent
    knowledge_base_id="kb_prod_v3",    # required
    pii_fields=["query", "content"],   # maps to query_encrypted, content_encrypted in schema
    tsa_url="https://timestamp.sectigo.com",  # RFC 3161, default
)

shield.audit.erase_user(user_id="u_123")  # destroys key, chain intact
```

**What remains after erasure:** chain hashes, timestamps, RFC 3161 tokens, doc IDs, content hashes, verdicts, scores. All structural audit data is preserved. Only the cleartext PII content is gone.

**Legal basis:** GDPR Article 6(1)(c) — retention necessary for compliance with a legal obligation (EU AI Act Article 12).

### 6.4 MCP Server (new, v0.2)

Exposes MemShield as an MCP tool. Any MCP-compatible agent (Claude Code, Cursor, etc.) gets shielded retrieval and Article 12 audit logging with one config entry.

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

**Tools exposed:**
- `shield_query(query, knowledge_base_id, top_k)` — runs shielded retrieval, returns clean chunks + `inference_id`
- `audit_inspect(inference_id)` — returns the full audit record for an inference
- `audit_verify()` — verifies chain integrity, returns pass/fail + any broken links

---

## 7. What MemShield Is Not

- **Not a full EU AI Act compliance solution.** Article 12 logging is one requirement. MemShield does not produce Annex IV technical documentation, Article 9 risk management plans, or Article 14 human oversight workflows.
- **Covers the retrieval layer only.** MemShield logs what was retrieved and validates it. It does not log the assembled prompt or LLM response. For complete Article 12 coverage, pair MemShield with an LLM observability tool (Langfuse, Helicone, or similar).
- **Not a ZK proof system.** This decision is final. Hash chains + Ed25519 + RFC 3161 is the trust model. No ZK.
- **Not a managed service.** Self-hosted only.
- **Not a TypeScript library.** Python only.

---

## 8. Integration Surface

- Chroma (`chromadb`)
- LangChain vectorstores (`langchain-core`)
- LlamaIndex retrievers
- Pinecone
- pgvector / asyncpg
- Qdrant
- MCP server

---

## 9. Success Metrics

**v0.2 launch (6 weeks):**
- memshield-bench published to HuggingFace before launch
- 5 developers using the audit log in production before August 2026
- 200 GitHub stars within 30 days of launch

**3 months post-launch:**
- 25 production integrations
- 2 case studies with named companies
- 1 developer citing MemShield in an enterprise compliance conversation

---

## 10. Risks

| Risk | Likelihood | Impact | Response |
|---|---|---|---|
| EU AI Act enforcement delayed to Dec 2027 | Medium | Medium | Security value prop stands regardless. Poisoning defense needs no compliance deadline to be useful. |
| Systima adds Python RAG retrieval logging | Medium | High | Ship in 6 weeks. First-mover + benchmark dataset + ISO 24970 alignment matter. |
| ISO/IEC 24970 final standard invalidates our schema | Low | Low | The draft is stable. Hash-chained, per-event, append-only logging will not be contradicted by the final standard. |
| RFC 3161 TSA (Sectigo free endpoint) goes down | Low | Low | Fall back to local timestamp + chain hash. TSA token is additive evidence, not the only tamper protection. |

---

## 11. Roadmap

### Pre-launch (this week)
- Publish memshield-bench to HuggingFace
- Write "EU AI Act Article 12 for RAG systems" technical blog post (owned search term, no competition)

### v0.3 — "Compliance + Integrations" (8-9 weeks)
- Article 12 audit log (SQLite + Postgres)
- GDPR crypto-shredding (AES-256-GCM, per-user keys)
- Ed25519 signing with key rotation
- RFC 3161 timestamp integration (Sectigo free TSA)
- ISO/IEC 24970 schema alignment
- CLI: verify, export, inspect, erase-user, keys rotate
- LlamaIndex adapter
- Pinecone adapter
- pgvector / asyncpg adapter
- Qdrant adapter
- MCP server
- Updated README with EU AI Act framing
