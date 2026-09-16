# mayaKB — Knowledge Base Design (v0.1, quick design)

**Status:** approved (2026-09-16). Build plan: `docs/plans/2026-09-16-mayakb-v0.1.md`
**Scope:** ingestion pipeline + retrieval library, packaged as Docker containers. Chat UI and
answer generation are a later part.

---

## 1. Goals and non-goals

**Goals**
- Ingest documents from different sources through separate, re-runnable **jobs**.
- Every stage sits behind an **interface**, with implementations picked by name in config.
- Retrieval uses **hybrid search** (dense + keyword), a **dynamic beam**, and a **reranker**,
  and returns ranked chunks with citations and a trace of what happened.
- Runs with no API keys by default. Online services are opt-in per knowledge base.
- Runs as **three containers** (app, Ollama, Postgres) with one compose file that works the
  same on a laptop and on an AWS EC2 instance.

**Non-goals for v0.1**
- Chat UI, answer generation, streaming "thinking" panel (next part).
- A network API for search. v0.1 is a library + CLI inside the app container.
- Knowledge graph traversal. NER output is stored and usable for filtering, not traversed.
- Enforcing source-system permissions (SharePoint/S3). An `acl` field is reserved in the
  contracts, index rows and `search()` so this can be added later without changing contracts
  or re-embedding. See §7.
- Queue/scheduler deployment. Jobs are plain functions; a queue can wrap them later.
- Deleting documents from the index when they disappear from the source.

---

## 2. Architecture overview

```
INGESTION (jobs)                         job function          interface
                                         ────────────────────  ─────────────────
source (local, S3, SharePoint…)
   │
   ▼
RawDocument          ◄────────────────── read_source           Connector
   │
   ▼
DocModel             ◄────────────────── understand_document   Parser
   │
   ▼
ChunkSet             ◄────────────────── chunk_document        ChunkingStrategy
   ├──────────────► EntitySet  ◄──────── extract_entities      EntityExtractor
   └──────────────► VectorSet  ◄──────── embed_chunks          Embedder
                        │
                        ▼
search index         ◄────────────────── index_chunks          SearchBackend

RETRIEVAL (library)
query ─► prepare ─► hybrid retrieve (dense + keyword, wide & cheap) ─► fuse
                                                                        │  e.g. 70 candidates
                                                                        ▼
                                                      dynamic beam (keep only likely ones)
                                                                        │  e.g. 9 candidates
                                                                        ▼
                                          context + citations ◄─ cut ◄─ rerank
```

Every job reads its inputs from and writes its output to an **ArtifactStore**. Nothing is
passed between stages in memory, so any stage can run alone, retry, or re-run with new settings.

**Where things run** (details in §6):

```
┌──────────────────────── docker network: mayakb ────────────────────────┐
│  app (Python)  ──── embeddings ────►  ollama                            │
│      │                                                                  │
│      └────────── SQL ─────────────►  db (Postgres + pgvector)           │
│                                       search index, job log, manifest   │
│  volumes: artifacts (/data), hf-cache │ ollama-models │ pgdata          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Ingestion

### 3.1 Stages and interfaces

| Stage (job function) | Interface | Input | Output artifact |
|---|---|---|---|
| `read_source` | `Connector` | source ref from `Connector.list()` | `RawDocument` |
| `understand_document` | `Parser` | `RawDocument` id | `DocModel` |
| `chunk_document` | `ChunkingStrategy` | `DocModel` id | `ChunkSet` |
| `extract_entities` | `EntityExtractor` | `ChunkSet` id | `EntitySet` |
| `embed_chunks` | `Embedder` | `ChunkSet` id | `VectorSet` |
| `index_chunks` | `SearchBackend` | `ChunkSet` + `VectorSet` (+ `EntitySet` if present) | index rows |

`index_chunks` is separate from `embed_chunks` so vectors can be loaded into a different
search backend without paying to embed again.

```python
class Connector:
    name: str
    def list(self) -> Iterator[SourceRef]: ...           # uri, etag, modified_at, size
    def fetch(self, ref: SourceRef) -> RawDocument: ...   # bytes + mime + source metadata

class Parser:
    name: str
    def parse(self, raw: RawDocument) -> DocModel: ...

class ChunkingStrategy:
    name: str
    def chunk(self, doc: DocModel) -> list[Chunk]: ...

class EntityExtractor:
    name: str
    def extract(self, chunks: list[Chunk]) -> list[Entity]: ...

class Embedder:
    name: str; model: str; dim: int
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...  # handles query/document prefixes

class SearchBackend:
    def upsert(self, rows: list[IndexRow], index: IndexId) -> None: ...
    def dense_search(self, vector, k: int, filters: Filters, index: IndexId) -> list[Hit]: ...
    def keyword_search(self, text: str, k: int, filters: Filters, index: IndexId) -> list[Hit]: ...
```

Implementations register by name (`@register("parser", "llamaparse")`) and are selected in
the knowledge base config.

### 3.2 Contracts (pydantic models)

- **`RawDocument`**: `id` (sha256 of bytes), `source_uri`, `connector`, `mime`, `etag`,
  `modified_at`, `acl`, `metadata`. Bytes stored alongside. `acl` is reserved: a list of
  principal ids allowed to read the document, or `None` for unrestricted; v0.1 connectors
  always set `None`. A separate `source_map` records `source_uri → content id`, so the same
  file at two locations is stored once but cited correctly.
- **`DocModel`**: the common output of every parser.
  `doc_id`, `parser`, `title`, `metadata`, `elements[]` where each element is
  `{id, type: heading|paragraph|list|table|figure|code, text, page, section_path[], level?, bbox?}`.
  Tables are rendered to markdown in `text`. Chunkers, NER and embedders only ever see this.
- **`Chunk`**: `chunk_id`, `doc_id`, `text`, `element_ids[]`, `page_start`, `page_end`,
  `section_path[]`, `position` (order within doc), `parent_id?`, `acl` (copied from its
  document), `metadata`.
- **`Entity`**: `chunk_id`, `text`, `label`, `normalized`, `start`, `end`, `score`.
- **`VectorSet`**: `chunk_ids[]`, `embedder`, `model`, `dim`, vectors (stored as `.npy`).

### 3.3 Job model

Every job has the same signature:

```python
def chunk_document(input_id: str, *, kb: KBConfig, store: ArtifactStore) -> JobResult: ...
```

- **Deterministic output ids.** `output_id = hash(stage, input_ids, implementation name +
  version, canonical params)`. If the artifact already exists, the job returns it without
  doing work. Changing the chunking strategy re-chunks but reuses parsed documents; this also
  avoids paying an online parser twice for the same file.
- **Atomic writes.** Artifacts are written to a temp file and renamed into place, so a crash
  or two runners on the same document never leave a half-written artifact.
- **Job log** (Postgres table `jobs`): `job_id, kb, stage, input_ids, output_id,
  implementation, params_hash, status (pending|running|succeeded|failed|skipped), attempts,
  error, started_at, finished_at`.
- **Failure isolation.** One bad document fails its own jobs only; the batch continues.
  Network implementations retry with backoff (3 attempts), then mark `failed`.
- **Runner.** v0.1 ships an in-process runner that walks the stage graph per document
  (`extract_entities` and `embed_chunks` may run in parallel). A queue or scheduler can call the
  same job functions later with no changes.
- **Incremental ingest.** `Connector.list()` returns etag/modified time; unchanged sources are
  skipped before fetching.

### 3.4 Storage

- **ArtifactStore** interface; v0.1 implementation is the filesystem, mounted in Docker as
  the `artifacts` volume at `/data`:
  `/data/<kb>/{raw,doc_model,chunks,entities,vectors}/<id>.{bin,json,npy}`. S3 implementation later.
- **Postgres holds everything queryable:** the search index, the job log, and the index
  manifest. Schema is created by numbered SQL migration files applied with `mayakb db init`
  (idempotent; runs automatically when the app container starts).
- **One index per (kb, embedder, model, dim).** Vectors from different models can't be mixed,
  and a pgvector column has a fixed dimension, so each index is its own table
  (`idx_<kb>_<embedder hash>`). The `indexes` manifest table maps kb + embedder + model + dim to
  its table. Switching embedders builds a new table beside the old one; retrieval reads the
  manifest to pick the matching query embedder.
- **Index table columns:** `chunk_id` (PK), `doc_id`, `text`, `embedding vector(dim)`,
  `tsv tsvector` (generated from `text`, language per KB, default `english`), `source_uri`,
  `title`, `page_start`, `page_end`, `section_path text[]`, `entities text[]`, `acl text[]`
  (null = unrestricted), `metadata jsonb`.
- **Table indexes:** HNSW on `embedding` (cosine); GIN on `tsv`, `entities` and `acl`.

### 3.5 Knowledge base config

```yaml
kb: finance-docs
source: {connector: s3, bucket: acme-docs, prefix: reports/}
parse:  {parser: local}                 # or llamaparse, azure_di, textract, ...
chunk:  {strategy: by_section, max_tokens: 512, overlap_tokens: 50}
ner:    {extractor: spacy, model: en_core_web_sm, labels: [PERSON, ORG, PRODUCT, DATE]}
embed:  {embedder: ollama, model: nomic-embed-text}
index:  {backend: postgres, text_search_language: english}
```

Connection details and secrets come from environment variables only, never from KB config:
`MAYAKB_DATABASE_URL`, `MAYAKB_OLLAMA_URL`, `MAYAKB_DATA_DIR`, `LLAMA_CLOUD_API_KEY`, …
On EC2, S3 access uses the instance's IAM role, so no AWS keys are set.

### 3.6 First implementations (confirmed)

| Interface | v0.1 | Later candidates |
|---|---|---|
| Connector | `local`, `s3` | `sharepoint` (Microsoft Graph), Google Drive, Confluence, web |
| Parser | `local` (PyMuPDF for PDF, python-docx for DOCX, plain text/markdown), `llamaparse` | Azure AI Document Intelligence, AWS Textract, Google Document AI, Unstructured API, Mistral OCR, Docling |
| ChunkingStrategy | `fixed` (tokens + overlap), `by_section` (split at headings, tables kept whole, oversize sections split) | `recursive`, `semantic`, `parent_child` |
| EntityExtractor | `spacy` | GLiNER, LLM via Ollama, AWS Comprehend, Azure AI Language |
| Embedder | `ollama` | sentence-transformers, OpenAI, Voyage, Cohere, Bedrock |
| SearchBackend | `postgres`: pgvector HNSW for dense, Postgres full-text search for keyword, filters and `acl` in the same SQL | ParadeDB `pg_search` (true BM25 inside Postgres; AGPL), OpenSearch |

Online implementations send document content to a third party; they are opt-in per KB.

---

## 4. Retrieval

### 4.1 API

```python
results = search(
    query="What was Q2 churn in EMEA?",
    kb="finance-docs",
    filters={"source_prefix": "reports/2026/", "entities": ["EMEA"]},   # optional
    options=SearchOptions(beam="adaptive", reranker="cross_encoder", token_budget=3000),
    principals=None,   # reserved: when set, only chunks whose acl is None or overlaps are searched
)
# results.chunks          -> ranked chunks with scores + citation {source_uri, title, pages, section_path}
# results.low_confidence  -> True when nothing cleared the rerank threshold
# results.trace           -> candidate counts, beam size and why each was kept, timings per step
```

### 4.2 Pipeline

1. **Prepare.** Normalize the query. Embed it with the index's embedder (`embed_query`).
   Optional, off by default: HyDE or multi-query variants.
2. **Hybrid retrieve.** Run `dense_search` and `keyword_search` in parallel, each returning
   `depth` hits with raw scores. This step is cheap, so it casts a wide net.
   - *Dense:* pgvector cosine distance over the HNSW index. Uses pgvector ≥ 0.8 iterative index
     scans so a filtered search still returns `depth` hits instead of fewer.
   - *Keyword:* `websearch_to_tsquery` against `tsv`, ranked with `ts_rank_cd`. This is not
     true BM25; RRF only uses ranks, so the difference matters less. ParadeDB is the upgrade
     path if the eval shows keyword recall is weak.
   - Filters (`source_prefix`, `entities`, `acl`) are **inside both SQL queries**, never
     applied afterwards, because post-filtering shrinks the candidate list unpredictably.
   - Optional third retriever: run NER on the query and match chunks tagged with those entities.
3. **Fuse.** Reciprocal Rank Fusion (`k=60`) by default, since dense and keyword scores aren't
   on the same scale. Weighted fusion over min-max normalized scores is the alternative. Each
   fused candidate keeps its dense and keyword scores for the beam.
4. **Dynamic beam** (§4.3): keep only the candidates likely to matter.
5. **Rerank** only the beam with a `Reranker`. Scores are passed through a sigmoid so
   thresholds read as 0–1.
6. **Cut** to the final set (§4.3).
7. **Assemble context.** Merge adjacent chunks from the same section, drop near-duplicates,
   order by rank, stop at the token budget, attach citations.

### 4.3 Dynamic beam

The reranker is the most expensive step in retrieval, so it should only see chunks that could
plausibly answer the query. Hybrid retrieval casts a wide, cheap net (e.g. 70 fused
candidates). The **beam** is the smaller set that goes on to the reranker (e.g. 9). Its size
changes per query: small when the first-pass results clearly point at a few chunks, larger
when they don't.

**Selecting the beam.** Walk the fused list in order and keep a candidate if any of these hold:
- *Agreement:* it is in the top `agree_top` of **both** the dense and keyword results.
- *Strong dense match:* its dense score is ≥ `dense_rel` × the best dense score for this query.
- *Strong keyword match:* its keyword score is ≥ `keyword_rel` × the best keyword score for this query.

Then trim to at most `max_beam`. If fewer than `min_beam` were kept, top up from the fused
order. Scores are compared against the best hit for the same query because raw dense and
keyword scores shift from query to query and model to model.

The trace records the beam size and why each candidate got in, e.g.
`{fused: 70, beam: 9, kept_by: {agreement: 4, dense: 3, keyword: 1, top_up: 1}}`.

**Final cut** after reranking. Final `k` is the smallest of:
- items with rerank score ≥ `min_score`,
- items with score ≥ `relative_to_top` × the best rerank score,
- what fits in the token budget,

and always at least `min_k`, at most `max_k`. If no item passes `min_score`, return the top
`min_k` and set `results.low_confidence = true`.

**Optional fallback (off by default).** With `fallback_widen: true`, if nothing in the beam
passes `min_score`, the next `max_beam` fused candidates are reranked once. This trades
latency for recall on queries where the beam was too narrow.

Default settings (tuned with the eval harness, §4.5):

```yaml
beam:
  policy: adaptive        # or fixed: send the top `size` fused candidates (baseline, size: 30)
  depth: 50               # hits per retriever
  min_beam: 5
  max_beam: 30
  agree_top: 20
  dense_rel: 0.9
  keyword_rel: 0.6
  fallback_widen: false
cut: {min_k: 2, max_k: 10, min_score: 0.3, relative_to_top: 0.5}
```

`BeamPolicy` is an interface with `FixedBeam` (the baseline to beat) and `AdaptiveBeam`.

### 4.4 Reranker

```python
class Reranker:
    name: str
    def score(self, query: str, texts: list[str]) -> list[float]: ...   # raw scores
```

| v0.1 | Later |
|---|---|
| `cross_encoder` via sentence-transformers: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, English) or `BAAI/bge-reranker-v2-m3` (multilingual, slower) | Cohere Rerank, Voyage rerank, Jina reranker, LLM-as-reranker |

Runs inside the app container on CPU. Model weights are cached in the `hf-cache` volume.
Thresholds in §4.3 are reranker-specific and live under the reranker's config.

### 4.5 Evaluation harness

Needed to tune the beam, not optional. Input: `queries.jsonl` with relevant chunk or document
ids per query. Reports:
- **beam recall**: how often a relevant chunk made it into the beam (what a narrow beam risks),
- **average beam size** (= reranker calls per query; what the beam saves),
- recall@k, nDCG@10, MRR, latency p50/p95.

Compares `FixedBeam` vs `AdaptiveBeam`, reranker on/off, dense-only vs keyword-only vs hybrid.
The target: a smaller average beam than `FixedBeam` with no drop in recall@k.

### 4.6 Degraded modes

- Reranker unavailable → return fused order, `trace.reranked = false`.
- Query embedder (Ollama) unavailable → keyword-only search, `trace.dense = false`.
- Empty results → empty list with trace; never an exception for "no hits".
- Database unavailable → clear error naming `MAYAKB_DATABASE_URL`; no partial results.

---

## 5. Package layout

```
mayaKB/
  mayakb/
    core/            contracts.py, ids.py (hashing), registry.py, config.py
    ingest/
      connectors/    base.py, local.py, s3.py
      parsers/       base.py, local.py, llamaparse.py
      chunking/      base.py, fixed.py, by_section.py
      ner/           base.py, spacy_ner.py
      embedders/     base.py, ollama.py
      jobs.py        read_source, understand_document, chunk_document,
                     extract_entities, embed_chunks, index_chunks
      runner.py      per-document stage graph
    storage/
      artifacts.py   ArtifactStore + filesystem implementation
      db.py          connection pool, migration runner
      migrations/    001_init.sql (jobs, indexes manifest), …
      joblog.py      jobs table access
      search/        base.py, postgres.py
    retrieval/
      hybrid.py      parallel dense + keyword
      fusion.py      rrf, weighted
      beam.py        FixedBeam, AdaptiveBeam, cut
      rerankers/     base.py, cross_encoder.py
      context.py     merge, dedupe, budget, citations
      search.py      search() entrypoint
      evaluate.py    eval harness
    cli.py
  configs/example.yaml
  docker/Dockerfile
  docker-compose.yml  docker-compose.gpu.yml
  tests/
  pyproject.toml  README.md  LICENSE  .env.example
```

**CLI** (inside the container: `docker compose run --rm app mayakb …`)
- `mayakb db init`: apply migrations (also runs on container start)
- `mayakb ingest --kb finance-docs`: list the source and run all stages for new or changed documents
- `mayakb run <stage> <input_id> --kb finance-docs`: run one job
- `mayakb jobs --kb finance-docs --status failed` / `mayakb retry <job_id>`
- `mayakb search --kb finance-docs "question" --trace`
- `mayakb eval --kb finance-docs --queries queries.jsonl`

---

## 6. Deployment (Docker)

### 6.1 Containers

One compose file, one internal network. Works the same on a laptop and on EC2.

| Service | Image | Role | Volumes | Published ports |
|---|---|---|---|---|
| `app` | built from `docker/Dockerfile` | CLI + job runner; search API / chat UI later | `artifacts:/data`, `hf-cache`, `./configs` (read-only) | none in v0.1 |
| `ollama` | `ollama/ollama` | embeddings (LLM later) | `ollama-models:/root/.ollama` | none |
| `ollama-pull` | `ollama/ollama` | one-shot: pulls the configured models, then exits | — | none |
| `db` | `pgvector/pgvector:pg17` | search index, job log, index manifest | `pgdata:/var/lib/postgresql/data` | none |

- **Start order.** `db` has a `pg_isready` healthcheck; `ollama` has a healthcheck;
  `app` waits for both to be healthy. `ollama-pull` is idempotent (already-pulled models are skipped).
- **Nothing is published by default.** Postgres and Ollama are reachable only on the internal
  network. Ollama has no authentication and must never be exposed. When a search API arrives,
  only its port is published.
- **Config** comes from `.env` (copied from `.env.example`), including the Postgres password.
  Nothing secret is baked into the image.

### 6.2 App image

- `python:3.12-slim`, multi-stage build, runs as a non-root user.
- Installs the **CPU-only PyTorch** build (for the reranker). The default PyTorch wheel on
  Linux bundles several GB of GPU libraries the app doesn't use.
- spaCy `en_core_web_sm` is baked into the image. Reranker weights download on first use into
  the `hf-cache` volume.
- Entrypoint runs `mayakb db init`, then the given command.

### 6.3 Variants

- **macOS development.** Docker on macOS can't use the Apple GPU, so Ollama in a container is
  CPU-only and slow. Run Ollama natively instead: set `MAYAKB_OLLAMA_URL=
  http://host.docker.internal:11434` and start without the `ollama` services (they sit in a
  compose profile, `ollama`, enabled by default in `.env.example`).
- **EC2 with a GPU.** `docker-compose.gpu.yml` gives the `ollama` container the NVIDIA GPU
  (host needs the NVIDIA Container Toolkit). Not needed for v0.1: embeddings and the reranker
  run fine on CPU. It matters once answer generation arrives (e.g. g5/g6 instances).

### 6.4 Moving to EC2

1. Install Docker + the compose plugin on the instance; copy the repo and `.env`.
2. `docker compose up -d` (add `-f docker-compose.gpu.yml` on GPU instances).
3. `docker compose run --rm app mayakb ingest --kb <kb>`.

- **Data** lives in Docker volumes on the instance's EBS disk. Back up with `pg_dump` and a copy
  of the `artifacts` volume (S3 `ArtifactStore` later removes the second step).
- **Moving data from a laptop:** `pg_dump`/`pg_restore` plus copying the `artifacts` volume. Or
  re-run ingest on EC2: it's idempotent, but embeddings and any online parsing are redone.
- **S3 access** uses the instance IAM role; no AWS keys in `.env`.
- **Managed Postgres later:** RDS for PostgreSQL supports pgvector. Point
  `MAYAKB_DATABASE_URL` at RDS and drop the `db` service; no code changes.

---

## 7. Decisions and open questions

**Decided (2026-09-16)**
- **Dynamic beam** = pruning the fused candidates so fewer chunks reach the reranker (§4.3).
- **First implementations** as listed in §3.6.
- **Storage:** Postgres + pgvector for the search index, job log and manifest; filesystem
  volume for artifacts. Replaces the earlier LanceDB + SQLite choice, because those run inside
  the Python process and can't be a separate storage container.
- **Deployment:** three containers (app, Ollama, Postgres) in one compose file, portable to EC2 (§6).
- **Permissions: reserve now, enforce later.** `acl` exists on `RawDocument`, `Chunk`, index
  rows and `search(principals=…)`, but v0.1 connectors leave it `None` and nothing is
  filtered. Adding enforcement later means a connector that fills `acl` in (mainly SharePoint;
  S3 access is controlled by IAM policies, not per-user lists) plus passing `principals` at
  search time. Existing index rows only need their `acl` field updated; nothing is re-embedded.

**Open**
1. **Relation to Maya.** Standalone project, or the document knowledge base for the Maya memory layer?
2. **Job triggering beyond the CLI.** Which queue or scheduler, if any, will call the job functions?

---

## 8. Testing

- Unit tests use fake implementations (fake parser, deterministic hash-based embedder, fake
  reranker, in-memory search backend). No network, no models, no API keys, no database.
- Contract tests: every registered implementation returns valid models (`DocModel`, `Chunk`, …).
- Job tests: same input + params → same output id and a `skipped` job; a failing document does
  not fail the batch; interrupted writes leave no partial artifact.
- Retrieval tests with fixed scores: RRF order; beam keeps agreed and strong candidates, drops
  the rest, respects `min_beam`/`max_beam`, and only the beam reaches the reranker; fallback
  runs only when enabled; cut rules; degraded modes.
- Postgres tests (skipped unless `MAYAKB_TEST_DATABASE_URL` is set; run against the compose
  `db` service): migrations apply twice cleanly; filters and `acl` are inside the SQL; a
  filtered dense search still returns `depth` hits; one table per embedder.
- Other integration tests gated by env: S3 (via moto), LlamaParse (API key), Ollama (running server).
- Docker smoke test: `docker compose up`, ingest a sample folder, run one search, check the
  result has a citation.
