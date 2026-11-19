# mayaKB Backlog

**Last updated:** 2026-09-16
**Baseline:** v0.1 on branch `feat/mayakb-v0.1`. All 27 plan tasks are done: 189 tests pass, and the Docker smoke test passes.
**Related:** [design](../specs/2026-09-16-mayakb-design.md) · [v0.1 plan](../plans/2026-09-16-mayakb-v0.1.md)

**Priority:**
- **P1:** fix before relying on search results
- **P2:** next increment
- **P3:** later or nice to have

**Size:**
- **S:** under half a day
- **M:** 1–2 days
- **L:** more than 2 days

Every item's status is **To do** unless the table says otherwise.

---

## 1. Bugs and quality issues (found running v0.1)

| ID | Item | Why / evidence | Size | Priority |
|---|---|---|---|---|
| MKB-01 | **Keyword search matches only chunks containing every query word** | `websearch_to_tsquery` ANDs all terms (`mayakb/storage/search/postgres.py:199`). In the smoke test, "How long do refunds take?" returned **0 keyword hits**, so hybrid search ran on vectors alone. Fix: match any query term and let `ts_rank_cd` order results. Add a Postgres test with a natural-language question. | S | P1 |
| MKB-02 | Results with a rerank score of ~0 are still returned | `cut.min_k = 2` (`mayakb/core/config.py:60`) forces two results. The smoke test returned the Warranty chunk with score 0.0 next to the right answer. Options: default `min_k` to 1, or drop near-zero results unless `low_confidence` is set. | S | P2 |
| MKB-03 | The same file at two locations is cited with only the first location | The parsed document is keyed by content hash, so the second URI reuses the first parse and its `source_uri`. Listed in the plan as a known limitation. Fix: keep all URIs per document and return them in citations. | M | P3 |
| MKB-04 | Files deleted from the source stay searchable | Ingest only adds or updates documents. A deletion step is needed: compare `Connector.list()` with the `source_map`, then delete index rows and mark the artifacts stale. (A spec non-goal for v0.1.) | M | P2 |
| MKB-05 | A document edited down to empty content leaves its old rows in the index | `index_chunks` skips the upsert when there are no chunks, so old rows for that `source_uri` are never removed. | S | P3 |
| MKB-06 | Evaluation harness has never run on real data | There's no labelled `queries.jsonl` yet, so the beam defaults (`dense_rel 0.9`, `keyword_rel 0.6`, `max_beam 30`) are untuned. Build 30–50 labelled queries for a real corpus, run `mayakb eval`, and tune. Do this after MKB-01. | M | P2 |

## 2. Setup, operations and security

| ID | Item | Why / evidence | Size | Priority |
|---|---|---|---|---|
| MKB-07 | Test database listens on all network interfaces | The documented command publishes `-p 55432:5432` (`README.md:79`, plan Task 14), so `0.0.0.0:55432` with password `test` is reachable on your network. Change it to `-p 127.0.0.1:55432:5432` in the README and plan. | S | P1 |
| MKB-08 | Docker credential helper isn't on the shell PATH on this Mac | `docker pull` failed with `docker-credential-desktop: executable file not found`. The workaround is prepending `/Applications/Docker.app/Contents/Resources/bin` to PATH. Fix Docker Desktop's CLI setup, or add a note to the README. | S | P2 |
| MKB-09 | `ollama-pull` re-runs on every `docker compose run app …` | Harmless (about a second) but noisy. Options: pull once by hand and drop the service, or keep it but skip the `app` dependency after first setup. | S | P3 |
| MKB-10 | Hugging Face downloads are unauthenticated | The reranker download warns `You are sending unauthenticated requests to the HF Hub`. Pass `HF_TOKEN` through `.env` → `app` for higher rate limits. | S | P3 |
| MKB-11 | EC2 deployment has never been tested | README steps, the IAM-role S3 access and `docker-compose.gpu.yml` are unverified on AWS. Do a real run on a CPU instance (and a GPU one later), and write down the instance size and cost. | M | P2 |
| MKB-12 | No CI | Add GitHub Actions: ruff, unit tests, and Postgres tests against a `pgvector/pgvector:pg17` service container. Blocked on MKB-14 (the repo has no remote). | S | P2 |
| MKB-13 | No structured logs or service metrics | Jobs and search print to stdout only. Add structured logs (kb, stage, job_id, timings) and health/latency metrics for the future search API. MKB-37 covers traces and experiments, not these. | M | P3 |
| MKB-14 | Branch not merged, repo has no remote | `main` has no commits; v0.1 lives on `feat/mayakb-v0.1`. Decide: merge locally, or create a GitHub repo and open a PR. | S | P1 |

## 3. Next features

| ID | Item | Why / evidence | Size | Priority |
|---|---|---|---|---|
| MKB-15 | **Search API service** | Every CLI search starts a new container and reloads the reranker (a few seconds). A long-running service (e.g. FastAPI in the `app` container) would keep models loaded and expose `/search`. Spec non-goal for v0.1; the natural next step. | M | P2 |
| MKB-16 | Answer generation and chat UI | Includes an LLM answer with `[Source N]` citations, a streaming "thinking" panel, Corrective RAG grading, and a semantic cache. Needs MKB-15. | L | P2 |
| MKB-17 | SharePoint connector | Microsoft Graph: list and fetch drive items, with etags for incremental ingest. | M | P2 |
| MKB-18 | Enforce document permissions (ACL) | The `acl` field and `search(principals=…)` are already reserved. Fill `acl` in from SharePoint permissions and pass the caller's principals at search time. Depends on MKB-17 and a decision on identity mapping. | L | P3 |
| MKB-19 | Query expansion: HyDE and multi-query | The spec marks these optional and off by default; not built in v0.1. Add them as `SearchOptions` switches inside `Searcher.search`. | S | P3 |
| MKB-20 | NER-on-query third retriever | Run the KB's entity extractor on the query and fuse the entity-matched chunks via RRF. Entity *filtering* already exists. | S | P3 |
| MKB-21 | Contextual chunk enrichment | Prepend an LLM-written context sentence to each chunk before embedding; improves retrieval on fragmentary chunks. Fits as an optional step between chunking and embedding. | M | P3 |
| MKB-22 | Job triggering beyond the CLI | Jobs are plain functions. Wrap them in a queue or scheduler (e.g. RQ, Celery, or a cron that runs `ingest`). Blocked on decision D-2. | M | P3 |
| MKB-23 | CLI additions | `mayakb docs --kb` (list indexed documents), `mayakb show <artifact id>` (inspect a job's output), `mayakb delete --kb --uri` (remove one document). | S | P3 |
| MKB-24 | S3 artifact store | Store artifacts in S3 instead of a Docker volume, so EC2 moves and backups don't need to copy a volume. It sits behind the existing `ArtifactStore` interface. | M | P3 |
| MKB-25 | Knowledge-graph traversal over entities | Entities are stored per chunk but not linked. Build entity co-occurrence and use it for multi-hop retrieval. | L | P3 |
| MKB-35 | **Move model serving from Ollama to vLLM** | vLLM gives much higher throughput and batching on GPUs, and an OpenAI-compatible API. That matters for bulk embedding and for answer generation (MKB-16). See [details below](#mkb-35-move-from-ollama-to-vllm). | L | P2 |
| MKB-36 | **Common gateway for frontier models** | One endpoint for every hosted model (OpenAI, Anthropic, Google, Mistral, Cohere, Voyage, AWS Bedrock, Azure OpenAI) and for local ones (vLLM, Ollama). Provider keys, routing, fallbacks, budgets and usage logs live in the gateway, not in mayaKB. See [details below](#mkb-36-common-gateway-for-frontier-models). | M–L | P2 |
| MKB-37 | **Observability and experimentation with MLflow** | Track every evaluation run (config, beam settings and metrics) as an MLflow experiment, so chunking, embedding, beam and reranker variants can be compared side by side. Trace every search and ingest job step by step. See [details below](#mkb-37-observability-and-experimentation-with-mlflow). | M–L | P2 |

## 4. More implementations (plug-ins)

Each item is one class plus a registry entry and tests. Priority depends on which sources and documents you actually have.

| ID | Interface | Candidates | Size each | Priority |
|---|---|---|---|---|
| MKB-26 | Parser | Azure AI Document Intelligence, AWS Textract, Google Document AI, Unstructured API, Mistral OCR, Docling (local, strong on scanned PDFs and tables) | M | P2 (pick one) |
| MKB-27 | Parser (licensing) | Replace PyMuPDF (**AGPL-3.0**) with pdfplumber or pypdf if mayaKB will be distributed commercially | M | P2 (if distributing) |
| MKB-28 | Chunking strategy | `recursive`, `semantic` (split where embedding similarity drops), `parent_child` (search small chunks, return larger ones) | S–M | P3 |
| MKB-29 | Entity extractor | GLiNER (custom labels), LLM via Ollama, AWS Comprehend, Azure AI Language | S–M | P3 |
| MKB-30 | Embedder | sentence-transformers (local), OpenAI, Voyage, Cohere, AWS Bedrock | S | P3 |
| MKB-31 | Reranker | Cohere Rerank, Voyage, Jina, LLM-as-reranker; `bge-reranker-v2-m3` for multilingual content | S | P3 |
| MKB-32 | Search backend | ParadeDB `pg_search` (true BM25 inside Postgres; **AGPL**), OpenSearch | M–L | P3 |
| MKB-33 | Connector | Google Drive, Confluence, web pages | M | P3 |
| MKB-34 | Verify real LlamaParse and Ollama integration tests | Both are gated and have never run against the real services. The LlamaParse v2 response shape was taken from docs only. | S | P2 |

## 5. Open decisions

| ID | Question | Notes |
|---|---|---|
| D-1 | Is mayaKB standalone, or the document knowledge base for the Maya memory layer? | Affects API shape, tenancy and storage sharing (Maya's design also uses Postgres). |
| D-2 | Which queue or scheduler, if any, should trigger jobs? | Needed for MKB-22. |
| D-3 | Enforce source permissions or not? | `acl` is reserved but not enforced. Needed before MKB-18. |
| D-4 | Will mayaKB be distributed commercially? | Decides MKB-27 (PyMuPDF) and whether ParadeDB (MKB-32) is acceptable. |
| D-5 | Which online parser first? | LlamaParse is built; pick the next based on real documents (MKB-26). |
| D-6 | After moving to vLLM, keep Ollama for local development on macOS? | vLLM can't use the Apple GPU and is impractical on a Mac. Keeping both behind the `Embedder` interface costs little. Needed for MKB-35. |
| D-7 | Which model gateway? | Self-hosted LiteLLM Proxy (data and keys stay in your infrastructure), hosted OpenRouter or Portkey (no ops, but traffic goes through a third party), or cloud-native AWS Bedrock (only Bedrock's models). Needed for MKB-36. |
| D-8 | Which providers may see which knowledge base's content? | Hosted models receive document text (chunks, queries, answers). Needs a per-KB allow-list. Needed for MKB-36. |
| D-9 | Where does the MLflow tracking server run? | A container in the mayaKB compose stack (backed by Postgres + a volume or S3), a shared team server, or a managed service (e.g. Amazon SageMaker with MLflow, or Databricks). Needed for MKB-37. |

---

## Details

### MKB-35: Move from Ollama to vLLM

**Goal:** serve embeddings (and later the answer LLM) with vLLM instead of Ollama, at least on GPU deployments such as EC2.

**Work:**
1. **New embedder `openai_compatible`:** calls `POST /v1/embeddings` with `base_url`, `model` and an optional `api_key`, and keeps the query/document prefix handling for models like nomic. It also works with OpenAI itself and Hugging Face TEI, which covers part of MKB-30.
2. **Compose:**
   - Replace the `ollama` and `ollama-pull` services with a `vllm` service (`vllm/vllm-openai` image) that serves the embedding model.
   - Model weights download from Hugging Face on first start into a volume, so no pull step is needed. That also removes MKB-09.
   - Keep the service unpublished, as today, and set `--api-key` anyway.
3. **One model per server:** a vLLM server process serves one model. Embeddings and the answer LLM (MKB-16) need **separate vLLM containers**, each sized for GPU memory. Decide how much of the GPU each gets.
4. **Reranker (optional):** vLLM can also serve cross-encoder scoring models through its score/rerank endpoints. Evaluate moving the reranker out of the app container, so the app no longer needs PyTorch and the image shrinks.
5. **Re-embedding:** the embedder name changes, so ingest builds a **new index table** and re-embeds every chunk. The old Ollama table stays until it's dropped. Plan the time and GPU cost for large knowledge bases.
6. **Settings:** add `MAYAKB_VLLM_URL` and `MAYAKB_VLLM_API_KEY`. Update `.env.example`, the README and the EC2 notes. The GPU override file becomes the default for vLLM.
7. **Tests:**
   - Unit tests with a mocked `/v1/embeddings`.
   - A gated integration test against a real vLLM server.
   - Re-run `mayakb eval` to compare retrieval quality with the Ollama index before switching the default.

**Risks and constraints:**
- **Hardware:** vLLM needs an NVIDIA GPU in practice (e.g. EC2 g5/g6). Its CPU backend is slow, and it can't use the Apple GPU on a Mac (see D-6).
- **Model packaging:** Ollama's `nomic-embed-text` corresponds to the Hugging Face model `nomic-ai/nomic-embed-text-v1.5`, which needs `trust_remote_code`. Check support and vectors against the vLLM version you pin.
- **Flag changes:** vLLM's command-line flags for embedding/pooling models have changed between releases. Pin a vLLM version and verify the exact serve command.

**Depends on:** D-6. **Related:** MKB-09, MKB-11 (EC2 GPU run), MKB-16, MKB-30, MKB-36.

### MKB-36: Common gateway for frontier models

**Goal:** every model call in mayaKB goes through one OpenAI-compatible gateway, so switching or mixing providers is a config change, not new code.

**Where mayaKB would use it:**

| Use | Today | Through the gateway |
|---|---|---|
| Embeddings | Ollama | OpenAI, Voyage, Cohere, Gemini, Bedrock Titan, or vLLM/Ollama behind the gateway |
| Reranking | local cross-encoder | Cohere / Voyage / Jina rerank, if the gateway supports rerank |
| Answer generation (MKB-16) | none | Hosted frontier models or a local model on vLLM |
| LLM steps (MKB-19, MKB-21, MKB-29) | none | Any chat model, cheap ones for bulk steps |

**Work:**
1. **Gateway service:** add it to Compose as its own container (e.g. LiteLLM Proxy, pending D-7).
   - Provider keys live only in its config or secrets, never in mayaKB's `.env` for the app.
   - It isn't published outside the Docker network, and it requires a key from mayaKB.
2. **Shared clients:** one OpenAI-compatible HTTP client in mayaKB, reused by:
   - the `openai_compatible` embedder (same class as MKB-35),
   - a new `Generator` interface for chat and answers, with streaming for the chat UI,
   - a gateway-backed `Reranker`.
3. **Per-KB model config:** e.g. `embed: {embedder: openai_compatible, model: voyage-3, base_url: gateway}`, plus a `generate:` section for the answer model. Pin exact model ids for reproducibility.
4. **Data governance (D-8):**
   - A per-KB list of allowed providers, enforced before calling the gateway.
   - A KB with no allow-list stays local-only, matching the spec's "online services are opt-in per knowledge base".
5. **Cost and usage:** tag every request with `kb`, `stage` and `job_id` so the gateway attributes spend. Add per-KB budgets or alerts, and record tokens and cost in the job log or search trace.
6. **Reliability:** retries, timeouts and rate limits in the gateway. Fallbacks go to an equivalent model for **chat only** (see risks).
7. **Tests:**
   - Unit tests with a mocked gateway.
   - A gated integration test through a real gateway to at least one hosted provider.
   - A `mayakb eval` comparison of hosted and local embedders and rerankers.

**Risks and constraints:**
- **Never fall back across embedding models:** vectors from different models can't be mixed, and each index table belongs to one embedder and model. A silent fallback would put incompatible vectors in the index or query with the wrong model. Disable fallbacks for embeddings, and fail loudly instead.
- **Data leaves your infrastructure:** chunks and queries go to the providers, so D-8 must be settled first. Check each provider's data-retention terms.
- **Lowest-common-denominator API:** a gateway normalizes to the OpenAI format, so provider features like Anthropic prompt caching or extended thinking may be lost or need passthrough settings. Check before relying on them.
- **Gateway caching:** response caching can store sensitive content. Keep it off by default, or scope it per KB.
- **Embedding gaps:** some providers have no embedding models (e.g. Anthropic recommends Voyage for embeddings), so pick embedders separately from chat models.
- **One more service:** a self-hosted gateway is another container to run, upgrade and secure. A hosted gateway removes that, but adds a third party in the data path (D-7).

**Depends on:** D-7, D-8. **Related:** MKB-16, MKB-19, MKB-21, MKB-29, MKB-30, MKB-31, MKB-35, MKB-37.

### MKB-37: Observability and experimentation with MLflow

**Goal:**
- **Experimentation:** every retrieval experiment is recorded and comparable.
- **Observability:** every search and ingest run can be inspected step by step, in one place.

**Experimentation:**
1. **`mayakb eval` logs to MLflow:** one run per variant (`fixed_beam`, `adaptive_beam`, …), grouped under an experiment per KB.
   - **Params:** chunker and its settings, embedder and model, index table, fusion, all beam and cut settings, reranker model.
   - **Metrics:** `beam_recall`, `avg_beam_size`, `recall_at_k`, `ndcg_at_10`, `mrr`, `latency_p50_ms`, `latency_p95_ms`.
   - **Artifacts:** the KB config YAML, the `queries.jsonl` used (plus its hash), per-query results.
2. **Sweeps:** `mayakb eval --sweep` runs a grid, e.g. `dense_rel` × `max_beam` × reranker, as child runs. This is the tool for MKB-06 beam tuning.
3. **Ingestion variants:** tag each run with the index table (embedder, model, dim) and chunk settings, so chunking and embedding choices can be compared on the same queries.
4. **Later:** evaluate answer quality (MKB-16) with MLflow's GenAI evaluation (LLM-judge scorers for faithfulness and relevance), and version answer prompts in its prompt registry.

**Observability (tracing):**
1. **Search traces:**
   - Spans for `embed`, `retrieve.dense`, `retrieve.keyword`, `fuse`, `beam`, `rerank`, `cut` and `assemble`.
   - Span attributes carry what `results.trace` already has: counts, `kept_by`, timings, `dense_error` / `rerank_error`, `low_confidence`.
2. **Ingest traces:** one trace per document, with a span per job. Attributes: stage, job_id, implementation, status, skipped, duration, error.
3. **Model calls:** once MKB-35/MKB-36 land, LLM and embedding calls become spans with model, tokens and cost.

**Setup:**
- **Settings:** `MAYAKB_MLFLOW_TRACKING_URI` and `MAYAKB_MLFLOW_TRACE_SAMPLE` (search trace sampling rate). **Unset means off:** all MLflow calls become no-ops, so tests and minimal setups need nothing new.
- **Compose:** an optional `mlflow` service under a profile (D-9). Its backend is a separate `mlflow` database in the existing `db` container, and artifacts go on a volume or S3. Publish the UI on `127.0.0.1` only.
- **Dependencies:** a lightweight MLflow client in the app image (check whether the tracing-only or skinny package is enough), so the image doesn't grow much.
- **Tests:** a unit test that logs to a local file/SQLite tracking URI in a temp dir and checks params, metrics and spans; a test that the no-op path works with MLflow unset.

**Risks and constraints:**
- **Sensitive data in traces:** search traces contain query text and chunk text. Add redaction options and a retention policy, and apply the same care as D-8.
- **UI security:** MLflow's built-in auth is basic, so keep the UI private (localhost, VPN or reverse proxy with auth).
- **Search latency:** trace logging must not slow searches. Use async or batched export and sampling.
- **API changes:** MLflow's tracing and GenAI APIs have changed across major versions. Pin a version.
- **Not a metrics system:** MLflow covers experiments and traces, not ops metrics like request rate or error alerts (see MKB-13).

**Depends on:** D-9. **Related:** MKB-06, MKB-13, MKB-15, MKB-16, MKB-35, MKB-36.

---

## Suggested order

1. **MKB-01** keyword search, **MKB-07** test DB binding, **MKB-14** merge or push.
2. **MKB-37** MLflow tracking and tracing, then **MKB-06** evaluation set and beam tuning (logged to MLflow), then **MKB-02** low-score results.
3. **MKB-15** search API, then **MKB-36** model gateway together with **MKB-35** vLLM (they share one OpenAI-compatible client, and vLLM can sit behind the gateway), then **MKB-16** answers and chat UI.
4. **MKB-04** deletion sync and **MKB-11** EC2 run.
5. Plug-ins driven by your real sources and documents: **MKB-17**, **MKB-26**, …
