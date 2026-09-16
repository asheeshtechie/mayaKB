# mayaKB

A knowledge base you can run anywhere Docker runs. Documents go through separate, re-runnable
ingestion jobs; search is hybrid (vector + keyword) with a dynamic beam and a reranker, and every
result carries a citation (file, pages, section).

Design: [`docs/specs/2026-09-16-mayakb-design.md`](docs/specs/2026-09-16-mayakb-design.md)

## How it works

**Ingestion** runs six jobs per document. Each is a plain Python function behind a swappable
interface, saves its output, and is skipped when that output already exists:

| Job | Interface | Built in |
|---|---|---|
| `read_source` | Connector | `local`, `s3` |
| `understand_document` | Parser | `local` (PDF, DOCX, Markdown, text), `llamaparse` |
| `chunk_document` | ChunkingStrategy | `fixed`, `by_section` |
| `extract_entities` | EntityExtractor | `spacy` |
| `embed_chunks` | Embedder | `ollama` |
| `index_chunks` | SearchBackend | `postgres` (pgvector + full-text search) |

**Retrieval**: vector and keyword search run in parallel with filters inside the SQL, results are
merged with Reciprocal Rank Fusion, the **dynamic beam** keeps only the candidates both searches are
confident about, only those go to the **reranker**, and a final cut picks how many to return.

## Run with Docker

```sh
cp .env.example .env          # set POSTGRES_PASSWORD
cp configs/example.yaml configs/docs.yaml   # edit: kb: docs, and your source
docker compose up -d          # db + ollama (pulls nomic-embed-text on first start)
docker compose build app
docker compose run --rm app mayakb ingest --kb docs
docker compose run --rm app mayakb search "How long do refunds take?" --kb docs --trace
```

Three containers: `app` (Python), `ollama`, `db` (Postgres + pgvector). No ports are published;
Postgres and Ollama are reachable only inside the compose network. Files for the `local` connector go
in `./sources` (mounted read-only at `/sources`).

**macOS:** Docker can't use the Apple GPU, so run Ollama natively (`ollama serve`) and in `.env` set
`COMPOSE_PROFILES=` and `MAYAKB_OLLAMA_URL=http://host.docker.internal:11434`.

**AWS EC2:** install Docker and the compose plugin, copy the repo and `.env`, then the same commands.
S3 access uses the instance's IAM role (no keys in `.env`). On GPU instances add
`-f docker-compose.gpu.yml`. Data lives in Docker volumes on the instance disk: back up with
`pg_dump` plus a copy of the `artifacts` volume. To use RDS instead of the `db` container, point
`MAYAKB_DATABASE_URL` at RDS (pgvector is supported there).

End-to-end check: `scripts/smoke.sh`.

## CLI

```
mayakb db init                                   create/upgrade the schema (runs on container start)
mayakb ingest --kb <kb>                          all stages for new or changed documents
mayakb run <stage> <input ids...> --kb <kb>      one job
mayakb jobs --kb <kb> [--status failed]          recent jobs
mayakb retry <job_id>                            re-run a job
mayakb search "<query>" --kb <kb> [--source-prefix P] [--entity E] [--beam fixed] [--no-rerank] [--trace] [--json]
mayakb eval --kb <kb> --queries queries.jsonl    fixed vs adaptive beam, reranker on/off, dense/keyword/hybrid
```

Knowledge bases are configured in `configs/<kb>.yaml` (see `configs/example.yaml`). Connection
details and secrets come only from environment variables: `MAYAKB_DATABASE_URL`, `MAYAKB_OLLAMA_URL`,
`MAYAKB_DATA_DIR`, `MAYAKB_CONFIGS_DIR`, `LLAMA_CLOUD_API_KEY`.

## Development

```sh
uv venv && uv pip install -e ".[ner,rerank,dev]"
pytest                                   # unit tests: no network, no models, no database
```

Postgres tests need a throwaway database (it is wiped):

```sh
docker run -d --name mayakb-test-db -e POSTGRES_PASSWORD=test -p 55432:5432 pgvector/pgvector:pg17
MAYAKB_TEST_DATABASE_URL=postgresql://postgres:test@localhost:55432/postgres pytest -m postgres
```

Integration tests call real services and run only when enabled: `LLAMA_CLOUD_API_KEY`,
`MAYAKB_TEST_OLLAMA_URL`, `MAYAKB_TEST_RERANKER=1`.

## License

MIT. Note that PyMuPDF, used by the `local` parser, is licensed AGPL-3.0.
