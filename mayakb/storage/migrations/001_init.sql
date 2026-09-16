CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS jobs (
    job_id          text PRIMARY KEY,
    kb              text NOT NULL,
    stage           text NOT NULL,
    input_ids       text[] NOT NULL,
    output_id       text,
    implementation  text NOT NULL,
    params_hash     text NOT NULL,
    status          text NOT NULL CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'skipped')),
    attempts        integer NOT NULL DEFAULT 0,
    error           text,
    started_at      timestamptz,
    finished_at     timestamptz
);

CREATE INDEX IF NOT EXISTS jobs_kb_status ON jobs (kb, status);

-- One row per search index table. A KB gets one table per (embedder, model, dim).
CREATE TABLE IF NOT EXISTS indexes (
    index_id              text PRIMARY KEY,
    kb                    text NOT NULL,
    embedder              text NOT NULL,
    model                 text NOT NULL,
    dim                   integer NOT NULL,
    table_name            text NOT NULL UNIQUE,
    text_search_language  text NOT NULL DEFAULT 'english',
    created_at            timestamptz NOT NULL DEFAULT now(),
    UNIQUE (kb, embedder, model, dim)
);
