CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    source      TEXT        NOT NULL,
    text        TEXT        NOT NULL,
    embedding   vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS memories_ts_idx ON memories (ts DESC);

CREATE INDEX IF NOT EXISTS memories_embedding_ivf
    ON memories USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Hybrid search: full-text index for BM25-style keyword retrieval. The
-- generated tsvector keeps itself in sync with `text` automatically.
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS text_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

CREATE INDEX IF NOT EXISTS memories_text_tsv_idx
    ON memories USING GIN (text_tsv);
