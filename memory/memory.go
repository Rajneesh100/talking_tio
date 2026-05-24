package memory

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
)

type Memory struct {
	ID        int64
	Timestamp time.Time
	Source    string
	Text      string
	Score     float32 // populated by Search; 1.0 = identical
}

type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

type Store struct {
	pool *pgxpool.Pool
	emb  Embedder
}

func New(ctx context.Context, postgresURI string, emb Embedder) (*Store, error) {
	pool, err := pgxpool.New(ctx, postgresURI)
	if err != nil {
		return nil, fmt.Errorf("memory: connect: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("memory: ping: %w", err)
	}
	return &Store{pool: pool, emb: emb}, nil
}

func (s *Store) Close() {
	s.pool.Close()
}

func (s *Store) Add(ctx context.Context, text, source string) error {
	vec, err := s.emb.Embed(ctx, text)
	if err != nil {
		return fmt.Errorf("memory: embed: %w", err)
	}
	_, err = s.pool.Exec(ctx,
		`INSERT INTO memories (source, text, embedding) VALUES ($1, $2, $3)`,
		source, text, pgvector.NewVector(vec),
	)
	if err != nil {
		return fmt.Errorf("memory: insert: %w", err)
	}
	return nil
}

// Search runs a hybrid retrieval over the memories table:
//   - vector-space cosine similarity using the ivfflat index, AND
//   - Postgres full-text search (BM25-ish via ts_rank) using a GIN index on
//     the generated text_tsv column.
//
// Top-20 candidates from each branch are unioned, re-scored as
// vec_sim + bm25_rank, and the global top-k returned.
//
// This rescues queries where the user references a memory by literal keywords
// ("remember that brick ball thing?") — vector search alone misses those
// because the embedding of the question is dominated by recall-words, not
// content words.
func (s *Store) Search(ctx context.Context, query string, k int) ([]Memory, error) {
	vec, err := s.emb.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("memory: embed: %w", err)
	}
	v := pgvector.NewVector(vec)

	rows, err := s.pool.Query(ctx, `
		WITH candidate_ids AS (
			(SELECT id FROM memories ORDER BY embedding <=> $1 LIMIT 20)
			UNION
			(SELECT id FROM memories
			 WHERE text_tsv @@ plainto_tsquery('english', $2)
			 ORDER BY ts_rank(text_tsv, plainto_tsquery('english', $2)) DESC
			 LIMIT 20)
		)
		SELECT m.id, m.ts, m.source, m.text,
		       (1 - (m.embedding <=> $1))
		         + COALESCE(ts_rank(m.text_tsv, plainto_tsquery('english', $2)), 0)
		         AS score
		FROM memories m
		WHERE m.id IN (SELECT id FROM candidate_ids)
		ORDER BY score DESC
		LIMIT $3
	`, v, query, k)
	if err != nil {
		return nil, fmt.Errorf("memory: search: %w", err)
	}
	defer rows.Close()

	var out []Memory
	for rows.Next() {
		var m Memory
		if err := rows.Scan(&m.ID, &m.Timestamp, &m.Source, &m.Text, &m.Score); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

func (s *Store) Recent(ctx context.Context, n int) ([]Memory, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT id, ts, source, text
		FROM memories
		ORDER BY ts DESC
		LIMIT $1
	`, n)
	if err != nil {
		return nil, fmt.Errorf("memory: recent: %w", err)
	}
	defer rows.Close()

	var out []Memory
	for rows.Next() {
		var m Memory
		if err := rows.Scan(&m.ID, &m.Timestamp, &m.Source, &m.Text); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}
