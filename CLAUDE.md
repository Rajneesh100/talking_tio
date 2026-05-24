# Talking Tío

Local, voice-driven conversational agent. Listens through the mic, reasons with a local LLM, speaks back, and remembers the conversation. Designed for **human-level latency** (first audio out within ~1.5s of speech end) and a **tool-extensible** agent loop so new capabilities (web search, email, calendar, etc.) can be added without changing the core.

Everything runs on the local machine. No cloud calls.

## Tech Stack

- **Language:** Go (single binary, `CGO_ENABLED=1` because whisper.cpp)
- **STT:** whisper.cpp (via official Go bindings), Apple Silicon CoreML acceleration when available
- **VAD:** WebRTC VAD (Go port)
- **Audio I/O:** miniaudio via `gen2brain/malgo` (cross-platform mic + speaker)
- **LLM:** Provider-agnostic. Today: Ollama (streaming NDJSON) or Gemini (streaming SSE). Selected by `LLM_PROVIDER`. Factory in `llm/factory.go`.
- **Embeddings:** Same provider as chat by default. Ollama `nomic-embed-text` and Gemini `text-embedding-004` are both 768-dim (matches the `vector(768)` column).
- **TTS:** macOS `say` (default). Pluggable backend interface — Piper / Kokoro can drop in.
- **Memory:** PostgreSQL + `pgvector` extension. Single Docker container. **Used as a module, not a service** — Go code talks to Postgres over `pgx` directly.
- **Config:** `.env` loaded by `godotenv`

## Project Layout

```
cmd/tio/main.go      — entry point, wires audio loop, agent, memory, tools
agent/               — agent loop, system prompt, structured-output schema
audio/               — mic capture (VAD-gated), speaker queue (sentence-buffered)
stt/                 — whisper.cpp CLI wrapper, model files cached under stt/models/
tts/                 — backend interface + macOS `say` impl
llm/                 — provider interfaces, factory, Ollama + Gemini impls
memory/              — pgvector-backed memory module + migrate.sql
tools/               — Tool interface, registry, built-in tools (clock smoke test)
config/              — .env loader, all config structs
docker-compose.yml   — postgres+pgvector only, port 5433
```

## End-to-End Turn

```
mic ──PCM──▶ VAD ──speech frames──▶ whisper.cpp ──text──▶ agent.Loop()
                                                                │
                                                                ▼
                                                memory.Search(query)  ◀── pgvector ANN
                                                                │
                                                                ▼
                                       llm.ChatStream(system + ctx + memories + user)
                                                                │
                       ┌────────────────────────────────────────┤
                       ▼                                        ▼
            tool_use_needed?                          tokens stream out
            execute tool, append, loop                 │
                                              sentence buffer
                                                       │
                                                       ▼
                                              speaker queue ──▶ say
                                                                │
                       memory.Store(user_text, "user")          │
                       memory.Store(reply,     "agent") ◀───────┘
```

**Latency hides behind streaming**: TTS speaks sentence N while the LLM is still generating sentence N+1. First-audio-out target ≤ 1.5s after speech ends.

## Agent Architecture

The agent loop (`agent/agent.go`) runs up to `maxAgentIterations = 4`:

1. Build messages: `system_prompt + recent_memory_snippets + conversation_window + user_query`
2. Call `llm.ChatJsonStream()` — LLM streams a structured response:
   ```json
   {
     "thought": "...",
     "tool_calls": [{"name": "...", "args": {...}}],
     "response": "..."
   }
   ```
3. If `tool_calls` non-empty: run each tool concurrently, append results as a `tool` role message, loop
4. If `response` present: stream it sentence-by-sentence to the speaker queue, persist to memory, return

Conversation window: last `AGENT_CONTEXT_MAX_MESSAGES` (default 8) turns, kept in-memory per session. System prompt is always index 0.

Streaming + tool calls: the LLM is instructed to emit `tool_calls` first (no spoken `response`) or `response` first (no `tool_calls`). The parser switches modes on the first non-whitespace token of the JSON value, so TTS can start as soon as `"response":` opens — without waiting for the closing brace.

## Memory

**Single Postgres + pgvector container.** No HTTP API. The agent imports `memory` and calls functions.

Schema (`memory/migrate.sql`):
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    source      TEXT NOT NULL,             -- 'user' | 'agent' | 'tool' | 'system'
    text        TEXT NOT NULL,
    embedding   vector(768) NOT NULL
);

CREATE INDEX memories_embedding_ivf ON memories
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

API:
```go
type Store interface {
    Add(ctx context.Context, text, source string) error
    Search(ctx context.Context, query string, k int) ([]Memory, error)
    Recent(ctx context.Context, n int) ([]Memory, error)
}
```

Embeddings are computed on-write via Ollama `nomic-embed-text`. Search uses pgvector cosine distance with an IVFFlat index.

## Tools

Tools implement `tools/tool.go → Tool`:

```go
type Tool interface {
    Descriptor() Descriptor
    Execute(ctx context.Context, args json.RawMessage) (string, error)
}

type Descriptor struct {
    Name        string
    Description string
    Schema      json.RawMessage   // JSON Schema for args
}
```

Register at startup in `tools/registry.go → NewRegistry()` via `r.Register(NewWebSearchTool())`. The system prompt builder (`agent/prompt.go → BuildSystemPrompt(reg)`) auto-includes every registered tool descriptor.

### Adding a New Tool

1. Create `tools/<name>.go` — implement `Tool` interface.
2. Register it in `tools/registry.go → NewRegistry()`.
3. That's it. The system prompt picks it up; the agent loop knows how to invoke it.

No core code touches a specific tool name. The agent's `tool_calls` dispatch is purely registry-driven.

### Planned (not yet implemented) Tools

- `web_search` — DuckDuckGo HTML scrape or SearXNG instance
- `read_email` — IMAP read-only over a Gmail app password
- `calendar_lookup` — CalDAV / Google Calendar API
- `clock` — current time / timezone helper (small, useful for testing the loop)

## Audio Pipeline

**Mic side** (`audio/mic.go`):
- `malgo` opens a 16kHz mono capture stream
- Frames are pushed through `webrtcvad` (aggressiveness 3)
- Once a speech segment ends (≥ 0.6s silence), the buffer is handed to `stt.Transcribe`
- Buffers shorter than 0.4s are discarded — kills most Whisper hallucinations at the source

**Speaker side** (`audio/speaker.go`):
- Worker goroutine reads from a sentence queue
- Each sentence goes to the configured `tts.Backend` (default `say`)
- Sentences are played strictly in order; the agent goroutine blocks on queue drain before releasing the interaction lock

## Concurrency Model

```
main goroutine        — startup, signal handling
mic goroutine         — capture + VAD + STT, emits user_text on chan
agent goroutine       — owns the LLM session, drives the agent loop, sentence-feeds the speaker
speaker goroutine     — pops sentences, blocks on TTS playback
spontaneous goroutine — periodic ticker, posts a synthetic user_text when idle
```

A single `interactionLock` (Go mutex) serializes mic-driven and spontaneous-driven turns. A 5-minute cooldown after voice input mutes the spontaneous goroutine.

## Key Configuration (`.env`)

```
LOG_LEVEL=info

LLM_PROVIDER=ollama                      # ollama | gemini

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_EMBED_MODEL=nomic-embed-text

GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=text-embedding-004

WHISPER_MODEL_PATH=./stt/models/ggml-small.bin
WHISPER_LANGUAGE=en

TTS_BACKEND=say                          # say | none
TTS_VOICE=                               # empty -> macOS system default

POSTGRES_URI=postgres://tio:tio@localhost:5433/tio?sslmode=disable

AGENT_CONTEXT_MAX_MESSAGES=8
AGENT_MAX_ITERATIONS=4
SPONTANEOUS_INTERVAL_SECONDS=35
VOICE_COOLDOWN_SECONDS=300
```

### Adding a new LLM provider

1. Create `llm/<name>.go` with a struct that satisfies both `ChatProvider` and `EmbedProvider`.
2. Add a case to the switch in `llm/factory.go → NewProvider`.
3. Add the provider's config fields to `config/config.go` and `.env.example`.

That's it — `agent.Agent` and `memory.Store` already take the interfaces, not concrete types, so no other code changes.

## Dev Commands

```bash
make dev          # go run ./cmd/tio
make build        # go build -o bin/tio ./cmd/tio
make db-up        # docker compose up -d postgres   (the only container)
make db-down      # docker compose down
make db-shell     # psql into local postgres
make db-migrate   # apply memory/migrate.sql
make whisper-pull # download ggml-small.bin into stt/models/
make lint         # go vet + staticcheck
make test         # go test ./...
```

## Conventions

- All errors are returned, not logged-and-ignored. Top-level (`cmd/tio/main.go`) logs with structured fields (`turn_id=`, `tool=`, `err=`).
- The agent's spoken response is **capped at 3 sentences** by the system prompt. The cap is a prompt instruction, not a hard truncation — keep it that way so the LLM can self-edit.
- TTS-friendly output is enforced in the system prompt: no markdown, no emojis, rich punctuation for prosody (commas, em-dashes, question marks). See `agent/prompt.go`.
- Whisper hallucinations (`"thanks for watching"`, `"see you in the next video"`, `"thank you for your time"`) are filtered in `stt/whisper.go` before they reach the agent. Add new phrases to `stt/hallucinations.go` as you find them.
- The mic VAD pauses **during** TTS playback. Otherwise the mic captures the agent's own voice through the speakers and starts a feedback loop.
- Postgres is the **only** containerized component. Do not split memory or any other concern into a separate service without a strong reason; we tried and the latency / ops overhead wasn't worth it.
- Model weights (whisper, etc.) live under the repo (e.g. `stt/models/`) and are gitignored individually. The folders themselves stay tracked so structure is documented.
