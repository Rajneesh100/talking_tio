# Talking Tío

Local-first voice + vision conversational agent. Listens through the mic, watches through the webcam, reasons with a local or hosted LLM, speaks back, and remembers the conversation. Designed for **human-level latency** (first audio out within ~1.5s of speech end) and a **tool-extensible** agent loop so new capabilities can be added without changing the core.

Default LLM is local (Ollama). Gemini is optional and switchable via env. Camera + mic data never leaves the machine; only Gemini chat tokens (if enabled) hit the network.

## Three-command UX

```bash
make setup        # one-shot: brew deps, model downloads, vision venv, postgres
make dev          # terminal #1 — Angela listens and talks
make vision-up    # terminal #2 — opens the camera so Angela can also see
```

`make setup` is idempotent — re-run anytime after deps drift. Anything inside `setup` is also exposed as a sub-target (`make whisper-pull`, `make vision-pull`, `make db-up`, etc.) for ad-hoc work.

## Tech Stack

- **Language:** Go (single binary, `CGO_ENABLED=1` because miniaudio + webrtcvad cgo deps)
- **STT:** `whisper-cpp` CLI subprocess, model `stt/models/ggml-small.bin`
- **VAD:** WebRTC VAD (Go port at `maxhawkins/go-webrtcvad`)
- **Audio I/O:** miniaudio via `gen2brain/malgo` (mic) + macOS `say` subprocess (speaker)
- **LLM:** Provider-agnostic. Ollama (streaming NDJSON) or Gemini (streaming SSE, JSON-schema constrained, thinking disabled). Picked by `LLM_PROVIDER`. Factory in `llm/factory.go`.
- **Embeddings:** Same provider as chat by default. Ollama `nomic-embed-text` and Gemini `gemini-embedding-001` both produce 768-dim vectors (matches the `vector(768)` column).
- **TTS:** macOS `say` (default). Picks the system default voice when `TTS_VOICE` is empty — the only path to a downloaded Siri voice, since `say -v "<name>"` is gated by Apple.
- **Memory:** PostgreSQL + `pgvector` + a generated `tsvector` column for **hybrid search** (vector cosine + BM25 keyword). Single Docker container. Used as a module — Go talks to Postgres over `pgx` directly.
- **Vision (optional):** Python sidecar at `vision/observe.py`. YOLOv8n for objects, MediaPipe Tasks API for face mesh + hand landmarks, OpenCV Haar + grayscale histograms for face recognition against `vision/images/<name>.jpg`. Writes `vision/visual_context.json` at ~1Hz; the Go agent reads it on each Turn.
- **Config:** `.env` loaded by `godotenv`.

## Project Layout

```
cmd/tio/main.go      — entry point, wires audio loop, agent, memory, tools, vision path
agent/               — agent loop, JSON envelope, engagement state machine, wake words, prompt
audio/               — mic capture (malgo + VAD) and sentence-buffered speaker queue
stt/                 — whisper.cpp CLI wrapper + hallucination filter
                       stt/models/   pinned whisper ggml model (gitignored)
tts/                 — Backend interface + macOS `say` impl
llm/                 — provider interfaces, factory, Ollama + Gemini impls
memory/              — pgvector store + hybrid search + migrate.sql
tools/               — Tool interface, Registry, every concrete tool
vision/              — python sidecar (observe.py) + Go consumer (context.go)
                       vision/models/   YOLO + face + hand models (gitignored)
                       vision/images/   reference photos for face recognition (gitignored)
config/              — .env loader, all config structs
docker-compose.yml   — postgres+pgvector only, port 5433
Makefile             — 3-command UX (setup / dev / vision-up) + sub-targets
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

### Tools currently registered (see `cmd/tio/main.go`)

| Tool | Purpose |
|------|---------|
| `clock` | Current ISO local time. |
| `youtube_music` | Resolve query via `yt-dlp` → open `music.youtube.com/watch?v=…` (autoplay). Auto-stops the previous song first. |
| `stop_music` | `nowplaying-cli pause` + close YouTube Music tabs across Chrome/Brave/Arc. |
| `memory_search` | Hybrid (vector + BM25) search over the `memories` table. |
| `memory_recent` | Last N memories chronologically. |
| `web_search` | DuckDuckGo Instant Answer API — Wikipedia-quality summary of top result. |

### Planned (not yet implemented)

- `read_email` — IMAP read-only over a Gmail app password
- `calendar_lookup` — CalDAV / Google Calendar API
- `set_reminder` — local timer + future spontaneous prompt

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

## Engagement state machine

The agent is not always "on". `Agent.state` is one of `StateIdle` / `StateActive`, plus a `musicActive` flag that overrides state to require a wake phrase whenever a song is playing.

Wake phrases live in `agent/wake.go` (`hey angela`, `ok angela`, `angela`, `hey`, `play`, `change`, `stop`, …). Any of those entering the transcribed input flips state IDLE → ACTIVE and strips the wake portion before the LLM sees it.

Once ACTIVE, fluid back-and-forth keeps state ACTIVE until either:
- `IdleTimeout` (90 s) elapses without a `respond` decision, or
- the model picks `"engagement": "dismiss"` (explicit goodbye).

Per-turn, the LLM also picks `"engagement": "skip"` for utterances that look like song lyrics, side-talk, or fragmentary noise. Skipped turns are still stored in memory (passive listening) but produce no audio.

While `youtube_music` is playing, `musicActive=true` forces wake-phrase-required mode even in ACTIVE — Angela won't talk over a song. `stop_music` clears the flag.

## Concurrency Model

```
main goroutine        — startup, signal handling, owns the conversation loop
mic goroutine         — malgo audio callback + VAD framing, emits Segments on a chan
agent goroutine       — per-turn: whisper subprocess, LLM stream, tool calls, mem writes
speaker goroutine     — pops sentences, blocks on `say` subprocess
turn-cancel ctx       — barge-in cancels in-flight HTTP read + kills `say` via CommandContext
```

No central interaction lock — barge-in (a new mic segment arriving while a turn is running) cancels the previous turn's context, which cascades into the LLM HTTP body read and the speaker's per-sentence `say` ctx. Memory writes are detached with their own timeout so they survive cancellation.

## Key Configuration (`.env`)

```
LOG_LEVEL=info

LLM_PROVIDER=ollama                      # ollama | gemini

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_EMBED_MODEL=nomic-embed-text

GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=gemini-embedding-001
# Gemini chat is forced to structured JSON output (responseSchema + thinkingBudget=0)
# so the agent's JSON envelope is always parseable.

VISION_CONTEXT_PATH=./vision/visual_context.json

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

The mandatory three:

```bash
make setup        # everything one-time: brew deps, model downloads, vision venv, postgres
make dev          # run the agent
make vision-up    # (separate terminal) start the camera sidecar
```

Sub-targets `make setup` already runs — listed for ad-hoc work:

```bash
# brew installs (all idempotent)
make ensure-whisper       # brew install whisper-cpp
make ensure-ytdlp         # brew install yt-dlp
make ensure-nowplaying    # brew install nowplaying-cli

# model downloads
make whisper-pull         # ggml-small.bin (466 MB) into stt/models/
make vision-pull          # yolov8n.pt + face/hand landmarker .task files

# vision sidecar
make vision-setup         # create vision/.venv + pip install (no model download)
make vision-status        # pretty-print latest vision/visual_context.json

# postgres
make db-up                # bring up postgres+pgvector
make db-down              # tear it down
make db-shell             # psql into the container
make db-migrate           # re-apply memory/migrate.sql (idempotent)

# go
make build                # go build -o bin/tio ./cmd/tio
make tidy / lint / test   # go mod tidy / go vet / go test ./...
```

## Conventions

- All errors are returned, not logged-and-ignored. Top-level (`cmd/tio/main.go`) logs with structured fields (`turn_id=`, `tool=`, `err=`).
- The agent's spoken response is **capped at 3 sentences** by the system prompt. The cap is a prompt instruction, not a hard truncation — keep it that way so the LLM can self-edit.
- TTS-friendly output is enforced in the system prompt: no markdown, no emojis, rich punctuation for prosody (commas, em-dashes, question marks). See `agent/prompt.go`.
- Whisper hallucinations (`"thanks for watching"`, `"see you in the next video"`, `"thank you for your time"`) are filtered in `stt/whisper.go` before they reach the agent. Add new phrases to `stt/hallucinations.go` as you find them.
- The mic VAD pauses **during** TTS playback. Otherwise the mic captures the agent's own voice through the speakers and starts a feedback loop.
- Visual context is a HINT to the LLM, never a verdict — audio still wins ties. The prompt explicitly tells Angela she CAN see, and includes worked examples for "now" / "recent (5s) motion" / "since last turn delta" lines.
- Music sleep: while a `youtube_music` tool call is active, the agent forces wake-phrase-required mode so Angela doesn't talk over lyrics. `stop_music` clears it.
- Model weights are downloaded into the repo (`stt/models/`, `vision/models/`) and gitignored individually so the folder structure stays tracked. Files >100 MB (i.e. `ggml-small.bin`) need Git LFS if you want to push them to GitHub.
- Postgres is the **only** containerized component. Do not split memory or any other concern into a separate service without a strong reason; we tried and the latency / ops overhead wasn't worth it.
- Model weights (whisper, etc.) live under the repo (e.g. `stt/models/`) and are gitignored individually. The folders themselves stay tracked so structure is documented.
