# Alexa-nxt

Local-first voice agent in Go. Hears you through the mic, watches you through
the camera, reasons with a local LLM (Ollama) or Gemini, talks back through
macOS `say`, remembers the conversation in Postgres + pgvector, and has tools
to play music, search the web, check the time, and look back through its own
memory.

For the architecture and conventions in detail, see [CLAUDE.md](CLAUDE.md).

> **Platform support — macOS only (for now).** TTS uses the macOS `say`
> binary, the `stop_music` tool drives the browser via AppleScript, and the
> setup script assumes Homebrew. The rest of the stack (Go agent, mic
> capture, whisper, vision sidecar, Postgres) is cross-platform — porting
> would mean swapping `tts/say.go` for a Linux backend (Piper) or Windows
> backend (PowerShell `System.Speech`), and replacing the AppleScript bits
> in `tools/media_control.go` with `playerctl` / `xdotool` on Linux. Both
> tracked as future work.

---

## Quick start (three commands)

Prereqs once: macOS, Homebrew, Docker Desktop, Go 1.22+, and either a running
local Ollama (with `gpt-oss:20b` + `nomic-embed-text` pulled) or a Gemini API
key in `.env`.

```bash
make setup        # one-shot: brew deps, model downloads, vision venv, postgres
make dev          # terminal #1 — Angela listens and talks
make vision-up    # terminal #2 — opens the camera so Angela can also see
```

That's the whole loop. `make setup` is idempotent so you can re-run it after
any deps change.

### What `make setup` actually does

| Step | What | Roughly |
|------|------|---------|
| brew | installs `whisper-cpp`, `yt-dlp`, `nowplaying-cli` if missing | seconds |
| whisper model | downloads `stt/models/ggml-small.bin` (466 MB) if missing | first run only |
| python venv | creates `vision/.venv` with python 3.10/11/12 + mediapipe + ultralytics + opencv | first run only |
| vision models | downloads YOLO + MediaPipe face + hand landmarker models (18 MB) | first run only |
| postgres | `docker compose up -d postgres` + applies `memory/migrate.sql` | seconds |

After that, `make dev` starts the agent, and (optionally) `make vision-up`
in a second terminal turns on the camera. macOS prompts for camera permission
the first time vision runs.

---

## What Angela can do

| Capability | Backed by |
|---|---|
| Speech in | `whisper-cpp` + WebRTC VAD via `gen2brain/malgo` (mic) |
| Speech out | macOS `say` (uses your system default voice — set a Siri voice in System Settings → Accessibility → Spoken Content → Manage Voices) |
| Chat LLM | Ollama or Gemini, picked via `LLM_PROVIDER` in `.env` |
| Long-term memory | Postgres + pgvector hybrid (semantic + BM25) keyword search |
| Engagement state | Wake-word gated; goes to sleep during music; LLM also picks `skip` for song lyrics / side-talk |
| Tools | `clock`, `youtube_music`, `stop_music`, `memory_search`, `memory_recent`, `web_search` |
| Vision (optional) | YOLO objects + MediaPipe face mesh + hand landmarks + OpenCV face recognition against `vision/images/<name>.jpg` |

---

## Repo layout

```
cmd/tio/main.go       entry point — wires audio, agent, memory, tools, vision path
agent/                agent loop, JSON envelope, engagement state machine, wake words, prompt
audio/                mic capture (malgo + VAD) and sentence-buffered speaker queue
stt/                  whisper.cpp CLI wrapper + hallucination filter
                      stt/models/   pinned whisper ggml model
tts/                  Backend interface + macOS `say` impl
llm/                  Ollama + Gemini providers behind a ChatProvider/EmbedProvider interface
memory/               pgvector store + hybrid search + migrate.sql
tools/                Tool interface, Registry, + every concrete tool
vision/               Python sidecar (observe.py) + Go consumer (context.go)
                      vision/models/   YOLO + face + hand model files
                      vision/images/   reference photos for face recognition
config/               .env loader
docker-compose.yml    only postgres (pgvector/pgvector:pg16)
Makefile              three-command UX: setup / dev / vision-up
```

---

## Adding a tool

1. Create `tools/<name>.go` implementing `tools.Tool`. See `tools/clock.go` for the minimal pattern.
2. Register it in `cmd/tio/main.go` with `reg.Register(tools.NewYourTool())`.
3. The agent's system prompt auto-includes every registered tool's descriptor — nothing else changes.

The agent calls tools by emitting a JSON envelope `{"tool_calls":[{"name":"...","args":"..."}], "response":"..."}`. The result of each tool is appended as a `tool`-role message and the agent loops up to `AGENT_MAX_ITERATIONS` times before answering.

---

## Adding a face for recognition

1. Drop a clear frontal photo into `vision/images/<name>.jpg` (the filename, without extension, becomes the recognized name).
2. Restart `make vision-up`. You'll see `✓ loaded face: <name>` on startup.
3. From then on the agent's visual context will read `name: <name>` instead of `Unknown` when that person is in frame.

`vision/images/*.jpg|jpeg|png` is gitignored on purpose — personal photos shouldn't get committed.

---

## Provider switch

```env
# .env
LLM_PROVIDER=gemini          # or "ollama"

# Ollama settings (when LLM_PROVIDER=ollama)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_EMBED_MODEL=nomic-embed-text

# Gemini settings (when LLM_PROVIDER=gemini)
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=gemini-embedding-001
```

Same `Agent` consumes either — the provider lives behind `llm.ChatProvider` / `llm.EmbedProvider` interfaces and is picked by the factory in `llm/factory.go`.

---

## Useful sub-targets

`make setup` covers all of these — listed here for ad-hoc work:

```
make whisper-pull    # download stt/models/ggml-small.bin only
make vision-setup    # create vision/.venv only (without model downloads)
make vision-pull     # download YOLO + MediaPipe models only
make db-up           # start postgres only
make db-down         # stop and remove postgres container
make db-shell        # psql into the running postgres
make db-migrate      # re-apply memory/migrate.sql (idempotent)
make vision-status   # pretty-print the latest visual_context.json
make build           # go build -o bin/tio ./cmd/tio
make tidy / lint / test
```

---

## Conventions

See [CLAUDE.md](CLAUDE.md) for the full set. Highlights:

- Memory is a module, not a service. Only Postgres runs in Docker.
- The LLM's spoken response is kept TTS-friendly via the system prompt (no markdown, no emojis, rich punctuation for prosody).
- Whisper hallucinations are filtered in `stt/hallucinations.go` before reaching the agent.
- Model weights are downloaded into the repo (`stt/models/`, `vision/models/`) and gitignored individually so the directory structure stays tracked.
- Visual context is a HINT to the LLM, never a verdict — audio content still wins ties.
- Music sleep: while a YouTube Music song is playing, the agent requires a wake phrase to engage, so it doesn't talk over lyrics.
