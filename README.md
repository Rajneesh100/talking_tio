# Talking Tío

Local, voice-driven conversational agent in Go. Listens through the mic, reasons with a local LLM (Ollama), speaks back through macOS `say`, and remembers the conversation in Postgres + pgvector. Designed for human-level latency and a tool-extensible agent loop.

For the full architecture spec, see [CLAUDE.md](CLAUDE.md).

## Status

Skeleton is in. What works today:

- ✅ Go module builds (`go build ./...`)
- ✅ Ollama streaming chat + embeddings (`llm/ollama.go`)
- ✅ pgvector-backed memory module (`memory/memory.go`)
- ✅ Tool framework + registry + a smoke-test `clock` tool (`tools/`)
- ✅ Agent loop with JSON-envelope output and tool dispatch (`agent/agent.go`)
- ✅ Sentence-buffered TTS speaker (`audio/speaker.go`)
- ✅ macOS `say` TTS backend (`tts/say.go`)
- ✅ Postgres+pgvector docker-compose with auto-applied schema

What's stubbed (TODO markers in source):

- ⬜ `audio/mic.go` — mic capture + VAD-gated segmentation (malgo + webrtcvad)
- ⬜ `stt/whisper.go` — whisper.cpp wrapper

`cmd/tio/main.go` currently reads user input from stdin so you can exercise the LLM → memory → tool → TTS path end-to-end before the audio side lands.

## Quick start

Prereqs: Go 1.22+, Docker, Ollama running locally with `gpt-oss:20b` and `nomic-embed-text` pulled.

```bash
cp .env.example .env          # only if .env doesn't already exist
make db-up                    # postgres+pgvector on localhost:5433 (schema auto-applied)
make dev                      # go run ./cmd/tio
```

You'll get a `> ` prompt. Type something; the agent embeds it, stores it, asks Ollama for a streaming reply, and speaks each sentence through `say` as it's generated.

## Layout

```
cmd/tio/main.go    entry point
agent/             agent loop + system prompt
audio/             sentence-buffered speaker (mic is stubbed)
stt/               whisper.cpp wrapper (stubbed) + hallucination filter
tts/               Backend interface + macOS say
llm/               Ollama HTTP client (streaming chat + embeddings)
memory/            Store interface + pgvector impl + migrate.sql
tools/             Tool interface, Registry, clock smoke-test tool
config/            .env loader
```

## Adding a tool

1. Create `tools/<name>.go` implementing the `Tool` interface (see `tools/clock.go` for the minimal pattern).
2. Register it in `cmd/tio/main.go` with `reg.Register(tools.NewYourTool())`.
3. The agent's system prompt picks it up automatically — no other code change.

The agent calls tools by emitting `{"tool_calls":[{"name":"...","args":{...}}]}` in its JSON response envelope, sees the result appended as a `tool`-role message, and loops up to `AGENT_MAX_ITERATIONS` times before responding.

## Conventions

See [CLAUDE.md](CLAUDE.md) for the full set. Highlights:

- Memory is a module, not a service. Only Postgres runs in Docker.
- The LLM's spoken `response` is enforced TTS-friendly via the system prompt (no markdown, no emojis, rich punctuation for prosody).
- Whisper hallucinations are filtered in `stt/hallucinations.go` before reaching the agent.
- Model weights live under the repo (`stt/models/`) and are gitignored.
