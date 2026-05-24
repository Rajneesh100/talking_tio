.PHONY: start setup dev build tidy lint test \
        ensure-whisper whisper-pull ensure-nowplaying ensure-ytdlp \
        db-up db-down db-shell db-migrate \
        clean

GO        ?= go
BIN       := bin/tio
WHISPER_MODEL := stt/models/ggml-small.bin

# ===== one-shot =====

# `make start` — everything: install deps, pull model, start postgres, run agent.
start: setup dev

# `make setup` — make sure all external prereqs exist. Idempotent.
setup: ensure-whisper whisper-pull ensure-ytdlp ensure-nowplaying db-up
	@echo
	@echo "Setup complete."

# ===== individual targets =====

dev:
	$(GO) run ./cmd/tio

build:
	mkdir -p bin
	$(GO) build -o $(BIN) ./cmd/tio

tidy:
	$(GO) mod tidy

lint:
	$(GO) vet ./...

test:
	$(GO) test ./...

# Install whisper.cpp CLI via Homebrew if missing.
ensure-whisper:
	@if command -v whisper-cli >/dev/null 2>&1; then \
		echo "[ok] whisper-cli already installed: $$(command -v whisper-cli)"; \
	else \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "ERROR: brew not found. Install Homebrew first: https://brew.sh"; \
			exit 1; \
		fi; \
		echo "Installing whisper-cpp via brew..."; \
		brew install whisper-cpp; \
	fi

# Install yt-dlp via Homebrew if missing — used by youtube_music to resolve
# search queries to a specific videoId so the browser opens watch?v=… (auto-
# plays) instead of dropping the user on a search results page.
ensure-ytdlp:
	@if command -v yt-dlp >/dev/null 2>&1; then \
		echo "[ok] yt-dlp already installed"; \
	else \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "WARN: brew not found, skipping yt-dlp install (youtube_music will fall back to search)"; \
		else \
			echo "Installing yt-dlp via brew..."; brew install yt-dlp; \
		fi; \
	fi

# Install nowplaying-cli via Homebrew if missing — used by stop_music to pause
# any media playing through macOS's Now Playing system. Optional: stop_music
# falls back to closing browser tabs if this isn't available.
ensure-nowplaying:
	@if command -v nowplaying-cli >/dev/null 2>&1; then \
		echo "[ok] nowplaying-cli already installed"; \
	else \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "WARN: brew not found, skipping nowplaying-cli (stop_music will rely on tab-close only)"; \
		else \
			echo "Installing nowplaying-cli via brew..."; brew install nowplaying-cli; \
		fi; \
	fi

# Download ggml-small.bin if missing.
whisper-pull:
	@mkdir -p stt/models
	@if [ -f $(WHISPER_MODEL) ]; then \
		echo "[ok] $(WHISPER_MODEL) already present ($$(du -h $(WHISPER_MODEL) | cut -f1))"; \
	else \
		echo "Downloading $(WHISPER_MODEL) (~466 MB)..."; \
		curl -L --fail \
			-o $(WHISPER_MODEL) \
			https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin; \
	fi

# Bring up Postgres + pgvector. Idempotent (docker compose up -d is a no-op
# if already running). Waits until pg is accepting connections.
db-up:
	@docker compose up -d postgres >/dev/null
	@printf "Waiting for postgres"
	@until docker exec tio_postgres pg_isready -U tio -d tio >/dev/null 2>&1; do \
		printf "."; sleep 0.5; \
	done
	@echo " ready (localhost:5433)"

db-down:
	docker compose down

db-shell:
	docker exec -it tio_postgres psql -U tio -d tio

db-migrate:
	docker exec -i tio_postgres psql -U tio -d tio < memory/migrate.sql

clean:
	rm -rf bin
