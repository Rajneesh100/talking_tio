.PHONY: start setup dev build tidy lint test \
        ensure-whisper whisper-pull ensure-nowplaying ensure-ytdlp \
        db-up db-down db-shell db-migrate \
        vision-setup vision-pull vision-up vision-status \
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

# ===== vision sidecar (optional) =====
# Run vision-setup once, then vision-up in a separate terminal alongside `make dev`.
# Camera permission is prompted by macOS the first time Python opens cam 0.
#
# Vision deps (mediapipe + ultralytics) only support Python 3.10/3.11/3.12 —
# NOT 3.13+. We resolve a compatible interpreter explicitly so the venv stays
# usable regardless of which python3 your shell happens to resolve to.

VISION_PY := $(shell command -v python3.11 2>/dev/null || command -v python3.12 2>/dev/null || command -v python3.10 2>/dev/null)

vision-setup:
	@if [ -z "$(VISION_PY)" ]; then \
		echo "ERROR: need python3.10, 3.11, or 3.12 on PATH."; \
		echo "Install with: brew install python@3.11"; \
		exit 1; \
	fi
	@if [ -d vision/.venv ] && ! vision/.venv/bin/python -c "import sys; sys.exit(0 if sys.version_info[:2] in [(3,10),(3,11),(3,12)] else 1)" >/dev/null 2>&1; then \
		echo "vision/.venv uses incompatible Python; recreating with $(VISION_PY)..."; \
		rm -rf vision/.venv; \
	fi
	@if [ ! -d vision/.venv ]; then \
		echo "Creating vision/.venv with $(VISION_PY)..."; \
		$(VISION_PY) -m venv vision/.venv; \
	fi
	vision/.venv/bin/pip install --upgrade pip
	vision/.venv/bin/pip install -r vision/requirements.txt
	@echo
	@echo "Vision deps installed. Run: make vision-up"

YOLO_MODEL := vision/models/yolov8n.pt
YOLO_URL   := https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

FACE_MODEL := vision/models/face_landmarker.task
FACE_URL   := https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

vision-pull:
	@mkdir -p vision/models
	@if [ -f $(YOLO_MODEL) ]; then \
		echo "[ok] $(YOLO_MODEL) already present ($$(du -h $(YOLO_MODEL) | cut -f1))"; \
	else \
		echo "Downloading $(YOLO_MODEL) (~6 MB) ..."; \
		curl -L --fail -o $(YOLO_MODEL) $(YOLO_URL); \
	fi
	@if [ -f $(FACE_MODEL) ]; then \
		echo "[ok] $(FACE_MODEL) already present ($$(du -h $(FACE_MODEL) | cut -f1))"; \
	else \
		echo "Downloading $(FACE_MODEL) (~4 MB) ..."; \
		curl -L --fail -o $(FACE_MODEL) $(FACE_URL); \
	fi

vision-up: vision-pull
	@if [ ! -x vision/.venv/bin/python ]; then \
		echo "ERROR: run 'make vision-setup' first"; exit 1; \
	fi
	vision/.venv/bin/python vision/observe.py

vision-status:
	@if [ -f vision/visual_context.json ]; then \
		cat vision/visual_context.json | python3 -m json.tool; \
	else \
		echo "no snapshot yet — start the sidecar with 'make vision-up'"; \
	fi

clean:
	rm -rf bin
