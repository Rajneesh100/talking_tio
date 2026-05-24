package config

import (
	"fmt"
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

type Config struct {
	LogLevel string

	// Selects which LLM provider drives both chat and embeddings.
	// "ollama" (default) or "gemini".
	LLMProvider string

	OllamaURL        string
	OllamaModel      string
	OllamaEmbedModel string

	GeminiAPIKey     string
	GeminiModel      string
	GeminiEmbedModel string

	WhisperModelPath string
	WhisperLanguage  string

	TTSBackend string
	TTSVoice   string

	PostgresURI string

	// VisionContextPath is the JSON snapshot file produced by
	// vision/observe.py. Empty disables visual context injection.
	VisionContextPath string

	AgentContextMaxMessages    int
	AgentMaxIterations         int
	SpontaneousIntervalSeconds int
	VoiceCooldownSeconds       int
}

func Load() (*Config, error) {
	_ = godotenv.Load()
	c := &Config{
		LogLevel:                   env("LOG_LEVEL", "info"),
		LLMProvider:                env("LLM_PROVIDER", "gemini"),
		OllamaURL:                  env("OLLAMA_URL", "http://localhost:11434"),
		OllamaModel:                env("OLLAMA_MODEL", "llama3.2"),
		OllamaEmbedModel:           env("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
		GeminiAPIKey:               env("GEMINI_API_KEY", ""),
		GeminiModel:                env("GEMINI_MODEL", "gemini-2.5-flash"),
		GeminiEmbedModel:           env("GEMINI_EMBED_MODEL", "text-embedding-004"),
		WhisperModelPath:           env("WHISPER_MODEL_PATH", "./stt/models/ggml-small.bin"),
		WhisperLanguage:            env("WHISPER_LANGUAGE", "en"),
		TTSBackend:                 env("TTS_BACKEND", "say"),
		TTSVoice:                   env("TTS_VOICE", ""),
		PostgresURI:                env("POSTGRES_URI", "postgres://tio:tio@localhost:5433/tio?sslmode=disable"),
		VisionContextPath:          env("VISION_CONTEXT_PATH", "./vision/visual_context.json"),
		AgentContextMaxMessages:    envInt("AGENT_CONTEXT_MAX_MESSAGES", 8),
		AgentMaxIterations:         envInt("AGENT_MAX_ITERATIONS", 4),
		SpontaneousIntervalSeconds: envInt("SPONTANEOUS_INTERVAL_SECONDS", 35),
		VoiceCooldownSeconds:       envInt("VOICE_COOLDOWN_SECONDS", 300),
	}
	return c, nil
}

func env(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			panic(fmt.Sprintf("config: %s must be int, got %q", key, v))
		}
		return n
	}
	return fallback
}
