package llm

import (
	"fmt"
	"strings"
)

// ProviderConfig is the minimal set of fields the factory needs.
// Decoupled from the config package so llm has no upward import.
type ProviderConfig struct {
	Provider string // "ollama" | "gemini" | future...

	OllamaURL        string
	OllamaModel      string
	OllamaEmbedModel string

	GeminiAPIKey     string
	GeminiModel      string
	GeminiEmbedModel string
}

// NewProvider constructs both a ChatProvider and an EmbedProvider for the
// selected backend. Today they're the same underlying client; if you later
// want chat=Gemini + embed=Ollama (local-only embeddings), add a second
// factory call site and swap one of the returned values.
//
// To add a new provider: implement ChatProvider + EmbedProvider in a new file,
// add a case here. Nothing else in the codebase needs to change.
func NewProvider(cfg ProviderConfig) (ChatProvider, EmbedProvider, error) {
	switch strings.ToLower(cfg.Provider) {
	case "", "ollama", "local":
		c := NewOllamaClient(cfg.OllamaURL, cfg.OllamaModel, cfg.OllamaEmbedModel)
		return c, c, nil
	case "gemini":
		if cfg.GeminiAPIKey == "" {
			return nil, nil, fmt.Errorf("llm: GEMINI_API_KEY required for provider %q", cfg.Provider)
		}
		c := NewGeminiClient(cfg.GeminiAPIKey, cfg.GeminiModel, cfg.GeminiEmbedModel)
		return c, c, nil
	default:
		return nil, nil, fmt.Errorf("llm: unknown LLM_PROVIDER %q", cfg.Provider)
	}
}
