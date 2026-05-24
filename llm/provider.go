package llm

import "context"

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

// ChatProvider streams an assistant reply. onChunk is invoked for every
// non-empty content delta; the full assembled reply is returned.
type ChatProvider interface {
	ChatStream(ctx context.Context, msgs []Message, onChunk func(string)) (string, error)
}

// EmbedProvider returns a fixed-dimension embedding for the given text.
// The dimension must match the Postgres `vector(N)` column (default 768).
type EmbedProvider interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}
