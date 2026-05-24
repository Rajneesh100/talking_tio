package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/rajneesh/talking_tio/memory"
)

// MemoryRecentTool returns the N most recently stored memories in
// chronological order. Used when the user refers to recent context without
// being specific enough for keyword/semantic search.
type MemoryRecentTool struct {
	store *memory.Store
}

func NewMemoryRecentTool(store *memory.Store) *MemoryRecentTool {
	return &MemoryRecentTool{store: store}
}

func (MemoryRecentTool) Descriptor() Descriptor {
	return Descriptor{
		Name: "memory_recent",
		Description: "Fetch the N most recently stored memories in chronological order. " +
			"Use this when the user refers to recent context without specific keywords " +
			"(\"what did we just talk about?\", \"go back to what you were saying\", " +
			"\"continue\"). Prefer memory_search when there are concrete keywords to search by.",
		Schema: json.RawMessage(`{
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many recent memories to return. Default 5, max 20."}
            },
            "additionalProperties": false
        }`),
	}
}

type memRecentArgs struct {
	N int `json:"n,omitempty"`
}

func (t *MemoryRecentTool) Execute(ctx context.Context, raw json.RawMessage) (string, error) {
	var args memRecentArgs
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &args); err != nil {
			return "", fmt.Errorf("memory_recent: bad args: %w", err)
		}
	}
	if args.N <= 0 {
		args.N = 5
	}
	if args.N > 20 {
		args.N = 20
	}

	mems, err := t.store.Recent(ctx, args.N)
	if err != nil {
		return "", fmt.Errorf("memory_recent: %w", err)
	}
	if len(mems) == 0 {
		return "Memory is empty — no prior conversation stored.", nil
	}

	return formatRecent(mems), nil
}

func formatRecent(mems []memory.Memory) string {
	out := fmt.Sprintf("Last %d memories (newest first):\n", len(mems))
	for _, m := range mems {
		out += fmt.Sprintf("[#%d %s %s] %s\n",
			m.ID,
			m.Timestamp.Local().Format("2006-01-02 15:04"),
			m.Source,
			collapseWhitespace(m.Text),
		)
	}
	return out
}
