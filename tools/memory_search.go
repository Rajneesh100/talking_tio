package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/rajneesh/talking_tio/memory"
)

// MemorySearchTool gives the agent first-class access to its own memory.
// Use this whenever the user references something past and the auto-injected
// memory header at turn-start didn't surface it.
type MemorySearchTool struct {
	store *memory.Store
}

func NewMemorySearchTool(store *memory.Store) *MemorySearchTool {
	return &MemorySearchTool{store: store}
}

func (MemorySearchTool) Descriptor() Descriptor {
	return Descriptor{
		Name: "memory_search",
		Description: "Search the long-term memory by keyword or semantic similarity. " +
			"Use this when the user references past conversations (\"do you remember…\", " +
			"\"earlier we talked about…\", \"what did I say about X?\") and the memories " +
			"you were given at the start of the turn don't already cover it. " +
			"Pick a query that captures the SUBJECT they're referencing, not the recall " +
			"phrasing — e.g. for \"remember that brick ball thing?\" search for \"brick ball\".",
		Schema: json.RawMessage(`{
            "type": "object",
            "properties": {
                "query":  {"type": "string", "description": "What to search for. Use concrete keywords from the topic."},
                "k":      {"type": "integer", "description": "How many memories to return. Default 5, max 10."}
            },
            "required": ["query"],
            "additionalProperties": false
        }`),
	}
}

type memSearchArgs struct {
	Query string `json:"query"`
	K     int    `json:"k,omitempty"`
}

func (t *MemorySearchTool) Execute(ctx context.Context, raw json.RawMessage) (string, error) {
	var args memSearchArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return "", fmt.Errorf("memory_search: bad args: %w", err)
	}
	if args.Query == "" {
		return "", fmt.Errorf("memory_search: query is required")
	}
	if args.K <= 0 {
		args.K = 5
	}
	if args.K > 10 {
		args.K = 10
	}

	mems, err := t.store.Search(ctx, args.Query, args.K)
	if err != nil {
		return "", fmt.Errorf("memory_search: %w", err)
	}
	return formatMemories(args.Query, mems), nil
}

// formatMemories renders the list as plain text the LLM can read in the next
// iteration. One memory per line, with id/time/source/score/snippet.
func formatMemories(query string, mems []memory.Memory) string {
	if len(mems) == 0 {
		return fmt.Sprintf("No memories matched %q.", query)
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "Found %d memories for %q:\n", len(mems), query)
	for _, m := range mems {
		fmt.Fprintf(&sb, "[#%d %s %s, %.2f] %s\n",
			m.ID,
			m.Timestamp.Local().Format("2006-01-02 15:04"),
			m.Source,
			m.Score,
			collapseWhitespace(m.Text),
		)
	}
	return sb.String()
}

func collapseWhitespace(s string) string {
	return strings.Join(strings.Fields(s), " ")
}
