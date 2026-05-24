package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/rajneesh/talking_tio/audio"
	"github.com/rajneesh/talking_tio/llm"
	"github.com/rajneesh/talking_tio/memory"
	"github.com/rajneesh/talking_tio/tools"
)

// memWriteTimeout caps detached memory writes so a slow embedding API can't
// pile up forever. Generous because the turn ctx is no longer governing.
const memWriteTimeout = 10 * time.Second

// storeAsync writes to memory on a detached goroutine. We don't want barge-in
// to cancel the user-input save (the user really did say it), and we don't
// want to block the turn on the embedding round-trip.
func storeAsync(mem *memory.Store, text, source string) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), memWriteTimeout)
		defer cancel()
		if err := mem.Add(ctx, text, source); err != nil {
			fmt.Fprintf(os.Stderr, "memory: add %s: %v\n", source, err)
		}
	}()
}

type AgentOutput struct {
	Thought   string     `json:"thought,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	Response  string     `json:"response,omitempty"`
}

type ToolCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type Agent struct {
	LLM           llm.ChatProvider
	Mem           *memory.Store
	Reg           *tools.Registry
	MaxIterations int
	MaxContext    int
	systemPrompt  string
	history       []llm.Message
}

func New(chat llm.ChatProvider, mem *memory.Store, reg *tools.Registry, maxIter, maxCtx int) *Agent {
	sys := BuildSystemPrompt(reg)
	return &Agent{
		LLM:           chat,
		Mem:           mem,
		Reg:           reg,
		MaxIterations: maxIter,
		MaxContext:    maxCtx,
		systemPrompt:  sys,
		history:       []llm.Message{{Role: llm.RoleSystem, Content: sys}},
	}
}

// Turn runs one user-input → spoken-response cycle. userText is the
// transcribed input. Speaker is the sentence-buffered TTS sink — the agent
// feeds it the final "response" field as soon as the LLM emits it.
//
// Returns the final spoken text (or empty if the agent had nothing to say).
func (a *Agent) Turn(ctx context.Context, userText string, speaker *audio.Speaker) (string, error) {
	// Order matters: search BEFORE storing, otherwise the just-stored row
	// shows up as the top match (cosine ≈ 1 + BM25 hit) and the model sees
	// the user's own current input as a "memory".
	mems, _ := a.Mem.Search(ctx, userText, 3)

	// Detached store after Search so a barge-in cancelling turnCtx doesn't
	// lose what the user just said.
	storeAsync(a.Mem, userText, "user")
	if len(mems) > 0 {
		sideLog("memory", "%d fetched (top %.2f)", len(mems), mems[0].Score)
		for _, m := range mems {
			continuation("#%d %s (%s, %.2f) %q",
				m.ID,
				m.Timestamp.Local().Format("15:04:05"),
				m.Source,
				m.Score,
				firstWords(m.Text, 20),
			)
		}
	} else {
		sideLog("memory", "none")
	}

	a.appendUser(userText, mems)

	var finalResponse string
	for i := 0; i < a.MaxIterations; i++ {
		raw, err := a.LLM.ChatStream(ctx, a.history, nil)
		if err != nil {
			return "", fmt.Errorf("agent: llm: %w", err)
		}
		a.history = append(a.history, llm.Message{Role: llm.RoleAssistant, Content: raw})

		out, err := parseOutput(raw)
		if err != nil {
			// JSON envelope malformed. Don't speak the raw output — it'd
			// include {"thought":"..."}, schema noise, etc. Log and bail.
			fmt.Fprintf(os.Stderr, "agent: parse failed (%v); raw=%q\n", err, raw)
			break
		}

		// Print thought to terminal for visibility, never to TTS.
		if out.Thought != "" {
			sideLog("thought", "%s", out.Thought)
		}

		// Speak whatever the model produced THIS iteration before doing
		// anything else. When tool_calls is also set, this becomes the
		// "Let me check that..." preamble that fills the silence while
		// the tool runs.
		if out.Response != "" {
			speaker.Feed(out.Response)
			finalResponse = out.Response // last iteration wins for memory
		} else if len(out.ToolCalls) > 0 {
			// Model forgot the preamble; insert a generic one so the user
			// doesn't sit through silence while the tool executes.
			speaker.Feed("One sec.")
		}

		if len(out.ToolCalls) > 0 {
			results := a.runTools(ctx, out.ToolCalls)
			a.history = append(a.history, llm.Message{
				Role:    llm.RoleTool,
				Content: results,
			})
			continue
		}
		break
	}

	speaker.Flush()

	if finalResponse != "" {
		storeAsync(a.Mem, finalResponse, "agent")
	}

	a.trimHistory()
	return finalResponse, nil
}

func (a *Agent) appendUser(userText string, mems []memory.Memory) {
	var sb strings.Builder
	if len(mems) > 0 {
		sb.WriteString("[relevant memories]\n")
		for _, m := range mems {
			sb.WriteString(fmt.Sprintf("- (%s, %.2f) %s\n", m.Source, m.Score, m.Text))
		}
		sb.WriteString("\n")
	}
	sb.WriteString(userText)
	a.history = append(a.history, llm.Message{Role: llm.RoleUser, Content: sb.String()})
}

func (a *Agent) runTools(ctx context.Context, calls []ToolCall) string {
	var sb strings.Builder
	for _, c := range calls {
		t, ok := a.Reg.Get(c.Name)
		if !ok {
			sideLog("tool", "%s → not found", c.Name)
			sb.WriteString(fmt.Sprintf("tool %q: not found\n", c.Name))
			continue
		}
		res, err := t.Execute(ctx, c.Args)
		if err != nil {
			sideLog("tool", "%s → error: %v", c.Name, err)
			sb.WriteString(fmt.Sprintf("tool %q: error: %v\n", c.Name, err))
			continue
		}
		sideLog("tool", "%s → %s", c.Name, truncate(res, 80))
		sb.WriteString(fmt.Sprintf("tool %q result: %s\n", c.Name, res))
	}
	return sb.String()
}

// trimHistory keeps the system prompt plus the most recent MaxContext
// non-system messages.
func (a *Agent) trimHistory() {
	if a.MaxContext <= 0 {
		return
	}
	if len(a.history) <= a.MaxContext+1 {
		return
	}
	keep := a.history[len(a.history)-a.MaxContext:]
	a.history = append([]llm.Message{{Role: llm.RoleSystem, Content: a.systemPrompt}}, keep...)
}

func parseOutput(raw string) (AgentOutput, error) {
	raw = strings.TrimSpace(raw)
	// Models sometimes wrap JSON in ```json ... ``` fences.
	if strings.HasPrefix(raw, "```") {
		raw = strings.TrimPrefix(raw, "```json")
		raw = strings.TrimPrefix(raw, "```")
		raw = strings.TrimSuffix(raw, "```")
		raw = strings.TrimSpace(raw)
	}
	var out AgentOutput
	if err := json.Unmarshal([]byte(raw), &out); err == nil {
		return out, nil
	}

	// Strict parse failed — most often this is a truncated stream where the
	// response field arrived fine but later fields were cut off. Try to
	// recover the response (and thought, if present) with a regex so we can
	// still speak something useful instead of swallowing the whole turn.
	if response := extractStringField(raw, "response"); response != "" {
		out.Response = response
		out.Thought = extractStringField(raw, "thought")
		fmt.Fprintln(os.Stderr, "agent: recovered partial response from truncated stream")
		return out, nil
	}

	return out, fmt.Errorf("parse failed and no response field recoverable")
}

// extractStringField pulls a single JSON string field out of raw text without
// requiring the surrounding object to be well-formed. Handles backslash
// escapes inside the string. Returns "" if the field isn't found.
func extractStringField(raw, name string) string {
	pattern := fmt.Sprintf(`"%s"\s*:\s*"((?:[^"\\]|\\.)*)"`, regexp.QuoteMeta(name))
	re := regexp.MustCompile(pattern)
	m := re.FindStringSubmatch(raw)
	if len(m) < 2 {
		return ""
	}
	// Unescape JSON string escapes (\n, \", \\, \uXXXX) by round-tripping
	// through json.Unmarshal of a synthetic quoted string.
	var s string
	if err := json.Unmarshal([]byte(`"`+m[1]+`"`), &s); err == nil {
		return s
	}
	return m[1]
}
