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
	"github.com/rajneesh/talking_tio/vision"
)

// memWriteTimeout caps detached memory writes so a slow embedding API can't
// pile up forever. Generous because the turn ctx is no longer governing.
const memWriteTimeout = 10 * time.Second

// IdleTimeout — after this much time with no `respond` decision while in
// ACTIVE state, drop back to IDLE so the agent stops speaking up.
const IdleTimeout = 90 * time.Second

// EngagementState controls whether the agent vocalises. In IDLE we only
// passively record what the mic transcribes; the LLM is not called and
// nothing is spoken. ACTIVE is normal conversational mode entered after a
// wake phrase.
type EngagementState int

const (
	StateIdle EngagementState = iota
	StateActive
)

// engagement values that may appear in the JSON envelope from the LLM.
const (
	engagementRespond  = "respond"
	engagementSkip     = "skip"
	engagementDismiss  = "dismiss"
)

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
	Thought    string     `json:"thought,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	Response   string     `json:"response,omitempty"`
	Engagement string     `json:"engagement,omitempty"` // respond | skip | dismiss
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

	// Engagement state machine (Axis 1 of the real-listener design).
	state        EngagementState
	lastEngageAt time.Time

	// VisionPath is the JSON snapshot file written by vision/observe.py.
	// Empty string disables visual context injection.
	VisionPath string
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
		state:         StateIdle, // explicit; iota default already 0
	}
}

// Turn runs one user-input → spoken-response cycle. userText is the
// transcribed input. Speaker is the sentence-buffered TTS sink.
//
// Engagement state machine:
//   - IDLE: every utterance is stored, no LLM call, no audio. Only a wake
//     phrase ("hey angela", etc.) transitions to ACTIVE.
//   - ACTIVE: normal flow, BUT the LLM may decide per-utterance to skip
//     (e.g. song lyrics, side-talk). After IdleTimeout with no respond, we
//     drop back to IDLE.
//
// Returns the final spoken text (or empty if the agent stayed silent).
func (a *Agent) Turn(ctx context.Context, userText string, speaker *audio.Speaker) (string, error) {
	// Always-store: passive recording happens regardless of state.
	storeAsync(a.Mem, userText, "user")

	// State gate — decide whether to engage the LLM at all.
	cleaned, woke := stripWakePhrase(userText)
	switch a.state {
	case StateIdle:
		if !woke {
			sideLog("idle", "passive (no wake): %q", firstWords(userText, 12))
			speaker.Flush() // close the unused speaker cleanly
			return "", nil
		}
		a.state = StateActive
		a.lastEngageAt = time.Now()
		sideLog("wake", "engaged")
		if cleaned == "" {
			// Bare wake ("Angela?") — quick audible ack, no LLM call.
			ack := "Yeah?"
			speaker.Feed(ack)
			speaker.Flush()
			storeAsync(a.Mem, ack, "agent")
			return ack, nil
		}
		userText = cleaned
	case StateActive:
		if time.Since(a.lastEngageAt) > IdleTimeout {
			a.state = StateIdle
			sideLog("idle", "timed out (passive): %q", firstWords(userText, 12))
			speaker.Flush()
			return "", nil
		}
		// If the user re-wakes mid-conversation, strip the phrase so the
		// LLM doesn't see "hey angela" as part of the question.
		if woke {
			userText = cleaned
		}
	}

	// ── from here on, we're committed to running the LLM ──

	// Order matters: search BEFORE storing, otherwise the just-stored row
	// shows up as the top match (cosine ≈ 1 + BM25 hit) and the model sees
	// the user's own current input as a "memory".
	mems, _ := a.Mem.Search(ctx, userText, 3)
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

	var (
		finalResponse string
		spoke         bool
		dismissed     bool
	)
	for i := 0; i < a.MaxIterations; i++ {
		raw, err := a.LLM.ChatStream(ctx, a.history, nil)
		if err != nil {
			return "", fmt.Errorf("agent: llm: %w", err)
		}

		out, perr := parseOutput(raw)
		if perr != nil {
			// JSON envelope malformed. Don't speak the raw output — it'd
			// include {"thought":"..."}, schema noise, etc.
			fmt.Fprintf(os.Stderr, "agent: parse failed (%v); raw=%q\n", perr, raw)
			break
		}

		if out.Thought != "" {
			sideLog("thought", "%s", out.Thought)
		}

		// Honor the engagement decision BEFORE appending to history or
		// speaking. Skipped iterations don't pollute the assistant history
		// — they were "I heard but chose not to engage" moments.
		switch out.Engagement {
		case engagementSkip:
			sideLog("skip", "(staying silent)")
			speaker.Flush()
			return "", nil
		case engagementDismiss:
			dismissed = true
			// fallthrough into respond — we still speak the goodbye.
			fallthrough
		default: // respond (or unset)
			a.history = append(a.history, llm.Message{Role: llm.RoleAssistant, Content: raw})
		}

		if out.Response != "" {
			speaker.Feed(out.Response)
			finalResponse = out.Response
			spoke = true
		} else if len(out.ToolCalls) > 0 {
			speaker.Feed("One sec.")
		}

		if len(out.ToolCalls) > 0 && !dismissed {
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
	if spoke {
		a.lastEngageAt = time.Now()
	}
	if dismissed {
		a.state = StateIdle
		sideLog("dismiss", "back to passive listening")
	}

	a.trimHistory()
	return finalResponse, nil
}

func (a *Agent) appendUser(userText string, mems []memory.Memory) {
	var sb strings.Builder

	// Visual context (if the sidecar is running and the snapshot is fresh).
	// Goes before memories so the model reads the "is this even for me?"
	// hint first.
	if snap, ok := vision.Read(a.VisionPath); ok {
		sideLog("visual", "%s (age %.1fs)", snap.Summary, snap.AgeSeconds)
		sb.WriteString("[visual context]\n")
		sb.WriteString(snap.Summary)
		sb.WriteString("\n\n")
	}

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
		args := normalizeArgs(c.Args)
		res, err := t.Execute(ctx, args)
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

// normalizeArgs handles both providers' tool-call args shapes:
//   - Ollama (json mode): args arrives as a raw JSON object, e.g. {"query":"x"}.
//   - Gemini (structured output): args arrives as a JSON-encoded STRING, e.g. "{\"query\":\"x\"}".
//
// We detect the string case (raw begins with a quote) and unwrap to inner JSON
// so every tool's Execute sees the same shape.
func normalizeArgs(raw json.RawMessage) json.RawMessage {
	trimmed := []byte(strings.TrimSpace(string(raw)))
	if len(trimmed) == 0 {
		return json.RawMessage("{}")
	}
	if trimmed[0] != '"' {
		return trimmed // already an object/array/literal
	}
	var inner string
	if err := json.Unmarshal(trimmed, &inner); err != nil {
		return trimmed
	}
	inner = strings.TrimSpace(inner)
	if inner == "" {
		return json.RawMessage("{}")
	}
	return json.RawMessage(inner)
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
