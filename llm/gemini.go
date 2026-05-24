package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

const geminiBase = "https://generativelanguage.googleapis.com/v1beta"

type GeminiClient struct {
	APIKey     string
	Model      string
	EmbedModel string
	HTTP       *http.Client
}

func NewGeminiClient(apiKey, model, embedModel string) *GeminiClient {
	if embedModel == "" {
		embedModel = "gemini-embedding-001"
	}
	return &GeminiClient{
		APIKey:     apiKey,
		Model:      model,
		EmbedModel: embedModel,
		HTTP:       &http.Client{},
	}
}

// Gemini API types

type geminiPart struct {
	Text    string `json:"text"`
	Thought bool   `json:"thought,omitempty"` // gemini-2.5+ marks reasoning parts
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"` // "user" | "model"
	Parts []geminiPart `json:"parts"`
}

type geminiSchema struct {
	Type        string                   `json:"type"`
	Properties  map[string]*geminiSchema `json:"properties,omitempty"`
	Items       *geminiSchema            `json:"items,omitempty"`
	Required    []string                 `json:"required,omitempty"`
	Description string                   `json:"description,omitempty"`
}

type geminiThinkingConfig struct {
	ThinkingBudget int `json:"thinkingBudget"` // 0 disables reasoning
}

type geminiGenerationConfig struct {
	ResponseMimeType string                `json:"responseMimeType,omitempty"`
	ResponseSchema   *geminiSchema         `json:"responseSchema,omitempty"`
	ThinkingConfig   *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
}

type geminiRequest struct {
	Contents          []geminiContent         `json:"contents"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

// agentOutputSchema mirrors agent.AgentOutput. Gemini enforces this on the
// server side, so the model literally cannot put prose around the JSON or
// invent top-level fields.
//
// Note: `args` is intentionally NOT in the tool_calls item schema. Gemini's
// structured output treats an OBJECT-typed field with no `properties` as an
// empty {}, which prevents the model from passing real tool arguments. With
// `args` omitted from the schema, the model falls back to the args spec in
// the system prompt (each tool's own JSON schema is listed there) and emits
// them freely.
var agentOutputSchema = &geminiSchema{
	Type: "OBJECT",
	Properties: map[string]*geminiSchema{
		"thought": {
			Type:        "STRING",
			Description: "One-line internal reasoning; never spoken or shown to the user.",
		},
		"tool_calls": {
			Type: "ARRAY",
			Items: &geminiSchema{
				Type: "OBJECT",
				Properties: map[string]*geminiSchema{
					"name": {
						Type:        "STRING",
						Description: "Tool name from the catalog in the system prompt.",
					},
					"args": {
						// Has to be STRING, not OBJECT. Gemini's structured
						// output collapses an OBJECT field with no properties
						// to {}, which means the model can't put real
						// arguments through. As a STRING, the model emits a
						// JSON-encoded args object — we unwrap it server-side.
						Type:        "STRING",
						Description: "JSON-encoded args object matching the tool's args schema. Example: \"{\\\"query\\\":\\\"bohemian rhapsody\\\"}\". Pass \"{}\" if the tool takes no args.",
					},
				},
				Required: []string{"name", "args"},
			},
		},
		"response": {
			Type:        "STRING",
			Description: "The literal words to speak this turn. No JSON, no schema, no quotes around fields.",
		},
		"engagement": {
			Type:        "STRING",
			Description: "One of: respond | skip | dismiss. Use \"respond\" by default. Use \"skip\" when the utterance is NOT directed at you (singing, song lyrics, side-talk to someone else in the room) — and leave \"response\" empty. Use \"dismiss\" only when the user explicitly ends the conversation.",
		},
	},
	Required: []string{"response", "engagement"},
}

type geminiStreamChunk struct {
	Candidates []struct {
		Content geminiContent `json:"content"`
		FinishReason string  `json:"finishReason,omitempty"`
	} `json:"candidates"`
}

// ChatStream consumes the JSON Server-Sent-Events form of streamGenerateContent.
// Each SSE event is a single JSON object; we extract text from the first
// candidate's parts and surface it via onChunk.
func (c *GeminiClient) ChatStream(ctx context.Context, msgs []Message, onChunk func(string)) (string, error) {
	contents, sysInstr := splitMessages(msgs)

	body, err := json.Marshal(geminiRequest{
		Contents:          contents,
		SystemInstruction: sysInstr,
		GenerationConfig: &geminiGenerationConfig{
			ResponseMimeType: "application/json",
			ResponseSchema:   agentOutputSchema,
			ThinkingConfig:   &geminiThinkingConfig{ThinkingBudget: 0},
		},
	})
	if err != nil {
		return "", err
	}

	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse&key=%s", geminiBase, c.Model, c.APIKey)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := bufio.NewReader(resp.Body).ReadBytes(0)
		return "", fmt.Errorf("gemini chat: %s — %s", resp.Status, string(raw))
	}

	var full bytes.Buffer
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var ch geminiStreamChunk
		if err := json.Unmarshal([]byte(payload), &ch); err != nil {
			continue
		}
		for _, cand := range ch.Candidates {
			for _, p := range cand.Content.Parts {
				if p.Text == "" {
					continue
				}
				// Skip internal reasoning parts — they're prose, would
				// corrupt our structured-JSON envelope if concatenated.
				if p.Thought {
					continue
				}
				full.WriteString(p.Text)
				if onChunk != nil {
					onChunk(p.Text)
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return full.String(), err
	}
	return full.String(), nil
}

// splitMessages maps our generic Message list into Gemini's (systemInstruction,
// contents) split. Gemini uses "model" for the assistant role.
func splitMessages(msgs []Message) ([]geminiContent, *geminiContent) {
	var sysParts []geminiPart
	var contents []geminiContent
	for _, m := range msgs {
		switch m.Role {
		case RoleSystem:
			sysParts = append(sysParts, geminiPart{Text: m.Content})
		case RoleAssistant:
			contents = append(contents, geminiContent{Role: "model", Parts: []geminiPart{{Text: m.Content}}})
		case RoleTool:
			// Surface tool output as a "user" message so the model can use it.
			contents = append(contents, geminiContent{Role: "user", Parts: []geminiPart{{Text: "[tool output]\n" + m.Content}}})
		default: // RoleUser
			contents = append(contents, geminiContent{Role: "user", Parts: []geminiPart{{Text: m.Content}}})
		}
	}
	var sys *geminiContent
	if len(sysParts) > 0 {
		sys = &geminiContent{Parts: sysParts}
	}
	return contents, sys
}

// Embeddings

// geminiEmbedDim matches our Postgres vector(768) column. gemini-embedding-001
// produces 3072-dim by default; outputDimensionality truncates server-side.
const geminiEmbedDim = 768

type geminiEmbedRequest struct {
	Model                string        `json:"model"`                // "models/<name>" — required
	Content              geminiContent `json:"content"`
	OutputDimensionality int           `json:"outputDimensionality"` // pin to 768
}

type geminiEmbedResponse struct {
	Embedding struct {
		Values []float32 `json:"values"`
	} `json:"embedding"`
}

func (c *GeminiClient) Embed(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(geminiEmbedRequest{
		Model:                "models/" + c.EmbedModel,
		Content:              geminiContent{Parts: []geminiPart{{Text: text}}},
		OutputDimensionality: geminiEmbedDim,
	})
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/models/%s:embedContent?key=%s", geminiBase, c.EmbedModel, c.APIKey)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		buf := new(bytes.Buffer)
		_, _ = buf.ReadFrom(resp.Body)
		return nil, fmt.Errorf("gemini embed: %s — %s", resp.Status, buf.String())
	}

	var er geminiEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&er); err != nil {
		return nil, err
	}
	if len(er.Embedding.Values) == 0 {
		return nil, fmt.Errorf("gemini embed: empty embedding")
	}
	return er.Embedding.Values, nil
}
