package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// WebSearchTool hits DuckDuckGo's Instant Answer API and returns a compact
// text summary the LLM can use to ground its next response.
//
// Picked when the user references something the model can't identify (a song
// title, person, event, place) and you'd otherwise have to ask them to
// clarify. No API key needed; DDG IA is best at Wikipedia-style topics,
// which covers most of what a voice agent encounters.
type WebSearchTool struct {
	HTTP *http.Client
}

func NewWebSearchTool() *WebSearchTool {
	return &WebSearchTool{HTTP: &http.Client{Timeout: 8 * time.Second}}
}

func (WebSearchTool) Descriptor() Descriptor {
	return Descriptor{
		Name: "web_search",
		Description: "Search the web for facts. Use this whenever the user mentions a song title, artist, movie, person, event, or topic you can't identify with confidence — call it BEFORE asking the user to clarify. Returns a short text summary of the top result.",
		Schema: json.RawMessage(`{
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short keyword search query. Skip filler words / question marks. e.g. \"as it was harry styles\" not \"who sings as it was?\"."
                }
            },
            "required": ["query"],
            "additionalProperties": false
        }`),
	}
}

type ddgIAResponse struct {
	AbstractText     string `json:"AbstractText"`
	AbstractSource   string `json:"AbstractSource"`
	Heading          string `json:"Heading"`
	Definition       string `json:"Definition"`
	DefinitionSource string `json:"DefinitionSource"`
	Type             string `json:"Type"`
	RelatedTopics    []struct {
		Text     string `json:"Text"`
		FirstURL string `json:"FirstURL"`
	} `json:"RelatedTopics"`
}

func (t *WebSearchTool) Execute(ctx context.Context, raw json.RawMessage) (string, error) {
	var args struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal(raw, &args); err != nil {
		return "", fmt.Errorf("web_search: bad args: %w", err)
	}
	if args.Query == "" {
		return "", fmt.Errorf("web_search: query is required")
	}

	endpoint := "https://api.duckduckgo.com/?" + url.Values{
		"q":             {args.Query},
		"format":        {"json"},
		"no_html":       {"1"},
		"skip_disambig": {"1"},
	}.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return "", fmt.Errorf("web_search: %w", err)
	}
	req.Header.Set("User-Agent", "talking_tio/1.0 (https://github.com/rajneesh/talking_tio)")

	resp, err := t.HTTP.Do(req)
	if err != nil {
		return "", fmt.Errorf("web_search: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("web_search: %s", resp.Status)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("web_search: read: %w", err)
	}

	var data ddgIAResponse
	if err := json.Unmarshal(body, &data); err != nil {
		return "", fmt.Errorf("web_search: parse: %w", err)
	}

	return formatDDG(args.Query, &data), nil
}

func formatDDG(query string, d *ddgIAResponse) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "Search %q:\n", query)

	wrote := false

	if d.AbstractText != "" {
		heading := d.Heading
		if heading == "" {
			heading = "summary"
		}
		fmt.Fprintf(&sb, "- %s: %s", heading, d.AbstractText)
		if d.AbstractSource != "" {
			fmt.Fprintf(&sb, " (source: %s)", d.AbstractSource)
		}
		sb.WriteString("\n")
		wrote = true
	} else if d.Definition != "" {
		fmt.Fprintf(&sb, "- definition: %s", d.Definition)
		if d.DefinitionSource != "" {
			fmt.Fprintf(&sb, " (source: %s)", d.DefinitionSource)
		}
		sb.WriteString("\n")
		wrote = true
	}

	shown := 0
	for _, rt := range d.RelatedTopics {
		if rt.Text == "" || shown >= 3 {
			continue
		}
		fmt.Fprintf(&sb, "- related: %s\n", rt.Text)
		shown++
		wrote = true
	}

	if !wrote {
		return fmt.Sprintf("Search %q: no direct result. Ask the user for more detail.", query)
	}
	return sb.String()
}
