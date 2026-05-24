package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

type OllamaClient struct {
	BaseURL    string
	Model      string
	EmbedModel string
	HTTP       *http.Client
}

func NewOllamaClient(baseURL, model, embedModel string) *OllamaClient {
	return &OllamaClient{
		BaseURL:    baseURL,
		Model:      model,
		EmbedModel: embedModel,
		HTTP:       &http.Client{},
	}
}

type ollamaChatRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	Stream    bool      `json:"stream"`
	Format    string    `json:"format,omitempty"`     // "json" forces valid-JSON output
	KeepAlive string    `json:"keep_alive,omitempty"`
}

type ollamaChatChunk struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
	Done bool `json:"done"`
}

func (c *OllamaClient) ChatStream(ctx context.Context, msgs []Message, onChunk func(string)) (string, error) {
	body, err := json.Marshal(ollamaChatRequest{
		Model:     c.Model,
		Messages:  msgs,
		Stream:    true,
		Format:    "json", // agent always wants structured output
		KeepAlive: "24h",
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/api/chat", bytes.NewReader(body))
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
		return "", fmt.Errorf("ollama chat: %s", resp.Status)
	}

	var full bytes.Buffer
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var ch ollamaChatChunk
		if err := json.Unmarshal(line, &ch); err != nil {
			continue
		}
		if ch.Message.Content != "" {
			full.WriteString(ch.Message.Content)
			if onChunk != nil {
				onChunk(ch.Message.Content)
			}
		}
		if ch.Done {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return full.String(), err
	}
	return full.String(), nil
}

type ollamaEmbedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type ollamaEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

func (c *OllamaClient) Embed(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(ollamaEmbedRequest{Model: c.EmbedModel, Input: text})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/api/embed", bytes.NewReader(body))
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
		return nil, fmt.Errorf("ollama embed: %s", resp.Status)
	}

	var er ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&er); err != nil {
		return nil, err
	}
	if len(er.Embeddings) == 0 {
		return nil, fmt.Errorf("ollama embed: no embeddings returned")
	}
	return er.Embeddings[0], nil
}
