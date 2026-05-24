package tools

import (
	"context"
	"encoding/json"
	"fmt"
)

// Descriptor is the metadata the agent's system prompt advertises to the LLM.
// Schema is a JSON Schema object describing the tool's args.
type Descriptor struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Schema      json.RawMessage `json:"schema"`
}

// Tool is implemented by anything the agent can call.
//
// Execute receives the args as raw JSON (matching the Descriptor.Schema) and
// returns a string the agent appends as a tool-role message.
type Tool interface {
	Descriptor() Descriptor
	Execute(ctx context.Context, args json.RawMessage) (string, error)
}

type Registry struct {
	tools map[string]Tool
}

func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]Tool)}
}

func (r *Registry) Register(t Tool) {
	d := t.Descriptor()
	if d.Name == "" {
		panic("tools: tool with empty name")
	}
	if _, exists := r.tools[d.Name]; exists {
		panic(fmt.Sprintf("tools: duplicate tool name %q", d.Name))
	}
	r.tools[d.Name] = t
}

func (r *Registry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

func (r *Registry) Descriptors() []Descriptor {
	out := make([]Descriptor, 0, len(r.tools))
	for _, t := range r.tools {
		out = append(out, t.Descriptor())
	}
	return out
}
