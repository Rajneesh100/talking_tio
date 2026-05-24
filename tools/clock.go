package tools

import (
	"context"
	"encoding/json"
	"time"
)

// ClockTool returns the current local time. Useful as a smoke-test of the
// tool framework before wiring real tools.
type ClockTool struct{}

func NewClockTool() *ClockTool { return &ClockTool{} }

func (ClockTool) Descriptor() Descriptor {
	return Descriptor{
		Name:        "clock",
		Description: "Returns the current local time as ISO-8601 with timezone.",
		Schema:      json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`),
	}
}

func (ClockTool) Execute(ctx context.Context, _ json.RawMessage) (string, error) {
	return time.Now().Format(time.RFC3339), nil
}
