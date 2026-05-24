package tts

import (
	"context"
	"fmt"
)

// Backend speaks text synchronously: it blocks until playback finishes.
// Implementations must be safe for serial use from a single worker goroutine.
type Backend interface {
	Speak(ctx context.Context, text string) error
}

func New(name, voice string) (Backend, error) {
	switch name {
	case "say":
		return NewSay(voice), nil
	case "none":
		return noopBackend{}, nil
	default:
		return nil, fmt.Errorf("tts: unknown backend %q", name)
	}
}

type noopBackend struct{}

func (noopBackend) Speak(context.Context, string) error { return nil }
