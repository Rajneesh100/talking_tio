package tts

import (
	"context"
	"os/exec"
)

// Say uses the macOS `say` binary. An empty voice falls back to the system
// default (set in System Settings → Accessibility → Spoken Content) — this is
// the only way to use a downloaded Siri voice, since `say -v "Voice 2"` is
// gated by Apple.
type Say struct {
	Voice string
}

func NewSay(voice string) *Say { return &Say{Voice: voice} }

func (s *Say) Speak(ctx context.Context, text string) error {
	args := []string{}
	if s.Voice != "" {
		args = append(args, "-v", s.Voice)
	}
	args = append(args, text)
	cmd := exec.CommandContext(ctx, "say", args...)
	return cmd.Run()
}
