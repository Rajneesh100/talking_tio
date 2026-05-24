package tts

import (
	"context"
	"os/exec"
	"time"
)

// sayTimeout caps how long a single `say` invocation can take. macOS's TTS
// sometimes stalls when the audio device is busy (e.g. YouTube Music started
// playing in another app) — without this cap the whole agent turn freezes on
// speaker.Flush waiting for a subprocess that will never return.
const sayTimeout = 30 * time.Second

// Say uses the macOS `say` binary. An empty voice falls back to the system
// default (set in System Settings → Accessibility → Spoken Content) — this is
// the only way to use a downloaded Siri voice, since `say -v "Voice 2"` is
// gated by Apple.
type Say struct {
	Voice string
}

func NewSay(voice string) *Say { return &Say{Voice: voice} }

func (s *Say) Speak(ctx context.Context, text string) error {
	ctx, cancel := context.WithTimeout(ctx, sayTimeout)
	defer cancel()
	args := []string{}
	if s.Voice != "" {
		args = append(args, "-v", s.Voice)
	}
	args = append(args, text)
	cmd := exec.CommandContext(ctx, "say", args...)
	return cmd.Run()
}
