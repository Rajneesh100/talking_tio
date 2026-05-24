package tools

import (
	"context"
	"encoding/json"
)

// StopMusicTool ends any music Angela started. Pauses macOS's Now Playing
// session (best-effort via nowplaying-cli) and closes every
// music.youtube.com tab across Chromium browsers.
//
// The LLM picks this when the user asks to stop, pause, or switch tracks.
// The youtube_music tool also calls the same underlying helper before
// opening a new song, so explicit invocation isn't strictly required for
// "play X then Y" flows — but having it as a first-class tool makes
// "stop the music" / "shut up" / "kill it" work cleanly.
type StopMusicTool struct{}

func NewStopMusicTool() *StopMusicTool { return &StopMusicTool{} }

func (StopMusicTool) Descriptor() Descriptor {
	return Descriptor{
		Name: "stop_music",
		Description: "Pause and close any YouTube Music playback Angela started. " +
			"Use whenever the user asks to stop, pause, end, or kill the music — " +
			"\"stop the music\", \"that's enough\", \"shut it off\", etc. " +
			"Also useful before launching a different song, though youtube_music " +
			"already does that internally.",
		Schema: json.RawMessage(`{
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }`),
	}
}

func (StopMusicTool) Execute(ctx context.Context, _ json.RawMessage) (string, error) {
	return stopBrowserMusic(ctx), nil
}
