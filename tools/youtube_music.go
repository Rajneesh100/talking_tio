package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os/exec"
	"runtime"
)

// YouTubeMusicTool opens YouTube Music in the default browser to a search
// results page. Picked by the LLM when the user asks to "play X", "search for
// Y", or otherwise wants to listen to music.
//
// For an even tighter UX (auto-play the top result instead of dropping the
// user on a results page), add a YOUTUBE_API_KEY and use the Data API to
// resolve the query to a videoId, then open
// https://music.youtube.com/watch?v=<videoId>. Left as a follow-up.
type YouTubeMusicTool struct{}

func NewYouTubeMusicTool() *YouTubeMusicTool { return &YouTubeMusicTool{} }

func (YouTubeMusicTool) Descriptor() Descriptor {
	return Descriptor{
		Name: "youtube_music",
		Description: "Open YouTube Music in the browser to search for or play music. " +
			"Use this whenever the user asks to play a song, artist, album, playlist, " +
			"or wants to find music to listen to. The query should be what they want " +
			"to hear (e.g. \"bohemian rhapsody\", \"taylor swift\", \"lo-fi beats\").",
		Schema: json.RawMessage(`{
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Song title, artist name, album, playlist, or genre to search for."
                }
            },
            "required": ["query"],
            "additionalProperties": false
        }`),
	}
}

type ytMusicArgs struct {
	Query string `json:"query"`
}

func (YouTubeMusicTool) Execute(ctx context.Context, raw json.RawMessage) (string, error) {
	var args ytMusicArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return "", fmt.Errorf("youtube_music: bad args: %w", err)
	}
	if args.Query == "" {
		return "", fmt.Errorf("youtube_music: query is required")
	}

	target := "https://music.youtube.com/search?q=" + url.QueryEscape(args.Query)

	opener, openerArgs := platformOpener()
	if opener == "" {
		return "", fmt.Errorf("youtube_music: no browser-open command on %s", runtime.GOOS)
	}
	cmd := exec.CommandContext(ctx, opener, append(openerArgs, target)...)
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("youtube_music: open: %w", err)
	}
	return fmt.Sprintf("opened YouTube Music search for %q at %s", args.Query, target), nil
}

// platformOpener picks the right "open this URL in default browser" command.
// macOS: `open`. Linux: `xdg-open`. Anything else: bail.
func platformOpener() (string, []string) {
	switch runtime.GOOS {
	case "darwin":
		return "open", nil
	case "linux":
		return "xdg-open", nil
	default:
		return "", nil
	}
}
