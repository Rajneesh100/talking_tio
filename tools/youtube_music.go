package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os/exec"
	"runtime"
	"strings"
	"time"
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

	// Try to resolve the query to a specific YouTube video ID via yt-dlp so
	// the browser opens `watch?v=…` (auto-plays) instead of the search page.
	// If yt-dlp isn't installed or times out, fall back to the search URL.
	var target, mode string
	if videoID := resolveVideoID(ctx, args.Query); videoID != "" {
		target = "https://music.youtube.com/watch?v=" + videoID
		mode = "play"
	} else {
		target = "https://music.youtube.com/search?q=" + url.QueryEscape(args.Query)
		mode = "search"
	}

	opener, openerArgs := platformOpener()
	if opener == "" {
		return "", fmt.Errorf("youtube_music: no browser-open command on %s", runtime.GOOS)
	}
	cmd := exec.CommandContext(ctx, opener, append(openerArgs, target)...)
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("youtube_music: open: %w", err)
	}
	return fmt.Sprintf("opened YouTube Music (%s mode) for %q at %s", mode, args.Query, target), nil
}

// resolveVideoID asks yt-dlp for the top YouTube result for the query and
// returns its 11-character video ID. Returns "" if yt-dlp isn't installed,
// times out, or returns something unexpected — caller falls back to a search
// URL in that case. Install with: brew install yt-dlp
func resolveVideoID(ctx context.Context, query string) string {
	if _, err := exec.LookPath("yt-dlp"); err != nil {
		return ""
	}
	resolveCtx, cancel := context.WithTimeout(ctx, 6*time.Second)
	defer cancel()
	out, err := exec.CommandContext(resolveCtx, "yt-dlp",
		"--skip-download",
		"--print", "id",
		"--default-search", "ytsearch1",
		"ytsearch1:"+query,
	).Output()
	if err != nil {
		return ""
	}
	id := strings.TrimSpace(string(out))
	// YouTube video IDs are 11 chars. Reject anything else (yt-dlp warnings,
	// playlist IDs, multiple lines, etc.) so we don't open a junk URL.
	if len(id) != 11 {
		return ""
	}
	return id
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
