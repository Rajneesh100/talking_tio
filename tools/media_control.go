package tools

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
)

// chromiumBrowsers are the macOS browser applications we attempt to close
// YouTube Music tabs in. Targeting each by app name keeps the AppleScript
// simple — Safari and Firefox would need different scripting models, and
// most users running this are on Chrome anyway. Errors per-browser are
// silently swallowed so a missing browser doesn't fail the whole stop.
var chromiumBrowsers = []string{
	"Google Chrome",
	"Brave Browser",
	"Arc",
}

// closeMusicTabsScript builds the per-browser AppleScript that finds and
// closes any tab on music.youtube.com. The outer `try` makes it a no-op when
// the browser isn't running.
func closeMusicTabsScript(appName string) string {
	return fmt.Sprintf(`
try
  tell application "%s"
    repeat with w in (every window)
      try
        set killTabs to (every tab of w whose URL contains "music.youtube.com")
        repeat with t in killTabs
          try
            close t
          end try
        end repeat
      end try
    end repeat
  end tell
end try
`, appName)
}

// stopBrowserMusic pauses anything the system is playing (via nowplaying-cli
// if installed) and closes every YouTube Music tab it can find across the
// known Chromium-based browsers.
//
// Best-effort, never fails: if nowplaying-cli isn't installed or AppleScript
// fails, we keep going. Returns a human-readable summary suitable for the
// tool's result string.
func stopBrowserMusic(ctx context.Context) string {
	var notes []string

	// 1. Pause anything currently in macOS's Now Playing system.
	//    Cheap if available, silently skipped if not.
	if _, err := exec.LookPath("nowplaying-cli"); err == nil {
		if err := exec.CommandContext(ctx, "nowplaying-cli", "pause").Run(); err == nil {
			notes = append(notes, "media paused")
		}
	}

	// 2. Close music.youtube.com tabs in each known browser.
	closed := 0
	for _, app := range chromiumBrowsers {
		script := closeMusicTabsScript(app)
		if err := exec.CommandContext(ctx, "osascript", "-e", script).Run(); err == nil {
			closed++
		}
	}
	if closed > 0 {
		notes = append(notes, fmt.Sprintf("tab-close attempted in %d browser(s)", closed))
	}

	if len(notes) == 0 {
		return "no media controllers available; nothing to do"
	}
	return strings.Join(notes, "; ")
}
