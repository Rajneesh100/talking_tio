package agent

import (
	"fmt"
	"strings"
)

// ANSI dim so the trace lines visually recede behind the bold You: / Angela:
// lines. Most modern terminals (Terminal.app, iTerm, vscode) handle this.
const (
	ansiDim   = "\033[2m"
	ansiReset = "\033[0m"
)

// sideLog prints a single side-channel line in this format:
//
//	  memory   · 3 fetched (top 0.91)
//	  thought  · user asking for the time, use clock
//	  tool     · clock → 2026-05-24T15:47:32+05:30
//
// Two-space indent, 8-char padded label, midline divider, dim ANSI so the
// reader's eye skips past these and lands on the conversational lines.
func sideLog(label, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	fmt.Printf("%s  %-8s · %s%s\n", ansiDim, label, msg, ansiReset)
}

// truncate shortens long tool results so they don't blow up the terminal.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// firstWords returns the first n whitespace-separated words of s, collapsing
// internal whitespace to single spaces. Adds an ellipsis if truncated.
func firstWords(s string, n int) string {
	fields := strings.Fields(s)
	if len(fields) <= n {
		return strings.Join(fields, " ")
	}
	return strings.Join(fields[:n], " ") + "…"
}

// continuation prints a side-channel line WITHOUT a label, aligning under
// the previous label's content column. Used for multi-row dumps like the
// memory preview.
func continuation(format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	fmt.Printf("%s           · %s%s\n", ansiDim, msg, ansiReset)
}
