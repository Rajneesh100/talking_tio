package stt

import (
	"regexp"
	"strings"
)

// Common phrases Whisper hallucinates when fed silent or near-silent audio.
// These come from YouTube training data and appear regardless of input.
// Append to this list as new ones surface.
var hallucinations = []string{
	"thanks for watching",
	"thank you for watching",
	"thank you so much for watching",
	"see you in the next video",
	"i'll see you in the next video",
	"thank you for your time",
	"subtitles by the amara.org community",
	"please subscribe",
	"like and subscribe",
}

// IsHallucination returns true if text looks like a Whisper artifact rather
// than real user speech.
//
// Three layers of filtering:
//  1. Whisper.cpp emits bracketed markers for non-speech audio:
//     [BLANK_AUDIO], [SILENCE], [MUSIC], [APPLAUSE], etc.
//     Anything that's only bracketed content gets dropped.
//  2. Whitespace/empty.
//  3. Known YouTube-training-data phrases that Whisper invents from silence.
func IsHallucination(text string) bool {
	t := strings.TrimSpace(text)
	if t == "" {
		return true
	}

	// Strip leading/trailing bracketed annotations like [BLANK_AUDIO].
	// If the whole utterance is bracketed (no real words), it's a marker.
	stripped := bracketStripper.ReplaceAllString(t, "")
	stripped = strings.TrimSpace(stripped)
	if stripped == "" {
		return true
	}

	low := strings.ToLower(stripped)
	low = strings.Trim(low, ".!?,'\" ")
	for _, h := range hallucinations {
		if low == h {
			return true
		}
	}
	return false
}

var bracketStripper = regexp.MustCompile(`[\[\(][^\]\)]*[\]\)]`)
