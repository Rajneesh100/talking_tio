package agent

import (
	"regexp"
	"strings"
)

// wakePhrases are matched case-insensitively as whole-word substrings.
// First match wins, and the matched phrase is stripped from the returned text.
var wakePhrases = []string{
	"hey angela",
	"ok angela",
	"okay angela",
	"yo angela",
	"angela",
	"hey",
	"play",
	"change",
	"stop",
}

// One regex per phrase, anchored at word boundaries. Compiled once at init.
var wakePhraseRes = compileWakePhraseRegexes()

func compileWakePhraseRegexes() []*regexp.Regexp {
	out := make([]*regexp.Regexp, 0, len(wakePhrases))
	for _, p := range wakePhrases {
		// (?i) case-insensitive, \b word boundary, regexp.QuoteMeta in case
		// we ever add phrases with punctuation.
		out = append(out, regexp.MustCompile(`(?i)\b`+regexp.QuoteMeta(p)+`\b`))
	}
	return out
}

// stripWakePhrase looks for any wake phrase in text. If found, returns the
// text with the FIRST occurrence removed and woke=true. If no wake phrase
// is present, returns the original text and woke=false.
//
// The returned cleaned text is whitespace-collapsed and trimmed of leading
// punctuation, so "Hey Angela, what time is it?" becomes "what time is it?".
func stripWakePhrase(text string) (cleaned string, woke bool) {
	for _, re := range wakePhraseRes {
		if loc := re.FindStringIndex(text); loc != nil {
			// cleaned = text[:loc[0]] + " " + text[loc[1]:]
			// cleaned = normalizeAfterStrip(cleaned)
			return text, true
		}
	}
	return text, false
}

// containsWakePhrase is a yes/no check without the cleaning work.
func containsWakePhrase(text string) bool {
	for _, re := range wakePhraseRes {
		if re.MatchString(text) {
			return true
		}
	}
	return false
}

// normalizeAfterStrip collapses whitespace and trims leading punctuation
// that a stripped wake phrase tends to leave behind (", what time is it?").
func normalizeAfterStrip(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	s = strings.TrimLeft(s, ",.;:!? \t")
	return strings.TrimSpace(s)
}
