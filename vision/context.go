// Package vision reads the JSON snapshot produced by vision/observe.py and
// hands the latest visual context to the agent for prompt injection.
//
// The Python sidecar writes to vision/visual_context.json (atomic temp+rename)
// at ~1Hz. This package reads it on demand at the top of each agent Turn. If
// the file is missing or stale, Read returns ok=false and the agent simply
// omits the visual block — vision is a strict enhancement, never a hard
// dependency.
//
// The JSON schema mirrors the local-talking-llm reference: a rolling
// `visual_history` of frames, each carrying YOLO object detections enriched
// with per-person face activity and hand-gesture analysis.
package vision

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
)

// freshnessLimit caps how stale a snapshot can be before we treat it as
// absent. Sidecar emits every ~1s; 5s gives headroom for slow frames.
const freshnessLimit = 5 * time.Second

// HandData mirrors observe.py's analyze_hand() output.
type HandData struct {
	Present       bool            `json:"present"`
	Fingers       map[string]bool `json:"fingers"`
	Open          bool            `json:"open"`
	Fist          bool            `json:"fist"`
	MiddleFinger  bool            `json:"middle_finger"`
}

// FaceActivity mirrors observe.py's face_activity dict.
type FaceActivity struct {
	Smiling       bool   `json:"smiling"`
	Talking       bool   `json:"talking"`
	HeadDirection string `json:"head_direction"`
}

// Person is the rich per-person payload attached to a "person" detection.
type Person struct {
	Name      string        `json:"name"`
	Face      *FaceActivity `json:"face"`
	LeftHand  *HandData     `json:"left_hand"`
	RightHand *HandData     `json:"right_hand"`
	HeadBBox  []int         `json:"head_bbox"` // nullable in JSON; nil if absent
}

// Detection is one row from the `detections` array. For "person" objects
// PersonData is populated; for everything else it's nil.
type Detection struct {
	Object     string  `json:"object"`
	PersonData *Person `json:"person_data,omitempty"`
	Confidence float64 `json:"confidence"`
	BBox       []int   `json:"bbox"`
}

// Frame is one timestamped entry from visual_history.
type Frame struct {
	Timestamp  string      `json:"timestamp"`
	Detections []Detection `json:"detections"`
}

type rawHistory struct {
	VisualHistory []Frame `json:"visual_history"`
}

// Snapshot is what the agent consumes — the most-recent frame plus a
// pre-built human-readable summary line for direct prompt injection.
type Snapshot struct {
	Timestamp    time.Time
	Frame        Frame   // newest frame
	AllFrames    []Frame // full rolling buffer (chronological, len ≤ 5)
	Summary      string  // one-line "now:" description of newest frame
	RecentMotion string  // diff across the rolling buffer ("just looked away", "picked up phone", ...)
	AgeSeconds   float64
}

// Read loads the latest snapshot from path. Returns ok=false if the file is
// absent, malformed, empty, or older than freshnessLimit.
func Read(path string) (Snapshot, bool) {
	if path == "" {
		return Snapshot{}, false
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Snapshot{}, false
	}
	var raw rawHistory
	if err := json.Unmarshal(data, &raw); err != nil {
		return Snapshot{}, false
	}
	if len(raw.VisualHistory) == 0 {
		return Snapshot{}, false
	}
	frame := raw.VisualHistory[len(raw.VisualHistory)-1]
	ts, err := parseTimestamp(frame.Timestamp)
	if err != nil {
		return Snapshot{}, false
	}
	age := time.Since(ts).Seconds()
	if age > freshnessLimit.Seconds() {
		return Snapshot{}, false
	}
	return Snapshot{
		Timestamp:    ts,
		Frame:        frame,
		AllFrames:    raw.VisualHistory,
		Summary:      BuildSummary(frame),
		RecentMotion: BuildRecentMotion(raw.VisualHistory),
		AgeSeconds:   age,
	}, true
}

// parseTimestamp accepts the two forms observe.py might emit: full RFC 3339
// and the older microsecond-precision iso form without a timezone.
func parseTimestamp(s string) (time.Time, error) {
	for _, layout := range []string{
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02T15:04:05.000000",
		"2006-01-02T15:04:05",
	} {
		if t, err := time.Parse(layout, s); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("unrecognised timestamp %q", s)
}

// BuildSummary renders the frame as a one-line, prompt-friendly string. The
// agent injects this directly under a "[visual context]" header.
func BuildSummary(frame Frame) string {
	var people []Person
	var otherObjects []string
	for _, d := range frame.Detections {
		if d.Object == "person" && d.PersonData != nil {
			people = append(people, *d.PersonData)
		} else if d.Object != "person" {
			otherObjects = append(otherObjects, d.Object)
		}
	}

	if len(people) == 0 {
		if len(otherObjects) > 0 {
			return "no one in frame; objects: " + strings.Join(dedup(otherObjects), ", ")
		}
		return "no one in frame"
	}

	pieces := []string{}
	if len(people) == 1 {
		p := people[0]
		who := p.Name
		if who == "" {
			who = "someone"
		}
		pieces = append(pieces, who)
		if p.Face != nil {
			if p.Face.Talking {
				pieces = append(pieces, "talking")
			}
			switch p.Face.HeadDirection {
			case "center":
				pieces = append(pieces, "looking at camera")
			case "left", "right":
				pieces = append(pieces, "looking "+p.Face.HeadDirection)
			}
			if p.Face.Smiling {
				pieces = append(pieces, "smiling")
			}
		}
		if gesture := handsSummary(p.LeftHand, p.RightHand); gesture != "" {
			pieces = append(pieces, gesture)
		}
	} else {
		pieces = append(pieces, fmt.Sprintf("%d people", len(people)))
		for _, p := range people {
			name := p.Name
			if name == "" {
				name = "someone"
			}
			pieces = append(pieces, name)
		}
	}

	if len(otherObjects) > 0 {
		pieces = append(pieces, "nearby: "+strings.Join(dedup(otherObjects), ", "))
	}
	return strings.Join(pieces, ", ")
}

// handsSummary picks the most informative gesture across left/right hands.
// Returns "" when nothing notable is happening.
func handsSummary(left, right *HandData) string {
	parts := []string{}
	for label, h := range map[string]*HandData{"left hand": left, "right hand": right} {
		if h == nil || !h.Present {
			continue
		}
		switch {
		case h.MiddleFinger:
			parts = append(parts, label+" flipping you off")
		case h.Fist:
			parts = append(parts, label+" closed fist")
		case h.Open:
			parts = append(parts, label+" open palm")
		default:
			extended := []string{}
			for name, ext := range h.Fingers {
				if ext {
					extended = append(extended, name)
				}
			}
			if len(extended) > 0 && len(extended) < 5 {
				parts = append(parts, fmt.Sprintf("%s: %s extended", label, strings.Join(extended, "+")))
			}
		}
	}
	return strings.Join(parts, ", ")
}

func dedup(in []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, v := range in {
		if seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}
	return out
}

// ---- frame accessors used by motion analysis ----

func firstPerson(f Frame) *Person {
	for _, d := range f.Detections {
		if d.Object == "person" && d.PersonData != nil {
			return d.PersonData
		}
	}
	return nil
}

func frameObjectSet(f Frame) map[string]bool {
	set := map[string]bool{}
	for _, d := range f.Detections {
		if d.Object != "person" {
			set[d.Object] = true
		}
	}
	return set
}

func handGesture(h *HandData) string {
	if h == nil || !h.Present {
		return ""
	}
	switch {
	case h.MiddleFinger:
		return "middle finger"
	case h.Fist:
		return "fist"
	case h.Open:
		return "open"
	default:
		return "partial"
	}
}

// BuildRecentMotion compares the most recent frame against the earliest frame
// in the rolling buffer (typically ~5s of history at 1Hz) and describes
// notable transitions. Returns "" when nothing notable changed.
//
// This is the "within-the-window" motion signal — useful for engagement
// decisions like "she just looked away" or "they just picked up a phone".
func BuildRecentMotion(frames []Frame) string {
	if len(frames) < 2 {
		return ""
	}
	earliest := frames[0]
	latest := frames[len(frames)-1]

	earlyP := firstPerson(earliest)
	curP := firstPerson(latest)

	var notes []string

	// Presence delta.
	switch {
	case earlyP == nil && curP != nil:
		notes = append(notes, "just came into frame")
	case earlyP != nil && curP == nil:
		notes = append(notes, "just left the frame")
	}

	// Face activity deltas (only meaningful when person is in both frames).
	if earlyP != nil && curP != nil && earlyP.Face != nil && curP.Face != nil {
		if !earlyP.Face.Talking && curP.Face.Talking {
			notes = append(notes, "just started talking")
		} else if earlyP.Face.Talking && !curP.Face.Talking {
			notes = append(notes, "just stopped talking")
		}
		if earlyP.Face.HeadDirection == "center" &&
			(curP.Face.HeadDirection == "left" || curP.Face.HeadDirection == "right") {
			notes = append(notes, "just looked "+curP.Face.HeadDirection)
		} else if (earlyP.Face.HeadDirection == "left" || earlyP.Face.HeadDirection == "right") &&
			curP.Face.HeadDirection == "center" {
			notes = append(notes, "just looked back at you")
		}
	}

	// Hand-gesture deltas (left + right tracked separately).
	if earlyP != nil && curP != nil {
		for _, side := range []struct {
			label string
			prev  *HandData
			now   *HandData
		}{
			{"left hand", earlyP.LeftHand, curP.LeftHand},
			{"right hand", earlyP.RightHand, curP.RightHand},
		} {
			prevG := handGesture(side.prev)
			nowG := handGesture(side.now)
			if prevG == "" && nowG != "" {
				notes = append(notes, fmt.Sprintf("just raised %s (%s)", side.label, nowG))
			} else if prevG != "" && nowG == "" {
				notes = append(notes, "dropped "+side.label)
			} else if prevG != "" && nowG != "" && prevG != nowG {
				notes = append(notes, fmt.Sprintf("%s: %s → %s", side.label, prevG, nowG))
			}
		}
	}

	// Object set delta.
	earlyObjs := frameObjectSet(earliest)
	curObjs := frameObjectSet(latest)
	for obj := range curObjs {
		if !earlyObjs[obj] {
			notes = append(notes, "picked up "+obj)
		}
	}
	for obj := range earlyObjs {
		if !curObjs[obj] {
			notes = append(notes, obj+" left")
		}
	}

	return strings.Join(notes, ", ")
}

// Delta describes what changed between two snapshots — typically the previous
// turn's snapshot and the current one. The gap between them can be any length
// (seconds to minutes), so this captures longer-horizon arcs than the 5-second
// rolling window covered by BuildRecentMotion.
//
// Returns "" when nothing notable changed.
func Delta(prev, current Snapshot) string {
	prevP := firstPerson(prev.Frame)
	curP := firstPerson(current.Frame)

	var notes []string

	switch {
	case prevP == nil && curP != nil:
		notes = append(notes, "came back into frame")
	case prevP != nil && curP == nil:
		notes = append(notes, "stepped out of frame")
	}

	if prevP != nil && curP != nil {
		if prevP.Name != curP.Name {
			switch {
			case prevP.Name == "Unknown" && curP.Name != "Unknown":
				notes = append(notes, "recognized as "+curP.Name)
			case prevP.Name != "Unknown" && curP.Name == "Unknown":
				notes = append(notes, "face no longer matched")
			default:
				notes = append(notes, prevP.Name+" → "+curP.Name)
			}
		}
		if prevP.Face != nil && curP.Face != nil {
			if prevP.Face.Talking && !curP.Face.Talking {
				notes = append(notes, "stopped talking")
			} else if !prevP.Face.Talking && curP.Face.Talking {
				notes = append(notes, "started talking")
			}
			if prevP.Face.HeadDirection != curP.Face.HeadDirection &&
				prevP.Face.HeadDirection != "" && curP.Face.HeadDirection != "" {
				notes = append(notes, "head "+prevP.Face.HeadDirection+" → "+curP.Face.HeadDirection)
			}
		}
	}

	prevObjs := frameObjectSet(prev.Frame)
	curObjs := frameObjectSet(current.Frame)
	for obj := range curObjs {
		if !prevObjs[obj] {
			notes = append(notes, "now has "+obj)
		}
	}
	for obj := range prevObjs {
		if !curObjs[obj] {
			notes = append(notes, "no longer has "+obj)
		}
	}

	return strings.Join(notes, ", ")
}
