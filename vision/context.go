// Package vision reads the JSON snapshot produced by vision/observe.py and
// hands the latest visual context to the agent for prompt injection.
//
// The Python sidecar writes to ./vision/visual_context.json atomically (temp+
// rename) at ~1Hz. This package reads it on demand at the top of each agent
// Turn. If the file is missing or stale, Read returns ok=false and the agent
// simply omits the visual block — vision is a strict enhancement, never a
// hard dependency.
package vision

import (
	"encoding/json"
	"os"
	"time"
)

// freshnessLimit is how old a snapshot can be before we treat it as stale.
// Sidecar emits every ~1s, so 5s gives plenty of headroom for slow frames.
const freshnessLimit = 5 * time.Second

// Person mirrors one entry from observe.py's `people` array.
type Person struct {
	Talking       bool   `json:"talking"`
	HeadDirection string `json:"head_direction"`
	Smiling       bool   `json:"smiling"`
}

// Snapshot is the parsed view of visual_context.json.
type Snapshot struct {
	TS            time.Time `json:"-"`
	TSRaw         string    `json:"ts"`
	People        []Person  `json:"people"`
	OtherObjects  []string  `json:"other_objects"`
	Summary       string    `json:"summary"`

	// AgeSeconds is filled in by Read — how stale the snapshot was when read.
	AgeSeconds float64 `json:"-"`
}

// Read loads the latest snapshot from path. Returns ok=false if the file is
// absent, malformed, or older than freshnessLimit. Cheap enough to call once
// per Turn — file is tiny (~hundreds of bytes) and read is synchronous.
func Read(path string) (Snapshot, bool) {
	if path == "" {
		return Snapshot{}, false
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Snapshot{}, false
	}
	var s Snapshot
	if err := json.Unmarshal(data, &s); err != nil {
		return Snapshot{}, false
	}
	ts, err := time.Parse(time.RFC3339, s.TSRaw)
	if err != nil {
		// Try the fractional-second form observe.py emits.
		ts, err = time.Parse("2006-01-02T15:04:05.000Z07:00", s.TSRaw)
		if err != nil {
			return Snapshot{}, false
		}
	}
	s.TS = ts
	s.AgeSeconds = time.Since(ts).Seconds()
	if s.AgeSeconds > freshnessLimit.Seconds() {
		return Snapshot{}, false
	}
	return s, true
}
