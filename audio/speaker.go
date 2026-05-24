package audio

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"

	"github.com/rajneesh/talking_tio/tts"
)

// sentenceEnd splits on '.', '!' or '?' followed by whitespace or buffer end.
var sentenceEnd = regexp.MustCompile(`([.!?])(\s|$)`)

// Speaker plays sentences in order on a background goroutine.
//
// Lifecycle: NewSpeaker -> Feed (n times) -> Flush. One-shot: do not reuse.
// Stop is callable from another goroutine to interrupt playback for barge-in;
// after Stop, Feed and Flush become no-ops and the worker exits.
type Speaker struct {
	backend tts.Backend
	queue   chan string
	wg      sync.WaitGroup

	ctx    context.Context
	cancel context.CancelFunc

	mu      sync.Mutex
	buffer  string
	closed  bool
}

func NewSpeaker(backend tts.Backend) *Speaker {
	ctx, cancel := context.WithCancel(context.Background())
	s := &Speaker{
		backend: backend,
		queue:   make(chan string, 64),
		ctx:     ctx,
		cancel:  cancel,
	}
	s.wg.Add(1)
	go s.run()
	return s
}

func (s *Speaker) run() {
	defer s.wg.Done()
	for sentence := range s.queue {
		if s.ctx.Err() != nil {
			continue // drain quickly after Stop
		}
		// Speak is bound to s.ctx; cancelling kills the in-flight `say`.
		// On error (most commonly: say subprocess timed out because the
		// audio device was busy), log and drop this sentence — keep going
		// with the next one so the agent doesn't hang.
		if err := s.backend.Speak(s.ctx, sentence); err != nil {
			fmt.Fprintf(os.Stderr, "tts: speak failed: %v (sentence: %q)\n", err, sentence)
		}
	}
}

// Feed adds streamed text. Whole sentences are emitted as they form; partial
// trailing text is held until Flush or the next Feed.
func (s *Speaker) Feed(chunk string) {
	if chunk == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.buffer += chunk
	for {
		loc := sentenceEnd.FindStringIndex(s.buffer)
		if loc == nil {
			break
		}
		sentence := strings.TrimSpace(s.buffer[:loc[1]])
		s.buffer = s.buffer[loc[1]:]
		if sentence != "" {
			s.queue <- sentence
		}
	}
}

// Flush emits the buffered tail and blocks until the queue drains.
func (s *Speaker) Flush() {
	s.mu.Lock()
	tail := strings.TrimSpace(s.buffer)
	s.buffer = ""
	if s.closed {
		s.mu.Unlock()
		return
	}
	if tail != "" {
		s.queue <- tail
	}
	s.closed = true
	close(s.queue)
	s.mu.Unlock()
	s.wg.Wait()
}

// Stop interrupts playback: cancels the current `say` subprocess via ctx and
// discards any queued sentences. Safe to call from any goroutine.
func (s *Speaker) Stop() {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.closed = true
	s.cancel()
	close(s.queue)
	s.mu.Unlock()
	s.wg.Wait()
}
