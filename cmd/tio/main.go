package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"

	"github.com/rajneesh/talking_tio/agent"
	"github.com/rajneesh/talking_tio/audio"
	"github.com/rajneesh/talking_tio/config"
	"github.com/rajneesh/talking_tio/llm"
	"github.com/rajneesh/talking_tio/memory"
	"github.com/rajneesh/talking_tio/stt"
	"github.com/rajneesh/talking_tio/tools"
	"github.com/rajneesh/talking_tio/tts"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "fatal:", err)
		os.Exit(1)
	}
}

func run() error {
	rootCtx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("config: %w", err)
	}

	chat, embed, err := llm.NewProvider(llm.ProviderConfig{
		Provider:         cfg.LLMProvider,
		OllamaURL:        cfg.OllamaURL,
		OllamaModel:      cfg.OllamaModel,
		OllamaEmbedModel: cfg.OllamaEmbedModel,
		GeminiAPIKey:     cfg.GeminiAPIKey,
		GeminiModel:      cfg.GeminiModel,
		GeminiEmbedModel: cfg.GeminiEmbedModel,
	})
	if err != nil {
		return fmt.Errorf("llm: %w", err)
	}

	mem, err := memory.New(rootCtx, cfg.PostgresURI, embed)
	if err != nil {
		return fmt.Errorf("memory: %w", err)
	}
	defer mem.Close()

	reg := tools.NewRegistry()
	reg.Register(tools.NewClockTool())
	reg.Register(tools.NewYouTubeMusicTool())
	reg.Register(tools.NewStopMusicTool())
	reg.Register(tools.NewMemorySearchTool(mem))
	reg.Register(tools.NewMemoryRecentTool(mem))

	ttsBackend, err := tts.New(cfg.TTSBackend, cfg.TTSVoice)
	if err != nil {
		return fmt.Errorf("tts: %w", err)
	}

	whisper, err := stt.New(cfg.WhisperModelPath, cfg.WhisperLanguage)
	if err != nil {
		return fmt.Errorf("stt: %w", err)
	}

	mic, err := audio.NewMic()
	if err != nil {
		return fmt.Errorf("mic: %w", err)
	}
	defer mic.Close()

	ag := agent.New(chat, mem, reg, cfg.AgentMaxIterations, cfg.AgentContextMaxMessages)

	bargeIn := strings.EqualFold(os.Getenv("BARGE_IN"), "true")
	mode := "mic muted during Tío's turn"
	if bargeIn {
		mode = "mic stays on (BARGE_IN=true) — headphones recommended"
	}

	fmt.Println("Tío is listening.", mode+".")
	fmt.Println("Provider:", cfg.LLMProvider, "  TTS:", cfg.TTSBackend, "  voice:", orDefault(cfg.TTSVoice, "<system default>"))
	fmt.Println("Ctrl-C to quit.")

	loop := newConversationLoop(rootCtx, ag, whisper, ttsBackend, mic, bargeIn)
	return loop.run()
}

func orDefault(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}

// conversationLoop owns the at-most-one in-flight turn. New mic segments
// interrupt any in-flight turn before starting a new one.
//
// bargeIn=false (default): mic is muted for the whole turn so the agent's own
// voice through speakers doesn't echo back as fake user input.
// bargeIn=true (set BARGE_IN=true): mic stays on — only safe with headphones.
type conversationLoop struct {
	rootCtx  context.Context
	agent    *agent.Agent
	stt      *stt.Whisper
	tts      tts.Backend
	mic      *audio.Mic
	bargeIn  bool

	mu          sync.Mutex
	turnCancel  context.CancelFunc
	turnSpeaker *audio.Speaker
	turnDone    chan struct{}
}

func newConversationLoop(ctx context.Context, ag *agent.Agent, w *stt.Whisper, t tts.Backend, m *audio.Mic, bargeIn bool) *conversationLoop {
	return &conversationLoop{rootCtx: ctx, agent: ag, stt: w, tts: t, mic: m, bargeIn: bargeIn}
}

func (l *conversationLoop) run() error {
	for {
		select {
		case <-l.rootCtx.Done():
			l.interrupt()
			return nil
		case seg, ok := <-l.mic.Segments():
			if !ok {
				return nil
			}
			// Anything new from the mic interrupts whatever Tío was saying.
			l.interrupt()

			text, err := l.stt.Transcribe(seg.PCM)
			if err != nil {
				fmt.Fprintln(os.Stderr, "stt error:", err)
				continue
			}
			if text == "" {
				continue
			}
			fmt.Println("\n--------------------------------------------------------------------")
			fmt.Println("\nYou:", text)

			l.startTurn(text)
		}
	}
}

// interrupt cancels the in-flight turn (if any) and waits for it to settle.
func (l *conversationLoop) interrupt() {
	l.mu.Lock()
	cancel := l.turnCancel
	speaker := l.turnSpeaker
	done := l.turnDone
	l.turnCancel = nil
	l.turnSpeaker = nil
	l.turnDone = nil
	l.mu.Unlock()

	if speaker != nil {
		speaker.Stop()
	}
	if cancel != nil {
		cancel()
	}
	if done != nil {
		<-done
	}
}

func (l *conversationLoop) startTurn(text string) {
	turnCtx, turnCancel := context.WithCancel(l.rootCtx)
	speaker := audio.NewSpeaker(l.tts)
	done := make(chan struct{})

	l.mu.Lock()
	l.turnCancel = turnCancel
	l.turnSpeaker = speaker
	l.turnDone = done
	l.mu.Unlock()

	// Mute the mic while Tío speaks, unless the user opted into barge-in.
	if !l.bargeIn {
		l.mic.Pause()
	}

	go func() {
		defer close(done)
		defer func() {
			if !l.bargeIn {
				l.mic.Resume()
			}
		}()
		reply, err := l.agent.Turn(turnCtx, text, speaker)
		if err != nil && !errors.Is(err, context.Canceled) {
			fmt.Fprintln(os.Stderr, "turn error:", err)
			return
		}
		if reply != "" {
			fmt.Println("\n Angela:", reply)
		}
	}()
}
