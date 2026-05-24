package audio

import (
	"encoding/binary"
	"fmt"
	"sync"
	"time"

	"github.com/gen2brain/malgo"
	"github.com/maxhawkins/go-webrtcvad"
)

const (
	micSampleRate   = 16000
	micChannels     = 1
	micFrameMs      = 30
	micFrameSamples = micSampleRate * micFrameMs / 1000 // 480 samples
	micFrameBytes   = micFrameSamples * 2               // int16 LE

	silenceEndMs   = 700 // end-of-utterance silence threshold
	minSpeechMs    = 400 // discard shorter clips (kills most hallucinations)
	silenceFrames  = silenceEndMs / micFrameMs
	minSpeechFrames = minSpeechMs / micFrameMs
)

// Segment is one utterance: PCM samples bracketed by detected silence.
type Segment struct {
	PCM        []int16
	SampleRate int
}

// Mic captures the system mic, frames it at 30ms, runs WebRTC VAD, and emits
// a Segment when an utterance ends (>= silenceEndMs of trailing silence).
//
// Always-on: Mic does not pause for TTS — interruption (barge-in) is handled
// at a higher level by the agent loop, which cancels in-flight TTS when a new
// segment arrives.
type Mic struct {
	ctx      *malgo.AllocatedContext
	device   *malgo.Device
	vad      *webrtcvad.VAD
	segments chan Segment

	mu             sync.Mutex
	residual       []byte  // leftover bytes <30ms from previous callback
	speechBuf      []int16 // current utterance buffer
	silenceCount   int     // consecutive non-speech frames since last speech frame
	startedSpeech  bool
	paused         bool    // when true, the capture callback drops frames
}

func NewMic() (*Mic, error) {
	vad, err := webrtcvad.New()
	if err != nil {
		return nil, fmt.Errorf("audio/mic: vad init: %w", err)
	}
	if err := vad.SetMode(3); err != nil {
		return nil, fmt.Errorf("audio/mic: vad mode: %w", err)
	}

	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		return nil, fmt.Errorf("audio/mic: malgo init: %w", err)
	}

	m := &Mic{
		ctx:      ctx,
		vad:      vad,
		segments: make(chan Segment, 4),
	}

	cfg := malgo.DefaultDeviceConfig(malgo.Capture)
	cfg.Capture.Format = malgo.FormatS16
	cfg.Capture.Channels = micChannels
	cfg.SampleRate = micSampleRate
	cfg.Alsa.NoMMap = 1

	device, err := malgo.InitDevice(ctx.Context, cfg, malgo.DeviceCallbacks{
		Data: func(_, in []byte, _ uint32) { m.onData(in) },
	})
	if err != nil {
		_ = ctx.Uninit()
		ctx.Free()
		return nil, fmt.Errorf("audio/mic: device init: %w", err)
	}
	m.device = device

	if err := device.Start(); err != nil {
		device.Uninit()
		_ = ctx.Uninit()
		ctx.Free()
		return nil, fmt.Errorf("audio/mic: device start: %w", err)
	}

	return m, nil
}

// Segments returns the channel of completed utterances. The channel closes
// when Close is called.
func (m *Mic) Segments() <-chan Segment { return m.segments }

func (m *Mic) Close() error {
	m.device.Uninit()
	_ = m.ctx.Uninit()
	m.ctx.Free()
	close(m.segments)
	return nil
}

// Pause makes the capture callback drop frames and discards any in-progress
// utterance. Used to mute the mic while TTS is playing — without this, the
// agent's own voice through speakers gets transcribed and fed back as input.
func (m *Mic) Pause() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.paused = true
	m.speechBuf = m.speechBuf[:0]
	m.silenceCount = 0
	m.startedSpeech = false
	m.residual = nil
}

// Resume re-enables frame processing.
func (m *Mic) Resume() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.paused = false
}

// onData processes whatever buffer miniaudio handed us. Frames are exactly
// 30ms; bytes that don't align are carried over.
func (m *Mic) onData(in []byte) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.paused {
		return
	}

	if len(m.residual) > 0 {
		in = append(m.residual, in...)
		m.residual = nil
	}

	for len(in) >= micFrameBytes {
		frame := in[:micFrameBytes]
		in = in[micFrameBytes:]
		m.processFrame(frame)
	}
	if len(in) > 0 {
		m.residual = append([]byte(nil), in...)
	}
}

func (m *Mic) processFrame(frameBytes []byte) {
	isSpeech, err := m.vad.Process(micSampleRate, frameBytes)
	if err != nil {
		return
	}

	samples := bytesToInt16(frameBytes)

	if isSpeech {
		m.speechBuf = append(m.speechBuf, samples...)
		m.silenceCount = 0
		m.startedSpeech = true
		return
	}

	// non-speech frame
	if !m.startedSpeech {
		return // still in pre-speech silence; ignore
	}

	// trailing silence after speech started
	m.speechBuf = append(m.speechBuf, samples...)
	m.silenceCount++
	if m.silenceCount >= silenceFrames {
		m.emit()
	}
}

func (m *Mic) emit() {
	defer func() {
		m.speechBuf = m.speechBuf[:0]
		m.silenceCount = 0
		m.startedSpeech = false
	}()

	totalFrames := len(m.speechBuf) / micFrameSamples
	if totalFrames < minSpeechFrames {
		return // discard too-short clip
	}
	pcm := make([]int16, len(m.speechBuf))
	copy(pcm, m.speechBuf)
	select {
	case m.segments <- Segment{PCM: pcm, SampleRate: micSampleRate}:
	default:
		// downstream slow / paused — drop rather than block the audio callback
	}
}

func bytesToInt16(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[2*i : 2*i+2]))
	}
	return out
}

// Duration of a Segment in milliseconds.
func (s Segment) DurationMs() int {
	if s.SampleRate == 0 {
		return 0
	}
	return int(time.Duration(len(s.PCM)) * 1000 / time.Duration(s.SampleRate))
}
