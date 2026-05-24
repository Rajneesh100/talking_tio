package stt

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// whisperBinary is the whisper.cpp CLI. Install on macOS via `brew install
// whisper-cpp`. The Go bindings (cgo) are faster but require building the
// library locally — drop them behind this same interface later if needed.
const whisperBinary = "whisper-cli"

type Whisper struct {
	modelPath string
	language  string
}

func New(modelPath, language string) (*Whisper, error) {
	if _, err := exec.LookPath(whisperBinary); err != nil {
		return nil, fmt.Errorf("stt: %q not found in PATH — `brew install whisper-cpp`", whisperBinary)
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("stt: model %q not found — `make whisper-pull`", modelPath)
	}
	return &Whisper{modelPath: modelPath, language: language}, nil
}

// Transcribe writes the PCM to a temp WAV, runs whisper-cli on it, and returns
// the transcript. Returns empty string for hallucinations.
func (w *Whisper) Transcribe(pcm []int16) (string, error) {
	tmp, err := os.CreateTemp("", "tio-*.wav")
	if err != nil {
		return "", fmt.Errorf("stt: tempfile: %w", err)
	}
	wavPath := tmp.Name()
	tmp.Close()
	defer os.Remove(wavPath)

	if err := writeWAV16k(wavPath, pcm); err != nil {
		return "", fmt.Errorf("stt: write wav: %w", err)
	}

	outPrefix := strings.TrimSuffix(wavPath, filepath.Ext(wavPath))
	cmd := exec.Command(whisperBinary,
		"-m", w.modelPath,
		"-l", w.language,
		"-nt",         // no timestamps
		"-otxt",       // write <prefix>.txt
		"-of", outPrefix,
		"-t", "4",     // threads
		wavPath,
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("stt: whisper-cli: %w (%s)", err, stderr.String())
	}

	txtPath := outPrefix + ".txt"
	defer os.Remove(txtPath)
	raw, err := os.ReadFile(txtPath)
	if err != nil {
		return "", fmt.Errorf("stt: read transcript: %w", err)
	}
	text := strings.TrimSpace(string(raw))
	if IsHallucination(text) {
		return "", nil
	}
	return text, nil
}

// writeWAV16k writes mono 16-bit PCM at 16kHz as a WAV file.
func writeWAV16k(path string, pcm []int16) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	dataLen := uint32(len(pcm) * 2)
	var hdr bytes.Buffer
	hdr.WriteString("RIFF")
	binary.Write(&hdr, binary.LittleEndian, uint32(36)+dataLen)
	hdr.WriteString("WAVE")
	hdr.WriteString("fmt ")
	binary.Write(&hdr, binary.LittleEndian, uint32(16))      // PCM chunk size
	binary.Write(&hdr, binary.LittleEndian, uint16(1))       // PCM format
	binary.Write(&hdr, binary.LittleEndian, uint16(1))       // mono
	binary.Write(&hdr, binary.LittleEndian, uint32(16000))   // sample rate
	binary.Write(&hdr, binary.LittleEndian, uint32(16000*2)) // byte rate
	binary.Write(&hdr, binary.LittleEndian, uint16(2))       // block align
	binary.Write(&hdr, binary.LittleEndian, uint16(16))      // bits per sample
	hdr.WriteString("data")
	binary.Write(&hdr, binary.LittleEndian, dataLen)

	if _, err := f.Write(hdr.Bytes()); err != nil {
		return err
	}
	return binary.Write(f, binary.LittleEndian, pcm)
}
