import sounddevice as sd
import numpy as np
import webrtcvad
import collections
import threading
import wave
import time
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
CHANNELS = 1

vad = webrtcvad.Vad()
vad.set_mode(2)  # 0=less aggressive, 3=more aggressive

model = WhisperModel("small", device="cpu")  # or "medium" for GPU


def transcribe_audio(audio_bytes):
    """Run Whisper transcription on the recorded voice chunk."""
    tmp = "temp.wav"
    with wave.open(tmp, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_bytes)

    segments, _ = model.transcribe(tmp)
    text = " ".join([seg.text.strip() for seg in segments])
    return text.strip()


def continuous_voice_listener(intraction_callback):
    """Continuously listen to the mic and call intraction_callback when human speech is detected."""
    ring_buffer = collections.deque(maxlen=10)
    speech_frames = []
    speaking = False
    silence_start = None

    def audio_callback(indata, frames, time_, status):
        ring_buffer.append(bytes(indata))

    def listen_loop():
        print("🎧 Voice listener started (always on)...")
        with sd.RawInputStream(samplerate=SAMPLE_RATE,
                               blocksize=FRAME_SIZE,
                               dtype='int16',
                               channels=CHANNELS,
                               callback=audio_callback):
            while True:
                if not ring_buffer:
                    time.sleep(0.01)
                    continue

                frame = ring_buffer.popleft()
                is_speech = vad.is_speech(frame, SAMPLE_RATE)

                nonlocal speaking, speech_frames, silence_start

                if is_speech:
                    if not speaking:
                        print("🎤 Detected speech...")
                        speaking = True
                        speech_frames = []
                    speech_frames.append(frame)
                    silence_start = None
                else:
                    if speaking:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > 0.8:
                            speaking = False
                            print("🛑 Speech ended. Transcribing...")
                            audio_bytes = b"".join(speech_frames)
                            text = transcribe_audio(audio_bytes)
                            if text:
                                print("🗣️ user input:", text)
                                intraction_callback(text, source="user")
                            speech_frames = []
                            silence_start = None

    t = threading.Thread(target=listen_loop, daemon=True)
    t.start()
