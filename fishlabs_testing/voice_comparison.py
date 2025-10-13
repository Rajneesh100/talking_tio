"""
Quick comparison of different TTS models and voices
"""

from TTS.api import TTS
import sounddevice as sd
import soundfile as sf
import numpy as np

print("=" * 80)

text = "Hello Rajneesh! How are you doing today? This is an example of text to speech."

# Model 1: LJSpeech (Single Female Voice)
print("\n1️⃣  MODEL: tts_models/en/ljspeech/tacotron2-DDC")


# sucks
tts_ljspeech = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
audio1 = tts_ljspeech.tts(text)
sf.write("ljspeech_sample.wav", audio1, tts_ljspeech.synthesizer.output_sample_rate)

print("🔊 Playing...")
sd.play(np.array(audio1), samplerate=tts_ljspeech.synthesizer.output_sample_rate)
sd.wait()



# Model 2: VCTK VITS (Multiple Voices)
print("\n" + "=" * 80)
print("2️⃣  MODEL: tts_models/en/vctk/vits")

tts_vctk = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
print(f"✅ Loaded! Available voices: {len(tts_vctk.speakers)}")

# Try 3 different voices
voices_to_try = [
    ("p225", "Young British Female"),
]

for voice_id, description in voices_to_try:
    print(f"\n🎙️  Voice: {voice_id} ({description})")
    audio = tts_vctk.tts(text, speaker=voice_id)
    
    filename = f"vctk_{voice_id}.wav"
    sf.write(filename, audio, tts_vctk.synthesizer.output_sample_rate)
    print(f"   ✅ Saved: {filename}")
    
    print("   🔊 Playing...")
    sd.play(np.array(audio), samplerate=tts_vctk.synthesizer.output_sample_rate)
    sd.wait()

