"""
Explore available TTS models and voices in Coqui TTS
"""

from TTS.api import TTS
import sounddevice as sd
import soundfile as sf
import numpy as np

print("=" * 80)
print("🎙️  COQUI TTS - AVAILABLE MODELS AND VOICES")
print("=" * 80)

# List all available TTS models
print("\n📋 Listing all available TTS models...\n")
print(TTS.list_models())

print("\n" + "=" * 80)
print("🔍 DETAILED MODEL INFORMATION")
print("=" * 80)

# Current model info
print("\n1️⃣  CURRENT MODEL: tts_models/en/ljspeech/tacotron2-DDC")
print("-" * 80)
print("Name: Tacotron2-DDC")
print("Language: English (en)")
print("Dataset: LJSpeech")
print("Voices: Single female voice (Linda Johnson)")
print("Quality: High quality, natural sounding")
print("Speed: Fast inference")
print("Description: Neural TTS using Tacotron2 architecture with Dynamic")
print("             Convolutional Attention. Great for general purpose TTS.")

# Multi-voice model info
print("\n2️⃣  MULTI-VOICE MODEL: tts_models/en/vctk/vits")
print("-" * 80)
print("Name: VITS")
print("Language: English (en)")
print("Dataset: VCTK (109 speakers)")
print("Voices: 109 different speakers (male & female)")
print("Quality: Very high quality, natural sounding")
print("Speed: Fast inference")
print("Description: State-of-the-art neural TTS with multiple speakers")

# Show available speakers for VCTK
print("\n📢 Loading VCTK model to show all voices...")
tts_vctk = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)

print(f"\n✅ Total voices available: {len(tts_vctk.speakers)}")
print("\n🎭 All Available Voices:")
print("-" * 80)

for i, speaker in enumerate(tts_vctk.speakers, 1):
    # Format in columns
    print(f"{speaker:8}", end="  ")
    if i % 8 == 0:
        print()  # New line every 8 speakers

print("\n\n" + "=" * 80)
print("🌟 RECOMMENDED VOICES FROM VCTK:")
print("=" * 80)

recommended = {
    "p225": "Female (British), Young, Clear",
    "p226": "Male (British), Young, Clear",
    "p227": "Male (British), Young, Deep",
    "p228": "Female (British), Young, Soft",
    "p229": "Female (British), Young, Professional",
    "p230": "Male (British), Young, Energetic",
    "p231": "Female (British), Young, Sweet",
    "p232": "Male (British), Young, Strong",
    "p233": "Female (British), Young, Warm",
    "p234": "Male (British), Young, Friendly",
    "p236": "Female (British), Young, Bright",
    "p239": "Female (British), Young, Cheerful",
    "p243": "Male (British), Young, Professional",
    "p244": "Female (British), Young, Elegant",
    "p245": "Male (British), Middle-aged, Authoritative",
    "p246": "Male (British), Young, Casual",
    "p247": "Female (British), Young, Dynamic",
    "p248": "Female (British), Young, Natural",
}

for voice_id, description in recommended.items():
    print(f"  {voice_id}: {description}")

print("\n" + "=" * 80)
print("🎵 GENERATING SAMPLE VOICES")
print("=" * 80)

text = "Hello Rajneesh, this is a sample of my voice."

# Generate samples with 3 different voices
samples = [
    ("p225", "Female - Young & Clear"),
    ("p226", "Male - Young & Clear"),
    ("p233", "Female - Warm & Friendly")
]

for voice_id, description in samples:
    print(f"\n🎙️  Generating: {voice_id} ({description})")
    audio = tts_vctk.tts(text, speaker=voice_id)
    
    filename = f"sample_{voice_id}.wav"
    sf.write(filename, audio, tts_vctk.synthesizer.output_sample_rate)
    print(f"   ✅ Saved: {filename}")
    
    print(f"   🔊 Playing...")
    sd.play(np.array(audio), samplerate=tts_vctk.synthesizer.output_sample_rate)
    sd.wait()

print("\n" + "=" * 80)
print("✅ DONE!")
print("=" * 80)
print("\n💡 To use a specific voice in your code:")
print("   tts = TTS(model_name='tts_models/en/vctk/vits')")
print("   audio = tts.tts(text='Your text here', speaker='p225')")
print("\n" + "=" * 80)

