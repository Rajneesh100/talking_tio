"""
Fish Audio OpenAudio S1-mini - Local Model Inference

To use the locally downloaded Fish Audio model, you need to:
1. Clone fish-speech repo
2. Install it
3. Use their inference tools

This script provides the setup instructions and a working alternative.
"""

import os
import sys

print("=" * 70)
print("🐟 FISH AUDIO OPENAUDIO S1-MINI - SETUP GUIDE")
print("=" * 70)

# Check if fish-speech is installed
try:
    import fish_speech
    print("✅ Fish Speech library is installed!")
    
    # If installed, try to run inference
    print("\n📋 To use Fish Audio with your local model, run:")
    print("\n  python -m fish_speech.tools.api \\")
    print("    --llama-checkpoint-path ./openaudio-s1-mini/model.pth \\")
    print("    --decoder-checkpoint-path ./openaudio-s1-mini/codec.pth \\")
    print("    --decoder-config-name firefly_gan_vq \\")
    print("    --device cuda")
    print("\nOr use their web UI:")
    print("  python -m fish_speech.webui")
    
except ImportError:
    print("⚠️  Fish Speech library not installed")
    print("\n" + "=" * 70)
    print("📦 INSTALLATION STEPS:")
    print("=" * 70)
    print("\n1️⃣  Clone Fish Speech repository:")
    print("   cd /Users/rajneesh.kumar/Desktop/gpt_tio/fishlabs")
    print("   git clone https://github.com/fishaudio/fish-speech.git")
    print("\n2️⃣  Install Fish Speech:")
    print("   cd fish-speech")
    print("   pip install -e .")
    print("\n3️⃣  Install additional dependencies:")
    print("   pip install hydra-core==1.3.2")
    print("   pip install nemo_text_processing")
    print("\n4️⃣  Run inference using their CLI:")
    print("   python tools/api.py \\")
    print("     --llama-checkpoint-path ../openaudio-s1-mini/model.pth \\")
    print("     --decoder-checkpoint-path ../openaudio-s1-mini/codec.pth")
    print("\nOr start their WebUI:")
    print("   python -m fish_speech.webui")
    print("=" * 70)

print("\n🔄 Running working TTS alternative instead...")
print("=" * 70)

# Use working TTS as alternative
try:
    from TTS.api import TTS
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    
    print("\n🎤 Loading Coqui TTS (high quality alternative)...")
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)
    
    # Support emotions in text
    text = "Hello Rajneesh! This is a high quality neural TTS system. While we set up Fish Audio, this works great!"
    
    print(f"\n🎙️  Text: {text}")
    print("🎵 Generating audio...")
    
    audio = tts.tts(text)
    
    # Save audio
    sf.write("output.wav", audio, tts.synthesizer.output_sample_rate)
    print("✅ Audio saved as output.wav")
    
    # Play audio
    print("🔊 Playing audio...")
    sd.play(np.array(audio), samplerate=tts.synthesizer.output_sample_rate)
    sd.wait()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nMake sure you have TTS installed:")
    print("  pip install TTS sounddevice soundfile")

print("\n" + "=" * 70)
