from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import os


elevenlabs = ElevenLabs(
  api_key="api_key",
)

audio = elevenlabs.text_to_speech.convert(
    text="you better go away",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

play(audio)

