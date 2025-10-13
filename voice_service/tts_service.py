# # tts_service.py
# import pyttsx3
# import time

# engine = pyttsx3.init()

# # choose voice if multiple
# voices = engine.getProperty("voices")
# # pick a voice index you like; may vary by system
# if len(voices) > 0:
#     # engine.setProperty("voice", voices[167].id)
#     engine.setProperty('voice', voices[87].id)  # Zira = sweet US female
#     engine.setProperty('rate', 170)            # slow, gentle
#     engine.setProperty('volume', 1.0)


# #  86 lekha
# def speak_text(text, emotion="neutral", intensity=0.5):
#     """
#     emotion: one of ['neutral','happy','sad','amused','nostalgic','angry']
#     intensity: 0.0 to 1.0
#     """
#     rate = 170
#     volume = 1.0

#     if emotion == "happy":
#         rate = int(150 + 60 * intensity)
#         volume = 1.0
#     elif emotion == "sad":
#         rate = int(120 - 40 * intensity)
#         volume = max(0.6, 1.0 - 0.4 * intensity)
#     elif emotion == "amused":
#         rate = int(190 + 30 * intensity)
#         volume = 1.0
#     elif emotion == "nostalgic":
#         rate = int(140 - 20 * intensity)
#         volume = 0.85
#     elif emotion == "angry":
#         rate = int(220 + 40 * intensity)
#         volume = 1.0

#     engine.setProperty('rate', rate)
#     engine.setProperty('volume', volume)
#     engine.say(text)
#     engine.runAndWait()
#     # tiny pause
#     time.sleep(0.15)


# def main():
#     """
#     Main function to run the text-to-speech example.
#     """
#     print("Running the text-to-speech example...")
#     speak_text("please to see you again ansh")
#     print("...example finished.")

# if __name__ == "__main__":
#     main()




from TTS.api import TTS
import sounddevice as sd
import numpy as np

# pre-trained neural TTS with sweet female voice
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
text = "how this happened, i knew they will stab me in the back"

sd.play(np.array(tts.tts(text)), samplerate=tts.synthesizer.output_sample_rate)
sd.wait()




# import sounddevice as sd
# import numpy as np
# from TTS.api import TTS
# import threading

# tts = TTS(model_name="tts_models/en/vctk/vits")

# def speak_live(text):
#     wav = []
#     for chunk in tts.stream_tts(text):  # pseudo API; some Coqui versions support this
#         sd.play(np.array(chunk), samplerate=tts.synthesizer.output_sample_rate)
#         sd.wait()

# speak_live("Hey Rajneesh, this should feel more instant!")

