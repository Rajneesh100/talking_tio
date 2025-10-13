from TTS.api import TTS

tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=True)
tts.tts_to_file(text="Hola amigo, cómo estás?", file_path="output.wav")
