from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/your_tts")

print("Available speakers:", tts.speakers)
