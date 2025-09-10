from TTS.api import TTS
import sounddevice as sd
import soundfile as sf

# Load the multi-speaker, multi-lingual model
tts = TTS("tts_models/multilingual/multi-dataset/your_tts")

# Pick a valid speaker (remove trailing \n if any) ['female-en-5', 'female-en-5\n', 'female-pt-4\n', 'male-en-2', 'male-en-2\n', 'male-pt-3\n']
speaker = "female-en-5"

# Malayalam text
text = "നമസ്കാരം, നിങ്ങൾക്ക് എങ്ങനെയാണ് ഇന്ന് സുഖം?"
output_file = "malayalam_output.wav"

# Generate WAV specifying both speaker and language
tts.tts_to_file(text=text, file_path=output_file, speaker=speaker, language="ml")
print(f"✅ Audio saved to {output_file}")

# Play the WAV
data, sr = sf.read(output_file, dtype="float32")
sd.play(data, sr)
sd.wait()
print("✅ Playback finished")
