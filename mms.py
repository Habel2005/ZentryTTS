import torch
from transformers import VitsModel, AutoTokenizer
import sounddevice as sd
import scipy.io.wavfile

def play_malayalam_tts(text):
    """
    Generates speech from Malayalam text using the MMS-TTS model and plays it.
    Also saves the audio to a file.
    """
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pre-trained model and tokenizer
    print("Loading model and tokenizer...")
    model = VitsModel.from_pretrained("facebook/mms-tts-mal").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-mal")
    
    print("Model loaded successfully.")

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Generate speech waveform
    print("Generating speech...")
    with torch.no_grad():
        output = model(**inputs).waveform

    # The output is a PyTorch tensor. We need to move it to the CPU and convert to a NumPy array for playback.
    # The tensor might have an extra batch dimension, so we squeeze it.
    speech_array = output.detach().cpu().numpy().squeeze()
    
    # Get the sampling rate from the model's configuration
    sampling_rate = model.config.sampling_rate
    
    # --- Play the audio ---
    print(f"Playing audio at {sampling_rate} Hz...")
    sd.play(speech_array, samplerate=sampling_rate)
    sd.wait()  # Wait for the audio to finish playing
    print("Playback finished.")

    # --- Save the audio to a .wav file ---
    output_filename = "malayalam_output12.wav"
    scipy.io.wavfile.write(output_filename, rate=sampling_rate, data=speech_array)
    print(f"Audio also saved to '{output_filename}'")


if __name__ == "__main__":
    malayalam_text = "സാങ്കേതികവിദ്യയുടെ വളർച്ചയോടെ, ആശയവിനിമയത്തിനുള്ള മാർഗ്ഗങ്ങൾ കൂടുതൽ ലളിതവും കാര്യക്ഷമവുമായി മാറിയിരിക്കുന്നു"
    play_malayalam_tts(malayalam_text)