import torch
from transformers import VitsModel, AutoTokenizer
import sounddevice as sd
import numpy as np

# --- Load model once (warm start) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = VitsModel.from_pretrained("facebook/mms-tts-mal").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-mal")

# Half precision on GPU for speed
if device != "cpu":
    model = model.half()

def tts_malayalam(text: str):
    """Generate Malayalam speech and play immediately"""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Inference
    with torch.no_grad():
        output = model(**inputs).waveform

    # Convert to numpy
    speech = output.squeeze().detach().cpu().numpy()

    # Normalize
    speech = speech / np.max(np.abs(speech))

    # Play async (non-blocking)
    sd.play(speech, samplerate=model.config.sampling_rate)
    return speech


if __name__ == "__main__":
    print("✅ Malayalam TTS ready. Type text (or 'quit' to exit).")
    while True:
        text = input("\nEnter Malayalam text: ")
        if text.strip().lower() == "quit":
            print("👋 Exiting...")
            break
        if text.strip():
            tts_malayalam(text)
