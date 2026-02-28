import torch
import soundfile as sf
from openunmix import utils
import soundfile as sf


# ================= CONFIG =================

MODEL_DIR = r"input here"  # change to your model directory (e.g., "PianoSeperationModelWeights")
INPUT_AUDIO = r"input here"  # change to your input audio file path
OUTPUT_AUDIO = r"input here"  # change to your output audio file path (e.g., "piano_only.wav")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================


def separate_piano():
    print("Loading separator...")

    separator = utils.load_separator(
        model_str_or_path=MODEL_DIR,
        targets=["Piano"],
        residual=True,
        niter=1,
        device=DEVICE,
        pretrained=True,
    )

    separator.freeze()
    separator.to(DEVICE)

    print("Loading audio...")
    audio, sr = sf.read(INPUT_AUDIO, always_2d=True)

    audio = torch.tensor(audio.T, dtype=torch.float32)
    audio = audio.unsqueeze(0)  # (1, channels, samples)

    print("Separating piano...")
    with torch.no_grad():
        estimates = separator(audio)
    estimates = separator.to_dict(estimates)

    piano = estimates["Piano"][0]        # (channels, samples)
    piano = piano.cpu().numpy().T        # (samples, channels)
    piano = estimates["Piano"][0].cpu().numpy().T
    out_path = OUTPUT_AUDIO
    sf.write(out_path, piano, samplerate=44100)
    print(f"Saved: {out_path}")


    print("Done ✔")


if __name__ == "__main__":
    separate_piano()
