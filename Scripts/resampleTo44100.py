import os
import soundfile as sf
import librosa
import numpy as np

ROOT = r"input here"  # change to your dataset root directory (e.g., "dataset")
SPLITS = ["train", "valid"]
TARGET_SR = 44100

def resample_if_needed(path):
    audio, sr = sf.read(path, always_2d=True)

    if sr != TARGET_SR:
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=TARGET_SR)
        audio = audio.T

        sf.write(path, audio, TARGET_SR)
        return True
    return False

def main():
    count = 0
    for split in SPLITS:
        split_path = os.path.join(ROOT, split)

        for song_dir in os.listdir(split_path):
            song_path = os.path.join(split_path, song_dir)
            if not os.path.isdir(song_path):
                continue

            mix_path = os.path.join(song_path, "mixture.wav")
            if os.path.exists(mix_path):
                if resample_if_needed(mix_path):
                    count += 1

    print(f"Resampled {count} files to {TARGET_SR} Hz")

if __name__ == "__main__":
    main()
