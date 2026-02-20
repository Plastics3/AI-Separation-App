import os
import soundfile as sf
import librosa
import numpy as np

ROOT = r"input here"  # change to your dataset root directory (e.g., "dataset")
SPLITS = ["train", "valid"]

def load_audio(path):
    audio, sr = sf.read(path, always_2d=True)
    return audio, sr

def resample(audio, orig_sr, target_sr):
    audio = audio.T  # (channels, samples)
    audio_rs = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio_rs.T

def trim_pair(mix_path, tgt_path):
    mix_audio, mix_sr = load_audio(mix_path)
    tgt_audio, tgt_sr = load_audio(tgt_path)

    # Fix sample rate mismatch
    if mix_sr != tgt_sr:
        print(
            f"Resampling: {os.path.basename(os.path.dirname(mix_path))} "
            f"(mix_sr={mix_sr}, tgt_sr={tgt_sr})"
        )
        tgt_audio = resample(tgt_audio, tgt_sr, mix_sr)
        tgt_sr = mix_sr

    # Fix channel mismatch
    if mix_audio.shape[1] != tgt_audio.shape[1]:
        raise ValueError(f"Channel mismatch in {mix_path}")

    # Trim to shortest
    min_len = min(len(mix_audio), len(tgt_audio))

    if len(mix_audio) != len(tgt_audio):
        print(
            f"Trimming: {os.path.basename(os.path.dirname(mix_path))} "
            f"(mix={len(mix_audio)}, tgt={len(tgt_audio)})"
        )

    mix_audio = mix_audio[:min_len]
    tgt_audio = tgt_audio[:min_len]

    sf.write(mix_path, mix_audio, mix_sr)
    sf.write(tgt_path, tgt_audio, tgt_sr)

def process_split(split):
    split_path = os.path.join(ROOT, split)

    for song_dir in sorted(os.listdir(split_path)):
        song_path = os.path.join(split_path, song_dir)
        if not os.path.isdir(song_path):
            continue

        mix_path = os.path.join(song_path, "mixture.wav")
        tgt_path = os.path.join(song_path, "Piano.wav")

        if not os.path.exists(mix_path) or not os.path.exists(tgt_path):
            print(f"Skipping (missing files): {song_path}")
            continue

        trim_pair(mix_path, tgt_path)

def main():
    for split in SPLITS:
        print(f"Processing {split}...")
        process_split(split)

    print("Done.")

if __name__ == "__main__":
    main()
