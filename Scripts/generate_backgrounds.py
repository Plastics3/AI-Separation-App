import os
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ================= CONFIG =================

#generate synthetic music for training/testing
SR = 44100

GUITAR_DIR = r"replace_with_your_guitar_wav_folder"  # put your guitar wavs here
OUT_DIR = "mixtures"

GAIN_GUITAR = 0.9
GAIN_MUSIC = 0.35

os.makedirs(OUT_DIR, exist_ok=True)

# ================= SYNTH =================

def adsr_env(n, sr, a=0.01, d=0.1, s=0.7, r=0.2):
    env = np.ones(n)

    A = int(a * sr)
    D = int(d * sr)
    R = int(r * sr)

    total = A + D + R
    if total >= n:
        scale = n / total
        A = int(A * scale)
        D = int(D * scale)
        R = n - A - D

    if A > 0:
        env[:A] = np.linspace(0, 1, A, endpoint=False)
    if D > 0:
        env[A:A+D] = np.linspace(1, s, D, endpoint=False)
    if R > 0:
        env[-R:] = np.linspace(s, 0, R)

    return env


def synth_note(freq, n, sr):
    t = np.arange(n) / sr
    signal = (
        np.sin(2 * np.pi * freq * t) +
        0.5 * np.sin(2 * np.pi * freq * 2 * t) +
        0.25 * np.sin(2 * np.pi * freq * 3 * t)
    )
    env = adsr_env(n, sr)
    return signal * env


def generate_music(n, sr):
    music = np.zeros(n)

    chord_freqs = [
        [261.63, 329.63, 392.00],  # C
        [293.66, 369.99, 440.00],  # D
        [329.63, 415.30, 493.88],  # E
        [349.23, 440.00, 523.25],  # F
    ]

    pos = 0
    while pos < n:
        dur = int(sr * np.random.uniform(0.4, 1.2))
        dur = min(dur, n - pos)

        chord = np.random.choice(len(chord_freqs))
        for f in chord_freqs[chord]:
            music[pos:pos+dur] += synth_note(f, dur, sr)

        pos += dur

    # simple rhythm modulation
    lfo = 0.5 * (1 + np.sin(2 * np.pi * 1.5 * np.arange(n) / sr))
    music *= lfo

    # normalize
    music /= np.max(np.abs(music) + 1e-8)
    return music


# ================= MIXING =================

def make_track(guitar_path):
    name = os.path.splitext(os.path.basename(guitar_path))[0]
    track_dir = os.path.join(OUT_DIR, name)
    os.makedirs(track_dir, exist_ok=True)

    guitar, sr = sf.read(guitar_path)
    if guitar.ndim > 1:
        guitar = guitar.mean(axis=1)

    n = len(guitar)

    music = generate_music(n, sr)

    mixture = (
        GAIN_GUITAR * guitar +
        GAIN_MUSIC * music
    )

    mixture /= np.max(np.abs(mixture) + 1e-8)

    sf.write(os.path.join(track_dir, "mixture.wav"), mixture, sr)
    sf.write(os.path.join(track_dir, "guitar.wav"), guitar, sr)
    sf.write(os.path.join(track_dir, "other.wav"), music, sr)
    


# ================= MAIN =================

def main():
    files = [
        os.path.join(GUITAR_DIR, f)
        for f in os.listdir(GUITAR_DIR)
        if f.lower().endswith(".wav")
    ]

    for g in tqdm(files, desc="Generating mixtures"):
        make_track(g)


if __name__ == "__main__":
    main()
