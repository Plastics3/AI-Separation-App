import os, sys, queue, threading, time, subprocess
import numpy as np
import sounddevice as sd
import sys
import os
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from openunmix import utils
import soundfile as sf
from pydub import AudioSegment


import onnxruntime as ort    # pip install onnxruntime
MDX_SESS = {}  # name -> onnxruntime.InferenceSession

#piano weights model path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "PianoSeparationModelWeights")
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")


# ========= SETTINGS =========
SR_MODEL       = 44100
SEGMENT_SEC    = 6.50 #default 6.0
OVERLAP        = 0.60 #default 0.80
HOP_SEC        = SEGMENT_SEC * (1.0 - OVERLAP)
OUTPUT_CHS     = 1
BLOCKSIZE      = 4096
PREROLL_BLOCKS = 6
#TARGETS        = ["vocals", "drums", "bass", "other"]
TARGETS = ["vocals", "drums", "bass", "other"]
VOLUME = {"gain": 1.0}

ACTIVE_STEMS = {
    "vocals": True,
    "drums": False,
    "bass": False,
    "piano": False,
    "other": False,
}

STEM_GAIN = {
    "vocals": 0.9, # reduce vocals by default
    "drums": 1.0,
    "bass": 1.2,   # boost bass by default
    "piano": 20.0,  # boost piano by default
    "other": 1.0,
}
# Optional: choose output device, e.g., Mac speakers were index 1
# sd.default.device = (None, 1)
# ============================

# Queues/flags shared with worker threads
blocks_q = queue.Queue(maxsize=32)
stop_flag = threading.Event()
threads_started = False
PIANO_SEPARATOR = None
piano = None
path = None


def run_onnx_single_target(sess: ort.InferenceSession, block_stereo_np: np.ndarray) -> np.ndarray:
    """Input: (N,2) float32. Output: (N,2) float32 single-instrument estimate."""
    x = block_stereo_np
    if x.ndim == 1 or x.shape[1] == 1:
        x = np.repeat(x.reshape(-1,1), 2, axis=1)
    x = x.T[None, ...].astype(np.float32)  # (1,2,N)
    inp = sess.get_inputs()[0].name
    y = sess.run(None, {inp: x})[0]        # (1,2,N) or (1,1,N) or (2,N)
    y = np.asarray(y)
    if y.ndim == 3: y = y[0]               # (C,N)
    if y.ndim == 2: y = y.T                # (N,C)
    if y.shape[1] == 1: y = np.repeat(y, 2, axis=1)
    return y.astype(np.float32)


def pick_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("using CUDA GPU.")
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("using Apple Silicon GPU. Make sure to install PyTorch with MPS support.")
    else:
        dev = torch.device("cpu")
        print("using CPU. Playback may be choppy.")
    return dev

DEVICE = pick_device()
separator = None  # loaded lazily after GUI starts

PIANO_SEPARATOR = utils.load_separator(
        model_str_or_path=MODEL_DIR,
        targets=["Piano"],
        residual=True,
        niter=1,
        device=DEVICE,
        pretrained=True,
    )

def ffmpeg_decode_to_tensor(path, sr=SR_MODEL, channels=2):
    """
    Native Python decoding using soundfile. 
    Bypasses subprocess to avoid [WinError 6].
    """
    try:
        # Read the file directly using the soundfile library
        data, file_sr = sf.read(path, dtype='float32')
        
        # Ensure we have the correct number of channels (N, C)
        if data.ndim == 1:
            # Mono to Stereo
            data = np.repeat(data[:, np.newaxis], channels, axis=1)
        elif data.shape[1] > channels:
            # Downmix to requested channels
            data = data[:, :channels]
        
        # Convert to the (C, N) tensor format the rest of your code expects
        wav = torch.from_numpy(data).t().contiguous()
        
        return wav, file_sr
        
    except Exception as e:
        raise RuntimeError(f"Could not load audio file: {e}")


def reader_worker(audio_path):
    """Decode whole file, then slide a window and enqueue windowed stems."""
    try:
        status_var.set(f"Decoding: {os.path.basename(audio_path)}")
        waveform, _ = ffmpeg_decode_to_tensor(audio_path, sr=SR_MODEL, channels=2)  # (C,N)
        x = waveform.t().numpy().astype("float32")  # (N,2)
        segN = int(SEGMENT_SEC * SR_MODEL)
        hopN = int(HOP_SEC * SR_MODEL)
        pos = 0
        while not stop_flag.is_set() and pos < x.shape[0]:
            end = min(pos + segN, x.shape[0])
            block = x[max(0, end - segN):end]      # (segN,2)
            separate_and_enqueue(block)
            pos += hopN
        stop_flag.set()
    except Exception as e:
        status_var.set("Error")
            
        stop_flag.set()


def separate_and_enqueue(block_in):
    """Run UMXHQ for 4 stems, optionally run ONNX single-target models for extras."""
    global separator
    if block_in.ndim == 1 or block_in.shape[1] == 1:
        block_in = np.repeat(block_in.reshape(-1,1), 2, axis=1)

    with torch.no_grad():
        x = torch.from_numpy(block_in.T).unsqueeze(0).to(DEVICE, torch.float32)  # (1,2,N)
        y = separator(x)                                                         # (1,4,2,T) or (4,2,T)
        est = y[0].cpu().numpy() if y.dim() == 4 else y.cpu().numpy()           # (4,2,T)

    stems_st = {
        "vocals": est[0].T, "drums": est[1].T, "bass": est[2].T, "other": est[3].T
    }  # each (N,2)

    # Optional: derive residual ‘other’ from sum, if you prefer: other = block_in - sum(known)
    # mix_est = stems_st["vocals"] + stems_st["drums"] + stems_st["bass"] + stems_st["other"]

    with torch.no_grad():
        px = torch.from_numpy(block_in.T).unsqueeze(0).to(DEVICE, torch.float32)
        p_est = PIANO_SEPARATOR(px)
        p_est = PIANO_SEPARATOR.to_dict(p_est)

    global piano
    piano = p_est["Piano"][0].cpu().numpy().T  # (N, 2)
    stems_st["piano"] = piano

    # Window & enqueue mono chunks
    N = next(iter(stems_st.values())).shape[0]
    w = np.hanning(N).astype(np.float32)
    out = {}
    for name, s in stems_st.items():
        mono = np.mean(s, axis=1) * w
        out[name] = mono.reshape(-1, 1)
    blocks_q.put(out, block=True)


def player_worker():
    """Overlap–add with hop slicing, feed to sounddevice."""
    segN = int(SEGMENT_SEC * SR_MODEL)
    hopN = int(HOP_SEC * SR_MODEL)
    ola = np.zeros((segN + hopN, 1), dtype=np.float32)
    write_pos = 0
    ready = np.zeros((0,1), dtype=np.float32)

    def callback(outdata, frames, tinfo, status):
        nonlocal ready, ola, write_pos
        # Fill ready buffer until we have enough frames
        while ready.shape[0] < frames and not stop_flag.is_set():
            try:
                blk_dict = blocks_q.get_nowait()
            except queue.Empty:
                break

            N = next(iter(blk_dict.values())).shape[0]
            # Initialize overlap-add buffer
            end = write_pos + N
            if end > ola.shape[0]:
                ola = np.vstack([ola, np.zeros((end - ola.shape[0], 1), dtype=np.float32)])

            # Mix all active stems
            out_block = np.zeros((N, 1), dtype=np.float32)
            for name, active in ACTIVE_STEMS.items():
                if active:
                    s = blk_dict.get(name)
                    if s is not None:
                        out_block += s * STEM_GAIN.get(name, 1.0)

            # Apply overlap-add
            ola[write_pos:end] += out_block

            # Take hop-sized slice for playback
            slice_end = write_pos + int(HOP_SEC * SR_MODEL)
            ready = np.vstack([ready, ola[write_pos:slice_end].copy()])
            ola[write_pos:slice_end] = 0.0
            write_pos += int(HOP_SEC * SR_MODEL)

        # Output frames
        if ready.shape[0] >= frames:
            out = ready[:frames] * VOLUME["gain"]   # apply global volume
            np.clip(out, -1.0, 1.0, out=out)       # avoid clipping
            outdata[:] = out
            ready = ready[frames:]
        else:
            # If not enough ready frames, output silence
            outdata[:] = 0.0

    try:
        with sd.OutputStream(
            samplerate=SR_MODEL,
            channels=OUTPUT_CHS,
            dtype='float32',
            blocksize=0, # Setting to 0 lets the OS/Laptop driver decide the best buffer
            callback=callback
        ):
            # pre-roll
            while blocks_q.qsize() < PREROLL_BLOCKS and not stop_flag.is_set():
                time.sleep(0.05)
            status_var.set("Playing…")
            while not stop_flag.is_set():
                time.sleep(0.05)
    except Exception as e:
        status_var.set("Error")
        # Instead of messagebox.showerror("...", str(e))
        root.after(0, lambda: messagebox.showerror("Error", str(e)))
        stop_flag.set()

# ---------------- GUI ----------------
def on_open_file():
    global path
    if threads_started:
        messagebox.showinfo("Busy", "Stop current playback before opening another file.")
        return
    path = filedialog.askopenfilename(
        title="Choose audio file",
        filetypes=[("Audio", "*.wav *.flac *.mp3 *.m4a *.aac *.ogg *.mp4"), ("All files","*.*")]
    )
    if not path:
        return
    start_playback(path)

def start_playback(path):
    global separator, threads_started
    try:
        status_var.set("Loading model…")
        # lazy-load (or reload) model
        if separator is None:
            separator = torch.hub.load(
                'sigsep/open-unmix-pytorch', 
                'umxhq', 
                verbose=False,
                trust_repo=True,
                skip_validation=True
            ).eval().to(DEVICE)
        # reset state
        while not blocks_q.empty():
            try: blocks_q.get_nowait()
            except queue.Empty: break
        stop_flag.clear()
        status_var.set("Preparing…")
        global PIANO_SEPARATOR

        if PIANO_SEPARATOR is None:
            PIANO_SEPARATOR = utils.load_separator(
                model_str_or_path=MODEL_DIR,
                targets=["Piano"],
                residual=True,
                niter=1,
                device=DEVICE,
                pretrained=True,
            )
            PIANO_SEPARATOR.freeze()
            PIANO_SEPARATOR.to(DEVICE)
        # start threads
        t1 = threading.Thread(target=reader_worker, args=(path,), daemon=True)
        t2 = threading.Thread(target=player_worker, daemon=True)
        t1.start(); t2.start()
        threads_started = True
    except Exception as e:
        status_var.set("Error")
        messagebox.showerror("Model error", str(e))
        stop_flag.set()
        threads_started = False


def on_stop():
    global threads_started
    stop_flag.set()
    threads_started = False
    status_var.set("Stopped")

def on_quit():
    on_stop()
    root.after(150, root.destroy)

def on_save():
    global piano, path, PIANO_SEPARATOR
    song_path = filedialog.askopenfilename(
        title="Choose audio file",
        filetypes=[("Audio", "*.wav *.flac *.mp3 *.m4a *.aac *.ogg *.mp4"), ("All files","*.*")]
    )
    out_path = filedialog.asksaveasfilename(
        title=f"Piano {str(path)}",
        defaultextension=".wav",
        filetypes=[("WAV", "*.wav"), ("All files","*.*")]
    )
    if not out_path:
        return
    
    audio, sr = sf.read(song_path)
    audio = torch.tensor(audio.T, dtype=torch.float32)
    audio = audio.unsqueeze(0)  # (1, channels, samples)

    print("Separating piano...")
    with torch.no_grad():
        estimates = PIANO_SEPARATOR(audio)
    estimates = PIANO_SEPARATOR.to_dict(estimates)

    piano = estimates["Piano"][0]        # (channels, samples)
    piano = piano.cpu().numpy().T        # (samples, channels)
    piano = estimates["Piano"][0].cpu().numpy().T
    sf.write(out_path, piano, samplerate=44100)
    print(f"Saved: {out_path}")
    song = AudioSegment.from_wav(out_path)
    song = song + 20
    song.export(out_path,"wav")
    

btns = {}

def toggle_stem(name):
    """Toggle a stem on/off for multi-stem playback"""
    ACTIVE_STEMS[name] = not ACTIVE_STEMS[name]
    # Update button appearance
    if name in btns:
        btns[name].config(relief="sunken" if ACTIVE_STEMS[name] else "raised")
    status_var.set(f"Active stems: {', '.join([n for n, a in ACTIVE_STEMS.items() if a])}")

# Build the window
root = tk.Tk()
root.title("UMX Real-Time Player")
root.geometry("480x550")
root.resizable(False, False)

top_frame = tk.Frame(root, padx=10, pady=10)
top_frame.pack(fill="x")

btn_open = tk.Button(top_frame, text="Open File…", width=12, command=on_open_file)
btn_open.pack(side="left")

btn_stop = tk.Button(top_frame, text="Stop", width=8, command=on_stop)
btn_stop.pack(side="left", padx=6)

btn_quit = tk.Button(top_frame, text="Quit", width=8, command=on_quit)
btn_quit.pack(side="left",padx=6)

Save_songs = tk.Button(top_frame, text="Save Songs", width=12, command=on_save)
Save_songs.pack(side="left", padx=6)

status_var = tk.StringVar(value=f"Device: {DEVICE}  |  Ready")
lbl_status = tk.Label(root, textvariable=status_var, anchor="w", padx=12)
lbl_status.pack(fill="x", pady=(0,6))

mid = tk.Frame(root, padx=10, pady=10)
mid.pack(fill="both", expand=True)


tk.Label(mid, text="Choose stem to play (mono):", anchor="w").pack(anchor="w", pady=(0,8))

row = tk.Frame(mid)
row.pack()
btns = {}  # store buttons for updating relief
for name, label in [("vocals","Vocals"), ("drums","Drums"),
                    ("bass","Bass"), ("piano","Piano"), ("other","Other")]:
    b = tk.Button(row, text=label, width=10,
                  relief="sunken" if ACTIVE_STEMS[name] else "raised",
                  command=lambda n=name: toggle_stem(n))
    b.pack(side="left", padx=6)
    btns[name] = b

for name in ACTIVE_STEMS.keys():
    tk.Label(mid, text=name.capitalize()).pack(anchor="w")
    s = tk.Scale(mid, from_=0.0, to=2.0, resolution=0.01,
                 orient="horizontal")
    s.set(1.0)
    s.pack(anchor="w", fill="x")

#volume slider
bottom = tk.Frame(root, padx=10, pady=6)
bottom.pack(side="bottom", fill="x")

vol_frame = tk.Frame(bottom)
vol_frame.pack(side="right")

def on_volume_change(val):
    VOLUME["gain"] = float(val)

tk.Label(vol_frame, text="🔊", font=("Segoe UI Emoji", 14)).pack(side="left", padx=(0,6))

vol_slider = tk.Scale(
    vol_frame,
    from_=0.0,
    to=2.0,
    resolution=0.01,
    orient="horizontal",
    length=100,
    showvalue=True, #show the current value = True
    command=on_volume_change
)
vol_slider.set(1.0)
vol_slider.pack(side="left")

root.protocol("WM_DELETE_WINDOW", on_quit)
root.mainloop()