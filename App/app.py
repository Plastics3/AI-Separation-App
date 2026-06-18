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


import onnxruntime as ort
MDX_SESS = {}

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "PianoSeparationModelWeights")
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")


# ========= SETTINGS =========
SR_MODEL       = 44100
SEGMENT_SEC    = 6.50
OVERLAP        = 0.60
HOP_SEC        = SEGMENT_SEC * (1.0 - OVERLAP)
OUTPUT_CHS     = 1
BLOCKSIZE      = 4096
PREROLL_BLOCKS = 6
TARGETS        = ["vocals", "drums", "bass", "other"]
VOLUME         = {"gain": 1.0}

ACTIVE_STEMS = {
    "vocals": True,
    "drums":  False,
    "bass":   False,
    "piano":  False,
    "other":  False,
}

STEM_BASE_GAIN = {   # fixed normalization boosts — never changed by sliders
    "vocals": 0.9,
    "drums":  1.0,
    "bass":   1.2,
    "piano":  20.0,
    "other":  1.0,
}
STEM_GAIN = {        # user slider multiplier — always starts at 1.0
    "vocals": 1.0,
    "drums":  1.0,
    "bass":   1.0,
    "piano":  1.0,
    "other":  1.0,
}
# ============================

blocks_q      = queue.Queue(maxsize=32)
stop_flag     = threading.Event()
threads_started = False
PIANO_SEPARATOR = None
piano = None
path  = None

# -------- waveform ring buffer --------
WAVE_BUF_SIZE = 4096          # samples kept for display
wave_ring     = np.zeros(WAVE_BUF_SIZE, dtype=np.float32)
wave_lock     = threading.Lock()

def _push_wave(mono_block: np.ndarray):
    """Thread-safe: push new samples into the ring buffer."""
    global wave_ring
    flat = mono_block.flatten().astype(np.float32)
    with wave_lock:
        wave_ring = np.roll(wave_ring, -len(flat))
        wave_ring[-len(flat):] = flat
# --------------------------------------


def run_onnx_single_target(sess, block_stereo_np):
    x = block_stereo_np
    if x.ndim == 1 or x.shape[1] == 1:
        x = np.repeat(x.reshape(-1,1), 2, axis=1)
    x = x.T[None, ...].astype(np.float32)
    inp = sess.get_inputs()[0].name
    y   = sess.run(None, {inp: x})[0]
    y   = np.asarray(y)
    if y.ndim == 3: y = y[0]
    if y.ndim == 2: y = y.T
    if y.shape[1] == 1: y = np.repeat(y, 2, axis=1)
    return y.astype(np.float32)


def pick_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("using CUDA GPU.")
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("using Apple Silicon GPU.")
    else:
        dev = torch.device("cpu")
        print("using CPU. Playback may be choppy.")
    return dev

DEVICE    = pick_device()
separator = None

PIANO_SEPARATOR = utils.load_separator(
    model_str_or_path=MODEL_DIR,
    targets=["Piano"],
    residual=True,
    niter=1,
    device=DEVICE,
    pretrained=True,
)


def ffmpeg_decode_to_tensor(path, sr=SR_MODEL, channels=2):
    try:
        data, file_sr = sf.read(path, dtype='float32')
        if data.ndim == 1:
            data = np.repeat(data[:, np.newaxis], channels, axis=1)
        elif data.shape[1] > channels:
            data = data[:, :channels]
        wav = torch.from_numpy(data).t().contiguous()
        return wav, file_sr
    except Exception as e:
        raise RuntimeError(f"Could not load audio file: {e}")


def reader_worker(audio_path):
    try:
        status_var.set(f"Decoding: {os.path.basename(audio_path)}")
        waveform, _ = ffmpeg_decode_to_tensor(audio_path, sr=SR_MODEL, channels=2)
        x    = waveform.t().numpy().astype("float32")
        segN = int(SEGMENT_SEC * SR_MODEL)
        hopN = int(HOP_SEC * SR_MODEL)
        pos  = 0
        while not stop_flag.is_set() and pos < x.shape[0]:
            end   = min(pos + segN, x.shape[0])
            block = x[max(0, end - segN):end]
            separate_and_enqueue(block)
            pos  += hopN
        stop_flag.set()
    except Exception as e:
        status_var.set("Error")
        stop_flag.set()


def separate_and_enqueue(block_in):
    global separator
    if block_in.ndim == 1 or block_in.shape[1] == 1:
        block_in = np.repeat(block_in.reshape(-1,1), 2, axis=1)

    with torch.no_grad():
        x   = torch.from_numpy(block_in.T).unsqueeze(0).to(DEVICE, torch.float32)
        y   = separator(x)
        est = y[0].cpu().numpy() if y.dim() == 4 else y.cpu().numpy()

    stems_st = {
        "vocals": est[0].T, "drums": est[1].T, "bass": est[2].T, "other": est[3].T
    }

    with torch.no_grad():
        px    = torch.from_numpy(block_in.T).unsqueeze(0).to(DEVICE, torch.float32)
        p_est = PIANO_SEPARATOR(px)
        p_est = PIANO_SEPARATOR.to_dict(p_est)

    global piano
    piano = p_est["Piano"][0].cpu().numpy().T
    stems_st["piano"] = piano

    N = next(iter(stems_st.values())).shape[0]
    w = np.hanning(N).astype(np.float32)
    out = {}
    for name, s in stems_st.items():
        mono = np.mean(s, axis=1) * w
        out[name] = mono.reshape(-1, 1)
    blocks_q.put(out, block=True)


def player_worker():
    segN  = int(SEGMENT_SEC * SR_MODEL)
    hopN  = int(HOP_SEC * SR_MODEL)
    ola   = np.zeros((segN + hopN, 1), dtype=np.float32)
    write_pos = 0
    ready = np.zeros((0,1), dtype=np.float32)

    def callback(outdata, frames, tinfo, status):
        nonlocal ready, ola, write_pos
        while ready.shape[0] < frames and not stop_flag.is_set():
            try:
                blk_dict = blocks_q.get_nowait()
            except queue.Empty:
                break

            N   = next(iter(blk_dict.values())).shape[0]
            end = write_pos + N
            if end > ola.shape[0]:
                ola = np.vstack([ola, np.zeros((end - ola.shape[0], 1), dtype=np.float32)])

            out_block = np.zeros((N, 1), dtype=np.float32)
            for name, active in ACTIVE_STEMS.items():
                if active:
                    s = blk_dict.get(name)
                    if s is not None:
                        out_block += s * STEM_BASE_GAIN.get(name, 1.0) * STEM_GAIN.get(name, 1.0)

            ola[write_pos:end] += out_block
            slice_end = write_pos + int(HOP_SEC * SR_MODEL)
            ready     = np.vstack([ready, ola[write_pos:slice_end].copy()])
            ola[write_pos:slice_end] = 0.0
            write_pos += int(HOP_SEC * SR_MODEL)

        if ready.shape[0] >= frames:
            out = ready[:frames] * VOLUME["gain"]
            np.clip(out, -1.0, 1.0, out=out)
            outdata[:] = out
            # ---- feed waveform ring buffer ----
            _push_wave(out)
            # -----------------------------------
            ready = ready[frames:]
        else:
            outdata[:] = 0.0

    try:
        with sd.OutputStream(
            samplerate=SR_MODEL,
            channels=OUTPUT_CHS,
            dtype='float32',
            blocksize=0,
            callback=callback
        ):
            while blocks_q.qsize() < PREROLL_BLOCKS and not stop_flag.is_set():
                time.sleep(0.05)
            status_var.set("Playing…")
            while not stop_flag.is_set():
                time.sleep(0.05)
    except Exception as e:
        status_var.set("Error")
        root.after(0, lambda: messagebox.showerror("Error", str(e)))
        stop_flag.set()


# ============================================================
#  WAVEFORM CANVAS WIDGET
# ============================================================

# Stem → color map for the waveform
STEM_COLORS = {
    "vocals": "#7F77DD",   # purple
    "drums":  "#D85A30",   # coral
    "bass":   "#1D9E75",   # teal
    "piano":  "#378ADD",   # blue
    "other":  "#888780",   # gray
}

class WaveformCanvas(tk.Canvas):
    """Animated waveform that reads from the shared ring buffer."""

    def __init__(self, master, **kw):
        kw.setdefault("bg", "#1a1a1a")
        kw.setdefault("highlightthickness", 0)
        super().__init__(master, **kw)
        self._after_id = None
        self._animate()

    def _animate(self):
        self._draw()
        self._after_id = self.after(30, self._animate)   # ~33 fps

    def stop(self):
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

    def _draw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or h < 2:
            return

        # Background grid lines
        mid = h // 2
        self.create_line(0, mid, w, mid, fill="#333333", width=1)
        for frac in (0.25, 0.75):
            y = int(h * frac)
            self.create_line(0, y, w, y, fill="#2a2a2a", width=1)

        # Decide waveform color from first active stem
        color = "#7F77DD"
        for name, active in ACTIVE_STEMS.items():
            if active:
                color = STEM_COLORS.get(name, "#7F77DD")
                break

        # Downsample ring buffer to canvas width
        with wave_lock:
            buf = wave_ring.copy()

        n      = len(buf)
        step   = max(1, n // w)
        points = []
        for px in range(w):
            start = px * step
            end   = min(start + step, n)
            chunk = buf[start:end]
            amp   = float(np.mean(np.abs(chunk))) if len(chunk) else 0.0
            amp   = min(amp, 1.0)
            half  = int(amp * (h // 2 - 4))
            y_top    = mid - half
            y_bottom = mid + half
            points.append((px, y_top, px, y_bottom))

        # Draw bars
        for (px, y1, px2, y2) in points:
            self.create_line(px, y1, px2, y2, fill=color, width=1)

        # Soft center glow line
        self.create_line(0, mid, w, mid, fill=color, width=1,
                         stipple="gray50")


# ============================================================
#  FULLSCREEN WINDOW
# ============================================================

fullscreen_win = None

def open_fullscreen():
    global fullscreen_win
    if fullscreen_win and tk.Toplevel.winfo_exists(fullscreen_win):
        fullscreen_win.lift()
        return

    fullscreen_win = tk.Toplevel(root)
    fullscreen_win.title("UMX – Fullscreen Visualizer")
    fullscreen_win.configure(bg="#1a1a1a")
    fullscreen_win.attributes("-fullscreen", True)

    # ---- top bar ----
    top = tk.Frame(fullscreen_win, bg="#1a1a1a")
    top.pack(fill="x", padx=16, pady=(10, 4))

    fs_status = tk.Label(top, textvariable=status_var,
                         bg="#1a1a1a", fg="#aaaaaa",
                         font=("Segoe UI", 11))
    fs_status.pack(side="left")

    tk.Button(top, text="✕ Exit fullscreen", bg="#2a2a2a", fg="#cccccc",
              relief="flat", cursor="hand2",
              command=fullscreen_win.destroy).pack(side="right")

    # ---- song label ----
    song_label_var = tk.StringVar(value="No file loaded")
    if path:
        song_label_var.set(os.path.basename(path))
    song_lbl = tk.Label(fullscreen_win, textvariable=song_label_var,
                        bg="#1a1a1a", fg="#ffffff",
                        font=("Segoe UI", 18, "bold"))
    song_lbl.pack(pady=(8, 0))

    # ---- big waveform ----
    wave = WaveformCanvas(fullscreen_win, bg="#1a1a1a")
    wave.pack(fill="both", expand=True, padx=24, pady=12)

    # ---- stem toggle row ----
    stem_row = tk.Frame(fullscreen_win, bg="#1a1a1a")
    stem_row.pack(pady=(0, 10))

    fs_btns = {}

    def _toggle(name, btn):
        ACTIVE_STEMS[name] = not ACTIVE_STEMS[name]
        _refresh_btn(name, btn)
        if name in btns:
            btns[name].config(relief="sunken" if ACTIVE_STEMS[name] else "raised")
        status_var.set(f"Active: {', '.join(n for n,a in ACTIVE_STEMS.items() if a)}")

    def _refresh_btn(name, btn):
        active  = ACTIVE_STEMS[name]
        color   = STEM_COLORS.get(name, "#888780")
        btn.config(
            bg     = color if active else "#2a2a2a",
            fg     = "#ffffff",
            relief = "flat"
        )

    for name, label in [("vocals","Vocals"), ("drums","Drums"),
                        ("bass","Bass"), ("piano","Piano"), ("other","Other")]:
        b = tk.Button(stem_row, text=label, width=9,
                      font=("Segoe UI", 11),
                      relief="flat", cursor="hand2",
                      padx=10, pady=6)
        b.config(command=lambda n=name, btn=b: _toggle(n, btn))
        _refresh_btn(name, b)
        b.pack(side="left", padx=6)
        fs_btns[name] = b

    # ---- volume slider ----
    vol_row = tk.Frame(fullscreen_win, bg="#1a1a1a")
    vol_row.pack(pady=(0, 16))

    tk.Label(vol_row, text="Volume", bg="#1a1a1a", fg="#888888",
             font=("Segoe UI", 10)).pack(side="left", padx=(0, 8))

    def _on_vol(val):
        VOLUME["gain"] = float(val)
        vol_slider.set(float(val))          # keep main slider in sync
        fullscreen_win._fs_vol = fs_vol     # store ref so main can push back

    fs_vol = tk.Scale(vol_row, from_=0.0, to=2.0, resolution=0.01,
                      orient="horizontal", length=200,
                      bg="#1a1a1a", fg="#cccccc",
                      troughcolor="#333333", highlightthickness=0,
                      showvalue=True, command=_on_vol)
    fs_vol.set(VOLUME["gain"])
    fs_vol.pack(side="left")

    # ESC closes fullscreen
    fullscreen_win.bind("<Escape>", lambda e: fullscreen_win.destroy())
    fullscreen_win.protocol("WM_DELETE_WINDOW", fullscreen_win.destroy)


# ============================================================
#  GUI
# ============================================================

def on_open_file():
    global path
    if threads_started:
        messagebox.showinfo("Busy", "Stop current playback before opening another file.")
        return
    path = filedialog.askopenfilename(
        title="Choose audio file",
        filetypes=[("Audio", "*.wav *.flac *.mp3 *.m4a *.aac *.ogg *.mp4"),
                   ("All files","*.*")]
    )
    if not path:
        return
    start_playback(path)

def start_playback(audio_path):
    global separator, threads_started
    try:
        status_var.set("Loading model…")
        if separator is None:
            separator = torch.hub.load(
                'sigsep/open-unmix-pytorch', 'umxhq',
                verbose=False, trust_repo=True, skip_validation=True
            ).eval().to(DEVICE)
        while not blocks_q.empty():
            try: blocks_q.get_nowait()
            except queue.Empty: break
        stop_flag.clear()
        status_var.set("Preparing…")
        global PIANO_SEPARATOR
        if PIANO_SEPARATOR is None:
            PIANO_SEPARATOR = utils.load_separator(
                model_str_or_path=MODEL_DIR, targets=["Piano"],
                residual=True, niter=1, device=DEVICE, pretrained=True,
            )
            PIANO_SEPARATOR.freeze()
            PIANO_SEPARATOR.to(DEVICE)
        t1 = threading.Thread(target=reader_worker, args=(audio_path,), daemon=True)
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
        filetypes=[("Audio", "*.wav *.flac *.mp3 *.m4a *.aac *.ogg *.mp4"),
                   ("All files","*.*")]
    )
    out_path = filedialog.asksaveasfilename(
        title=f"Piano {str(path)}",
        defaultextension=".wav",
        filetypes=[("WAV", "*.wav"), ("All files","*.*")]
    )
    if not out_path:
        return
    audio, sr = sf.read(song_path)
    audio = torch.tensor(audio.T, dtype=torch.float32).unsqueeze(0)
    print("Separating piano...")
    with torch.no_grad():
        estimates = PIANO_SEPARATOR(audio)
    estimates = PIANO_SEPARATOR.to_dict(estimates)
    piano = estimates["Piano"][0].cpu().numpy().T
    sf.write(out_path, piano, samplerate=44100)
    print(f"Saved: {out_path}")
    song = AudioSegment.from_wav(out_path)
    song = song + 20
    song.export(out_path, "wav")

btns = {}

def toggle_stem(name):
    ACTIVE_STEMS[name] = not ACTIVE_STEMS[name]
    if name in btns:
        btns[name].config(relief="sunken" if ACTIVE_STEMS[name] else "raised")
    status_var.set(f"Active stems: {', '.join(n for n,a in ACTIVE_STEMS.items() if a)}")


# ---- Build main window ----
root = tk.Tk()
root.title("UMX Real-Time Player")
root.geometry("480x620")
root.resizable(False, False)

# -- top bar --
top_frame = tk.Frame(root, padx=10, pady=10)
top_frame.pack(fill="x")

btn_open = tk.Button(top_frame, text="Open File…", width=12, command=on_open_file)
btn_open.pack(side="left")

btn_stop = tk.Button(top_frame, text="Stop", width=8, command=on_stop)
btn_stop.pack(side="left", padx=6)

btn_quit = tk.Button(top_frame, text="Quit", width=8, command=on_quit)
btn_quit.pack(side="left", padx=6)

btn_save = tk.Button(top_frame, text="Save Songs", width=12, command=on_save)
btn_save.pack(side="left", padx=6)

btn_fs = tk.Button(top_frame, text="⛶ Fullscreen", width=12, command=open_fullscreen)
btn_fs.pack(side="left", padx=6)

status_var = tk.StringVar(value=f"Device: {DEVICE}  |  Ready")
lbl_status = tk.Label(root, textvariable=status_var, anchor="w", padx=12)
lbl_status.pack(fill="x", pady=(0, 4))

# -- mini waveform in main window --
mini_wave = WaveformCanvas(root, height=70, bg="#1a1a1a")
mini_wave.pack(fill="x", padx=10, pady=(0, 6))

# -- stem controls --
mid = tk.Frame(root, padx=10, pady=6)
mid.pack(fill="both", expand=True)

tk.Label(mid, text="Choose stem to play (mono):", anchor="w").pack(anchor="w", pady=(0, 6))

row = tk.Frame(mid)
row.pack()
btns = {}
for name, label in [("vocals","Vocals"), ("drums","Drums"),
                    ("bass","Bass"), ("piano","Piano"), ("other","Other")]:
    b = tk.Button(row, text=label, width=10,
                  relief="sunken" if ACTIVE_STEMS[name] else "raised",
                  command=lambda n=name: toggle_stem(n))
    b.pack(side="left", padx=6)
    btns[name] = b

stem_sliders = {}
for name in ACTIVE_STEMS.keys():
    tk.Label(mid, text=name.capitalize()).pack(anchor="w")
    def _make_gain_cb(n):
        def _cb(val): STEM_GAIN[n] = float(val)
        return _cb
    s = tk.Scale(mid, from_=0.0, to=2.0, resolution=0.01,
                 orient="horizontal", command=_make_gain_cb(name))
    s.set(1.0)   # slider always starts at 1.0; base boost is applied separately
    s.pack(anchor="w", fill="x")
    stem_sliders[name] = s

# -- volume slider --
bottom = tk.Frame(root, padx=10, pady=6)
bottom.pack(side="bottom", fill="x")

vol_frame = tk.Frame(bottom)
vol_frame.pack(side="right")

def on_volume_change(val):
    VOLUME["gain"] = float(val)
    # push to fullscreen slider if it's open
    if fullscreen_win and tk.Toplevel.winfo_exists(fullscreen_win):
        fs_vol_ref = getattr(fullscreen_win, "_fs_vol", None)
        if fs_vol_ref:
            fs_vol_ref.set(float(val))

tk.Label(vol_frame, text="🔊", font=("Segoe UI Emoji", 14)).pack(side="left", padx=(0, 6))

vol_slider = tk.Scale(
    vol_frame, from_=0.0, to=2.0, resolution=0.01,
    orient="horizontal", length=100,
    showvalue=True, command=on_volume_change
)
vol_slider.set(1.0)
vol_slider.pack(side="left")

root.protocol("WM_DELETE_WINDOW", on_quit)
root.mainloop()