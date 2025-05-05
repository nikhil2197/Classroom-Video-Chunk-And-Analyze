import subprocess
import tempfile
import numpy as np
import torch
import pathlib
import time
from scipy.io.wavfile import write as wavwrite
from scipy.io import wavfile
from pathlib import Path
from demucs.apply import apply_model
from demucs.pretrained import get_model

SR = 16_000                         # Whisper default
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Video helpers ----------
def split_video(input_path: str, chunk_sec: int = 60) -> list[str]:
    """
    Slice <input_path> into N x <chunk_sec> MP4s (stream‑copy, no re‑encode).
    Returns a sorted list of chunk paths.
    """
    stem = pathlib.Path(input_path).stem
    tmp_dir = pathlib.Path(tempfile.gettempdir()) / f"{stem}_chunks"
    tmp_dir.mkdir(exist_ok=True)
    out_tmpl = tmp_dir / "chunk_%03d.mp4"

    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", input_path,
        "-c", "copy", "-map", "0",
        "-f", "segment",
        "-segment_time", str(chunk_sec),
        out_tmpl.as_posix()
    ]
    subprocess.run(cmd, check=True)
    return sorted([f.as_posix() for f in tmp_dir.glob("chunk_*.mp4")])

def extract_audio(video_fp: str, sr: int = SR) -> str:
    wav_out = pathlib.Path(tempfile.gettempdir()) / (pathlib.Path(video_fp).stem + ".wav")
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", video_fp, "-ac", "1", "-ar", str(sr), wav_out.as_posix()]
    subprocess.run(cmd, check=True)
    return str(wav_out)

# ---------- Demucs ----------
_model = None                       # lazy‑load once

def separate_vocals(wav_fp: str) -> str:
    global _model,_device

    if _model is None:
        print(f"🚀 [Demucs] loading mdx_extra_q on {_device} …")
        _model = get_model("mdx_extra_q").to(_device)

    # ---------- 1) resample to 16 kHz mono with ffmpeg ----------
    tmp_wav = Path(tempfile.gettempdir()) / (Path(wav_fp).stem + "_16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", wav_fp, "-ac", "1", "-ar", "16000", tmp_wav.as_posix()],
        check=True)

    # ---------- 2) load & dup mono → stereo ----------
    sr, wav_np = wavfile.read(tmp_wav)
    if wav_np.dtype == np.int16:
        wav_np = wav_np.astype(np.float32) / 32768.0
    if wav_np.ndim == 1:
        wav_np = np.stack([wav_np, wav_np])          # [2, N]

    wav_tensor = torch.tensor(wav_np).unsqueeze(0).to(_device)  # [1,2,N]

    # ---------- 3) run Demucs (fast settings) ----------
    print(f"🎧 [Demucs] start on {wav_fp}  →  {wav_tensor.shape[-1]/sr:.1f}s")
    t0 = time.time()
    sources = apply_model(
    _model,
    wav_tensor,          # [1, 2, N]
    segment=15,
    overlap=0.25,
    shifts=0,
    device=_device,
)             
    dt = time.time() - t0
    print(f"✅ [Demucs] finished in {dt:.2f}s")

    # ---------- 4) extract vocals stem ----------
    vocal_idx = _model.sources.index("vocals")
    vocals = sources[0, vocal_idx].cpu().numpy()
    vocals = (vocals / np.max(np.abs(vocals))) * 32767
    vocals = vocals.astype(np.int16)

    out_fp = wav_fp.replace(".wav", "_vocals.wav")
    wavwrite(out_fp, sr, vocals.T)   # scipy wav write
    return out_fp
# ----------------------------------------------------------------