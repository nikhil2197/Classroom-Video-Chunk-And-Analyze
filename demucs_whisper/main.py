import os, sys, json, pathlib, time
#from dotenv import load_dotenv
import whisper
from openai import OpenAI
from openai.types.chat import ChatCompletion
from utils import split_video, extract_audio, separate_vocals
import torch

def main(mp4_path: str, chunk_len: int = 60):
    print(f"[DEBUG] Starting main() with file: {mp4_path}")

# ---------- init ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
device = "cuda" if torch.cuda.is_available() else "cpu"
wmodel = whisper.load_model("large-v3", device=device)                # <- forces fp16
print("🤖 Whisper weights on", next(wmodel.parameters()).device, "dtype", next(wmodel.parameters()).dtype)
from pathlib import Path
BASE = Path(__file__).resolve().parent        # …/demucs_whisper
OUT = BASE / "outputs"                        # …/demucs_whisper/outputs
OUT.mkdir(exist_ok=True)

def transcribe(chunk_wav: str) -> str:
    res = wmodel.transcribe(chunk_wav, fp16=True, language="en")
    txt = res["text"].strip()
    (pathlib.Path(chunk_wav).with_suffix(".txt")).write_text(txt)
    return txt

def main(mp4_path: str, chunk_len: int = 60):
    t0 = time.time()
    print(f"🔪 Splitting into {chunk_len}s chunks …")
    chunks = split_video(mp4_path, chunk_len)

    all_txt = []
    for i, chunk in enumerate(chunks, 1):
        print(f"\n⏩  Chunk {i}/{len(chunks)}  ({pathlib.Path(chunk).name})")
        wav = extract_audio(chunk)
        vocals = separate_vocals(wav)
        print("✅ Vocals isolated for", wav)
        txt = transcribe(vocals)
        all_txt.append(f"[Chunk {i}] {txt}")

    full_transcript = "\n".join(all_txt)
    stem = pathlib.Path(mp4_path).stem
    tpath = OUT / f"{stem}_transcript.txt"
    tpath.write_text(full_transcript)
    print(f"\n📝 Transcript saved → {tpath}")

    # ---------- GPT‑4o feedback ----------
    prompt = f"""
You are evaluating a preschool teacher. Here is the auto‑extracted transcript (teacher‑dominant) in 60‑second chunks:

{full_transcript}

Return JSON with keys:
  strengths: list[str]
  areas_for_improvement: list[str]
  summary: str
"""
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content": prompt}],
        temperature=0.3
    )
    feedback = chat.choices[0].message.content
    fpath = OUT / f"{stem}_feedback.json"
    fpath.write_text(feedback)
    print(f"✅ Feedback saved → {fpath}")
    print(f"\n🏁 Completed in {int(time.time()-t0)} s")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 demucs_whisper/main.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
