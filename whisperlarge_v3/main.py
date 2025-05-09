#!/usr/bin/env python3
"""
Audio transcription pipeline:
  - Extracts audio from input MP4 and splits into 10-second WAV chunks using ffmpeg
  - Transcribes each chunk using Hugging Face's whisper-large-v3 model
  - Outputs JSON with start/end times and transcription text for each chunk
Usage:
  python main.py path/to/video.mp4
Output:
  Creates an 'outputs' directory containing chunked WAVs and a JSON transcript
"""
import argparse
import os
import subprocess
import glob
import json

import torch
from transformers import pipeline
from tqdm import tqdm


def split_video(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    chunk_pattern = os.path.join(output_dir, "chunk%03d.wav")
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                   # drop video stream
        "-acodec", "pcm_s16le",  # WAV PCM 16-bit little endian
        "-ar", "16000",          # 16 kHz sample rate
        "-f", "segment",
        "-segment_time", "10",
        "-reset_timestamps", "1",
        chunk_pattern,
    ]
    subprocess.run(cmd, check=True)
    files = sorted(glob.glob(os.path.join(output_dir, "chunk*.wav")))
    return files


def get_duration(file_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def transcribe_chunks(chunk_files, model_name="openai/whisper-large-v3", device=None):
    device = device if device is not None else (0 if torch.cuda.is_available() else -1)
    recognizer = pipeline("automatic-speech-recognition", model=model_name, device=device)
    transcripts = []
    for idx, chunk in enumerate(tqdm(chunk_files, desc="Transcribing chunks")):
        # compute start time based on 10-second segments
        start_time = idx * 10.0
        result = recognizer(chunk)
        text = result.get("text", "").strip()
        duration = get_duration(chunk)
        end_time = start_time + duration
        transcripts.append({
            "start": start_time,
            "end": end_time,
            "text": text,
        })
    return transcripts


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio from an MP4 video into JSON chunks")
    parser.add_argument("video_path", help="Path to input MP4 video")
    args = parser.parse_args()

    video_path = args.video_path
    if not os.path.isfile(video_path):
        print(f"Error: File not found: {video_path}")
        exit(1)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting audio and splitting into 10-second WAV chunks...")
    chunk_files = split_video(video_path, output_dir)
    if not chunk_files:
        print("No chunks were created; aborting.")
        exit(1)

    print(f"Created {len(chunk_files)} chunk(s) in '{output_dir}'")
    print("Starting transcription on GPU" if torch.cuda.is_available() else "Starting transcription on CPU")
    transcripts = transcribe_chunks(chunk_files)

    output_json = os.path.join(output_dir, f"{base_name}_transcription.json")
    with open(output_json, "w") as f:
        json.dump(transcripts, f, indent=2)

    print(f"Transcription complete. JSON output saved to: {output_json}")


if __name__ == "__main__":
    main()