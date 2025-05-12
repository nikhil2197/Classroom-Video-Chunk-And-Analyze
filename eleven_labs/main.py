#!/usr/bin/env python3
"""
Transcribe MP4 videos using Eleven Labs Scribe v1 model and save JSON transcripts.

Usage:
  python main.py path/to/video.mp4

Requires:
  - Python 3.6+
  - requests library (pip install -r requirements.txt)
  - ELEVEN_LABS_API_KEY environment variable set to your Eleven Labs API key
"""
import os
import sys
import argparse
import json

try:
    import requests
except ImportError:
    sys.stderr.write("Error: this script requires the 'requests' library. Install with 'pip install -r requirements.txt'\n")
    sys.exit(1)

def transcribe(video_path, api_key, model, response_format, language=None):
    """
    Send the video file to Eleven Labs speech-to-text API and return the parsed JSON response.
    """
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": api_key}
    with open(video_path, "rb") as vf:
        files = {"file": vf}
        # Eleven Labs API expects 'model_id' for the speech-to-text model identifier
        data = {"model_id": model, "response_format": response_format}
        if language:
            data["language"] = language
        resp = requests.post(url, headers=headers, files=files, data=data)
    try:
        result = resp.json()
    except ValueError:
        resp.raise_for_status()
        raise
    if resp.status_code != 200:
        sys.stderr.write(f"API request failed [{resp.status_code}]: {result}\n")
        sys.exit(1)
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe MP4 videos using Eleven Labs Scribe v1"
    )
    parser.add_argument(
        "video_path",
        help="Path to the MP4 video file to transcribe"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs",
        help="Directory to save the JSON transcript (default: outputs)"
    )
    parser.add_argument(
        "--model", "-m",
        default=os.getenv("ELEVEN_LABS_SCRIBE_MODEL", "scribe_v1"),
        help="Eleven Labs model ID to use (default: scribe_v1)"
    )
    parser.add_argument(
        "--response-format", "-r",
        default="verbose_json",
        choices=["json", "verbose_json", "text"],
        help="Response format: json, verbose_json (with timestamps), or text (default: verbose_json)"
    )
    parser.add_argument(
        "--language", "-l",
        help="Language code (e.g., en), optional"
    )
    args = parser.parse_args()

    video_path = args.video_path
    if not os.path.isfile(video_path):
        sys.stderr.write(f"Error: file not found: {video_path}\n")
        sys.exit(1)

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        sys.stderr.write("Error: set ELEVEN_LABS_API_KEY environment variable\n")
        sys.exit(1)

    transcript = transcribe(
        video_path,
        api_key,
        args.model,
        args.response_format,
        language=args.language
    )

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}.json")

    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(transcript, out_f, ensure_ascii=False, indent=2)

    print(f"Transcript saved to {out_path}")

if __name__ == "__main__":
    main()