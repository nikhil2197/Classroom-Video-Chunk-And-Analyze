# GCP_Video_AI/main.py

import os
import sys
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")  # if you're in GCP_Video_AI dir

from chunk_and_annotate import process_long_video
from parse_annotations import extract_relevant_data
from generate_feedback import generate_feedback

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <video_path>")
        return

    video_path = sys.argv[1]
    chunk_dir = os.path.join("chunks", os.path.basename(video_path).replace(".mp4", ""))
    os.makedirs(chunk_dir, exist_ok=True)

    print("🎬 Step 1: Chunking & Annotating each video part")
    merged_annotations = process_long_video(video_path, chunk_dir)

    print("🧩 Step 2: Parsing combined annotations")
    parsed_data = extract_relevant_data(merged_annotations)

    print("🧠 Step 3: Sending to GPT-4o for feedback")
    feedback = generate_feedback(parsed_data)

    print("\n✅ Final Feedback:\n")
    print(feedback)

if __name__ == "__main__":
    main()
