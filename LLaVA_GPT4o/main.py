import sys
import os

# Add utils/ directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from frame_extractor import extract_frames
from llava_inference import run_llava_on_frames
from gpt4o_feedback import generate_feedback
from video_utils import prepare_output_dir


def main(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = prepare_output_dir(video_name)

    print("🔹 Extracting frames...")
    frame_dir = extract_frames(video_path, output_dir)

    print("🔹 Running LLaVA on frames...")
    llava_json_path = run_llava_on_frames(frame_dir, output_dir)

    print("🔹 Generating GPT-4o feedback...")
    generate_feedback(llava_json_path, output_dir)

    print(f"✅ Done. Check output in {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])
