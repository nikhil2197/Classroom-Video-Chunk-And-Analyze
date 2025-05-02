import whisper
from openai import OpenAI
from openai.types.chat import ChatCompletion
import sys
import os
import subprocess
import tempfile

# === Step 0: Set your OpenAI API Key ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Step 1: Get the input file path ===
input_file = sys.argv[1]  # e.g. python transcribe_and_evaluate.py path/to/video.mp4
model = whisper.load_model("base")

# === Step 2: Chunk video into 1-minute segments ===
def split_video(input_path, output_dir, chunk_length=60):
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "chunk_%03d.mp4")
    command = [
        "ffmpeg", "-i", input_path,
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", str(chunk_length),
        output_template
    ]
    subprocess.run(command, check=True)
    return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp4")])

# === Step 3: Transcribe each chunk and combine ===
def transcribe_chunks(chunk_paths):
    full_transcript = ""
    for i, chunk_path in enumerate(chunk_paths):
        print(f"Transcribing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")
        result = model.transcribe(chunk_path)
        full_transcript += result["text"].strip() + " "
    return full_transcript.strip()

# === Step 4: Evaluate teacher based on transcript ===
def evaluate_transcript(transcript):
    prompt = f"""
Here is a transcript of a classroom session led by a teacher:

{transcript}

Please evaluate the teacher's performance based on:
1. Clarity and articulation
2. Engagement and questioning style
3. Classroom management (as inferred from dialogue)
4. Use of feedback and encouragement
5. Structure and progression of instruction

Return your evaluation as:
- Key Strengths (bullet points)
- Areas for Improvement (bullet points)
- Overall Summary (2-3 lines)
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert classroom evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content

# === MAIN ===
if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            chunk_paths = split_video(input_file, tmpdir)
            full_transcript = transcribe_chunks(chunk_paths)
            print("\n--- FULL TRANSCRIPTION ---\n")
            print(full_transcript)

            analysis = evaluate_transcript(full_transcript)
            print("\n--- TEACHER PERFORMANCE ANALYSIS ---\n")
            print(analysis)
        except Exception as e:
            print("❌ Error during processing:", e)
