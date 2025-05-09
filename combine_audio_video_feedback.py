#!/usr/bin/env python3
import os
import sys
import json
import argparse
from openai import OpenAI

def load_audio_transcript(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_image_captions(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    image_map = {}
    for val in data.values():
        key = round(val.get("time_s", 0.0), 3)
        image_map[key] = {k: v for k, v in val.items() if k not in ("time_s", "time_min")}
    return image_map

def combine_transcript(audio_data, image_map):
    combined = []
    for entry in audio_data:
        start_key = round(entry.get("start", 0.0), 3)
        combined_entry = {
            "start": entry.get("start"),
            "end": entry.get("end"),
            "transcript": entry.get("text", "").strip()
        }
        image_entry = image_map.get(start_key, {})
        combined_entry["image"] = image_entry
        combined.append(combined_entry)
    return combined

def transcript_to_plaintext(chunk):
    parts = []
    for idx, seg in enumerate(chunk, 1):
        part_lines = [
            f"Segment {idx}: {seg.get('start', 0.0)}s to {seg.get('end', 0.0)}s",
            f"Transcript: {seg.get('transcript', '')}"
        ]
        image = seg.get("image", {})
        if image:
            part_lines.append("Image Analysis:")
            for k, v in image.items():
                part_lines.append(f"- {k.replace('_', ' ').capitalize()}: {v}")
        parts.append("\n".join(part_lines))
    return "\n\n".join(parts)

def summarize_chunk(chunk, client, model):
    content = transcript_to_plaintext(chunk)
    messages = [
        {"role": "system", "content": "You are an expert preschool education evaluator. Summarize the following transcript chunk. Focus on classroom setup, child engagement, prop usage, body language, and teacher communication. Provide a concise summary."},
        {"role": "user", "content": content}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def generate_feedback(summary, client, model):
    messages = [
        {"role": "system", "content": "You are a knowledgeable preschool education evaluator."},
        {"role": "user", "content": (
            "Below is the summarized combined transcript of a preschool storytelling class:\n\n"
            f"{summary}\n\n"
            "Based on this, provide comprehensive, actionable feedback focusing on classroom setup, "
            "child engagement, prop usage, body language, and teacher communication."
        )}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def main():
    parser = argparse.ArgumentParser(
        description="Combine audio and image transcripts, summarize, and generate feedback."
    )
    parser.add_argument("--audio_json", required=True, help="Path to audio transcription JSON")
    parser.add_argument("--image_json", required=True, help="Path to image caption JSON")
    parser.add_argument("--output_transcript", default="combined_transcript.json", help="Path to write combined transcript JSON")
    parser.add_argument("--output_feedback", default="Feedback_on_combined_transcript.txt", help="Path to write final feedback text")
    parser.add_argument("--chunk_size", type=int, default=10, help="Number of segments per summarization chunk")
    parser.add_argument("--model", default="gpt-4o", help="Model to use for final feedback")
    parser.add_argument("--summary_model", default="gpt-4o", help="Model to use for summarization")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    print("Loading transcripts...")
    audio_data = load_audio_transcript(args.audio_json)
    image_map = load_image_captions(args.image_json)

    print("Combining transcripts...")
    combined = combine_transcript(audio_data, image_map)
    with open(args.output_transcript, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"Combined transcript saved to {args.output_transcript}")

    # Summarize in chunks if needed
    if len(combined) > args.chunk_size:
        chunks = [combined[i : i + args.chunk_size] for i in range(0, len(combined), args.chunk_size)]
        summaries = []
        for idx, chunk in enumerate(chunks, 1):
            print(f"Summarizing chunk {idx}/{len(chunks)}...")
            summary = summarize_chunk(chunk, client, args.summary_model)
            summaries.append(summary)
        combined_summary = "\n\n".join(summaries)
    else:
        print("No summarization needed; transcript is small.")
        combined_summary = transcript_to_plaintext(combined)

    print("Generating final feedback...")
    feedback = generate_feedback(combined_summary, client, args.model)
    with open(args.output_feedback, "w", encoding="utf-8") as f:
        f.write(feedback)
    print(f"Feedback saved to {args.output_feedback}")

if __name__ == "__main__":
    main()
