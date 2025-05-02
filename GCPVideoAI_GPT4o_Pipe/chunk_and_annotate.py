# GCP_Video_AI/chunk_and_annotate.py

import os
import subprocess
from google.cloud import videointelligence_v1 as vi
from google.protobuf.json_format import MessageToDict
import json

def split_video(input_path, output_dir, chunk_length=60):
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "chunk_%03d.mp4")
    command = [
        "ffmpeg", "-i", input_path,
        "-c", "copy", "-map", "0",
        "-f", "segment", "-segment_time", str(chunk_length),
        output_template
    ]
    subprocess.run(command, check=True)
    return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp4")])

def annotate_chunk(video_path):
    client = vi.VideoIntelligenceServiceClient()
    features = [
        vi.Feature.SPEECH_TRANSCRIPTION,
        vi.Feature.LABEL_DETECTION,
        vi.Feature.PERSON_DETECTION
    ]
    with open(video_path, "rb") as f:
        input_content = f.read()
    operation = client.annotate_video(
        request={
            "features": features,
            "input_content": input_content,
            "video_context": {
                "speech_transcription_config": {
                    "language_code": "en-US",
                    "enable_automatic_punctuation": True
                }
            }
        }
    )
    print(f"⏳ Annotating {video_path}...")
    result = operation.result(timeout=300)

    # ✅ Guaranteed conversion
    return MessageToDict(result._pb)

def merge_annotations(results):
    merged = {
        "annotationResults": []
    }
    for r in results:
        merged["annotationResults"].extend(r.get("annotationResults", []))
    return merged

def process_long_video(video_path, chunk_dir):
    chunks = split_video(video_path, chunk_dir)
    results = []

    for chunk in chunks:
        print(f"⏳ Annotating {chunk}...")
        result = annotate_chunk(chunk)
        results.append(result)

        # ✅ Save raw annotation for inspection
        base_name = os.path.splitext(os.path.basename(chunk))[0]
        json_path = os.path.join(chunk_dir, f"{base_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return merge_annotations(results)