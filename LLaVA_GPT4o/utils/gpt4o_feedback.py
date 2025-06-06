"""
gpt4o_feedback.py
────────────────────────────────────────────────────────────────────────────
Generate expert pre‑school–observer feedback from LLaVA frame‑by‑frame
descriptions *without* breaching GPT‑4o’s 30 k TPM rate‑limit tier.

Core features
─────────────
1.  **Run‑length compression** of consecutive, identical descriptions
    (e.g. `F041–F057: children colouring at table`) → keeps chronology *and*
    duration info while saving tokens.

2.  **Adaptive chunking** based on the org’s TPM limit so every API call
    stays safely below the quota (input + output + buffer ≤ 30 000 tokens).

3.  **Two‑stage summarisation**
      • per‑chunk feedback  
      • single synthesis call that merges the partial notes in *chronological*
        order into one concise report (Key Strengths / Areas for Improvement /
        Overall Summary).
"""

from __future__ import annotations
import os
import re
import time
import json
from pathlib import Path
import tiktoken
from openai import OpenAI
from collections import OrderedDict

import sys, pathlib

#Ensure parent folder is in the import path
#sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from config import OPENAI_MODEL, ANALYSIS_PROMPTS

# Configuration
TPM_LIMIT = 28000
MAX_REPLY_TOKENS = 1200
SAFETY_BUFFER = 3000
MAX_INPUT_TOKENS = TPM_LIMIT - MAX_REPLY_TOKENS - SAFETY_BUFFER
SLEEP_SECONDS = 75

PROMPT_REMOVER = re.compile(
    r"\s*This image is from a pre-school storytelling class\. I would like you to analyze the scene and provide a detailed breakdown of what is happening\. Specifically, focus on the following aspects:\s*"
    r"1\. Children's Actions: Describe what the children are doing, including any notable gestures, facial expressions, or group activities\.\s*"
    r"2\. Teacher's Role: Observe how the teacher is interacting with the children, including their use of props and storytelling techniques\.\s*"
    r"3\. Classroom Management: Highlight any visible strategies used by the teacher to maintain engagement and order, such as gestures, positioning, or visual aids\.\s*"
    r"4\. Props and Visual Aids: Identify any props or materials being used and explain how they are integrated into the storytelling\.\s*"
    r"5\. Overall Atmosphere: Describe the general mood or energy of the class and how effectively the storytelling setup seems to support learning and engagement\.\s*"
    r"Please provide a clear and comprehensive analysis based on these points\.\s*",
    re.I
)


# Token encoding
enc = tiktoken.encoding_for_model(OPENAI_MODEL)

def message_tokens(text: str) -> int:
    """Calculate the number of tokens in the given text."""
    return len(enc.encode(text)) + 4  # 4 tokens for role and JSON formatting

def _clean(desc: str) -> str:
    """Clean redundant prompt text and excess whitespace."""
    return PROMPT_REMOVER.sub(" ", desc).strip()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Feedback generation function
def generate_feedback(llava_json_path: str | Path, output_dir: str | Path) -> None:
    try:                                                                                                                                                                        
        with open(llava_json_path, "r") as f:                                                                                                                                   
            # raw may map frame names to either a single string or a dict of prompt-name -> caption                                                                             
            raw: dict[str, object] = json.load(f)  
    except Exception as e:
        print(f"❌ Failed to load JSON file: {e}")
        sys.exit(1)

    # Run-length compression
    compressed: list[str] = []
    last_desc, first_frame, last_frame = None, None, None

    for frame, item in raw.items():                                                                                                                                             
        # Flatten multi-prompt entries or use single string                                                                                                                     
        if isinstance(item, dict):                                                                                                                                              
            parts: list[str] = []                                                                                                                                               
            # iterate in config order to keep consistency                                                                                                                       
            for p in ANALYSIS_PROMPTS:                                                                                                                                          
                key = p["name"]                                                                                                                                                 
                if key in item and item[key]:                                                                                                                                   
                    parts.append(f"[{key}] {item[key].strip()}")                                                                                                                
            desc_str = "\n".join(parts)                                                                                                                                         
        else:                                                                                                                                                                   
            desc_str = str(item)                                                                                                                                                
        desc = _clean(desc_str)   
        if desc == last_desc:
            last_frame = frame
            continue
        if last_desc is not None:
            if first_frame == last_frame:
                compressed.append(f"{first_frame}: {last_desc}")
            else:
                compressed.append(f"{first_frame}–{last_frame}: {last_desc}")
        first_frame = last_frame = frame
        last_desc = desc

    if last_desc is not None:
        if first_frame == last_frame:
            compressed.append(f"{first_frame}: {last_desc}")
        else:
            compressed.append(f"{first_frame}–{last_frame}: {last_desc}")

    # Adaptive chunking
    bucket, chunks, bucket_tok = [], [], 0
    for line in compressed:
        t = message_tokens(line + "\n")
        if bucket_tok + t > MAX_INPUT_TOKENS:
            chunks.append("\n".join(bucket))
            bucket, bucket_tok = [line], t
        else:
            bucket.append(line)
            bucket_tok += t
    if bucket:
        chunks.append("\n".join(bucket))

    # Process each chunk
    partial_notes = []
    CHUNK_TEMPLATE = """
You are an expert pre-school classroom observer.

For each frame description below, infer what the teacher and children are
doing, judge teaching quality, and provide constructive, actionable feedback. Grade them basis the following criteria:
1. Classroom setup
2. Warm-up and settling-down
3. Props and visual aids
4. Body language
5. Child interaction
6. Wrap-up


OUTPUT FORMAT
- Key Strengths
- Areas for Improvement
- Overall Summary

{context}
"""
    for i, context in enumerate(chunks, 1):
        prompt = CHUNK_TEMPLATE.format(context=context)
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_REPLY_TOKENS,
            )
            partial_notes.append(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"❌ Error during API call: {e}")
            sys.exit(1)

        if i < len(chunks):
            time.sleep(SLEEP_SECONDS)

    # Final synthesis
    merged_notes = "\n\n".join(partial_notes)
    SYNTH_PROMPT = f"""
You are an instructional-coaching expert.

**Important:** The notes below appear in chronological order. Combine them
into ONE cohesive report, keeping the same sequence so the reader can follow
the lesson flow.

Notes to merge:
{merged_notes}

- Key Strengths
- Areas for Improvement
- Overall Summary

Merge duplicates, eliminate contradictions, and prioritize clarity and actionability.
"""
    final = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": SYNTH_PROMPT}],
        max_tokens=MAX_REPLY_TOKENS,
    ).choices[0].message.content.strip()

    # Save output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "final_feedback.txt"
    with open(output_path, "w") as f:
        f.write(final)
    print(f"✅ Feedback saved to {output_path}")

#Main execution
#if __name__ == "__main__":
#    if len(sys.argv) < 2:
#        print("Usage: python gpt4o_feedback.py <llava_json_path> [output_dir]")
#        sys.exit(1)
#
#    llava_json_path = sys.argv[1]
#    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(llava_json_path)
#
#    print(f"Input file: {llava_json_path}")
#    print(f"Output directory: {output_dir}")
#
#    try:
#        generate_feedback(llava_json_path, output_dir)
#    except Exception as e:
#        print(f"❌ Error: {e}")
