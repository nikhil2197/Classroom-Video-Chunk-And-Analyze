# GCP_Video_AI/generate_feedback.py

import os
from openai import OpenAI
from openai.types.chat import ChatCompletion

# === Init OpenAI client ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === GPT Feedback ===
def generate_feedback(data):
    prompt = f"""
Analyze the following classroom video data and provide actionable feedback for the teacher.

**Teacher Transcript:**
{data['transcript']}

**Speech Sentiment Analysis:**
{data['sentiments']}

**Pose and Gesture Observations (Summarized):**
{data['pose_summary']}

**Classroom Context (from Label Detection):**
{', '.join(data['labels'])}

Return your evaluation as:
- Key Strengths (bullet points)
- Areas for Improvement (bullet points)
- Overall Summary (2-3 lines)
"""

    # === Save transcript for reference ===
    with open("transcript_used_for_feedback.txt", "w", encoding="utf-8") as f:
        f.write(data['transcript'])

    # === GPT-4o call ===
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert classroom evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content
