# 🎥 Chunk, Transcribe, and Analyze Classroom Video

This Python tool automates the process of evaluating classroom teaching sessions. It:
1. Splits a video into 1-minute chunks
2. Transcribes each chunk using OpenAI's Whisper
3. Evaluates the teacher's performance using GPT-4o

---

## 🚀 Features

- Automatically chunks long classroom videos
- Uses Whisper for accurate audio transcription
- Evaluates teaching performance based on 5 key metrics:
  - Clarity and articulation
  - Engagement and questioning style
  - Classroom management
  - Use of feedback and encouragement
  - Instructional structure

---

## 📦 Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/download.html)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Whisper](https://github.com/openai/whisper)

Install dependencies:

```bash
pip install openai whisper
```

Make sure `ffmpeg` is installed and in your system path.

---

## 🔑 Setup

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## ▶️ Usage

```bash
python transcribe_and_evaluate.py path/to/video.mp4
```

Output:
- Full transcript printed to console
- Teacher performance analysis printed after transcription

---

## ⚠️ Note on Large Files

Do **not** commit `.mp4` files to the repo. Use a `.gitignore` to avoid tracking them:

```gitignore
*.mp4
```

---

## 📁 File Structure

```
transcribe_and_evaluate.py   # Main script
README.md                    # This file
```

