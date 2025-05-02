# 🎥 Chunk, Transcribe, and Analyze Classroom Video

This repository contains a collection of basic scripts I've written to experiment with different approaches to analyzing classroom video. The goal is to automate the process of evaluating teaching sessions through a combination of video chunking, transcription, and performance analysis.

The scripts in this repo include attempts using:

* **OpenAI Whisper** for transcription and evaluation
* **Google Cloud Video Intelligence API** for activity recognition and segmentation passed through to **GPT 4o** for scoring
* Other approaches I’m testing for better accuracy, speed, and cost-effectiveness

Each script reflects a different approach. Below are descriptions of individual attempts, observations, and how to run them.

---

## `whisperaudio_chunk_transcribe_analyze.py`

### 🚀 What It Does

* Uses Whisper's `base` model to transcribe audio from 1-minute video chunks
* Sends the combined transcript to GPT-4o for evaluation on:

  * Clarity and articulation
  * Engagement and questioning style
  * Classroom management
  * Use of feedback and encouragement
  * Instructional structure

### 📈 Early Observations

* **Pros:**

  * Whisper is relatively fast and accurate for clean audio
  * GPT-4o generates thoughtful evaluations when given full transcripts

* **Cons:**

  * Whisper struggles with noisy environments or crosstalk
  * GPT evaluation depends heavily on transcription quality
  * Lacks speaker separation and timestamped analysis (next goal)

### 📦 Requirements

* Python 3.9+
* [ffmpeg](https://ffmpeg.org/download.html)
* [OpenAI Python SDK](https://github.com/openai/openai-python)
* [Whisper](https://github.com/openai/whisper)

Install dependencies:

```bash
pip install openai whisper
```

Make sure `ffmpeg` is installed and in your system path.

### 🔑 Setup

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

### ▶️ Usage

```bash
python transcribe_and_evaluate.py path/to/video.mp4
```

Output:

* Full transcript printed to console
* Teacher performance analysis printed after transcription

---

## `GCPVideoAI_GPT4o_Pipe`

### 🚀 What It Does

* Uses **Google Cloud Video Intelligence API** to generate activity labels and speech transcripts from classroom video
* Passes both labels and transcripts **into GPT-4o** to analyze teacher performance within a structured timeline

### 📈 Early Observations

* **What worked:**

  * Successfully extracted both transcript and activity labels

* **What didn’t work:**

  * Poor video quality (shot on a phone from many angles) degraded both transcription and video labeling quality
  * Inconsistent framing and audio resulted in a weak input timeline for GPT-4o evaluation

### 📦 Requirements

* Python 3.9+
* [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
* `google-cloud-videointelligence` Python package
* [OpenAI Python SDK](https://github.com/openai/openai-python)

Install dependencies:

```bash
pip install google-cloud-videointelligence openai
```

### 🔑 Setup

1. Enable the **Video Intelligence API** in your Google Cloud Console.
2. Download your service account key JSON file.
3. Set the environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-file.json"
export OPENAI_API_KEY="sk-..."
```

### ▶️ Usage

```bash
python main.py path/to/video.mp4
```

Output:

* Activity labels and transcript printed or saved
* Performance analysis generated via GPT-4o

---

## ⚠️ Note on Large Files

Do **not** commit `.mp4` files to the repo. Use a `.gitignore` to avoid tracking them:

```gitignore
*.mp4
```

---

## 📁 File Structure

```
whisperaudio_chunk_transcribe_analyze.py   # Whisper-based pipeline
GCPVideoAI_GPT4o_Pipe/                     # Google Cloud Video AI + GPT-4o pipeline
README.md                                  # This file
<future files>                             # Other experiments (diarization, etc.)
```

---

More updates and experimental results coming soon.
