# 🎥 Classroom Video Analysis and Feedback Tool

This repository contains a collection of scripts for experimenting with different approaches to analyzing classroom video. The primary goal is to automate the evaluation of teaching sessions through a combination of video chunking, transcription, and performance analysis.

The scripts in this repo include attempts using:

- **OpenAI Whisper (v2 on API)** for audio-only transcription and GPT-4o-based evaluation
- **OpenAI Whisper (large-v3 on GPU)** for audio-only transcription <- this is the latest and greatest in transcription for this repo. It is on `whisperlarge_v3` subfolder
- **Google Cloud Video Intelligence API** for activity recognition and segmentation passed through to **GPT-4o** for scoring
- **Demucs + Whisper (GPU)** for teacher voice isolation
- **LLaVA (Language-Image Alignment Vision Assistant)** with feature-based prompts (e.g., setup, prop_usage, engagement) and 1 frame per 10 s extraction for visual analysis of classroom scenes
- **Combined Audio-Video Feedback Script** that merges audio and video JSON transcripts into a unified GPT-4o evaluation

### Evolution of Approaches

#### Early Fast Approaches (Low Signal)
Initially, we tried basic methods using Whisper and Google Cloud Video Intelligence API. These approaches were fast and simple to implement but produced limited or no actionable insights, primarily due to noisy environments and poor video quality.

#### GPU-Based Custom Models (Higher Signal)
To overcome the limitations of earlier methods, we moved to a GPU-based setup using a GCP NVIDIA T4 GPU with the machine type: **n1-standard-16 (16 vCPUs, 60 GB memory)**. The following components were integrated:

1. **Demucs + Whisper (Audio)** - Accurate teacher voice isolation from noisy classroom audio.
2. **LLaVA (Visual)** - Frame-by-frame scene analysis to understand classroom dynamics visually.
3. **Combine Script** - Integrates audio and visual feedback to generate a holistic report.

### Whisper-large-v3 Audio Transcription
We’ve added a standalone audio transcription pipeline using **Whisper-large-v3**. This script chunks the input video into 1-minute segments, transcribes each with Whisper-large-v3, and sends the combined transcript to **GPT-4o** for detailed feedback.

#### Usage
```bash
python whisperaudio_chunk_transcribe_analyze.py path/to/video.mp4
```

### Demucs + Whisper Pipeline (Audio)
The Demucs + Whisper pipeline isolates the teacher’s voice from noisy classroom environments and transcribes it using Whisper large-v3. This approach significantly improves transcription accuracy when compared to raw audio processing.

#### Usage (within the Demucs + Whisper file path)
```bash
python main.py path/to/video.mp4
```

### LLaVA Video Analysis (Visual)
The LLaVA pipeline has been updated to use **feature-based prompts** rather than a single monolithic prompt. We now query each extracted frame for specific feature categories—such as **setup** (classroom arrangement), **prop_usage** (teacher’s use of visual aids), **engagement** (student participation), **classroom_management**, etc.—to focus the model on actionable aspects. Frame extraction has also been adjusted to capture **one frame every 10 seconds**, balancing temporal coverage against token use.

#### Usage (within the LLaVA file path)
```bash
python main.py path/to/video.mp4
```

### Combining Audio and Video Feedback
The combined feedback generator now accepts **two JSON transcript files** (audio and visual), merges them into a consolidated transcript, and feeds the combined content into **GPT-4o** for a unified evaluation.

#### Usage
```bash
python combine_audio_video_feedback.py --audio-json path/to/audio_feedback.json --video-json path/to/video_feedback.json
```

### Issues and Limitations
- **Audio Quality**: The current audio transcript still contains noise from adjacent classes (picked up by the microphone), which affects accuracy. Human reviewers can often ignore background sounds, but the audio processing currently lacks this ability. We may address this by using larger models and improving microphone setup.
- **Visual Context**: The LLaVA model sometimes guesses the class type (e.g., 'Art and Craft') when it is uncertain. Providing more explicit context or providing a fully contained activity rather than a small clip of an activity may yield better results. 
    - This was verified by simply attempting a more focused LLaVA prompt, hence further improvements will be made in this direction. The difference in results can be observed in the outputs for _teacher1_ (without a detailed config prompt, only snippet of activity) v/s. _teacher2_ (with a detailed config prompt and fully contained activity)

Even though these issues exist, the output still catches actionable feedback such as teacher dominance, usage of open-ended questions, and personalized attention.

### Sample Final Output
See the file `Combined_Pipeline_Outputs` folder for the detailed outputs from the 2 runs. Inside this folder you can see the intermediate outputs from the Audio and Video processing steps as well as the final feedback. 
- **Teacher 1**: Only a snippet of an art & craft activity was provided and the prompt used by LLaVA was a stub prompt. 
- **Teacher 2**: A clip of a full storytelling activity was provided with the LLaVA prompt containing both context about the activity and directions w.reg. to what to analyze in the images. 
- **Teacher 2**: Updated feedback with the 09/05/2025 changes made.

### ⚠️ Important
All code runs on a GCP NVIDIA T4 GPU (n1-standard-16) for optimal performance. Ensure you have the appropriate environment configured.

More updates and experimental results coming soon.
