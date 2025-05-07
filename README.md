# 🎥 Classroom Video Analysis and Feedback Tool

This repository contains a collection of scripts for experimenting with different approaches to analyzing classroom video. The primary goal is to automate the evaluation of teaching sessions through a combination of video chunking, transcription, and performance analysis.

The scripts in this repo include attempts using:

- **OpenAI Whisper** for transcription and evaluation
- **Google Cloud Video Intelligence API** for activity recognition and segmentation passed through to **GPT-4o** for scoring
- **Demucs + Whisper (GPU)** for teacher voice isolation
- **LLaVA (Language-Image Alignment Vision Assistant)** for visual analysis of classroom scenes
- **Combined Audio-Video Feedback Script** for synthesizing audio and image-based feedback into a comprehensive evaluation

### Evolution of Approaches

#### Early Fast Approaches (Low Signal)
Initially, we tried basic methods using Whisper and Google Cloud Video Intelligence API. These approaches were fast and simple to implement but produced limited or no actionable insights, primarily due to noisy environments and poor video quality.

#### GPU-Based Custom Models (Higher Signal)
To overcome the limitations of earlier methods, we moved to a GPU-based setup using a GCP NVIDIA T4 GPU with the machine type: **n1-standard-16 (16 vCPUs, 60 GB memory)**. The following components were integrated:

1. **Demucs + Whisper (Audio)** - Accurate teacher voice isolation from noisy classroom audio.
2. **LLaVA (Visual)** - Frame-by-frame scene analysis to understand classroom dynamics visually.
3. **Combine Script** - Integrates audio and visual feedback to generate a holistic report.

### Demucs + Whisper Pipeline (Audio)
The Demucs + Whisper pipeline isolates the teacher’s voice from noisy classroom environments and transcribes it using Whisper large-v3. This approach significantly improves transcription accuracy when compared to raw audio processing.

#### Usage (within the Demucs + Whisper file path)
```bash
python main.py path/to/video.mp4
```

### LLaVA Video Analysis (Visual)
The LLaVA approach focuses on analyzing classroom scenes from videos using image recognition models. It breaks down classroom interactions visually, identifying group activities, individual engagement, and teacher movements.
**Note on speed** - you can expect to process about 9 frames (1 frame = 1 minute of the video) per clock minute in the current implementation on the NVIDIA T4 GPU mentioned. 

#### Usage (within the LLaVA file path)
```bash
python main.py path/to/video.mp4
```

### Combining Audio and Video Feedback
The combined approach runs the Demucs pipeline first, followed by the LLaVA pipeline. The outputs from both are then integrated using the combine script to generate a comprehensive evaluation report.

#### Usage (after the Audio and Video runs have been completed and path added)
```bash
python combine_audio_video_feedback.py
```

### Issues and Limitations
- **Audio Quality**: The current audio transcript still contains noise from adjacent classes (picked up by the microphone), which affects accuracy. Human reviewers can often ignore background sounds, but the audio processing currently lacks this ability. We may address this by using larger models and improving microphone setup.
- **Visual Context**: The LLaVA model sometimes guesses the class type (e.g., 'Art and Craft') when it is uncertain. Providing more explicit context or providing a fully contained activity rather than a small clip of an activity may yield better results. 
    - This was verified by simply attempting a more focused LLaVA prompt, hence further improvements will be made in this direction. The difference in results can be observed in the outputs for _teacher1_ (without a detailed config prompt, only snippet of activity) v/s. _teacher2_ (with a detailed config prompt and fully contained activity)

Even though these issues exist, the output still catches actionable feedback such as teacher dominance, usage of open-ended questions, and personalized attention.

### Sample Final Output
See the file `sample_feedback.txt` for a complete example of the output feedback.

### ⚠️ Important
All code runs on a GCP NVIDIA T4 GPU (n1-standard-16) for optimal performance. Ensure you have the appropriate environment configured.

More updates and experimental results coming soon.
