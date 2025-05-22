# 🎥 Classroom Video Analysis and Feedback Tool

This repository contains a collection of scripts for experimenting with different approaches to analyzing classroom video. The primary goal is to automate the evaluation of teaching sessions through a combination of video chunking, transcription, and performance analysis.

See the `Results and Next Steps` section at the bottom for details w.reg. the outcome of the project. 

The scripts in this repo include attempts using:

- **OpenAI Whisper (v2 on API)** for audio-only transcription and GPT-4o-based evaluation
- **OpenAI Whisper (large-v3 on GPU)** for audio-only transcription using 10s chunks instead of 1 minute chunks. It is on `whisperlarge_v3` subfolder
- **ElevenLabs Scribe API** to transcribe the full audio in 1 go <- this is currently the latest and greatest in audio transcription for this repo. You can access this in the `eleven_labs` folder.
- **Google Cloud Video Intelligence API** for activity recognition and segmentation passed through to **GPT-4o** for scoring
- **Demucs + Whisper (GPU)** for teacher voice isolation
- **LLaVA (Language-Image Alignment Vision Assistant)** with feature-based prompts (e.g., setup, prop_usage, engagement) and 1 frame per 10 s extraction for visual analysis of classroom scenes
- **Combined Audio-Video Feedback Script** that merges audio and video JSON transcripts into a unified GPT-4o evaluation

### Evolution of Approaches

#### Early Fast Approaches (Low Signal)
Initially, we tried basic methods using Whisper v2 on the Open AI API (`whisperaudio_chunk_transcribe_analyze`) and Google Cloud Video Intelligence API (`GCPVideoAI_GPT4o_Pipe`). These approaches were fast and simple to implement but produced limited or no actionable insights, primarily due to noisy environments and poor video quality.

#### GPU-Based Custom Models (Higher Signal)
To overcome the limitations of earlier methods, we moved to a GPU-based setup using a GCP NVIDIA T4 GPU with the machine type: **n1-standard-16 (16 vCPUs, 60 GB memory)**. The following components were integrated:

1. **Demucs + Whisper large-v3 (Audio)** - Isolate the teachers voice and only transcribe that. 
2. **LLaVA (Visual)** - Frame-by-frame scene analysis to understand classroom dynamics visually.
3. **Combine Script** - Integrates audio and visual feedback to generate a holistic report.
4.  **Whisper large-v3 (Audio)** - Transcribing the full session audio.
5.  **ElevenLabs Scribe API** - Transcribing the full session audio.

### ElevenLabs Scribe API
Used the elevenlabs scribev1 api to transcribe the full recording in 1 go and generate the output json. This was then re-processed to allign with the frame wise processing done for visual component of the project and then passed into the combined feedback generator. This did a significantly better job at handling the indian accent used in the classroom and handling the background noise. An added benefit is that it is an API call that can be run remotely v/s. whisper-large which requires GPU processing currently. 

#### Usage (within eleven_labs folder)
```bash
python main.py path/to/video.mp4
python split_to_intervals.py path/to/output.json
```

### Demucs + Whisper Pipeline (Audio)
The Demucs + Whisper pipeline isolates the teacher’s voice from noisy classroom environments and transcribes it using Whisper large-v3. This approach significantly improves transcription accuracy when compared to the non GPU models but was not able to accurately capture the teachers voice in a noisy classroom, as a result, we switched to using only Whisper-large with a much smaller chunk size. 

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
The combined feedback generator now accepts **two JSON transcript files** (audio and visual), merges them into a consolidated transcript, and feeds the combined content into **GPT-4o** for a unified evaluation. Doing this was an attempt to increase the context provided during the final analysis run, however token limits led us having to do a 2 step analysis by chunking the original transcript into 8 different segments, analyzing them first and then analyzing the analysis.
_The next efforts will be focused here to improve the analysis of the timeline to match the desired parameters and without losing a lot of the detail provided in the timeline_

#### Usage
```bash
python combine_audio_video_feedback.py --audio-json path/to/audio_feedback.json --video-json path/to/video_feedback.json
```

### Issues and Limitations
- **Audio Quality**: The current audio transcript is within acceptable tolerance using the eleven labs scribe v1 API. Further improvements will come from improving the microphone setup in classroom.
    - We are currently missing prosody (pitch / volume / intonation) features which are a crucial component of classroom facilitaion for children of this age and are working on identifying the best methods to add these features into the combined transcript to make it richer. 
- **Visual Context**: The LLaVA model sometimes guesses the class type (e.g., 'Art and Craft') when it is uncertain. Providing more explicit context or providing a fully contained activity rather than a small clip of an activity may yield better results. 
    - This was verified by simply attempting a more focused LLaVA prompt, hence further improvements will be made in this direction. The difference in results can be observed in the outputs for _teacher1_ (without a detailed config prompt, only snippet of activity) v/s. _teacher2_ (with a detailed config prompt and fully contained activity)
    - We tried using  YOLOv8 to pre-process images before giving them to LLaVA however, the pre-trained weights available don't seem to be suitable for this step giving us very generic output. We might attempt to fine tune our own version of YOLO however this will be a much later improvement. 

Even though these issues exist, the output still catches actionable feedback such as teacher dominance, usage of open-ended questions, and personalized attention.

### Sample Final Output
See the file `Combined_Pipeline_Outputs` folder for the detailed outputs from the 2 runs. Inside this folder you can see the intermediate outputs from the Audio and Video processing steps as well as the final feedback. 
- **Teacher 1**: Only a snippet of an art & craft activity was provided and the prompt used by LLaVA was a stub prompt. 
- **Teacher 2**: A clip of a full storytelling activity was provided with the LLaVA prompt containing both context about the activity and directions w.reg. to what to analyze in the images. 
- **Teacher 2**: Updated feedback with the 09/05/2025 using whisper_v3_large audio processing.
- **Teacher 2**: Updated feedback with the 12/05/2025 changes using the elevenlabs audio processing.

### Results

Given that visual evaluation significantly outperformed audio-based evaluation, we refined our rubrics to assess only two criteria — **Room Setup** and **Activity Setup** — purely through images.

We benchmarked approximately **200 model-generated evaluations** (on a 0, 0.5, 1 scale) against human grader scores:

- **45%** were a **perfect match** with human evaluations  
- An additional **44%** were **within half a band** of the human score  
- All half-band mismatches were **qualitatively explainable**, demonstrating sound model reasoning

This level of accuracy met the predefined MVP threshold, allowing us to proceed to production deployment.

### Next Steps

- Automate image **input and output** processes  
- Scale backend **APIs** to handle increased load  
- Build a more **intuitive front-end** for end users
- 
### ⚠️ Important
All code runs on a GCP NVIDIA T4 GPU (n1-standard-16) for optimal performance. Ensure you have the appropriate environment configured.

More updates and experimental results coming soon.
