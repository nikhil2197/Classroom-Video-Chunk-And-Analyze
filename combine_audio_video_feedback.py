import os
from openai import OpenAI
from openai.types.chat import ChatCompletion
import json

# Paths to feedback files
audio_feedback_path = "" #ENTER PATH OF AUDIO_FEEDBACK.JSON FROM THE demucs_whisper run 
image_feedback_path = "" #ENTER PATH OF IMAGE_FEEDBACK.TXT FROM THE LLaVA_GPT4o run 

# OpenAI API key setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_audio_feedback():
    with open(audio_feedback_path, 'r') as f:
        return json.load(f)

def load_image_feedback():
    with open(image_feedback_path, 'r') as f:
        return f.read()

def generate_combined_feedback(audio_feedback, image_feedback):
    prompt = f"Audio-based feedback: {audio_feedback}\nImage-based feedback: {image_feedback}\n\n You have been provided with feedback from an teachers audio transcript of the class and feedback from the frame by frame video of the class without Audio. As an expert pre-school evaluator understand both and provide a combined, actionable, and comprehensive feedback for the preschool class based on the inputs."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an expert in preschool education feedback."},
                 {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Load feedback
    audio_feedback = load_audio_feedback()
    image_feedback = load_image_feedback()

    # Generate combined feedback
    combined_feedback = generate_combined_feedback(audio_feedback, image_feedback)

    # Save combined feedback
    output_path = "/home/nikhilramesh/Chunk-Transcribe-Analyze/combined_feedback.txt"
    with open(output_path, 'w') as f:
        f.write(combined_feedback)

    print(f"Combined feedback saved to {output_path}")
