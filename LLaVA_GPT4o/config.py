FPS = 1  # Frames per second to extract
FRAME_SIZE = (224, 224)  # Resize frames for LLaVA
LLaVA_PROMPT = """
<image>\n
This image is from a pre-school storytelling class. I would like you to analyze the scene and provide a detailed breakdown of what is happening. Specifically, focus on the following aspects:

1. Children's Actions: Describe what the children are doing, including any notable gestures, facial expressions, or group activities.
2. Teacher's Role: Observe how the teacher is interacting with the children, including their use of props and storytelling techniques.
3. Classroom Management: Highlight any visible strategies used by the teacher to maintain engagement and order, such as gestures, positioning, or visual aids.
4. Props and Visual Aids: Identify any props or materials being used and explain how they are integrated into the storytelling.
5. Overall Atmosphere: Describe the general mood or energy of the class and how effectively the storytelling setup seems to support learning and engagement.

Please provide a clear and comprehensive analysis based on these points.
"""
OPENAI_MODEL = "gpt-4o"
