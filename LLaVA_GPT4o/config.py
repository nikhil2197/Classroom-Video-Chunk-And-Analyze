FPS = 1/10  # Frames per second to extract
FRAME_SIZE = (224, 224)  # Resize frames for LLaVA
ANALYSIS_PROMPTS = [
        {
            "name": "setup",
            "prompt": "<image>\nComment on the setup of the classroom such as the general cleanliness, organization of the classroom, decoration of the classroom etc.",
            "start_min": 0,
            "end_min": 1,
        },
        {
            "name": "child_engagement",
            "prompt": "<image>\n Are the children appearing to be engaged e.g doing an action, playing with a toy or listening to the teacher.",
            "start_min": 2,
            "end_min": None,
        },
        {
            "name": "props",
            "prompt": "<image>\n Is the teacher using any props or toys to engage the children ? Are the children playing with any toys or props ?",
            "start_min": 2,
            "end_min": None,
        },
        {
            "name": "body_language",
            "prompt": "<image>\n Is the teacher using hand gestures such as clapping, counting or facial gestures like smile or scowling or movement such as standing up, dancing etc. to engage the children ?",
            "start_min": 2,
            "end_min": None,
        },
    ]
OPENAI_MODEL = "gpt-4o"
