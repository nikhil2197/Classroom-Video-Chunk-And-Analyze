import os

def prepare_output_dir(video_name):
    output_dir = os.path.join("outputs", video_name)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
    return output_dir
