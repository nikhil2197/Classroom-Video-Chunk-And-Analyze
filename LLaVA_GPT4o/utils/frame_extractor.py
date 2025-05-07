import os
import ffmpeg
from config import FPS

def extract_frames(video_path, output_dir):
    frame_dir = os.path.join(output_dir, "frames")
    (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=FPS)
        .output(os.path.join(frame_dir, 'frame_%04d.jpg'), start_number=0)
        .run(overwrite_output=True)
    )
    return frame_dir
