"""
Replace the original video's audio track with the newly assembled English
dub, keeping the original video stream untouched (no re-encoding of video,
so this step is fast and lossless for the picture).
"""

import subprocess
from pathlib import Path


def mux_audio_into_video(video_path, audio_path, output_path) -> Path:
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr}")

    return output_path
