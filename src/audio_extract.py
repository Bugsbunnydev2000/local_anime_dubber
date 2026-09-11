"""
Extract a clean mono WAV from the source video, and read the video's
total duration (needed later to size the final assembled audio track).

Requires the `ffmpeg` and `ffprobe` binaries to be installed and on PATH.
"""

import subprocess
from pathlib import Path


def extract_audio(video_path, output_wav_path, sample_rate: int = 16000) -> Path:
    """Extract mono PCM audio from a video file using ffmpeg.

    16kHz mono is what Whisper expects, so we standardize on that here.
    """
    video_path = Path(video_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                      # no video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",                 # mono
        str(output_wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")

    return output_wav_path


def get_media_duration(path) -> float:
    """Return the duration (in seconds) of a media file using ffprobe."""
    path = Path(path)
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed to read duration:\n{result.stderr}")

    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {result.stdout!r}") from exc
