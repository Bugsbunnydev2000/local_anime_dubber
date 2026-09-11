"""
Split the source audio into a vocals stem and a background (music + SFX)
stem using Demucs. This lets us:
  - transcribe/clone from a cleaner vocal-only signal
  - keep the original background music intact in the final dub, instead of
    replacing the whole audio track
"""

import subprocess
from pathlib import Path


def separate_vocals(audio_path, output_dir, model: str = "htdemucs", device: str = "cuda"):
    """Run Demucs in two-stems mode. Returns (vocals_path, background_path).

    Demucs writes output to: <output_dir>/<model>/<audio filename stem>/
      - vocals.wav
      - no_vocals.wav   (everything else: music, ambience, most SFX)
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demucs_device = "cuda" if device == "cuda" else "cpu"

    cmd = [
        "demucs",
        "-n", model,
        "--two-stems", "vocals",
        "-d", demucs_device,
        "-o", str(output_dir),
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs separation failed:\n{result.stderr}")

    stem_dir = output_dir / model / audio_path.stem
    vocals_path = stem_dir / "vocals.wav"
    background_path = stem_dir / "no_vocals.wav"

    if not vocals_path.exists() or not background_path.exists():
        raise RuntimeError(
            f"Expected Demucs output not found in {stem_dir}. "
            f"Demucs stdout:\n{result.stdout}"
        )

    return vocals_path, background_path
