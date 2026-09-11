"""
Ensure the voice reference clip fed to OmniVoice is a short, clean sample
rather than an entire source file. OmniVoice's own docs warn that longer
reference audio degrades cloning quality and increases memory/generation
time (it recommends 3-10 seconds).

This transcribes the reference clip FIRST (with the same faster-whisper
model already loaded for the main pipeline), then trims to the end of the
last complete sentence that fits within the target duration, and builds
ref_text from exactly those same sentences. Audio and text are therefore
guaranteed to match, and the cut can never land mid-word.

An earlier version tried to find a clean cut point via amplitude-based
silence detection (ffmpeg's silencedetect). That failed for speakers with
very short pauses between sentences (well under the ~200ms minimum needed
to register as "silence"): no candidate cut point was ever found, so it
always fell back to a blind cutoff at the max duration -- which regularly
landed mid-word and corrupted the reference transcript. Using the model's
own linguistic sentence boundaries instead of acoustic silence sidesteps
that failure mode entirely, since it doesn't depend on there being a real
pause at all.
"""

from pathlib import Path

from src.audio_extract import extract_audio, get_media_duration

import config

REFERENCE_SAMPLE_RATE = 24000  # OmniVoice's native sample rate


def prepare_reference_clip(whisper_model, reference_path, work_dir):
    """Returns (trimmed_path, ref_text).

    whisper_model: an already-loaded faster_whisper.WhisperModel instance
    (reused from the main ASR stage -- no extra model load needed).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    standardized_path = work_dir / "reference_standardized.wav"
    extract_audio(reference_path, standardized_path, sample_rate=REFERENCE_SAMPLE_RATE)

    segments_iter, _info = whisper_model.transcribe(
        str(standardized_path),
        language=config.SOURCE_LANG,
        task="transcribe",
        vad_filter=True,
    )
    segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in segments_iter
        if s.text.strip()
    ]

    if not segments:
        # Nothing usable detected; return the standardized clip as-is and
        # let OmniVoice fall back to auto-transcribing it internally.
        return standardized_path, ""

    duration = get_media_duration(standardized_path)
    if duration <= config.REFERENCE_MAX_SECONDS:
        return standardized_path, " ".join(s["text"] for s in segments)

    # Greedily include complete segments, in order, for as long as the
    # cumulative span stays within the max duration. The first segment is
    # always included even if it alone exceeds the max (an oversized but
    # intact single sentence is far better than a mid-word cut).
    included = [segments[0]]
    window_start = segments[0]["start"]
    for seg in segments[1:]:
        if seg["end"] - window_start > config.REFERENCE_MAX_SECONDS:
            break
        included.append(seg)

    start = max(0.0, included[0]["start"] - 0.1)
    end = included[-1]["end"]

    trimmed_path = work_dir / "reference_trimmed.wav"
    _trim_audio(standardized_path, trimmed_path, start, end)

    ref_text = " ".join(s["text"] for s in included)
    return trimmed_path, ref_text


def _trim_audio(input_path, output_path, start: float, end: float):
    import subprocess

    duration = max(end - start, 0.1)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg reference trimming failed:\n{result.stderr}")
