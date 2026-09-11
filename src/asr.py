"""
Transcribe Japanese speech with Whisper-large-v3-turbo, via faster-whisper
(CTranslate2) instead of the transformers pipeline.

WHY THE SWITCH: the previous version used transformers' `chunk_length_s`
long-form chunking to guarantee no audio was skipped (fixing an earlier
"missing speech" bug) -- but transformers' own docs flag that mechanism as
"very experimental with seq2seq models," and in practice it produced
overlapping, sometimes duplicate-start segment timestamps (e.g. two
segments both starting at 0.00s). Overlapping segments mean two lines of
generated dialogue get summed onto the same stretch of the output track --
audibly overlapping/mixed voices, which is what broke an earlier test run.

faster-whisper uses CTranslate2's mature, purpose-built long-form
transcription algorithm (proper sequential windowing with correct
timestamp continuation) plus a well-tested built-in VAD filter, without
that experimental-chunking caveat. This loads a CTranslate2-converted copy
of your EXISTING local Whisper weights -- see README for the one-time,
fully local conversion command (no new Hugging Face download required).

The loaded model returned by load_whisper() is also passed directly to
src.reference_audio.prepare_reference_clip(), which handles reference-clip
transcription and trimming itself -- see that module for why.
"""

from faster_whisper import WhisperModel

import config


def load_whisper(device: str = "cuda") -> WhisperModel:
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(
        config.WHISPER_CT2_MODEL_PATH,
        device=device,
        compute_type=compute_type,
    )


def transcribe_full(model: WhisperModel, vocals_path):
    """Transcribe the entire vocals track. Returns:
    [{"start": float, "end": float, "text": str}, ...]

    faster-whisper's segments are already sequential and non-overlapping
    (unlike the previous transformers-chunking approach), so no extra
    de-overlapping logic is needed here.
    """
    segments_iter, _info = model.transcribe(
        str(vocals_path),
        language=config.SOURCE_LANG,
        task="transcribe",
        vad_filter=True,
        vad_parameters=dict(
            threshold=config.ASR_VAD_THRESHOLD,
            min_speech_duration_ms=config.ASR_VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=config.ASR_VAD_MIN_SILENCE_MS,
            speech_pad_ms=config.ASR_VAD_SPEECH_PAD_MS,
        ),
        condition_on_previous_text=False,
    )

    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({"start": float(seg.start), "end": float(seg.end), "text": text})

    return segments


def enforce_non_overlapping(segments, min_gap: float = 0.05, min_duration: float = 0.3):
    """Guarantee the final segment list is strictly ordered and
    non-overlapping, regardless of any residual overlap left by ASR
    timestamp estimation, VAD padding, or mouth-alignment.

    No ASR system produces perfectly non-overlapping timestamps on tightly
    paced, pause-free speech -- boundary estimation always has some
    imprecision, and padding added to avoid clipping word onsets can push
    neighboring segments into each other. Since segments are placed on the
    output track by timestamp and overlapping audio gets SUMMED, even a
    fraction of a second of overlap is audible as two voices briefly
    speaking at once. This is applied as the final step, after every other
    timing adjustment, specifically to make that impossible: each segment
    is only ever pushed later (never earlier), so earlier segments keep
    their original timing and only later ones absorb the correction.
    """
    if not segments:
        return segments

    ordered = sorted(segments, key=lambda s: s["start"])
    fixed = []
    last_end = float("-inf")

    for seg in ordered:
        seg = dict(seg)
        if seg["start"] < last_end + min_gap:
            seg["start"] = last_end + min_gap
        if seg["end"] < seg["start"] + min_duration:
            seg["end"] = seg["start"] + min_duration
        fixed.append(seg)
        last_end = seg["end"]

    return fixed
