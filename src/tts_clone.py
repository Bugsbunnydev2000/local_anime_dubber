"""
Clone the original speaker's voice and synthesize the translated English
line for each segment, using OmniVoice (k2-fsa/OmniVoice).

Voice cloning mode: a single reference audio clip of the original voice is
reused for every segment, with an EXPLICIT ref_text (transcribed once by
our own Whisper pass in src/asr.py, with the source language forced) --
rather than letting OmniVoice auto-transcribe the reference clip itself
internally on every call, with undirected language detection. Leaving
ref_text unset was the likely cause of brief non-English artifacts
bleeding into the start of some generated lines: without an accurate,
correctly-forced-language anchor for where the reference ends, the model
has more room to blend reference content into the generated output.

Pronunciation handling: text is preprocessed (src/pronunciation.py) before
synthesis, and generation explicitly declares language="en" and requests
text normalization, rather than relying on auto-detection. Note that
OmniVoice's own docs state cross-lingual cloning (our Japanese reference
voice speaking English text) inherently carries some accent from the
reference language -- these settings reduce mispronunciation, they don't
eliminate the underlying cross-lingual accent behavior entirely.
"""

from pathlib import Path

import soundfile as sf
import torch
from omnivoice import OmniVoice

import config
from src.pronunciation import prepare_text_for_tts

OMNIVOICE_SAMPLE_RATE = 24000  # OmniVoice always outputs 24kHz audio


def load_omnivoice():
    """Load OmniVoice onto the GPU. Call this only when needed, and delete
    the returned object + call free_gpu_memory() as soon as you're done."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model = OmniVoice.from_pretrained(
        config.OMNIVOICE_MODEL_ID,
        device_map=device,
        dtype=dtype,
    )
    return model


def _generate(model, text: str, reference_wav_path: str, reference_text: str, target_duration: float):
    """Try the fullest-featured generate() call first, and progressively
    fall back to simpler calls if:
      - the installed OmniVoice build doesn't support a given kwarg
        (e.g. normalize_text requires an optional extra to be installed), or
      - the requested duration isn't feasible for this line's length, or
      - the provided ref_text is somehow rejected.
    This keeps a single problematic line from failing the whole run.

    NOTE: we deliberately do NOT pass postprocess_output=False anymore.
    Forcing exact duration at the cost of skipping the model's own output
    cleanup was a likely contributor to audible artifacts at the start of
    some lines -- losing perfectly exact per-line duration is an acceptable
    trade, since final assembly already tolerates natural length variation.
    """
    base_kwargs = dict(text=text, ref_audio=reference_wav_path, language=config.TARGET_LANG)
    if reference_text:
        base_kwargs["ref_text"] = reference_text

    attempts = [
        {**base_kwargs, "duration": target_duration, "normalize_text": True},
        {**base_kwargs, "duration": target_duration},
        {**base_kwargs},
        {"text": text, "ref_audio": reference_wav_path},  # last resort: minimal call
    ]

    last_error = None
    for kwargs in attempts:
        try:
            return model.generate(**kwargs)
        except TypeError as exc:
            # Unsupported kwarg in this installed OmniVoice build -- try a
            # simpler call.
            last_error = exc
            continue
        except Exception as exc:
            # e.g. infeasible duration for this text length, or a rejected
            # ref_text -- try a simpler call.
            last_error = exc
            continue

    raise RuntimeError(f"OmniVoice generation failed for text: {text!r}") from last_error


def synthesize_segments(model, segments, reference_wav_path, reference_text, out_dir):
    """Generate cloned English speech for each segment, targeting each
    segment's original duration directly (via OmniVoice's `duration` param)
    rather than generating freely and stretching the waveform afterward --
    stretching introduces pitch/formant artifacts, while duration-guided
    generation just paces the delivery naturally, the way a real dubbing
    actor would speed up or slow down a line.

    Returns a new list of segment dicts with "wav_path" and "sample_rate" added.
    Segments with empty translated text are skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_wav_path = str(reference_wav_path)

    results = []
    for i, seg in enumerate(segments):
        text_en = seg.get("text_en", "").strip()
        if not text_en:
            continue

        tts_text = prepare_text_for_tts(text_en)
        target_duration = seg["end"] - seg["start"]

        audio_list = _generate(model, tts_text, reference_wav_path, reference_text, target_duration)
        wav = audio_list[0]  # list of np.ndarray, shape (T,) at 24kHz

        out_path = out_dir / f"seg_{i:03d}.wav"
        sf.write(str(out_path), wav, OMNIVOICE_SAMPLE_RATE)

        results.append({**seg, "wav_path": out_path, "sample_rate": OMNIVOICE_SAMPLE_RATE})

    return results
