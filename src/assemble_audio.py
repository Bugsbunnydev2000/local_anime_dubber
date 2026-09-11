"""
Assemble the final dub track:
  1. Place each synthesized English line at its original timestamp (now
     that OmniVoice generates each line targeting that slot's duration
     directly, timestamp placement works much better than it did with
     freely-generated, variable-length lines).
  2. Load the preserved background stem (music/ambience/SFX) and duck its
     volume under dialogue, with a short fade so the volume change isn't
     audible as a click.
  3. Mix the two together and pad/trim to exactly match the video's length.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def _place_voice_track(segments, sample_rate: int, min_length: int) -> np.ndarray:
    track = np.zeros(min_length, dtype=np.float32)

    for seg in segments:
        y, sr = librosa.load(str(seg["wav_path"]), sr=None)
        if sr != sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)

        start = int(seg["start"] * sample_rate)
        end = start + len(y)

        if end > len(track):
            track = np.pad(track, (0, end - len(track)))

        track[start:end] += y

    return track


def _build_duck_envelope(
    segments, length: int, sample_rate: int, duck_level: float, fade_seconds: float
) -> np.ndarray:
    envelope = np.ones(length, dtype=np.float32)
    fade_samples = max(1, int(fade_seconds * sample_rate))

    for seg in segments:
        start = max(0, int(seg["start"] * sample_rate))
        end = min(length, int(seg["end"] * sample_rate))
        if end <= start:
            continue

        seg_env = np.ones(length, dtype=np.float32)

        fade_in_end = min(start + fade_samples, end)
        seg_env[start:fade_in_end] = np.linspace(1.0, duck_level, fade_in_end - start)

        sustain_end = max(start, end - fade_samples)
        seg_env[fade_in_end:sustain_end] = duck_level

        fade_out_start = max(fade_in_end, sustain_end)
        seg_env[fade_out_start:end] = np.linspace(duck_level, 1.0, end - fade_out_start)

        # Combine with min() rather than overwrite, so overlapping segments
        # (e.g. two lines close together) never accidentally un-duck a
        # region that an earlier segment already ducked.
        envelope = np.minimum(envelope, seg_env)

    return envelope


def assemble_track(
    segments,
    background_path,
    total_duration: float,
    sample_rate: int,
    out_path,
    duck_level: float = 0.35,
    duck_fade_seconds: float = 0.15,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = int(total_duration * sample_rate) + 1

    # --- Voice track: each line at its original timestamp -------------------
    voice_track = _place_voice_track(segments, sample_rate, total_samples)

    # --- Background track: load, resample, pad/trim to full length ----------
    background, bg_sr = librosa.load(str(background_path), sr=None, mono=True)
    if bg_sr != sample_rate:
        background = librosa.resample(background, orig_sr=bg_sr, target_sr=sample_rate)

    # Match lengths across voice_track / background before mixing (a
    # duration-guided line can still occasionally run slightly past its
    # slot; padding here keeps array shapes aligned for the elementwise mix).
    common_length = max(len(voice_track), len(background), total_samples)
    voice_track = np.pad(voice_track, (0, common_length - len(voice_track)))
    background = np.pad(background, (0, common_length - len(background)))

    # --- Duck background under dialogue, then mix ----------------------------
    envelope = _build_duck_envelope(segments, common_length, sample_rate, duck_level, duck_fade_seconds)
    mixed = background * envelope + voice_track

    # --- Final length must exactly match the video ---------------------------
    if len(mixed) < total_samples:
        mixed = np.pad(mixed, (0, total_samples - len(mixed)))
    else:
        mixed = mixed[:total_samples]

    peak = np.max(np.abs(mixed)) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak * 0.98

    sf.write(str(out_path), mixed, sample_rate)
    return out_path
