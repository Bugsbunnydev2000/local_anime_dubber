"""
Rewrite translated lines that are noticeably too long or too short for the
time slot they need to fit into, using qwen2.5:7b served locally by Ollama
(http://localhost:11434) -- rather than loading a model directly into this
process via transformers.

This runs AFTER M2M100 translation and BEFORE synthesis. M2M100 translates
for meaning only, with no awareness of timing, so English lines often end
up much longer or shorter than the original Japanese -- this pass nudges
phrasing back toward a natural length for the slot while trying to keep the
meaning intact. It's a rough pacing aid, not a substitute for professional
dubbing-script adaptation.

We only call Ollama when a line is meaningfully off-target (see
config.REWRITE_TOLERANCE), both to save time and because every rewrite is a
small risk of drifting from the original meaning.

NOTE ON VRAM: Ollama runs as its own server process and manages its own
model loading/unloading independently -- it is NOT covered by this
project's free_gpu_memory() sequencing between stages. If you hit VRAM
pressure, see the README note on running this pipeline alongside Ollama.
"""

import requests

import config
from src.text_utils import contains_japanese


def _call_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {config.OLLAMA_HOST}. "
            "Make sure Ollama is running (`ollama serve`) and that the "
            f"model is available (`ollama pull {config.OLLAMA_MODEL}`)."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Ollama returned an error for model '{config.OLLAMA_MODEL}': {exc}\n"
            f"Response body: {response.text}"
        ) from exc

    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def estimate_word_budget(duration_seconds: float) -> int:
    return max(1, round(duration_seconds * config.WORDS_PER_SECOND))


def _rewrite_line(text_en: str, target_words: int) -> str:
    prompt = (
        "You are a dubbing script editor. Rewrite the following English line "
        f"so it can be spoken naturally in about {target_words} words, keeping "
        "the same meaning and tone as closely as possible. Output ONLY the "
        "rewritten line, with no explanation, quotes, or extra text.\n\n"
        f"Line: {text_en}"
    )
    rewritten = _call_ollama(prompt).strip().strip('"').strip()
    return rewritten if rewritten else text_en


def rewrite_segments(segments, device: str = "cuda"):
    """Add a length-adjusted "text_en" to segments that are off-pace.
    Segments already close to their target length are left untouched, and
    Ollama is never called for them.

    `device` is accepted for interface compatibility with the rest of the
    pipeline (main.py passes it uniformly to every stage) but is unused
    here -- Ollama manages its own device placement independently.
    """
    result = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        target_words = estimate_word_budget(duration)
        current_words = len(seg["text_en"].split())

        if current_words == 0:
            result.append(seg)
            continue

        off_by = abs(current_words - target_words) / target_words
        if off_by <= config.REWRITE_TOLERANCE:
            result.append(seg)
            continue

        new_text = _rewrite_line(seg["text_en"], target_words)

        if contains_japanese(new_text):
            # Safety net: never let the rewrite pass reintroduce Japanese
            # into an already-translated line. Keep the pre-rewrite text
            # instead of risking a corrupted TTS input.
            print(f"    Warning: rewrite introduced Japanese text, keeping original: {seg['text_en']!r}")
            result.append(seg)
            continue

        result.append({**seg, "text_en": new_text})

    return result
