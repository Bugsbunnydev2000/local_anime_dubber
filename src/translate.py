"""
Translate each transcribed Japanese line to English using qwen2.5:7b
served locally by Ollama -- replacing M2M100-418M, which produced noticeably
worse, less natural translations.

Includes an automatic correction loop: if the model's output still
contains Japanese characters (it sometimes leaves ambiguous or invented
terms untranslated rather than committing to an English rendering), the
line is sent back with an explicit correction request, up to a few times,
rather than silently shipping a partially-translated line.
"""

import requests

import config
from src.text_utils import contains_japanese

MAX_TRANSLATION_ATTEMPTS = 3


def _call_ollama_chat(prompt: str) -> str:
    try:
        response = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.2},
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
    text = data.get("message", {}).get("content", "").strip()
    return text.strip('"').strip()


def _build_translate_prompt(text_ja: str) -> str:
    return (
        "Translate the following Japanese anime dialogue line into natural, "
        "spoken English suitable for dubbing. Preserve the tone and meaning "
        "of the original as closely as possible. Spell out any numbers or "
        "abbreviations as full words (e.g. \"123\" -> \"one hundred twenty-three\"). "
        "Translate EVERY word into English -- do not leave any Japanese "
        "characters, words, or phrases untranslated in your answer. If a "
        "term is ambiguous, invented, or has no exact English equivalent, "
        "give your best English approximation rather than leaving it in "
        "Japanese. Output ONLY the English translation -- no romaji, notes, "
        "quotes, or explanation.\n\n"
        f"Japanese: {text_ja}"
    )


def _build_correction_prompt(text_ja: str, previous_attempt: str) -> str:
    return (
        "Your previous translation still contained Japanese characters: "
        f"\"{previous_attempt}\". Rewrite it as a FULLY English translation "
        "with absolutely no Japanese characters remaining anywhere in your "
        "answer. If a word or name has no exact English equivalent, use "
        "your best English approximation instead of leaving it in "
        "Japanese. Output ONLY the corrected English translation.\n\n"
        f"Original Japanese line: {text_ja}"
    )


def translate_line(text_ja: str) -> str:
    text_en = _call_ollama_chat(_build_translate_prompt(text_ja))

    attempt = 1
    while contains_japanese(text_en) and attempt < MAX_TRANSLATION_ATTEMPTS:
        attempt += 1
        text_en = _call_ollama_chat(_build_correction_prompt(text_ja, text_en))

    if contains_japanese(text_en):
        print(f"    Warning: translation may still contain untranslated Japanese: {text_en!r}")

    return text_en


def translate_segments(segments, device: str = "cuda"):
    """Add a "text_en" field to each segment dict. Returns a new list.

    `device` is accepted for interface compatibility with the rest of the
    pipeline (main.py passes it uniformly to every stage) but is unused --
    Ollama manages its own device placement independently.
    """
    translated = []
    for seg in segments:
        text_en = translate_line(seg["text"])
        translated.append({**seg, "text_en": text_en})
    return translated
