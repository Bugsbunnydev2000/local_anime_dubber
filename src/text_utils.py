"""Small shared text-checking helpers."""

import re

# Hiragana, Katakana, CJK Unified Ideographs (kanji), and halfwidth katakana.
_JAPANESE_CHAR_PATTERN = re.compile(
    r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF66-\uFF9F]'
)


def contains_japanese(text: str) -> bool:
    return bool(_JAPANESE_CHAR_PATTERN.search(text))
