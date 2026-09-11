"""
Light text preprocessing applied to translated lines before they reach
OmniVoice, aimed at reducing English mispronunciation.

IMPORTANT CAVEAT: OmniVoice's own documentation states that cross-lingual
voice cloning (a Japanese reference voice speaking English text, our case)
inherently carries some accent from the reference audio's language -- this
is architectural, not a bug. Nothing here eliminates that entirely. What
this module DOES help with:
  1. Explicitly declaring the text's language (see tts_clone.py's
     `language="en"` parameter) instead of relying on auto-detection,
     which is especially unreliable on short lines.
  2. Fixing specific words OmniVoice keeps getting wrong, using its inline
     CMU-dictionary phoneme override syntax: [PH ON EME1 S].
  3. Normalizing abbreviations/symbols that are otherwise read literally
     instead of spoken naturally (a safety net alongside OmniVoice's own
     normalize_text=True option, in case that extra isn't installed).

Add your own character names / invented terms / loanwords to
PRONUNCIATION_GLOSSARY as you notice mispronunciations in your output --
this is meant to be edited over time, not a complete solution out of the
box. Look up CMU phonemes for a word at:
http://www.speech.cs.cmu.edu/cgi-bin/cmudict
"""

import re

# Format: "word as it appears in text" -> "CMU phonemes, space-separated,
# each with a stress digit (0/1/2) on the vowel". Matching is
# case-insensitive and whole-word only.
# Example:
#   "Levi": "L IY1 V AY2",
#   "Eren": "EH1 R AH0 N",
PRONUNCIATION_GLOSSARY = {
    # Add entries here as you find them.
}

# Safety-net normalization in case OmniVoice's own normalize_text=True
# extra isn't installed. Extend as needed.
_ABBREVIATION_MAP = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miss",
    "Dr.": "Doctor",
    "St.": "Saint",
    "&": "and",
    "%": "percent",
}


def apply_glossary(text: str) -> str:
    """Wrap glossary words with OmniVoice's inline phoneme-override syntax."""
    for word, phonemes in PRONUNCIATION_GLOSSARY.items():
        pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
        text = pattern.sub(f"[{phonemes}]", text)
    return text


def apply_fallback_normalization(text: str) -> str:
    """Basic abbreviation/symbol expansion."""
    for abbr, expanded in _ABBREVIATION_MAP.items():
        text = text.replace(abbr, expanded)
    return text


def prepare_text_for_tts(text: str) -> str:
    text = apply_fallback_normalization(text)
    text = apply_glossary(text)
    return text
