"""
Central configuration for the anime dubbing pipeline.
Adjust the *_MODEL_PATH values if your downloaded models live elsewhere
(e.g. a different drive, or the raw Hugging Face cache path).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Model locations -------------------------------------------------------
# Point these at the folders you downloaded from Hugging Face.
# They can be local snapshot folders (containing config.json, weights, etc.)
MODELS_DIR = PROJECT_ROOT / "models"

# Original transformers-format snapshot -- kept only as the SOURCE for the
# one-time local CTranslate2 conversion (see README). The pipeline itself
# no longer loads this path directly.
WHISPER_MODEL_PATH = str(MODELS_DIR / "whisper-large-v3-turbo")

# CTranslate2-converted copy of the same weights above, used by
# faster-whisper. Produced locally by `ct2-transformers-converter` -- no
# download from Hugging Face required, since it converts your existing
# WHISPER_MODEL_PATH weights in place.
WHISPER_CT2_MODEL_PATH = str(MODELS_DIR / "whisper-large-v3-turbo-ct2")

# OmniVoice can be either the Hugging Face Hub id (auto-downloaded and
# cached under ~/.cache/huggingface on first run) or a local snapshot
# folder path if you've already downloaded the weights yourself.
OMNIVOICE_MODEL_ID = "k2-fsa/OmniVoice"

# Translation AND the length-aware rewrite pass are both served by a local
# Ollama instance running qwen2.5:7b (ollama pull qwen2.5:7b). M2M100 is no
# longer used -- Qwen gives noticeably better, more natural translations.
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# --- Source separation (Demucs) ----------------------------------------------
DEMUCS_MODEL = "htdemucs"

# --- ASR / VAD (faster-whisper's built-in filter) ----------------------------
# faster-whisper transcribes the ENTIRE vocals track using CTranslate2's
# mature long-form algorithm (proper sequential windowing with correct
# timestamp continuation), with its own built-in VAD filter to skip
# non-speech. This replaced an earlier transformers-pipeline approach
# (`chunk_length_s` long-form chunking) that transformers itself flags as
# "very experimental with seq2seq models" -- in practice it produced
# overlapping/duplicate segment timestamps, which caused multiple dubbed
# lines to be summed onto the same stretch of audio (audibly overlapping
# voices). faster-whisper does not have that failure mode.
ASR_VAD_THRESHOLD = 0.35          # lower = more sensitive to quiet/soft speech
ASR_VAD_MIN_SPEECH_MS = 150
ASR_VAD_MIN_SILENCE_MS = 200
ASR_VAD_SPEECH_PAD_MS = 120       # pad detected speech regions to avoid clipped word edges

# --- Final overlap safety net -------------------------------------------------
# No ASR system produces perfectly non-overlapping timestamps on tightly
# paced speech -- boundary estimation imprecision plus padding above can
# leave adjacent segments overlapping by anywhere from a fraction of a
# second to a few seconds. Since overlapping segments get summed on the
# output track (audibly simultaneous voices), this is enforced as a hard
# rule after every other timing adjustment.
SEGMENT_MIN_GAP_S = 0.05
SEGMENT_MIN_DURATION_S = 0.3

# --- Timing / rewrite pass ----------------------------------------------------
# Rough English speaking rate used to estimate how many words fit a given
# duration. ~2.3-2.6 words/sec is typical for natural spoken English.
WORDS_PER_SECOND = 2.4
# Only trigger the LLM rewrite pass when a translated line's word count is
# off from its target by more than this fraction (avoids unnecessary rewrites
# -- and their small risk of drifting from the original meaning -- on lines
# that are already close enough).
REWRITE_TOLERANCE = 0.25

# --- Mouth-timing alignment (lip-sync via retiming, not pixel generation) ----
# Instead of generating new mouth pixels (Wav2Lip-style neural lip-sync,
# unreliable on non-photorealistic/stylized faces), we detect when the
# on-screen mouth is actually moving and snap dubbed-line timing to match.
MOUTH_SAMPLE_EVERY_N_FRAMES = 1        # increase (e.g. 2-3) to speed up long clips
MOUTH_OPEN_FRACTION_THRESHOLD = 0.15   # fraction of this clip's own open/close range
MOUTH_MIN_ACTIVE_DURATION_S = 0.1
MOUTH_MIN_GAP_S = 0.15
MOUTH_SEGMENT_OVERLAP_TOLERANCE_S = 0.3
MOUTH_MAX_DURATION_GROWTH_FACTOR = 2.5  # cap against visual noise ballooning a line

# --- Voice reference clip auto-trimming --------------------------------------
# OmniVoice's own docs warn that long reference audio degrades cloning
# quality and increases memory/generation time (recommends 3-10s). Rather
# than relying on you to manually cut a clip every time, this trims
# automatically to a clean, contiguous speech window in that range -- both
# the start AND end of the trim are aligned to natural pauses in speech
# (not just the start), since ending mid-word was found to corrupt the
# reference transcript and cause mispronunciation artifacts on every line.
REFERENCE_MIN_SECONDS = 3.0
REFERENCE_MAX_SECONDS = 10.0

# --- Background music mixing --------------------------------------------------
# Volume multiplier applied to the background stem under dialogue
# (0.35 = background drops to 35% volume while a line is playing).
DUCK_LEVEL = 0.35
DUCK_FADE_SECONDS = 0.15

# --- I/O directories ---------------------------------------------------------
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
WORK_DIR = PROJECT_ROOT / "work"  # intermediate files: extracted audio, per-segment wavs

# --- Language settings -------------------------------------------------------
SOURCE_LANG = "ja"  # Whisper language hint
TARGET_LANG = "en"  # Passed to OmniVoice's `language` parameter

# --- Audio settings -----------------------------------------------------------
# Sample rate used for the final assembled/mixed audio track. Higher than
# OmniVoice's native 24kHz output since we're mixing in the original
# (typically 44.1/48kHz) background music -- everything gets resampled to
# this rate during mixing.
FINAL_SAMPLE_RATE = 44100
