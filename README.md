# Anime Dubber (MVP) — JA → EN local AI dubbing

A local, offline AI dubbing pipeline: takes a Japanese video (anime or live-action) + a short voice sample, and produces an English-dubbed version with the original character's voice cloned, background music/SFX preserved, and dialogue timing aligned to the video — no cloud APIs, no subscription costs, running entirely on your system.

# Current architecture (7 stages) : 

Video → extract audio →

Demucs: split vocals from music/SFX
                              ↓
                              
MediaPipe: detect mouth movement] (normal mode only)
                              ↓
                              
faster-whisper: transcribe reference clip + full vocals track, align timing to mouth movement, force zero overlap
                              ↓
                              
Qwen/Ollama: translate JA→EN, auto-retry if Japanese leaks through
                              ↓
                              
  Qwen/Ollama: rewrite lines too long/short for their time slot
                              ↓
                              
  OmniVoice: clone the voice, synthesize English, target exact duration]
                              ↓
                              
   duck background music under dialogue, mix, mux into final video


   --------------------------------

# Features currently working : 

1- Voice cloning + translation: Whisper (ASR) → Qwen (translation) → OmniVoice (cloned TTS), fully local

2- Background music/SFX preserved, not replaced — ducked under dialogue, not silenced

3-Duration-guided synthesis: lines paced to fit their original timing slot, not generated freely then stretched (which caused artifacts)

4-Mouth-timing alignment: retimes dialogue to match visible mouth movement (not pixel-level lip-sync — see caveats below)

5-Zero-overlap guarantee: enforced as a hard rule regardless of any upstream timing imprecision

6-Pronunciation controls: explicit language forcing, editable phoneme glossary for recurring mispronounced words

7-Auto-trimmed, sentence-accurate reference clips: no manual clip prep needed

8-Automatic mistranslation detection: catches and retries lines where Japanese leaked into the "English" output

9- --mode {anime,normal}: one flag switches duck level, reference length, accent-forcing, and whether mouth-alignment even runs — since these two content types need genuinely different settings, not just different luck
