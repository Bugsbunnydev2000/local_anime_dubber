# Anime Dubber (MVP) — JA → EN local AI dubbing

A local, offline AI dubbing pipeline: takes a Japanese video (anime or live-action) + a short voice sample, and produces an English-dubbed version with the original character's voice cloned, background music/SFX preserved, and dialogue timing aligned to the video — no cloud APIs, no subscription costs, running entirely on your system.

# Current architecture (7 stages) : 

Video → extract audio → [Demucs: split vocals from music/SFX]
                              ↓
              [MediaPipe: detect mouth movement] (normal mode only)
                              ↓
   [faster-whisper: transcribe reference clip + full vocals track,
    align timing to mouth movement, force zero overlap]
                              ↓
     [Qwen/Ollama: translate JA→EN, auto-retry if Japanese leaks through]
                              ↓
        [Qwen/Ollama: rewrite lines too long/short for their time slot]
                              ↓
    [OmniVoice: clone the voice, synthesize English, target exact duration]
                              ↓
   [duck background music under dialogue, mix, mux into final video]
