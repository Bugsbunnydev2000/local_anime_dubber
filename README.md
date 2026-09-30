# Anime Dubber (MVP) — JA → EN local AI dubbing

Local pipeline: Demucs (source separation) → MediaPipe Face Mesh
(mouth-timing alignment) → faster-whisper/CTranslate2 (full-coverage ASR)
→ qwen2.5:7b via Ollama (translation + length-aware rewrite) → OmniVoice
(voice cloning + duration-guided TTS, with pronunciation controls) →
ffmpeg (mux). No cloud APIs, no API keys.




