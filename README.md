# Anime Dubber (MVP) — JA → EN local AI dubbing

Local pipeline: Demucs (source separation) → MediaPipe Face Mesh
(mouth-timing alignment) → faster-whisper/CTranslate2 (full-coverage ASR)
→ qwen2.5:7b via Ollama (translation + length-aware rewrite) → OmniVoice
(voice cloning + duration-guided TTS, with pronunciation controls) →
ffmpeg (mux). No cloud APIs, no API keys.

## 1. Install system dependency: ffmpeg

**Windows:** download from https://ffmpeg.org/download.html and add the
`bin` folder to your PATH, or `winget install ffmpeg`.

**Linux:** `sudo apt install ffmpeg`

Verify it worked:
```
ffmpeg -version
ffprobe -version
```

## 2. Install PyTorch 2.8.x (required by OmniVoice)

```
pip uninstall torch torchvision torchaudio -y
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
```

Verify:
```
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```
You want `torch.cuda.is_available()` to print `True`.

## 3. Set up Ollama for translation + rewrite

You've already pulled `qwen2.5:7b`. Just make sure the Ollama server is
running before you run `main.py`:

```
ollama serve
```

(If Ollama is installed as a background service/app on your system, it may
already be running — check with `ollama list` or by visiting
http://localhost:11434 in a browser, which should return "Ollama is
running".)

## 4. Install the rest of the dependencies

```
pip install -r requirements.txt
```

If you previously installed `silero-vad`, it's no longer used and can be
removed (`pip uninstall silero-vad`) — faster-whisper has its own built-in
VAD filter now.

`k2-fsa/OmniVoice` and Demucs's `htdemucs` model still auto-download from
the Hugging Face Hub on first run (cached locally afterward).

## 5. Convert your Whisper model to CTranslate2 format (one-time, local, no download)

This is the key step for this update. It converts your **already
downloaded** `whisper-large-v3-turbo` folder into the format
`faster-whisper` needs — entirely on your machine, reading your existing
weights and writing a converted copy. No new download from Hugging Face.

```
ct2-transformers-converter --model models/whisper-large-v3-turbo --output_dir models/whisper-large-v3-turbo-ct2 --quantization float16
```

This is installed as part of the `ctranslate2` package from step 4, and
typically finishes in well under a couple of minutes on a normal CPU for a
model this size. Your original `models/whisper-large-v3-turbo/` folder can
stay where it is (or be deleted afterward if you want the disk space back
— the converted copy is self-contained).

## 6. Place your input files

```
anime-dubber/input/clip.mp4        # your video clip
anime-dubber/input/voice_ref.wav   # a clean sample of the original voice
```

The reference clip no longer needs to be manually trimmed — see "What
changed" below. Any length works; long clips are automatically shortened.

## 7. Run it

```
python main.py --video input/clip.mp4 --voice-ref input/voice_ref.wav --output output/clip_dubbed.mp4
```

You'll see console output for all 8 stages: reference trimming,
separation, detected mouth-movement intervals, transcription, translation,
any lines rewritten for pacing, voice cloning/synthesis, and final mixing.

## What changed in this version

**1. Fixed overlapping/mixed voices.** The previous ASR approach
(transformers' `chunk_length_s` long-form chunking) is explicitly flagged
by transformers itself as "very experimental with seq2seq models" — and in
practice it produced broken timestamps: segments starting at the same
time, overlapping by several seconds. Since dialogue lines are placed on
the output track by timestamp and overlapping audio gets summed, this
caused multiple lines of generated speech to play simultaneously — the
"voices mixed together" you heard. **Fixed by switching to faster-whisper
(CTranslate2)**, which uses a mature, purpose-built long-form algorithm
with proper sequential timestamp handling and a well-tested built-in VAD
filter — see step 5 above for the one-time local conversion this requires.
The standalone Silero-VAD-then-snap workaround from the previous version
is no longer needed and has been removed, since faster-whisper's own VAD
filtering handles this natively and more reliably.

**2. Automatic voice reference trimming.** OmniVoice's own docs warn that
long reference audio (yours was the entire 49.5s clip) degrades cloning
quality and slows generation, recommending 3-10 seconds. `src/reference_audio.py`
now automatically finds where speech actually starts (via ffmpeg's silence
detection, so it doesn't cut into a word) and trims to a clean window in
that range — no manual clip prep needed. Runs once per video, before
anything else.

**3. Mouth-timing alignment carried over from the previous version**,
unchanged in approach — but note it can only be fairly judged now that the
timestamp corruption above is fixed; in the run that produced overlapping
voices, mouth-alignment was operating on already-broken segment
boundaries and couldn't meaningfully help.

## VRAM note (RTX 4060, 8GB)

Each model this project loads directly (Demucs, faster-whisper, OmniVoice)
is loaded, used, and explicitly freed before the next one loads — they
never sit in VRAM at the same time. MediaPipe/OpenCV/ffmpeg-based steps run
entirely on CPU. faster-whisper in `float16` on GPU uses noticeably less
VRAM than the old transformers pipeline did for the same model.

**Ollama is different**: it runs as its own separate server process, so
`qwen2.5:7b` sits in Ollama's own VRAM allocation independently of this
project's sequencing, for as long as Ollama keeps it loaded (default
timeout ~5 minutes). If you hit an out-of-memory error around the TTS
stage after translation/rewrite has run:
- Run `ollama stop qwen2.5:7b` right after stage 6 finishes, before stage 7
  starts (manual for now)
- Or lower Ollama's keep-alive: `OLLAMA_KEEP_ALIVE=0 ollama serve`
- Or close other GPU-using apps

## What this MVP still does NOT do

- No pixel-level lip-sync (no new mouth pixels are generated — timing is
  aligned to existing mouth movement, not the mouth shape itself). This is
  a coarser form of sync than frame-accurate viseme matching — lines start
  and end near the right moments, not word-for-word mouth shapes.
- No speaker diarization (one reference voice is used for the whole clip —
  fine for a single-character clip, not scenes with multiple speakers)
- Background separation isn't perfect — SFX tightly layered under dialogue
  can partially bleed into either stem; quality will vary by scene
- Cross-lingual accent (Japanese reference voice speaking English) isn't
  fully solved, only reduced — see OmniVoice's own documented behavior
- Mouth-timing alignment depends on a detectable face; anime/stylized
  content may get little or no benefit from it, since MediaPipe is built
  and validated primarily on real faces



for run : python main.py --video input/clip.mp4 --voice-ref input/voice_ref.mp3 --output output/clip_dubbed.mp4

python main.py --video input/test.mp4 --voice-ref input/test-a.mp3 --output output/clip_dubbed.mp4


