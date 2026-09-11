"""
Local AI anime dubbing pipeline (Japanese -> English).

Pipeline:
  1. Extract full audio from the source video
  2. Separate vocals from background music/SFX (Demucs)
  3. Detect visible mouth-movement intervals in the VIDEO (MediaPipe Face
     Mesh) -- used later to align dubbed-line timing to when the character
     is actually seen talking (lip-sync via retiming, not pixel generation)
  4. Load faster-whisper once and use it for two things:
     a. Prepare the voice reference clip: transcribe it, trim to the end
        of the last complete sentence that fits OmniVoice's recommended
        length, and build ref_text from exactly those same sentences (so
        audio and text always match -- see src/reference_audio.py)
     b. Transcribe the FULL vocals track (proper long-form timestamp
        handling, built-in VAD filter), align segment timing to nearby
        visible mouth movement where available, then enforce strictly
        non-overlapping timing as a final hard rule
  5. Translate each line to English with qwen2.5:7b via Ollama, with an
     automatic correction pass if any Japanese characters are left
     untranslated in the output
  6. Rewrite lines that are too long/short for their slot (qwen2.5:7b)
  7. Clone the voice and synthesize each line, targeting its (now
     mouth-aligned) slot duration directly, with explicit
     language/pronunciation controls (OmniVoice)
  8. Assemble the dub: place each line at its aligned timestamp, duck the
     preserved background track under dialogue, and mix the two together
  9. Mux the mixed audio into the original video

Models are loaded and freed one stage at a time to fit an 8GB GPU (except
Ollama, which runs as its own separate server process, and MediaPipe/
OpenCV/ffmpeg-based steps, which run on CPU -- see README).
"""

import argparse

import config
from src.audio_extract import extract_audio, get_media_duration
from src.reference_audio import prepare_reference_clip
from src.separate_audio import separate_vocals
from src.mouth_activity import detect_mouth_active_intervals, align_segments_to_mouth
from src.asr import load_whisper, transcribe_full, enforce_non_overlapping
from src.translate import translate_segments
from src.rewrite import rewrite_segments
from src.tts_clone import load_omnivoice, synthesize_segments
from src.assemble_audio import assemble_track
from src.mux_video import mux_audio_into_video
from src.model_manager import free_gpu_memory, get_device, print_vram


def run(video_path: str, reference_wav_path: str, output_path: str) -> None:
    device = get_device()
    print(f"Using device: {device}")

    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extracted_audio_path = config.WORK_DIR / "source_audio.wav"
    extract_audio(video_path, extracted_audio_path)
    video_duration = get_media_duration(video_path)
    print(f"Video duration: {video_duration:.2f}s")

    # --- Stage 1: Source separation -------------------------------------------
    print("\n[1/7] Separating vocals from background music/SFX (Demucs)...")
    separation_dir = config.WORK_DIR / "separated"
    vocals_path, background_path = separate_vocals(
        extracted_audio_path, separation_dir, model=config.DEMUCS_MODEL, device=device
    )
    free_gpu_memory()
    print_vram("after separation")

    # --- Stage 2: Mouth-activity detection (video) ----------------------------
    print("\n[2/7] Detecting visible mouth movement (MediaPipe Face Mesh)...")
    mouth_intervals = detect_mouth_active_intervals(
        video_path,
        sample_every_n_frames=config.MOUTH_SAMPLE_EVERY_N_FRAMES,
        open_fraction_threshold=config.MOUTH_OPEN_FRACTION_THRESHOLD,
        min_active_duration_s=config.MOUTH_MIN_ACTIVE_DURATION_S,
        min_gap_s=config.MOUTH_MIN_GAP_S,
    )
    if mouth_intervals:
        print(f"  Found {len(mouth_intervals)} visible mouth-movement interval(s).")
    else:
        print("  No face/mouth movement detected -- lines will fall back to audio-only timing.")

    # --- Stage 3: ASR (reference clip + full vocals track) ---------------------
    print("\n[3/7] Transcribing Japanese speech (faster-whisper, full-coverage)...")
    whisper_model = load_whisper(device=device)

    prepared_reference_path, reference_text = prepare_reference_clip(
        whisper_model, reference_wav_path, config.WORK_DIR
    )
    trimmed_duration = get_media_duration(prepared_reference_path)
    print(f"  Reference clip: {prepared_reference_path} ({trimmed_duration:.2f}s)")
    if reference_text:
        print(f"  Reference clip transcript (for TTS conditioning): {reference_text}")
    else:
        print("  Warning: could not transcribe the reference clip; OmniVoice will "
              "fall back to auto-transcribing it internally.")

    segments = transcribe_full(whisper_model, vocals_path)
    if not segments:
        raise RuntimeError("No speech was transcribed. Check the input audio.")
    segments = align_segments_to_mouth(
        segments,
        mouth_intervals,
        overlap_tolerance=config.MOUTH_SEGMENT_OVERLAP_TOLERANCE_S,
        max_growth_factor=config.MOUTH_MAX_DURATION_GROWTH_FACTOR,
    )
    segments = enforce_non_overlapping(
        segments,
        min_gap=config.SEGMENT_MIN_GAP_S,
        min_duration=config.SEGMENT_MIN_DURATION_S,
    )
    for s in segments:
        print(f"    [{s['start']:.2f}-{s['end']:.2f}] {s['text']}")

    del whisper_model
    free_gpu_memory()
    print_vram("after ASR")

    # --- Stage 4: Translation ----------------------------------------------------
    print("\n[4/7] Translating to English (qwen2.5:7b via Ollama)...")
    segments = translate_segments(segments, device=device)
    for s in segments:
        print(f"    JA: {s['text']}")
        print(f"    EN: {s['text_en']}")
    free_gpu_memory()
    print_vram("after translation")

    # --- Stage 5: Length-aware rewrite --------------------------------------------
    print("\n[5/7] Rewriting lines that are off-pace for their slot...")
    segments = rewrite_segments(segments, device=device)
    for s in segments:
        print(f"    EN(final): {s['text_en']}")
    free_gpu_memory()
    print_vram("after rewrite")

    # --- Stage 6: Voice cloning + duration-guided TTS -----------------------------
    print("\n[6/7] Cloning voice and synthesizing English speech (OmniVoice)...")
    omnivoice_model = load_omnivoice()
    tts_out_dir = config.WORK_DIR / "tts_segments"
    segments = synthesize_segments(omnivoice_model, segments, prepared_reference_path, reference_text, tts_out_dir)
    if not segments:
        raise RuntimeError("No segments were synthesized (all translated text was empty).")
    del omnivoice_model
    free_gpu_memory()
    print_vram("after TTS")

    # --- Stage 7: Assemble, mix with background, and mux --------------------------
    print("\n[7/7] Assembling dub, mixing with background, and muxing into video...")
    sample_rate = config.FINAL_SAMPLE_RATE
    final_audio_path = config.WORK_DIR / "dubbed_audio.wav"
    assemble_track(
        segments,
        background_path,
        video_duration,
        sample_rate,
        final_audio_path,
        duck_level=config.DUCK_LEVEL,
        duck_fade_seconds=config.DUCK_FADE_SECONDS,
    )

    mux_audio_into_video(video_path, final_audio_path, output_path)

    print(f"\nDone. Dubbed video saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local AI anime dubbing pipeline (JA -> EN)")
    parser.add_argument("--video", required=True, help="Path to the input video clip")
    parser.add_argument("--voice-ref", required=True, help="Path to a reference WAV of the original voice")
    parser.add_argument(
        "--output",
        default=str(config.OUTPUT_DIR / "dubbed_output.mp4"),
        help="Path to write the final dubbed video",
    )
    args = parser.parse_args()

    run(args.video, args.voice_ref, args.output)
