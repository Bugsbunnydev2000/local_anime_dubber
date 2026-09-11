"""
Detect when the on-screen character's mouth is open/moving, using
MediaPipe Face Mesh lip landmarks. This produces a visual, frame-based
timeline of "the mouth is actually talking here" -- independent of audio --
which we then use to snap our audio-derived segment timing to match real
mouth movement.

This is the "mouth-timing alignment" approach to lip-sync: instead of
generating new mouth pixels frame-by-frame (Wav2Lip-style neural lip-sync,
which struggles badly on non-photorealistic/stylized faces), we align the
DUBBED AUDIO's timing to the EXISTING mouth movement already in the
footage -- the same principle professional dubbing uses when timing a
voice recording to match an actor's or character's existing performance.

Runs entirely on CPU via MediaPipe/OpenCV; does not compete for GPU/VRAM
with the rest of the pipeline.
"""

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "opencv-python is required for mouth-timing alignment. Install it with: "
        "pip install opencv-python"
    ) from exc

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "mediapipe is required for mouth-timing alignment. Install it with: "
        "pip install mediapipe"
    ) from exc

# MediaPipe Face Mesh landmark indices (468-point mesh)
_UPPER_INNER_LIP = 13
_LOWER_INNER_LIP = 14
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263


def _frame_openness(landmarks, frame_w: int, frame_h: int):
    """Mouth-opening amount for one frame, normalized by interocular
    distance so it's roughly scale/distance-from-camera invariant. Returns
    None if landmarks are degenerate (shouldn't normally happen once a
    face is detected, but guards against divide-by-zero)."""

    def point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * frame_w, lm.y * frame_h])

    upper = point(_UPPER_INNER_LIP)
    lower = point(_LOWER_INNER_LIP)
    eye_l = point(_LEFT_EYE_OUTER)
    eye_r = point(_RIGHT_EYE_OUTER)

    interocular = np.linalg.norm(eye_r - eye_l)
    if interocular < 1e-3:
        return None

    mouth_gap = np.linalg.norm(upper - lower)
    return float(mouth_gap / interocular)


def extract_mouth_openness_series(video_path, sample_every_n_frames: int = 1):
    """Returns (timestamps, openness_values). openness_values[i] is None
    wherever no face was detected in that sampled frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for mouth detection: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    timestamps = []
    openness_values = []

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % sample_every_n_frames == 0:
                timestamp = frame_index / fps
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    openness = _frame_openness(landmarks, w, h)
                else:
                    openness = None

                timestamps.append(timestamp)
                openness_values.append(openness)

            frame_index += 1

    cap.release()
    return timestamps, openness_values


def _smooth(values, window: int = 5):
    """Moving-average smoothing that tolerates None gaps (no face detected
    in that frame) by only averaging over available numeric neighbors."""
    smoothed = []
    n = len(values)
    half = window // 2
    for i in range(n):
        window_vals = [
            values[j] for j in range(max(0, i - half), min(n, i + half + 1))
            if values[j] is not None
        ]
        smoothed.append(sum(window_vals) / len(window_vals) if window_vals else None)
    return smoothed


def detect_mouth_active_intervals(
    video_path,
    sample_every_n_frames: int = 1,
    open_fraction_threshold: float = 0.15,
    min_active_duration_s: float = 0.1,
    min_gap_s: float = 0.15,
):
    """Returns a list of {"start": float, "end": float} in seconds marking
    periods where the on-screen face's mouth is open/moving noticeably
    above its own resting (closed) baseline for THIS clip.

    Returns an empty list if essentially no face could be detected anywhere
    in the clip (character never on screen, camera angle loses the face,
    or -- for stylized/anime content -- a style MediaPipe can't detect at
    all). Callers should treat an empty list as "no visual data available"
    and fall back to audio-only timing; this is not treated as an error.
    """
    timestamps, raw_values = extract_mouth_openness_series(video_path, sample_every_n_frames)
    values = _smooth(raw_values, window=5)

    valid = [v for v in values if v is not None]
    if len(valid) < 5:
        return []

    baseline = float(np.percentile(valid, 10))  # resting/closed mouth level
    peak = float(np.percentile(valid, 90))       # typical wide-open level
    span = max(peak - baseline, 1e-6)
    threshold = baseline + open_fraction_threshold * span

    active_flags = [v is not None and v > threshold for v in values]

    raw_intervals = []
    start_idx = None
    for i, active in enumerate(active_flags):
        if active and start_idx is None:
            start_idx = i
        elif not active and start_idx is not None:
            raw_intervals.append((timestamps[start_idx], timestamps[i - 1]))
            start_idx = None
    if start_idx is not None:
        raw_intervals.append((timestamps[start_idx], timestamps[-1]))

    if not raw_intervals:
        return []

    merged = [list(raw_intervals[0])]
    for start, end in raw_intervals[1:]:
        if start - merged[-1][1] <= min_gap_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [
        {"start": s, "end": e} for s, e in merged
        if (e - s) >= min_active_duration_s
    ]


def align_segments_to_mouth(segments, mouth_intervals, overlap_tolerance: float = 0.3, max_growth_factor: float = 2.5):
    """For each segment, if mouth-active intervals overlap (or lie within
    overlap_tolerance of) its audio-derived [start, end] window, replace
    the segment's timing with the union of those intervals -- so
    duration-guided TTS and final placement target when the mouth is
    ACTUALLY moving on screen, not just when Whisper/VAD detected audio.

    Segments with no nearby visual mouth activity (off-screen character,
    narration, a cutaway -- exactly the case where lip-sync isn't
    meaningful anyway) are left with their original audio-derived timing.

    max_growth_factor caps how much a segment's duration can expand from
    matching mouth intervals, as a sanity check against visual noise (e.g.
    a yawn or an unrelated nearby mouth movement) ballooning a short line.
    """
    if not mouth_intervals:
        return segments

    aligned = []
    for seg in segments:
        overlapping = [
            m for m in mouth_intervals
            if m["end"] >= seg["start"] - overlap_tolerance
            and m["start"] <= seg["end"] + overlap_tolerance
        ]

        if not overlapping:
            aligned.append(seg)
            continue

        new_start = min(m["start"] for m in overlapping)
        new_end = max(m["end"] for m in overlapping)

        original_duration = seg["end"] - seg["start"]
        new_duration = new_end - new_start
        if original_duration > 0 and new_duration > original_duration * max_growth_factor:
            aligned.append(seg)
            continue

        aligned.append({**seg, "start": new_start, "end": new_end})

    return aligned
