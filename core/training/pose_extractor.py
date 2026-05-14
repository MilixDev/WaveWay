"""MediaPipe Pose wrapper that outputs COCO-17 keypoints.

Supports both API generations automatically:
  - Legacy  (mediapipe < 0.10.15): mp.solutions.pose
  - Tasks   (mediapipe >= 0.10.15): mediapipe.tasks.python.vision.PoseLandmarker
            Requires a .task model file — downloaded automatically on first use.

MediaPipe landmark → COCO-17 index mapping (33-landmark model, same for both APIs)
------------------------------------------------------------------------------------
 0  nose            → 0
 1  left_eye        → 2
 2  right_eye       → 5
 3  left_ear        → 7
 4  right_ear       → 8
 5  left_shoulder   → 11
 6  right_shoulder  → 12
 7  left_elbow      → 13
 8  right_elbow     → 14
 9  left_wrist      → 15
10  right_wrist     → 16
11  left_hip        → 23
12  right_hip       → 24
13  left_knee       → 25
14  right_knee      → 26
15  left_ankle      → 27
16  right_ankle     → 28
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Bounding-box helpers (used by DataCollector to normalise poses)
# ----------------------------------------------------------------------

# Strict requirements for an acceptable training sample.
# COCO-17 indices: 5,6 = shoulders; 11,12 = hips. These four define the torso
# anchor — without them the bbox is meaningless.
_REQUIRED_JOINTS = {5, 6, 11, 12}
_MIN_VISIBLE_TOTAL = 10   # of 17 — ensures most of the body is in frame


def compute_bbox(
    keypoints: List[Tuple[float, float, float]],
    vis_thresh: float = 0.3,
    pad: float = 0.10,
) -> Optional[Tuple[float, float, float, float]]:
    """Tight bbox around visible keypoints, padded and clipped to [0,1].

    Requires the torso (both shoulders + both hips) plus enough total joints
    visible. Returns None when the sample is incomplete — those frames must
    be discarded so the model isn't trained on truncated bodies.
    """
    visible_set = {i for i, (_, _, v) in enumerate(keypoints) if v >= vis_thresh}
    if not _REQUIRED_JOINTS.issubset(visible_set):
        return None
    if len(visible_set) < _MIN_VISIBLE_TOTAL:
        return None

    visible = [(keypoints[i][0], keypoints[i][1]) for i in visible_set]
    xs = [p[0] for p in visible]
    ys = [p[1] for p in visible]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    if w < 1e-3 or h < 1e-3:
        return None
    x0 = max(0.0, x0 - pad * w)
    y0 = max(0.0, y0 - pad * h)
    x1 = min(1.0, x1 + pad * w)
    y1 = min(1.0, y1 + pad * h)
    if (x1 - x0) < 0.05 or (y1 - y0) < 0.05:
        return None
    return (x0, y0, x1, y1)


def bbox_diagnosis(
    keypoints: List[Tuple[float, float, float]],
    vis_thresh: float = 0.3,
) -> str:
    """Short human-readable reason why compute_bbox returned None (or 'ok')."""
    visible_set = {i for i, (_, _, v) in enumerate(keypoints) if v >= vis_thresh}
    missing_torso = _REQUIRED_JOINTS - visible_set
    if missing_torso:
        names = {5: "hombro izq", 6: "hombro der", 11: "cadera izq", 12: "cadera der"}
        return "falta " + ", ".join(names[i] for i in sorted(missing_torso))
    if len(visible_set) < _MIN_VISIBLE_TOTAL:
        return f"pocas articulaciones visibles ({len(visible_set)}/17)"
    return "ok"


def normalize_to_bbox(
    keypoints: List[Tuple[float, float, float]],
    bbox: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float]]:
    """Re-express keypoint (x, y) in bbox-relative coords [0, 1]. Visibility kept as-is."""
    x0, y0, x1, y1 = bbox
    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)
    return [
        (float((x - x0) / w), float((y - y0) / h), float(v))
        for x, y, v in keypoints
    ]

_COCO17_FROM_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "training_data" / "pose_landmarker_lite.task"


def _download_model() -> Path:
    """Download the Tasks API model file if not already present."""
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MediaPipe pose model → %s …", _MODEL_PATH)
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    logger.info("Model downloaded (%d KB)", _MODEL_PATH.stat().st_size // 1024)
    return _MODEL_PATH


class PoseExtractor:
    """Extracts COCO-17 keypoints from a webcam frame using MediaPipe Pose.

    Usage:
        ex = PoseExtractor()
        keypoints = ex.extract(bgr_frame)  # None if no person found
    """

    def __init__(self, model_complexity: int = 1) -> None:
        self._pose = None        # solutions API handle
        self._landmarker = None  # tasks API handle
        self._api: Optional[str] = None
        self._available = False

        # --- Try legacy solutions API (mediapipe < ~0.10.15) ---
        try:
            import mediapipe.solutions.pose as _mp_pose
            self._pose = _mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._api = "solutions"
            self._available = True
            logger.info("MediaPipe Pose (solutions API) initialised")
            return
        except (ImportError, AttributeError):
            pass  # fall through to Tasks API

        # --- Try Tasks API (mediapipe >= 0.10.15) ---
        try:
            from mediapipe.tasks import python as _mp_python
            from mediapipe.tasks.python import vision as _mp_vision

            model_path = _download_model()

            base_opts = _mp_python.BaseOptions(model_asset_path=str(model_path))
            opts = _mp_vision.PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=_mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = _mp_vision.PoseLandmarker.create_from_options(opts)
            self._api = "tasks"
            self._available = True
            logger.info("MediaPipe Pose (Tasks API) initialised")
        except ImportError:
            logger.warning("mediapipe not installed — pose extraction unavailable")
        except Exception as exc:
            logger.warning("MediaPipe Tasks API init error: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def extract(
        self, bgr_frame: np.ndarray
    ) -> Optional[List[Tuple[float, float, float]]]:
        """Return 17 COCO keypoints or None.

        Each keypoint is (x, y, visibility) normalised to [0,1].
        x is horizontal, y is vertical (top=0).
        """
        if not self._available:
            return None
        try:
            import cv2
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            if self._api == "solutions":
                return self._extract_solutions(rgb)
            else:
                return self._extract_tasks(rgb)
        except Exception as exc:
            logger.debug("extract error: %s", exc)
            return None

    def _extract_solutions(self, rgb: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        return [(float(lm[i].x), float(lm[i].y), float(lm[i].visibility))
                for i in _COCO17_FROM_MP]

    def _extract_tasks(self, rgb: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0]
        return [(float(lm[i].x), float(lm[i].y), float(lm[i].visibility))
                for i in _COCO17_FROM_MP]

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
