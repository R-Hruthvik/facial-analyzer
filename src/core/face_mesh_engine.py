"""
Core Face Mesh Engine — wraps MediaPipe Face Landmarker for real-time 478-landmark tracking.

Uses the newer ``mediapipe.tasks`` API (FaceLandmarker) which replaces the
deprecated ``mp.solutions.face_mesh`` in MediaPipe 0.10.x.

Optimisations for CPU-constrained environments:
    - Frame skipping (configurable via FRAME_SKIP).
    - Resolution downscaling before inference (RESIZE_SCALE).
    - Early exit when no face is detected.
"""

import os
import time
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import settings, logger

# ---------------------------------------------------------------------------
# Model file
# ---------------------------------------------------------------------------
_MODEL_FILENAME = "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ---------------------------------------------------------------------------
# Eye contour connections — extracted from FaceLandmarksConnections
# ---------------------------------------------------------------------------
FACEMESH_LEFT_EYE_INDICES = [
    (c.start, c.end)
    for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
]
FACEMESH_RIGHT_EYE_INDICES = [
    (c.start, c.end)
    for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
]

# Indices used for head-pose estimation (6 standard points).
# First 468 landmarks are identical between the old 468-point and new 478-point models.
HEAD_POSE_LANDMARKS = [
    1,     # nose tip
    4,     # nose bridge
    33,    # left eye outer corner
    263,   # right eye outer corner
    133,   # left eye inner corner
    362,   # right eye inner corner
]

# Number of landmarks in the new model
NUM_LANDMARKS = 478


def _ensure_model(model_path: str) -> str:
    """Download the FaceLandmarker model if it does not exist."""
    if os.path.isfile(model_path):
        return model_path
    logger.info("Downloading FaceLandmarker model from %s …", _MODEL_URL)
    import urllib.request

    urllib.request.urlretrieve(_MODEL_URL, model_path)
    logger.info("Model saved to %s", model_path)
    return model_path


class FaceMeshEngine:
    """
    Lightweight wrapper around MediaPipe Face Landmarker (Tasks API).

    Usage:
        engine = FaceMeshEngine()
        result = engine.process(frame_rgb)
        landmarks = engine.extract_landmarks(result, 0)
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

        # Resolve model path — default to project root or CWD
        if model_path is None:
            # Search locations in order
            for candidate in (
                os.path.join(str(settings.BASE_DIR), _MODEL_FILENAME),
                os.path.join(os.getcwd(), _MODEL_FILENAME),
            ):
                if os.path.isfile(candidate):
                    model_path = candidate
                    break
            else:
                # Fall back to project root and download
                model_path = os.path.join(str(settings.BASE_DIR), _MODEL_FILENAME)

        self._model_path = _ensure_model(model_path)

        # Build FaceLandmarker options
        base_options = python.BaseOptions(
            model_asset_path=self._model_path,
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=settings.FACEMESH_MAX_NUM_FACES,
            min_face_detection_confidence=settings.FACEMESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.FACEMESH_MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        # Performance bookkeeping
        self._frame_count: int = 0
        self._inference_times: list[float] = []
        self._avg_inference_ms: float = 0.0

        self._logger.info(
            "FaceMeshEngine initialised (max_faces=%s, model=%s)",
            settings.FACEMESH_MAX_NUM_FACES,
            self._model_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame_rgb: np.ndarray) -> Optional[vision.FaceLandmarkerResult]:
        """
        Run MediaPipe inference on an RGB frame.

        Returns
        -------
        ``FaceLandmarkerResult`` with ``face_landmarks``, or ``None``.
        """
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        t0 = time.perf_counter()
        result = self._landmarker.detect(mp_image)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._frame_count += 1
        self._inference_times.append(elapsed_ms)

        # Keep rolling window of ~100 samples for stats
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self._avg_inference_ms = np.mean(self._inference_times).item()

        return result

    @staticmethod
    def extract_landmarks(
        result, face_idx: int = 0
    ) -> Optional[np.ndarray]:
        """
        Convert FaceLandmarker result to a (NUM_LANDMARKS, 3) float32 numpy array
        of normalised (x, y, z) coordinates.

        Parameters
        ----------
        result : ``FaceLandmarkerResult`` (or compatible object with ``face_landmarks``).
        face_idx : Index of the face to extract (default 0).

        Returns
        -------
        np.ndarray of shape (NUM_LANDMARKS, 3) or None if no face detected.
        """
        if not result or not result.face_landmarks:
            return None
        if face_idx >= len(result.face_landmarks):
            return None

        landmarks = result.face_landmarks[face_idx]
        arr = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            arr[i] = [lm.x, lm.y, lm.z]
        return arr

    @staticmethod
    def denormalise_landmarks(
        landmarks: np.ndarray, frame_w: int, frame_h: int
    ) -> np.ndarray:
        """Convert normalised landmark coordinates to pixel space."""
        denorm = landmarks.copy()
        denorm[:, 0] *= frame_w
        denorm[:, 1] *= frame_h
        # z remains in MediaPipe's metric scale (approx mm)
        return denorm

    @staticmethod
    def draw_mesh(
        frame: np.ndarray,
        landmarks: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        point_radius: int = 1,
    ) -> None:
        """Draw face-mesh landmarks onto the frame **in-place**."""
        h, w = frame.shape[:2]
        for x_norm, y_norm, _ in landmarks:
            x, y = int(x_norm * w), int(y_norm * h)
            cv2.circle(frame, (x, y), point_radius, color, -1)

    @staticmethod
    def draw_connections(
        frame: np.ndarray,
        landmarks: np.ndarray,
        connections: list,
        color: Tuple[int, int, int] = (0, 165, 255),
        thickness: int = 1,
    ) -> None:
        """Draw connection lines specified by ``connections`` (list of (start, end) tuples)."""
        h, w = frame.shape[:2]
        for start_idx, end_idx in connections:
            sx, sy = int(landmarks[start_idx, 0] * w), int(
                landmarks[start_idx, 1] * h
            )
            ex, ey = int(landmarks[end_idx, 0] * w), int(
                landmarks[end_idx, 1] * h
            )
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness)

    # ------------------------------------------------------------------
    # Performance helpers
    # ------------------------------------------------------------------

    @property
    def avg_inference_ms(self) -> float:
        return self._avg_inference_ms

    @property
    def frames_processed(self) -> int:
        return self._frame_count

    def reset_stats(self) -> None:
        self._frame_count = 0
        self._inference_times.clear()
        self._avg_inference_ms = 0.0

    def close(self) -> None:
        self._landmarker.close()
        self._logger.info("FaceMeshEngine closed.")
