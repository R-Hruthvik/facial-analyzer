"""
Facial metrics computation — Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR).

All functions operate on the normalised (x, y) landmark array from MediaPipe.
References
----------
- EAR : T. Soukupová and J. Čech, "Real-Time Eye Blink Detection using Facial
        Landmarks", CI2CV 2016.
- MAR : standard mouth-opening metric used in drowsiness detection.
"""

import logging
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe Face Mesh)
# ---------------------------------------------------------------------------
# Left eye  (indices from MediaPipe canonical model)
LEFT_EYE_IDX = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
# For EAR we only need the 6 points around the slit:
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]      # p1–p6

RIGHT_EYE_IDX = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]     # p1–p6

# Mouth landmarks
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13]
MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17]
MOUTH_MAR = [61, 39, 0, 267, 269, 291, 375, 321]   # simplified 6-point set

# ---------------------------------------------------------------------------
# Euclidean distance
# ---------------------------------------------------------------------------

def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1[:2] - p2[:2]))   # ignore z


# ---------------------------------------------------------------------------
# Eye Aspect Ratio
# ---------------------------------------------------------------------------

def calculate_ear(landmarks: np.ndarray, eye_indices: List[int]) -> float:
    """
    Compute the Eye Aspect Ratio for one eye.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Parameters
    ----------
    landmarks : (468, 3) array of normalised coordinates.
    eye_indices : 6-element list of landmark indices [p1..p6].

    Returns
    -------
    float — the EAR value.
    """
    pts = landmarks[eye_indices, :2]      # (6, 2)
    d1 = np.linalg.norm(pts[1] - pts[5])
    d2 = np.linalg.norm(pts[2] - pts[4])
    d3 = np.linalg.norm(pts[0] - pts[3])
    ear = (d1 + d2) / (2.0 * d3 + 1e-6)
    return float(ear)


def calculate_ear_both(landmarks: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute EAR for left eye, right eye, and their average.

    Returns (left_ear, right_ear, avg_ear).
    """
    left_ear = calculate_ear(landmarks, LEFT_EYE_EAR)
    right_ear = calculate_ear(landmarks, RIGHT_EYE_EAR)
    avg_ear = (left_ear + right_ear) / 2.0
    return left_ear, right_ear, avg_ear


# ---------------------------------------------------------------------------
# Mouth Aspect Ratio
# ---------------------------------------------------------------------------

def calculate_mar(landmarks: np.ndarray) -> float:
    """
    Compute the Mouth Aspect Ratio.

    MAR = (|p2-p8| + |p3-p7| + |p4-p6|) / (2 * |p1-p5|)

    Uses the outer-lip eight-point set defined in MOUTH_MAR.
    """
    pts = landmarks[MOUTH_MAR, :2]
    d1 = np.linalg.norm(pts[1] - pts[7])
    d2 = np.linalg.norm(pts[2] - pts[6])
    d3 = np.linalg.norm(pts[3] - pts[5])
    d4 = np.linalg.norm(pts[0] - pts[4])
    mar = (d1 + d2 + d3) / (2.0 * d4 + 1e-6)
    return float(mar)


def calculate_mouth_opening_ratio(landmarks: np.ndarray) -> float:
    """
    Alternative metric — vertical opening of the inner mouth
    normalised by mouth width.
    """
    # Upper inner lip (13) -> Lower inner lip (14)
    upper = landmarks[13, :2]
    lower = landmarks[14, :2]
    # Mouth width: left corner (61) -> right corner (291)
    left = landmarks[61, :2]
    right = landmarks[291, :2]

    height = np.linalg.norm(upper - lower)
    width = np.linalg.norm(left - right)
    return float(height / (width + 1e-6))


# ---------------------------------------------------------------------------
# Tracking classes with temporal smoothing
# ---------------------------------------------------------------------------

class EyeAspectRatioTracker:
    """
    Tracks EAR over a sliding window, counts blinks, and flags drowsiness.

    A blink is registered when the average EAR drops below ``threshold``
    for at least ``consecutive_frames`` frames.
    """

    def __init__(
        self,
        threshold: float = settings.EAR_THRESHOLD,
        consecutive_frames: int = settings.EAR_CONSECUTIVE_FRAMES,
        window_size: int = 30,
    ) -> None:
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self._history = deque(maxlen=window_size)
        self._below_counter = 0
        self._blink_count = 0
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def blink_count(self) -> int:
        return self._blink_count

    @property
    def avg_ear(self) -> float:
        return float(np.mean(self._history)) if self._history else 0.0

    def update(self, ear: float) -> float:
        """
        Feed a new EAR value. Returns the **smoothed** (averaged) EAR.
        """
        self._history.append(ear)
        smoothed = float(np.mean(self._history))

        if smoothed < self.threshold:
            self._below_counter += 1
        else:
            if self._below_counter >= self.consecutive_frames:
                self._blink_count += 1
                self._logger.debug("Blink detected (EAR=%.3f)", smoothed)
            self._below_counter = 0

        return smoothed

    def reset(self) -> None:
        self._history.clear()
        self._below_counter = 0
        self._blink_count = 0


class MouthAspectRatioTracker:
    """
    Tracks MAR and flags yawning/speaking events.
    """

    def __init__(
        self, threshold: float = settings.MAR_THRESHOLD, window_size: int = 15
    ) -> None:
        self.threshold = threshold
        self._history = deque(maxlen=window_size)
        self._open_duration_frames = 0
        self._yawn_count = 0
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def yawn_count(self) -> int:
        return self._yawn_count

    @property
    def avg_mar(self) -> float:
        return float(np.mean(self._history)) if self._history else 0.0

    def update(self, mar: float) -> float:
        self._history.append(mar)
        smoothed = float(np.mean(self._history))

        if smoothed > self.threshold:
            self._open_duration_frames += 1
            # Count a yawn if mouth stays open for >10 consecutive frames
            if self._open_duration_frames == 12:
                self._yawn_count += 1
                self._logger.debug("Yawn detected (MAR=%.3f)", smoothed)
        else:
            self._open_duration_frames = 0

        return smoothed

    def reset(self) -> None:
        self._history.clear()
        self._open_duration_frames = 0
        self._yawn_count = 0
