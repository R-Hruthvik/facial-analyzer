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
    Compute the Mouth Aspect Ratio based on inner mouth vertical opening
    normalized by mouth width to capture clean talking and yawning dynamics.
    """
    return calculate_mouth_opening_ratio(landmarks)


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
        window_size: int = 5,
    ) -> None:
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self._history = deque(maxlen=window_size)
        self._below_counter = 0
        self._above_counter = 0
        self._blink_in_progress = False
        self._blink_count = 0
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def blink_count(self) -> int:
        return self._blink_count

    @property
    def avg_ear(self) -> float:
        return float(np.mean(self._history)) if self._history else 0.0

    def update(self, ear: float) -> float:
        """Feed a new EAR value. Returns the smoothed EAR."""
        self._history.append(ear)
        smoothed = float(np.mean(self._history))

        # Check raw EAR against threshold
        if ear < self.threshold:
            self._below_counter += 1
            self._above_counter = 0
            # Only count blink if we had enough consecutive frames below threshold
            if self._below_counter >= self.consecutive_frames and not self._blink_in_progress:
                self._blink_count += 1
                self._blink_in_progress = True
                self._logger.debug("Blink detected (EAR=%.3f, frames=%d)", ear, self._below_counter)
        else:
            self._above_counter += 1
            # Reset counter and blink state when eye opens for at least 2 frames
            if self._above_counter >= 2:
                self._blink_in_progress = False
                self._below_counter = 0

        return smoothed

    def reset(self) -> None:
        self._history.clear()
        self._below_counter = 0
        self._above_counter = 0
        self._blink_in_progress = False
        self._blink_count = 0


class MouthAspectRatioTracker:
    """
    Tracks MAR and flags yawning/speaking events.
    """

    def __init__(
        self, threshold_yawn: float = settings.MAR_THRESHOLD, threshold_talk: float = 0.35, window_size: int = 5
    ) -> None:
        self.threshold_yawn = threshold_yawn
        self.threshold_talk = threshold_talk
        self._history = deque(maxlen=window_size)
        
        self._yawn_frames = 0
        self._talk_frames = 0
        self._yawn_count = 0
        self._talk_count = 0
        
        self._yawn_in_progress = False
        self._talk_in_progress = False
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def yawn_count(self) -> int:
        return self._yawn_count

    @property
    def talk_count(self) -> int:
        return self._talk_count

    @property
    def is_talking(self) -> bool:
        return self._talk_in_progress

    @property
    def avg_mar(self) -> float:
        return float(np.mean(self._history)) if self._history else 0.0

    def update(self, mar: float) -> float:
        self._history.append(mar)
        smoothed = float(np.mean(self._history))

        # Yawn logic
        if mar >= self.threshold_yawn:
            self._yawn_frames += 1
            if self._yawn_frames >= 15 and not self._yawn_in_progress:
                self._yawn_count += 1
                self._yawn_in_progress = True
                self._logger.debug("Yawn detected (MAR=%.3f)", mar)
        else:
            self._yawn_frames = 0
            if mar < self.threshold_talk:  # Fully closed mouth resets yawn cooldown
                self._yawn_in_progress = False

        # Talking logic
        if self.threshold_talk <= mar < self.threshold_yawn:
            self._talk_frames += 1
            if self._talk_frames >= 1 and not self._talk_in_progress:
                self._talk_count += 1
                self._talk_in_progress = True
        else:
            self._talk_frames = 0
            if mar < self.threshold_talk:
                self._talk_in_progress = False

        return smoothed

    def reset(self) -> None:
        self._history.clear()
        self._yawn_frames = 0
        self._talk_frames = 0
        self._yawn_count = 0
        self._talk_count = 0
        self._yawn_in_progress = False
        self._talk_in_progress = False


def calculate_gaze_distraction(landmarks: np.ndarray) -> bool:
    """
    Determine if user is looking away based on horizontal or vertical iris offset.
    """
    if len(landmarks) < 478:
        return False
        
    # Left eye gaze (outer: 33, inner: 133, iris: 468, top: 159, bottom: 145)
    left_center_x = (landmarks[33, 0] + landmarks[133, 0]) / 2.0
    left_width = abs(landmarks[33, 0] - landmarks[133, 0]) + 1e-6
    left_offset_x = (landmarks[468, 0] - left_center_x) / left_width
    
    left_center_y = (landmarks[159, 1] + landmarks[145, 1]) / 2.0
    left_height = abs(landmarks[159, 1] - landmarks[145, 1]) + 1e-6
    left_offset_y = (landmarks[468, 1] - left_center_y) / left_height
    
    # Right eye gaze (outer: 263, inner: 362, iris: 473, top: 386, bottom: 374)
    right_center_x = (landmarks[263, 0] + landmarks[362, 0]) / 2.0
    right_width = abs(landmarks[263, 0] - landmarks[362, 0]) + 1e-6
    right_offset_x = (landmarks[473, 0] - right_center_x) / right_width
    
    right_center_y = (landmarks[386, 1] + landmarks[374, 1]) / 2.0
    right_height = abs(landmarks[386, 1] - landmarks[374, 1]) + 1e-6
    right_offset_y = (landmarks[473, 1] - right_center_y) / right_height
    
    avg_offset_x = (abs(left_offset_x) + abs(right_offset_x)) / 2.0
    avg_offset_y = (abs(left_offset_y) + abs(right_offset_y)) / 2.0
    
    # Gaze thresholds: horizontal displacement > 0.18 or vertical displacement > 0.28
    return avg_offset_x > 0.18 or avg_offset_y > 0.28
