"""
Engagement Scorer

Combines EAR, MAR, head-pose, and blink-rate metrics into a single
0–100 % engagement score.

The formula is a weighted combination:
    - Eye engagement (EAR)         : 40 %
    - Head pose / focus            : 30 %
    - Blink rate normalisation     : 15 %
    - Mouth state (MAR)            : 15 %
"""

import logging
from typing import Optional

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class EngagementScorer:
    """
    Computes a weighted engagement score from facial telemetry.

    The score is designed so that:
    - 90–100 % → Fully engaged
    - 70–89 %  → Mildly distracted
    - 50–69 %  → Distracted
    - < 50 %   → Disengaged / fatigued
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    def compute(
        self,
        avg_ear: float = 0.3,
        min_ear: float = 0.25,
        avg_mar: float = 0.2,
        looking_away_ratio: float = 0.0,
        blink_rate: float = 15.0,
    ) -> float:
        """
        Calculate the engagement score.

        Parameters
        ----------
        avg_ear : Mean Eye Aspect Ratio across the window.
        min_ear : Minimum EAR observed (captures eye-closure events).
        avg_mar : Mean Mouth Aspect Ratio.
        looking_away_ratio : Fraction of frames where the user looked away.
        blink_rate : Blinks per minute.

        Returns
        -------
        float between 0 and 100.
        """
        ear_score = np.clip(avg_ear / 0.30, 0.0, 1.0) * 40.0

        focus_score = (1.0 - looking_away_ratio) * 30.0

        if blink_rate < 6:
            blink_score = (blink_rate / 6.0) * 15.0
        elif blink_rate > 30:
            blink_score = max(0.0, 1.0 - (blink_rate - 30) / 30.0) * 15.0
        else:
            blink_score = 15.0

        mar_score = np.clip(1.0 - (avg_mar / 0.7), 0.0, 1.0) * 15.0

        total = ear_score + focus_score + blink_score + mar_score
        return round(max(0.0, min(100.0, total)), 2)

    def score_to_label(self, score: float) -> str:
        """Map a numerical score to a qualitative label."""
        if score >= 90:
            return "Fully Engaged"
        elif score >= 70:
            return "Engaged"
        elif score >= 50:
            return "Mildly Distracted"
        elif score >= 30:
            return "Distracted"
        else:
            return "Disengaged / Fatigued"
