"""
Prompt-Based Workflow Mapping

Translates raw numerical telemetry into structured, natural-language insight
prompts that can be fed into an LLM for behavioural coaching summaries.

This layer bridges the gap between raw engagement metrics and human-readable
feedback.
"""

import logging
from typing import Dict, List, Optional

from src.config import logger


SEVERITY_LABELS = ["low", "moderate", "high", "severe"]


class PromptMapper:
    """
    Converts engagement telemetry into natural-language insights.

    Each insight is a short sentence describing a behavioural observation.
    These can be passed directly to an LLM for coaching-style summaries.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_insights(
        self,
        avg_ear: Optional[float] = None,
        min_ear: Optional[float] = None,
        blink_count: int = 0,
        blink_rate: Optional[float] = None,
        avg_mar: Optional[float] = None,
        yawn_count: int = 0,
        looking_away_count: int = 0,
        looking_away_ratio: float = 0.0,
        engagement_score: Optional[float] = None,
    ) -> List[str]:
        """
        Produce a list of human-readable insight strings.

        These insights can be passed to an LLM prompt like:

            "Based on the following observations, write a coaching summary: ..."
        """
        insights: List[str] = []

        # --- Engagement score ---
        if engagement_score is not None:
            if engagement_score >= 80:
                insights.append(
                    f"High engagement ({engagement_score:.0f}%) — the user was "
                    f"consistently focused on the screen."
                )
            elif engagement_score >= 50:
                insights.append(
                    f"Moderate engagement ({engagement_score:.0f}%) — some "
                    f"attention lapses detected."
                )
            else:
                insights.append(
                    f"Low engagement ({engagement_score:.0f}%) — significant "
                    f"attention loss observed."
                )

        # --- EAR / blink ---
        if avg_ear is not None and avg_ear < 0.18:
            insights.append(
                f"Low average Eye Aspect Ratio ({avg_ear:.3f}) suggests "
                f"possible drowsiness or partial eye closure."
            )
        if min_ear is not None and min_ear < 0.15:
            insights.append(
                f"Min EAR dropped to {min_ear:.3f}, indicating moments of "
                f"significant eye closure."
            )
        if blink_rate is not None:
            if blink_rate > 30:
                insights.append(
                    f"Elevated blink rate ({blink_rate:.1f} blinks/min) may "
                    f"indicate eye strain or fatigue."
                )
            elif blink_rate < 6:
                insights.append(
                    f"Low blink rate ({blink_rate:.1f} blinks/min) — the user "
                    f"may be hyper-focused."
                )
            else:
                insights.append(
                    f"Normal blink rate ({blink_rate:.1f} blinks/min)."
                )
        if blink_count > 0:
            insights.append(
                f"Total of {blink_count} blinks detected during the session."
            )

        # --- MAR / yawning ---
        if avg_mar is not None and avg_mar > 0.5:
            insights.append(
                f"Elevated Mouth Aspect Ratio ({avg_mar:.3f}) — possible "
                f"yawning or talking detected."
            )
        if yawn_count > 0:
            insights.append(
                f"{yawn_count} yawn(s) detected; this may correlate with "
                f"fatigue or boredom."
            )

        # --- Head pose / looking away ---
        if looking_away_count > 0:
            insights.append(
                f"User looked away from the screen {looking_away_count} time(s) "
                f"({looking_away_ratio * 100:.1f}% of frames)."
            )
        if looking_away_ratio > 0.2:
            insights.append(
                f"High looking-away ratio ({looking_away_ratio * 100:.1f}%) — "
                f"the user was frequently distracted or multi-tasking."
            )
        else:
            insights.append(
                f"User maintained good screen focus "
                f"({(1 - looking_away_ratio) * 100:.1f}% of frames)."
            )

        # --- Overall behavioural prompt (LLM-ready) ---
        insights.append(self._build_llm_prompt(
            avg_ear=avg_ear,
            blink_count=blink_count,
            blink_rate=blink_rate,
            yawn_count=yawn_count,
            looking_away_ratio=looking_away_ratio,
            engagement_score=engagement_score,
        ))

        return insights

    def _build_llm_prompt(
        self,
        avg_ear: Optional[float] = None,
        blink_count: int = 0,
        blink_rate: Optional[float] = None,
        yawn_count: int = 0,
        looking_away_ratio: float = 0.0,
        engagement_score: Optional[float] = None,
    ) -> str:
        """
        Build a structured prompt that can be sent to an LLM for a coaching
        summary.
        """
        lines = [
            "=== LLM BEHAVIOURAL COACHING PROMPT ===",
            "You are an expert behavioural coach. Based on the following "
            "real-time facial telemetry, write a concise coaching summary "
            "(2-3 paragraphs) for the user. Be constructive and specific.",
            "",
            "Telemetry Data:",
        ]
        if avg_ear is not None:
            lines.append(f"- Eye Aspect Ratio (EAR): {avg_ear:.3f}")
        lines.append(f"- Blinks detected: {blink_count}")
        if blink_rate is not None:
            lines.append(f"- Blink rate: {blink_rate:.1f} blinks/min")
        lines.append(f"- Yawns detected: {yawn_count}")
        lines.append(
            f"- Looking-away ratio: {looking_away_ratio * 100:.1f}%"
        )
        if engagement_score is not None:
            lines.append(f"- Engagement score: {engagement_score:.0f}%")
        lines.append("")
        lines.append(
            "Please provide actionable recommendations to improve engagement "
            "and reduce fatigue."
        )

        return "\n".join(lines)

    def to_dict(self, insights: List[str]) -> Dict[str, str]:
        """Group insights by category for structured output."""
        return {
            "engagement_summary": insights[0] if len(insights) > 0 else "",
            "eye_analysis": insights[1] if len(insights) > 1 else "",
            "mouth_analysis": insights[2] if len(insights) > 2 else "",
            "head_pose_analysis": insights[3] if len(insights) > 3 else "",
            "llm_prompt": insights[-1] if insights else "",
        }
