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
        duration_seconds: Optional[float] = None,
        avg_pitch: Optional[float] = None,
        avg_yaw: Optional[float] = None,
        total_frames: Optional[int] = None,
    ) -> List[str]:
        """
        Produce a list of human-readable insight strings followed by a
        rich LLM coaching prompt as the final element.
        """
        insights: List[str] = []

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
                    f"may be hyper-focused or experiencing screen-induced dryness."
                )
            else:
                insights.append(
                    f"Normal blink rate ({blink_rate:.1f} blinks/min)."
                )
        if blink_count > 0:
            insights.append(
                f"Total of {blink_count} blinks detected during the session."
            )

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

        insights.append(self._build_llm_prompt(
            avg_ear=avg_ear,
            min_ear=min_ear,
            blink_count=blink_count,
            blink_rate=blink_rate,
            avg_mar=avg_mar,
            yawn_count=yawn_count,
            looking_away_count=looking_away_count,
            looking_away_ratio=looking_away_ratio,
            engagement_score=engagement_score,
            duration_seconds=duration_seconds,
            avg_pitch=avg_pitch,
            avg_yaw=avg_yaw,
            total_frames=total_frames,
        ))

        return insights

    def _classify_ear(self, ear: float) -> str:
        if ear >= 0.30:
            return "normal (eyes fully open)"
        elif ear >= 0.22:
            return "slightly low (mild fatigue possible)"
        elif ear >= 0.15:
            return "low (significant drowsiness risk)"
        else:
            return "very low (severe eye closure detected)"

    def _classify_blink_rate(self, rate: float) -> str:
        if rate > 30:
            return "elevated — possible eye strain or irritation"
        elif rate > 20:
            return "slightly high — mild fatigue"
        elif rate >= 12:
            return "normal range"
        elif rate >= 6:
            return "slightly low — screen-focused state"
        else:
            return "very low — possible hyper-focus or dry eyes risk"

    def _classify_engagement(self, score: float) -> str:
        if score >= 85:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "moderate — improvement possible"
        else:
            return "poor — significant distraction present"

    def _build_llm_prompt(
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
        duration_seconds: Optional[float] = None,
        avg_pitch: Optional[float] = None,
        avg_yaw: Optional[float] = None,
        total_frames: Optional[int] = None,
    ) -> str:
        """
        Build a rich, structured prompt for an LLM coaching summary.

        The prompt provides:
          - Role and output format instructions
          - Interpreted (not just raw) telemetry with context labels
          - Clear section structure for the LLM to follow
          - Specific coaching constraints (tone, length, actionability)
        """

        session_lines = [
            "Hey! I need your help analyzing my recent screen-based work session.",
            "I've been tracking my facial engagement metrics and I'd love your expert advice, tips, and detailed feedback.",
            "",
            "Here's the data from my session:"
        ]
        
        if duration_seconds is not None:
            mins = int(duration_seconds // 60)
            secs = int(duration_seconds % 60)
            duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            session_lines.append(f"- The session lasted for {duration_str}.")
        if total_frames is not None:
            session_lines.append(f"- A total of {total_frames} frames were analyzed.")

        eye_lines = ["\nEye & Attention Metrics:"]
        if avg_ear is not None:
            eye_lines.append(f"- My average Eye Aspect Ratio (EAR) was {avg_ear:.3f}, which suggests {self._classify_ear(avg_ear).lower()}.")
        if min_ear is not None:
            eye_lines.append(f"- The lowest EAR (my deepest blink or eye closure) was {min_ear:.3f} ({'within normal range' if min_ear >= 0.15 else 'significant closure detected'}).")
        if blink_rate is not None:
            eye_lines.append(f"- My blink rate was {blink_rate:.1f} blinks per minute, meaning {self._classify_blink_rate(blink_rate).lower()}.")
        eye_lines.append(f"- I blinked a total of {blink_count} times.")

        fatigue_lines = ["\nFatigue & Alertness Signals:"]
        if avg_mar is not None:
            mar_label = "elevated (possible yawning or talking)" if avg_mar > 0.5 else "normal (mouth generally closed)"
            fatigue_lines.append(f"- My average Mouth Aspect Ratio (MAR) was {avg_mar:.3f}, which is {mar_label}.")
        fatigue_lines.append(f"- Yawn count: {yawn_count} {'(fatigue signal present)' if yawn_count > 0 else '(none — good alertness)'}.")

        pose_lines = ["\nHead Pose & Distraction:"]
        if avg_pitch is not None:
            pitch_label = "forward lean or nodding" if avg_pitch < -10 else "upright and alert" if abs(avg_pitch) <= 10 else "head tilted back"
            pose_lines.append(f"- Average head pitch: {avg_pitch:.1f}° ({pitch_label}).")
        if avg_yaw is not None:
            yaw_label = "facing left" if avg_yaw < -10 else "centred and forward" if abs(avg_yaw) <= 10 else "facing right"
            pose_lines.append(f"- Average head yaw: {avg_yaw:.1f}° ({yaw_label}).")
        pose_lines.append(f"- I had {looking_away_count} moments where I was looking away from the screen.")
        pose_lines.append(
            f"- Overall, I was distracted {looking_away_ratio * 100:.1f}% of the time, "
            f"which is {'concerning' if looking_away_ratio > 0.2 else 'acceptable'}."
        )

        engagement_lines = ["\nOverall Engagement:"]
        if engagement_score is not None:
            engagement_lines.append(
                f"- My composite engagement score was calculated at {engagement_score:.1f}%, "
                f"indicating {self._classify_engagement(engagement_score).lower()}."
            )

        lines = []
        lines += session_lines
        lines += eye_lines
        lines += fatigue_lines
        lines += pose_lines
        lines += engagement_lines
        
        lines.append("")
        lines.append(
            "Based on this data, could you give me a really detailed, structured coaching summary? "
            "Please act as an expert behavioural and cognitive performance coach. I'd love it if you could:\n\n"
            "1. Give me a detailed performance summary (what these metrics collectively say about my session, going beyond just repeating the numbers).\n"
            "2. Highlight 3 to 4 key observations (point out the most significant positive or negative findings and what they mean for my real-world productivity).\n"
            "3. Provide me with 3 to 4 highly actionable, specific coaching recommendations or tips I can use to improve my focus and alertness before my next session.\n\n"
            "Please be honest, encouraging, and write directly to me. Don't hold back on the details!"
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