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
        # Extended params — pass from session summary if available
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

        # --- LLM coaching prompt (always last element) ---
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

        # --- Session context block ---
        session_lines = ["SESSION CONTEXT:"]
        if duration_seconds is not None:
            mins = int(duration_seconds // 60)
            secs = int(duration_seconds % 60)
            duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            session_lines.append(f"  - Session duration : {duration_str}")
        if total_frames is not None:
            session_lines.append(f"  - Frames analysed  : {total_frames}")

        # --- Eye & attention block ---
        eye_lines = ["EYE & ATTENTION METRICS:"]
        if avg_ear is not None:
            eye_lines.append(
                f"  - Avg Eye Aspect Ratio (EAR) : {avg_ear:.3f}  "
                f"→ {self._classify_ear(avg_ear)}"
            )
        if min_ear is not None:
            eye_lines.append(
                f"  - Min EAR (lowest blink)     : {min_ear:.3f}  "
                f"({'within normal range' if min_ear >= 0.15 else 'significant closure detected'})"
            )
        if blink_rate is not None:
            eye_lines.append(
                f"  - Blink rate                 : {blink_rate:.1f} blinks/min  "
                f"→ {self._classify_blink_rate(blink_rate)}"
            )
        eye_lines.append(f"  - Total blinks               : {blink_count}")

        # --- Fatigue signals block ---
        fatigue_lines = ["FATIGUE & ALERTNESS SIGNALS:"]
        if avg_mar is not None:
            mar_label = (
                "elevated — possible yawning or talking" if avg_mar > 0.5
                else "normal — mouth generally closed"
            )
            fatigue_lines.append(
                f"  - Avg Mouth Aspect Ratio (MAR): {avg_mar:.3f}  → {mar_label}"
            )
        fatigue_lines.append(
            f"  - Yawns detected              : {yawn_count}  "
            f"{'(fatigue signal present)' if yawn_count > 0 else '(none — good alertness)'}"
        )

        # --- Head pose & distraction block ---
        pose_lines = ["HEAD POSE & DISTRACTION:"]
        if avg_pitch is not None:
            pitch_label = (
                "forward lean / nodding" if avg_pitch < -10
                else "upright / alert" if abs(avg_pitch) <= 10
                else "head tilted back"
            )
            pose_lines.append(
                f"  - Avg Pitch                  : {avg_pitch:.1f}°  → {pitch_label}"
            )
        if avg_yaw is not None:
            yaw_label = (
                "facing left" if avg_yaw < -10
                else "centred / forward" if abs(avg_yaw) <= 10
                else "facing right"
            )
            pose_lines.append(
                f"  - Avg Yaw                    : {avg_yaw:.1f}°  → {yaw_label}"
            )
        pose_lines.append(
            f"  - Looking-away events         : {looking_away_count}"
        )
        pose_lines.append(
            f"  - % of time distracted        : {looking_away_ratio * 100:.1f}%  "
            f"→ {'concerning' if looking_away_ratio > 0.2 else 'acceptable'}"
        )

        # --- Engagement summary block ---
        engagement_lines = ["ENGAGEMENT SUMMARY:"]
        if engagement_score is not None:
            engagement_lines.append(
                f"  - Composite engagement score : {engagement_score:.1f}%  "
                f"→ {self._classify_engagement(engagement_score)}"
            )

        # --- Assemble full prompt ---
        lines = [
            "╔══════════════════════════════════════════╗",
            "║      LLM BEHAVIOURAL COACHING PROMPT     ║",
            "╚══════════════════════════════════════════╝",
            "",
            "ROLE:",
            "You are an expert behavioural and cognitive performance coach "
            "specialising in screen-based work sessions. You interpret facial "
            "engagement telemetry and provide constructive, evidence-based feedback.",
            "",
            "OUTPUT FORMAT:",
            "Write exactly 3 paragraphs:",
            "  1. PERFORMANCE SUMMARY  — What the metrics collectively say about "
            "this session. Mention the engagement score and key signals. Avoid "
            "simply repeating numbers; interpret what they mean behaviourally.",
            "  2. KEY OBSERVATIONS     — Highlight the 2-3 most significant "
            "findings (positive or negative). Be specific: name the metric and "
            "explain its real-world implication for the user.",
            "  3. ACTIONABLE COACHING  — Give 2-3 specific, practical "
            "recommendations the user can act on before their next session. "
            "Tie each recommendation directly to an observed metric.",
            "",
            "TONE: Encouraging but honest. Do not sugarcoat poor metrics. "
            "Do not be clinical or robotic. Write as if speaking to the user directly.",
            "",
        ]

        lines += session_lines + [""]
        lines += eye_lines + [""]
        lines += fatigue_lines + [""]
        lines += pose_lines + [""]
        lines += engagement_lines + [""]

        lines.append(
            "Now write the 3-paragraph coaching summary following the OUTPUT FORMAT above."
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