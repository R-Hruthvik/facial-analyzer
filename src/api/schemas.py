"""
Pydantic schemas for the REST API.

These models are used for request validation and response serialisation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Frame / Stream ingestion
# ---------------------------------------------------------------------------

class FrameData(BaseModel):
    """
    A single frame's worth of landmark data submitted to the processing
    endpoint.
    """
    session_id: str = Field(..., description="Unique session identifier.")
    timestamp: float = Field(
        ..., description="Unix timestamp (seconds) of this frame."
    )
    landmarks: List[List[float]] = Field(
        ...,
        description="Flattened landmark array of shape (468, 3) or None.",
    )
    frame_width: int = Field(640, ge=1)
    frame_height: int = Field(480, ge=1)


class FrameBatch(BaseModel):
    """Batch of frames for off-line video processing."""
    frames: List[FrameData]


# ---------------------------------------------------------------------------
# Metrics & Telemetry
# ---------------------------------------------------------------------------

class PerFrameMetrics(BaseModel):
    timestamp: float
    left_ear: Optional[float] = None
    right_ear: Optional[float] = None
    avg_ear: Optional[float] = None
    mar: Optional[float] = None
    mouth_opening: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    roll: Optional[float] = None
    is_looking_away: bool = False
    is_distracted: bool = False
    distraction_type: Optional[str] = ""
    inference_ms: Optional[float] = None


class EngagementSummary(BaseModel):
    """Aggregated telemetry for a complete session."""
    session_id: str
    duration_seconds: float
    total_frames: int
    avg_ear: Optional[float] = None
    min_ear: Optional[float] = None
    blink_count: int = 0
    blink_rate_per_min: Optional[float] = None
    avg_mar: Optional[float] = None
    yawn_count: int = 0
    avg_pitch: Optional[float] = None
    avg_yaw: Optional[float] = None
    avg_roll: Optional[float] = None
    looking_away_count: int = 0
    looking_away_ratio: float = 0.0
    engagement_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall engagement 0–100 %"
    )
    insights: List[str] = Field(
        default_factory=list,
        description="Natural-language insight messages.",
    )
    started_at: Optional[str] = None
    ended_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None
