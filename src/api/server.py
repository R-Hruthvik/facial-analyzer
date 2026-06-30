"""
FastAPI application — REST endpoints for the Facial Engagement Analyzer.

Endpoints
---------
- GET  /health                   → Health check
- POST /api/process-frame        → Ingest a single frame's landmarks
- POST /api/process-batch        → Ingest a batch of frames
- GET  /api/telemetry/summary/{session_id} → Engagement summary for a session
- GET  /api/telemetry/live/{session_id}    → Latest metrics for live dashboard
"""

import time
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    EngagementSummary,
    FrameBatch,
    FrameData,
    HealthResponse,
    PerFrameMetrics,
)
from src.config import settings, logger
from src.pipeline.engagement_scorer import EngagementScorer
from src.pipeline.prompt_mapper import PromptMapper

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Facial Engagement Analyzer API",
    version="1.0.0",
    description="Real-time facial engagement telemetry via MediaPipe landmark analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
# In production, replace with Redis / PostgreSQL.

_sessions: Dict[str, Dict] = {}
_start_time: float = time.time()

_scorer = EngagementScorer()
_mapper = PromptMapper()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.post("/api/process-frame")
async def process_frame(payload: FrameData):
    """
    Receive a single frame's landmark data and update session metrics.
    """
    session_id = payload.session_id

    # Initialise session if new
    if session_id not in _sessions:
        _sessions[session_id] = {
            "id": session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "frames": [],
            "metrics": [],
            "blink_count": 0,
            "yawn_count": 0,
            "looking_away_count": 0,
        }

    session = _sessions[session_id]

    # Convert landmarks to numpy
    landmarks_arr = None
    if payload.landmarks and len(payload.landmarks) == 468:
        landmarks_arr = np.array(payload.landmarks, dtype=np.float32)

    # Compute per-frame metrics
    metrics = _compute_frame_metrics(
        landmarks_arr, payload.frame_width, payload.frame_height
    )
    metrics.timestamp = payload.timestamp
    session["metrics"].append(metrics)

    # Accumulate counters
    ear = metrics.avg_ear
    if ear is not None:
        if ear < settings.EAR_THRESHOLD:
            session["blink_count"] += 1   # simplified blink detection
    if metrics.is_looking_away:
        session["looking_away_count"] += 1

    session["frames"].append(payload.timestamp)

    return {"status": "accepted", "session_id": session_id, "frame_count": len(session["frames"])}


@app.post("/api/process-batch")
async def process_batch(payload: FrameBatch):
    """Process a batch of frames (off-line video upload scenario)."""
    results = []
    for frame in payload.frames:
        resp = await process_frame(frame)
        results.append(resp)
    return {"status": "completed", "frames_processed": len(results)}


@app.get(
    "/api/telemetry/summary/{session_id}",
    response_model=EngagementSummary,
)
async def telemetry_summary(session_id: str):
    """Return the aggregated engagement summary for a session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    metrics: List[PerFrameMetrics] = session["metrics"]
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics recorded.")

    summary = _aggregate_summary(session_id, session, metrics)
    return summary


@app.get("/api/telemetry/live/{session_id}")
async def telemetry_live(session_id: str):
    """
    Return the most recent metrics for the live dashboard (polled by Streamlit).
    """
    session = _sessions.get(session_id)
    if not session or not session["metrics"]:
        raise HTTPException(status_code=404, detail="No data yet.")

    recent = session["metrics"][-1]
    return {
        "session_id": session_id,
        "latest": recent.model_dump(),
        "frame_count": len(session["frames"]),
        "blink_count": session["blink_count"],
        "yawn_count": session["yawn_count"],
        "looking_away_count": session["looking_away_count"],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_frame_metrics(
    landmarks: Optional[np.ndarray], fw: int, fh: int
) -> PerFrameMetrics:
    """Derive EAR, MAR, and head-pose metrics from a landmark array."""
    if landmarks is None:
        return PerFrameMetrics(timestamp=0.0, is_looking_away=False)

    from src.core.metrics import calculate_ear_both, calculate_mar, calculate_mouth_opening_ratio
    from src.core.head_pose import HeadPoseEstimator

    left_ear, right_ear, avg_ear = calculate_ear_both(landmarks)
    mar = calculate_mar(landmarks)
    mouth_opening = calculate_mouth_opening_ratio(landmarks)

    pose_est = HeadPoseEstimator()
    pose = pose_est.estimate(landmarks, fw, fh)

    is_looking_away = False
    pitch = yaw = roll = None
    if pose:
        pitch, yaw, roll = pose["pitch"], pose["yaw"], pose["roll"]
        is_looking_away = (
            abs(pitch) > settings.HEAD_PITCH_THRESHOLD
            or abs(yaw) > settings.HEAD_YAW_THRESHOLD
            or abs(roll) > settings.HEAD_ROLL_THRESHOLD
        )

    return PerFrameMetrics(
        timestamp=0.0,
        left_ear=round(left_ear, 4),
        right_ear=round(right_ear, 4),
        avg_ear=round(avg_ear, 4),
        mar=round(mar, 4),
        mouth_opening=round(mouth_opening, 4),
        pitch=round(pitch, 2) if pitch is not None else None,
        yaw=round(yaw, 2) if yaw is not None else None,
        roll=round(roll, 2) if roll is not None else None,
        is_looking_away=is_looking_away,
    )


def _aggregate_summary(
    session_id: str,
    session: Dict,
    metrics: List[PerFrameMetrics],
) -> EngagementSummary:
    """Aggregate per-frame metrics into an EngagementSummary."""
    ears = [m.avg_ear for m in metrics if m.avg_ear is not None]
    mars = [m.mar for m in metrics if m.mar is not None]
    pitches = [m.pitch for m in metrics if m.pitch is not None]
    yaws = [m.yaw for m in metrics if m.yaw is not None]
    rolls = [m.roll for m in metrics if m.roll is not None]
    away_count = sum(1 for m in metrics if m.is_looking_away)

    duration = (
        metrics[-1].timestamp - metrics[0].timestamp
        if len(metrics) > 1 and metrics[-1].timestamp > 0
        else 0.0
    )
    # Estimate duration from frame count if timestamps are synthetic
    if duration <= 0:
        duration = len(metrics) / 30.0  # assume 30 fps

    blink_rate = (
        session["blink_count"] / (duration / 60.0) if duration > 0 else 0.0
    )

    # Compute engagement score
    score = _scorer.compute(
        avg_ear=float(np.mean(ears)) if ears else 0.0,
        min_ear=float(np.min(ears)) if ears else 0.0,
        avg_mar=float(np.mean(mars)) if mars else 0.0,
        looking_away_ratio=away_count / max(len(metrics), 1),
        blink_rate=blink_rate,
    )

    # Generate natural-language insights
    insights = _mapper.generate_insights(
        avg_ear=float(np.mean(ears)) if ears else None,
        min_ear=float(np.min(ears)) if ears else None,
        blink_count=session["blink_count"],
        blink_rate=blink_rate,
        avg_mar=float(np.mean(mars)) if mars else None,
        yawn_count=session["yawn_count"],
        looking_away_count=away_count,
        looking_away_ratio=away_count / max(len(metrics), 1),
        engagement_score=score,
    )

    return EngagementSummary(
        session_id=session_id,
        duration_seconds=round(duration, 1),
        total_frames=len(metrics),
        avg_ear=round(float(np.mean(ears)), 4) if ears else None,
        min_ear=round(float(np.min(ears)), 4) if ears else None,
        blink_count=session["blink_count"],
        blink_rate_per_min=round(blink_rate, 2),
        avg_mar=round(float(np.mean(mars)), 4) if mars else None,
        yawn_count=session["yawn_count"],
        avg_pitch=round(float(np.mean(pitches)), 2) if pitches else None,
        avg_yaw=round(float(np.mean(yaws)), 2) if yaws else None,
        avg_roll=round(float(np.mean(rolls)), 2) if rolls else None,
        looking_away_count=away_count,
        looking_away_ratio=round(away_count / max(len(metrics), 1), 4),
        engagement_score=round(score, 1),
        insights=insights,
        started_at=session.get("started_at"),
        ended_at=datetime.now(timezone.utc).isoformat(),
    )
