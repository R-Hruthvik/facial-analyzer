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

import os
os.environ['MEDIAPIPE_DISABLE_TELEMETRY'] = '1'

import time
import logging
import uuid
import threading
import queue
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
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
# In-memory session store & Global ML Engine
# ---------------------------------------------------------------------------
# In production, replace with Redis / PostgreSQL.

_sessions: Dict[str, Dict] = {}
_start_time: float = time.time()
_shared_face_engine = None

_scorer = EngagementScorer()
_mapper = PromptMapper()

from src.core.head_pose import HeadPoseEstimator
_head_pose_estimator = HeadPoseEstimator()

_SESSION_TTL_SECONDS = 30 * 60  # 30 minutes

def _evict_stale_sessions():
    """Remove sessions inactive for longer than _SESSION_TTL_SECONDS."""
    now = time.time()
    stale = [sid for sid, s in _sessions.items()
             if now - s.get("last_active_at", now) > _SESSION_TTL_SECONDS]
    for sid in stale:
        del _sessions[sid]
    if stale:
        logger.info("Evicted %d stale session(s): %s", len(stale), stale)


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
    if payload.landmarks and len(payload.landmarks) in (468, 478):
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

    left_ear, right_ear, avg_ear = calculate_ear_both(landmarks)
    mar = calculate_mar(landmarks)
    mouth_opening = calculate_mouth_opening_ratio(landmarks)

    pose = _head_pose_estimator.estimate(landmarks, fw, fh)

    is_looking_away = False
    pitch = yaw = roll = None
    if pose:
        pitch, yaw, roll = pose["pitch"], pose["yaw"], pose["roll"]
        head_looking_away = (
            abs(pitch) > settings.HEAD_PITCH_THRESHOLD
            or abs(yaw) > settings.HEAD_YAW_THRESHOLD
            or abs(roll) > settings.HEAD_ROLL_THRESHOLD
        )
        from src.core.metrics import calculate_gaze_distraction
        eye_looking_away = calculate_gaze_distraction(landmarks)
        is_looking_away = head_looking_away or eye_looking_away

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
    # For live sessions, looking_away_count is accumulated in seconds.
    # Fallback to estimating seconds from metrics if not present.
    duration = (
        metrics[-1].timestamp - metrics[0].timestamp
        if len(metrics) > 1 and metrics[-1].timestamp > 0
        else 0.0
    )
    if duration <= 0:
        duration = len(metrics) / 30.0

    away_count = session.get("looking_away_count", 0)
    if away_count == 0 and len(metrics) > 0:
        # Fallback approximation: frames / 30
        away_count = int(sum(1 for m in metrics if m.is_looking_away) / 30.0)

    away_ratio = away_count / max(duration, 1.0)

    blink_rate = (
        session["blink_count"] / (duration / 60.0) if duration > 0 else 0.0
    )

    # Compute engagement score
    score = _scorer.compute(
        avg_ear=float(np.mean(ears)) if ears else 0.0,
        min_ear=float(np.min(ears)) if ears else 0.0,
        avg_mar=float(np.mean(mars)) if mars else 0.0,
        looking_away_ratio=away_ratio,
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
        looking_away_ratio=away_ratio,
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
        looking_away_ratio=round(away_ratio, 4),
        engagement_score=round(score, 1),
        insights=insights,
        started_at=session.get("started_at"),
        ended_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Background processing and WebSocket globals
# ---------------------------------------------------------------------------

active_websockets = set()
main_event_loop = None

async def broadcast_message(message: dict):
    if not active_websockets:
        return
    import json
    text = json.dumps(message)
    tasks = [ws.send_text(text) for ws in active_websockets]
    await asyncio.gather(*tasks, return_exceptions=True)

def _log_broadcast_errors(future):
    try:
        future.result()
    except Exception:
        logger.exception("broadcast_message failed")

def log_listener(entry):
    if not main_event_loop or not active_websockets:
        return
    payload = {
        "type": "log",
        "data": {
            "timestamp": entry.timestamp.strftime("%H:%M:%S"),
            "level": entry.level,
            "name": entry.name,
            "message": entry.message,
            "filename": entry.filename,
            "lineno": entry.lineno,
            "traceback": entry.traceback,
        }
    }
    asyncio.run_coroutine_threadsafe(
        broadcast_message(payload),
        main_event_loop
    )

# Global pre-warmed camera and engine to fix late launches
prewarmed_camera = None

@app.on_event("startup")
async def startup_event():
    global main_event_loop, prewarmed_camera
    main_event_loop = asyncio.get_running_loop()
    from src.dashboard.logs import get_log_manager
    log_manager = get_log_manager()
    log_manager.get_handler().add_listener(log_listener)
    
    logger.info("Pre-warming camera and ML models to prevent late launches...")
    import threading
    def _prewarm():
        global _shared_face_engine
        try:
            from src.core.face_mesh_engine import FaceMeshEngine
            _shared_face_engine = FaceMeshEngine()  # Loads the TF Lite model into memory and caches it
            logger.info("Pre-warm complete.")
        except Exception as e:
            logger.error("Failed to pre-warm: %s", e)
            
    threading.Thread(target=_prewarm, daemon=True).start()
    
    logger.info("FastAPI backend initialized, WebSocket logs hook active.")

class VideoSessionManager:
    def __init__(self):
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.session_id = None
        self.processor = None
        self.video_source = "camera"
        self.camera_id = 0
        self.video_path = None
        self.config = {}
        self._has_viewers = threading.Event()
        
    def start(self, video_source: str, source_value: str, config: dict):
        if self.running:
            self.stop()
            
        self.running = True
        self.video_source = video_source
        self.config = config
        self.session_id = str(uuid.uuid4())[:8]
        
        # Initialize session in global store
        _sessions[self.session_id] = {
            "id": self.session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_active_at": time.time(),
            "frames": [],
            "metrics": [],
            "blink_count": 0,
            "yawn_count": 0,
            "looking_away_count": 0,
        }
        
        if video_source == "camera":
            self.camera_id = int(source_value) if source_value.isdigit() else 0
            self.video_path = None
        else:
            self.video_path = source_value
            self.camera_id = None
            
        self.frame_queue = queue.Queue(maxsize=5)
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Session %s started. Source: %s (%s)", self.session_id, video_source, source_value)
        
    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        logger.info("Session %s stopped.", self.session_id)
        
    def _run_loop(self):
        import cv2
        from src.pipeline.frame_processor import FrameProcessor
        from src.api.schemas import PerFrameMetrics
        
        self.processor = FrameProcessor(
            ear_threshold=float(self.config.get("ear_threshold", settings.EAR_THRESHOLD)),
            mar_threshold=float(self.config.get("mar_threshold", settings.MAR_THRESHOLD)),
            frame_skip=int(self.config.get("frame_skip", settings.FRAME_SKIP)),
            resize_scale=float(self.config.get("resize_scale", settings.RESIZE_SCALE)),
            face_mesh_engine=_shared_face_engine,
        )
        
        def on_frame(frame_bgr):
            if not self._has_viewers.is_set():
                return
            ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                jpeg_bytes = buffer.tobytes()
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(jpeg_bytes)
                
        try:
            if self.video_source == "camera":
                generator = self.processor.run_camera(self.camera_id, on_frame=on_frame)
            else:
                generator = self.processor.run_video(self.video_path, on_frame=on_frame)
                
            for result in generator:
                if not self.running:
                    break
                    
                session = _sessions.get(self.session_id)
                if session:
                    session["last_active_at"] = time.time()
                    if not result.frame_skipped:
                        # Accumulate counters
                        session["blink_count"] = result.blink_count
                        session["yawn_count"] = result.yawn_count
                        session["looking_away_count"] = int(result.looking_away_seconds)
                        session["frames"].append(result.timestamp)
                        
                        metrics = PerFrameMetrics(
                            timestamp=result.timestamp,
                            left_ear=round(result.left_ear, 4),
                            right_ear=round(result.right_ear, 4),
                            avg_ear=round(result.avg_ear, 4),
                            mar=round(result.mar, 4),
                            mouth_opening=round(result.mouth_opening, 4),
                            pitch=round(result.pitch, 2) if result.pitch is not None else None,
                            yaw=round(result.yaw, 2) if result.yaw is not None else None,
                            roll=round(result.roll, 2) if result.roll is not None else None,
                            is_looking_away=result.is_looking_away,
                            is_distracted=result.is_distracted,
                            distraction_type=result.distraction_type,
                            inference_ms=round(result.inference_ms, 2),
                        )
                        session["metrics"].append(metrics)
                        
                    # Broadcast telemetry
                    payload = {
                        "type": "telemetry",
                        "data": {
                            "session_id": self.session_id,
                            "timestamp": result.timestamp,
                            "avg_ear": result.avg_ear,
                            "left_ear": result.left_ear,
                            "right_ear": result.right_ear,
                            "mar": result.mar,
                            "mouth_opening": result.mouth_opening,
                            "pitch": result.pitch,
                            "yaw": result.yaw,
                            "roll": result.roll,
                            "is_looking_away": result.is_looking_away,
                            "is_distracted": result.is_distracted,
                            "distraction_type": result.distraction_type,
                            "is_talking": result.is_talking,
                            "engagement_score": result.engagement_score,
                            "blink_count": session["blink_count"],
                            "yawn_count": session["yawn_count"],
                            "looking_away_count": session["looking_away_count"],
                            "fps": result.fps,
                            "frame_skipped": result.frame_skipped,
                            "pose_axes_2d": result.pose_axes_2d,
                        }
                    }
                    if main_event_loop:
                        fut = asyncio.run_coroutine_threadsafe(
                            broadcast_message(payload),
                            main_event_loop
                        )
                        fut.add_done_callback(_log_broadcast_errors)
            
            # End of stream indicator
            payload = {
                "type": "finished",
                "data": {
                    "session_id": self.session_id
                }
            }
            if main_event_loop:
                fut = asyncio.run_coroutine_threadsafe(
                    broadcast_message(payload),
                    main_event_loop
                )
                fut.add_done_callback(_log_broadcast_errors)
                
        except Exception as exc:
            logger.exception("Exception in OpenCV processing loop thread")
        finally:
            self.running = False
            if self.processor:
                try:
                    self.processor.close()
                except Exception:
                    pass
                self.processor = None

video_manager = VideoSessionManager()

# ---------------------------------------------------------------------------
# Frontend WebSocket & Streaming Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/session/verify-camera")
async def verify_camera():
    """Verify if the default webcam is physically available and not locked by another process."""
    import cv2
    try:
        camera_idx = int(settings.CAMERA_ID) if str(settings.CAMERA_ID).isdigit() else settings.CAMERA_ID
        import sys
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return {"status": "detected"}
    except Exception:
        pass
    return {"status": "unavailable"}

@app.websocket("/api/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    
    # Send history of logs to this client
    from src.dashboard.logs import get_log_manager
    log_manager = get_log_manager()
    recent = log_manager.get_recent_logs(limit=50)
    for entry in recent:
        payload = {
            "type": "log",
            "data": {
                "timestamp": entry.timestamp.strftime("%H:%M:%S"),
                "level": entry.level,
                "name": entry.name,
                "message": entry.message,
                "filename": entry.filename,
                "lineno": entry.lineno,
                "traceback": entry.traceback,
            }
        }
        await websocket.send_json(payload)
        
    try:
        while True:
            # Keep open
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        active_websockets.remove(websocket)

@app.get("/api/video-feed")
async def video_feed():
    async def frame_generator():
        loop = asyncio.get_event_loop()
        video_manager._has_viewers.set()
        try:
            while True:
                if not video_manager.running:
                    await asyncio.sleep(0.1)
                    continue
                try:
                    jpeg_bytes = await loop.run_in_executor(
                        None, video_manager.frame_queue.get, True, 0.033
                    )
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                except queue.Empty:
                    pass
        finally:
            video_manager._has_viewers.clear()
                
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

class SessionConfig(BaseModel):
    video_source: str
    camera_id: str = "0"
    video_filename: Optional[str] = None
    ear_threshold: float = 0.20
    mar_threshold: float = 0.60
    frame_skip: int = 0
    resize_scale: float = 1.0

@app.post("/api/session/start")
async def start_session(config: SessionConfig):
    _evict_stale_sessions()
    source_val = config.camera_id if config.video_source == "camera" else config.video_filename
    if config.video_source == "video_file":
        if not source_val:
            raise HTTPException(status_code=400, detail="Missing video_filename parameter.")
        from src.dashboard.utils import RESULTS_DIR
        video_path = RESULTS_DIR / source_val
        if not video_path.exists():
            raise HTTPException(status_code=400, detail=f"Uploaded video file '{source_val}' not found in results directory.")
        source_val = str(video_path)
        
    video_manager.start(
        video_source=config.video_source,
        source_value=source_val,
        config={
            "ear_threshold": config.ear_threshold,
            "mar_threshold": config.mar_threshold,
            "frame_skip": config.frame_skip,
            "resize_scale": config.resize_scale,
        }
    )
    return {
        "status": "started",
        "session_id": video_manager.session_id
    }

@app.post("/api/session/stop")
async def stop_session():
    _evict_stale_sessions()
    video_manager.stop()
    return {"status": "stopped", "session_id": video_manager.session_id}

@app.post("/api/session/reset")
async def reset_session():
    video_manager.stop()
    if video_manager.session_id in _sessions:
        del _sessions[video_manager.session_id]
    video_manager.session_id = None
    return {"status": "reset"}

@app.post("/api/session/upload")
async def upload_video(file: UploadFile = File(...)):
    from src.dashboard.utils import RESULTS_DIR
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = RESULTS_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    logger.info("Video file '%s' uploaded successfully to %s", file.filename, file_path)
    return {"filename": file.filename}

# Serve static dashboard client
import os
os.makedirs("src/static", exist_ok=True)
app.mount("/", StaticFiles(directory="src/static", html=True), name="static")
