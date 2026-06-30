"""
Frame Processor — the central orchestration loop.

It ties together the camera/video source, FaceMeshEngine, metric trackers,
head-pose estimator, and engagement scorer.  Optimisations for CPU-bound
environments:
    - Frame skipping (process every N-th frame).
    - Resolution downscaling before inference.
    - Logging of per-frame latency.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Generator, List, Optional, Tuple

import cv2
import numpy as np

from src.config import settings, logger
from src.core.face_mesh_engine import FaceMeshEngine
from src.core.head_pose import HeadPoseEstimator
from src.core.metrics import (
    EyeAspectRatioTracker,
    MouthAspectRatioTracker,
    calculate_ear_both,
    calculate_mar,
)
from src.pipeline.engagement_scorer import EngagementScorer
from src.pipeline.prompt_mapper import PromptMapper


@dataclass
class FrameResult:
    """Snapshot of metrics computed for one processed frame."""
    timestamp: float
    avg_ear: float
    left_ear: float
    right_ear: float
    mar: float
    mouth_opening: float
    pitch: Optional[float]
    yaw: Optional[float]
    roll: Optional[float]
    is_looking_away: bool
    engagement_score: float
    blink_count: int
    yawn_count: int
    inference_ms: float
    total_latency_ms: float
    fps: float = 0.0
    frame_skipped: bool = False


@dataclass
class SessionSummary:
    """Final aggregation after processing stops."""
    total_frames: int
    processed_frames: int
    avg_inference_ms: float
    duration_seconds: float
    avg_ear: float
    min_ear: float
    blink_count: int
    blink_rate_per_min: float
    avg_mar: float
    yawn_count: int
    avg_pitch: Optional[float]
    avg_yaw: Optional[float]
    avg_roll: Optional[float]
    looking_away_count: int
    looking_away_ratio: float
    engagement_score: float
    insights: List[str] = field(default_factory=list)


class FrameProcessor:
    """
    High-level processor that reads frames from a source and yields
    ``FrameResult`` objects for the dashboard or API.

    Usage
    -----
        processor = FrameProcessor()
        for result in processor.run_video("path/to/video.mp4"):
            print(result.avg_ear, result.engagement_score)
        summary = processor.summarise()
    """

    def __init__(
        self,
        ear_threshold: float = settings.EAR_THRESHOLD,
        mar_threshold: float = settings.MAR_THRESHOLD,
        frame_skip: int = settings.FRAME_SKIP,
        resize_scale: float = settings.RESIZE_SCALE,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._frame_skip = frame_skip
        self._resize_scale = resize_scale

        # Core engines
        self._face_mesh = FaceMeshEngine()
        self._pose_estimator = HeadPoseEstimator()
        self._engagement_scorer = EngagementScorer()
        self._prompt_mapper = PromptMapper()

        # Metric trackers
        self._ear_tracker = EyeAspectRatioTracker(
            threshold=ear_threshold,
            consecutive_frames=settings.EAR_CONSECUTIVE_FRAMES,
        )
        self._mar_tracker = MouthAspectRatioTracker(threshold=mar_threshold)

        # Accumulated data for summary
        self._results: List[FrameResult] = []
        self._start_time: Optional[float] = None

        # FPS tracking and performance adaptation
        from collections import deque
        self._fps_timestamps = deque(maxlen=30)
        self._low_fps_counter = 0

        self._logger.info(
            "FrameProcessor ready (frame_skip=%s, resize_scale=%s)",
            frame_skip,
            resize_scale,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_camera(
        self, camera_id: int = settings.CAMERA_ID, on_frame: Callable = None
    ) -> Generator[FrameResult, None, None]:
        """
        Run the processing loop on a live camera feed.

        Parameters
        ----------
        camera_id : int — device index for ``cv2.VideoCapture``.
        on_frame : optional callback invoked with the annotated frame (BGR).

        Yields ``FrameResult`` for each **processed** frame (skipped frames
        yield a result with ``frame_skipped=True`` and no metrics).
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        self._start_time = time.time()
        try:
            yield from self._process_loop(cap, on_frame)
        finally:
            cap.release()

    def run_video(
        self, video_path: str, on_frame: Callable = None
    ) -> Generator[FrameResult, None, None]:
        """
        Run the processing loop on a pre-recorded video file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self._start_time = time.time()
        try:
            yield from self._process_loop(cap, on_frame)
        finally:
            cap.release()

    def summarise(self) -> SessionSummary:
        """Aggregate all processed results into a ``SessionSummary``."""
        if not self._results:
            return SessionSummary(
                total_frames=0,
                processed_frames=0,
                avg_inference_ms=0.0,
                duration_seconds=0.0,
                avg_ear=0.0,
                min_ear=0.0,
                blink_count=0,
                blink_rate_per_min=0.0,
                avg_mar=0.0,
                yawn_count=0,
                avg_pitch=None,
                avg_yaw=None,
                avg_roll=None,
                looking_away_count=0,
                looking_away_ratio=0.0,
                engagement_score=0.0,
            )

        processed = [r for r in self._results if not r.frame_skipped]
        if not processed:
            return SessionSummary(
                total_frames=len(self._results),
                processed_frames=0,
                avg_inference_ms=0.0,
                duration_seconds=0.0,
                avg_ear=0.0,
                min_ear=0.0,
                blink_count=0,
                blink_rate_per_min=0.0,
                avg_mar=0.0,
                yawn_count=0,
                avg_pitch=None,
                avg_yaw=None,
                avg_roll=None,
                looking_away_count=0,
                looking_away_ratio=0.0,
                engagement_score=0.0,
            )

        ears = [r.avg_ear for r in processed]
        mars = [r.mar for r in processed]
        pitches = [r.pitch for r in processed if r.pitch is not None]
        yaws = [r.yaw for r in processed if r.yaw is not None]
        rolls = [r.roll for r in processed if r.roll is not None]
        away_count = sum(1 for r in processed if r.is_looking_away)

        duration = (
            processed[-1].timestamp - processed[0].timestamp
            if self._start_time
            else len(processed) / 30.0
        )
        blink_rate = (
            self._ear_tracker.blink_count / (duration / 60.0)
            if duration > 0
            else 0.0
        )

        score = self._engagement_scorer.compute(
            avg_ear=float(np.mean(ears)),
            min_ear=float(np.min(ears)),
            avg_mar=float(np.mean(mars)),
            looking_away_ratio=away_count / max(len(processed), 1),
            blink_rate=blink_rate,
        )

        insights = self._prompt_mapper.generate_insights(
            avg_ear=float(np.mean(ears)),
            min_ear=float(np.min(ears)),
            blink_count=self._ear_tracker.blink_count,
            blink_rate=blink_rate,
            avg_mar=float(np.mean(mars)),
            yawn_count=self._mar_tracker.yawn_count,
            looking_away_count=away_count,
            looking_away_ratio=away_count / max(len(processed), 1),
            engagement_score=score,
        )

        return SessionSummary(
            total_frames=len(self._results),
            processed_frames=len(processed),
            avg_inference_ms=self._face_mesh.avg_inference_ms,
            duration_seconds=round(duration, 1),
            avg_ear=round(float(np.mean(ears)), 4),
            min_ear=round(float(np.min(ears)), 4),
            blink_count=self._ear_tracker.blink_count,
            blink_rate_per_min=round(blink_rate, 2),
            avg_mar=round(float(np.mean(mars)), 4),
            yawn_count=self._mar_tracker.yawn_count,
            avg_pitch=round(float(np.mean(pitches)), 2) if pitches else None,
            avg_yaw=round(float(np.mean(yaws)), 2) if yaws else None,
            avg_roll=round(float(np.mean(rolls)), 2) if rolls else None,
            looking_away_count=away_count,
            looking_away_ratio=round(away_count / max(len(processed), 1), 4),
            engagement_score=round(score, 1),
            insights=insights,
        )

    # ------------------------------------------------------------------
    # Internal processing loop
    # ------------------------------------------------------------------

    def _process_loop(
        self, cap: cv2.VideoCapture, on_frame: Callable = None
    ) -> Generator[FrameResult, None, None]:
        """Shared loop for both camera and video sources."""
        frame_idx = -1

        while True:
            t0 = time.perf_counter()
            ret, frame_bgr = cap.read()
            t1 = time.perf_counter()
            self._logger.warning("cap.read() took %.1f ms", (t1 - t0) * 1000)
            if not ret:
                break

            frame_idx += 1
            timestamp = time.time()

            # --- FPS Tracking ---
            self._fps_timestamps.append(timestamp)
            if len(self._fps_timestamps) > 1:
                elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
                current_fps = (len(self._fps_timestamps) - 1) / elapsed if elapsed > 0 else 30.0
            else:
                current_fps = 30.0

            # --- Frame skipping ---
            if frame_idx % (self._frame_skip + 1) != 0:
                yield FrameResult(
                    timestamp=timestamp,
                    avg_ear=0.0,
                    left_ear=0.0,
                    right_ear=0.0,
                    mar=0.0,
                    mouth_opening=0.0,
                    pitch=None,
                    yaw=None,
                    roll=None,
                    is_looking_away=False,
                    engagement_score=0.0,
                    blink_count=self._ear_tracker.blink_count,
                    yawn_count=self._mar_tracker.yawn_count,
                    inference_ms=0.0,
                    total_latency_ms=0.0,
                    fps=current_fps,
                    frame_skipped=True,
                )
                continue

            loop_start = time.perf_counter()

            # --- Resolution downscaling ---
            if self._resize_scale < 1.0:
                h, w = frame_bgr.shape[:2]
                new_w, new_h = (
                    int(w * self._resize_scale),
                    int(h * self._resize_scale),
                )
                frame_bgr = cv2.resize(
                    frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_bgr.shape[:2]

            # --- MediaPipe inference ---
            results = self._face_mesh.process(frame_rgb)

            result = FrameResult(
                timestamp=timestamp,
                avg_ear=0.0,
                left_ear=0.0,
                right_ear=0.0,
                mar=0.0,
                mouth_opening=0.0,
                pitch=None,
                yaw=None,
                roll=None,
                is_looking_away=False,
                engagement_score=0.0,
                blink_count=self._ear_tracker.blink_count,
                yawn_count=self._mar_tracker.yawn_count,
                inference_ms=self._face_mesh.avg_inference_ms,
                total_latency_ms=0.0,
                fps=current_fps,
            )

            landmarks = self._face_mesh.extract_landmarks(results)
            if landmarks is not None:
                # --- Compute metrics ---
                left_ear, right_ear, avg_ear = calculate_ear_both(landmarks)
                mar = calculate_mar(landmarks)
                result.left_ear = left_ear
                result.right_ear = right_ear
                result.avg_ear = self._ear_tracker.update(avg_ear)
                result.mar = self._mar_tracker.update(mar)
                result.blink_count = self._ear_tracker.blink_count
                result.yawn_count = self._mar_tracker.yawn_count

                # --- Head pose ---
                pose = self._pose_estimator.estimate(landmarks, w, h)
                if pose:
                    result.pitch = pose["pitch"]
                    result.yaw = pose["yaw"]
                    result.roll = pose["roll"]
                    result.is_looking_away = (
                        abs(result.pitch) > settings.HEAD_PITCH_THRESHOLD
                        or abs(result.yaw) > settings.HEAD_YAW_THRESHOLD
                        or abs(result.roll) > settings.HEAD_ROLL_THRESHOLD
                    )

                # --- Engagement score ---
                result.engagement_score = self._engagement_scorer.compute(
                    avg_ear=result.avg_ear,
                    min_ear=self._ear_tracker.avg_ear,
                    avg_mar=result.mar,
                    looking_away_ratio=(
                        sum(1 for r in self._results if r.is_looking_away)
                        / max(len(self._results), 1)
                    ),
                    blink_rate=(
                        self._ear_tracker.blink_count
                        / max((timestamp - (self._start_time or timestamp)), 1)
                        * 60.0
                    ),
                )
                
                # Log face detection and metrics
                self._logger.debug(
                    "Frame %d: EAR=%.3f, MAR=%.3f, Engagement=%.1f%%, "
                    "LookingAway=%s, Pitch=%.1f, Yaw=%.1f, Roll=%.1f",
                    frame_idx,
                    result.avg_ear,
                    result.mar,
                    result.engagement_score,
                    result.is_looking_away,
                    result.pitch or 0.0,
                    result.yaw or 0.0,
                    result.roll or 0.0,
                )

                # --- Draw mesh on frame ---
                FaceMeshEngine.draw_mesh(frame_bgr, landmarks)
                self._annotate_frame(frame_bgr, result)

            loop_elapsed = (time.perf_counter() - loop_start) * 1000.0
            result.total_latency_ms = round(loop_elapsed, 2)

            # Performance adaptation (CPU protection)
            if current_fps < 12.0:
                self._low_fps_counter += 1
                if self._low_fps_counter >= 60:  # ~2-5 seconds of sustained low FPS
                    adapted = False
                    if self._resize_scale > 0.5:
                        self._resize_scale = max(0.5, self._resize_scale - 0.15)
                        adapted = True
                    elif self._frame_skip < 3:
                        self._frame_skip += 1
                        adapted = True
                    if adapted:
                        self._logger.warning(
                            "Sustained low FPS (%.1f) detected. Adapting pipeline: "
                            "resize_scale=%.2f, frame_skip=%d to reduce CPU load.",
                            current_fps, self._resize_scale, self._frame_skip
                        )
                    self._low_fps_counter = 0
            else:
                self._low_fps_counter = max(0, self._low_fps_counter - 1)

            # Cap maximum frame rate at 30 FPS to prevent 100% CPU usage
            elapsed_sec = loop_elapsed / 1000.0
            target_period = 1.0 / 30.0
            if elapsed_sec < target_period:
                time.sleep(target_period - elapsed_sec)

            self._results.append(result)
            yield result

            # Optional callback for display
            if on_frame and callable(on_frame):
                on_frame(frame_bgr)

    # ------------------------------------------------------------------
    # Frame annotation
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_frame(frame: np.ndarray, result: FrameResult) -> None:
        """Overlay key metrics on the frame (in-place)."""
        h, w = frame.shape[:2]
        lines = [
            f"FPS: {result.fps:.1f}",
            f"EAR: {result.avg_ear:.3f}",
            f"MAR: {result.mar:.3f}",
            f"Blink: {result.blink_count}",
            f"Engagement: {result.engagement_score:.0f}%",
        ]
        if result.pitch is not None:
            lines.append(
                f"P/Y/R: {result.pitch:.1f}/{result.yaw:.1f}/{result.roll:.1f}"
            )
        if result.is_looking_away:
            lines.append("** LOOKING AWAY **")

        y0 = 30
        for i, text in enumerate(lines):
            y = y0 + i * 25
            color = (0, 0, 255) if "LOOKING AWAY" in text else (0, 255, 0)
            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
