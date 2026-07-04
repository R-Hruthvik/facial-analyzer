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
from collections import deque
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, List, Optional, Tuple

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
    calculate_gaze_distraction,
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
    is_talking: bool = False
    is_distracted: bool = False
    distraction_type: str = ""
    pose_axes_2d: Optional[Dict] = None
    looking_away_seconds: float = 0.0


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
        face_mesh_engine=None,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._frame_skip = frame_skip
        self._resize_scale = resize_scale

        self._owns_face_mesh = (face_mesh_engine is None)
        self._face_mesh = face_mesh_engine if face_mesh_engine else FaceMeshEngine()
        self._pose_estimator = HeadPoseEstimator()
        self._engagement_scorer = EngagementScorer()
        self._prompt_mapper = PromptMapper()

        self._ear_tracker = EyeAspectRatioTracker(
            threshold=ear_threshold,
            consecutive_frames=settings.EAR_CONSECUTIVE_FRAMES,
        )
        self._mar_tracker = MouthAspectRatioTracker(threshold_yawn=mar_threshold)

        self._results = deque(maxlen=30)

        self._sum_ear = 0.0
        self._sum_mar = 0.0
        self._min_ear = float('inf')
        self._sum_pitch = 0.0
        self._sum_yaw = 0.0
        self._sum_roll = 0.0
        self._count_pitch = 0
        self._count_ear = 0
        self._count_mar = 0
        self._start_time: Optional[float] = None
        self._last_pose: Optional[Tuple[float, float, float]] = None
        self._consecutive_looking_away_frames = 0

        self._calibrated = False
        self._calibration_frames = []
        self._pitch_offset = 0.0
        self._yaw_offset = 0.0
        self._roll_offset = 0.0

        self._looking_away_frames_total = 0
        self._processed_frames_total = 0
        self._looking_away_seconds = 0.0
        self._last_frame_time = None
        
        self._consecutive_looking_away_seconds = 0.0
        self._consecutive_closed_eyes_seconds = 0.0
        self._consecutive_no_face_seconds = 0.0

        self._fps_timestamps = deque(maxlen=30)
        self._low_fps_counter = 0

        self._logger.info(
            "FrameProcessor ready (frame_skip=%s, resize_scale=%s)",
            frame_skip,
            resize_scale,
        )

    def run_camera(
        self, camera_id: int = settings.CAMERA_ID, on_frame: Callable = None
    ) -> Generator[FrameResult, None, None]:
        """
        Run the processing loop on a live camera feed.
        """
        import sys
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
        n_ear = self._count_ear
        n_mar = self._count_mar
        n_pitch = self._count_pitch

        if n_ear == 0:
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

        avg_ear = self._sum_ear / n_ear
        min_ear = self._min_ear if self._min_ear != float('inf') else 0.0
        avg_mar = self._sum_mar / n_mar if n_mar else 0.0
        avg_pitch = self._sum_pitch / n_pitch if n_pitch else None
        avg_yaw = self._sum_yaw / n_pitch if n_pitch else None
        avg_roll = self._sum_roll / n_pitch if n_pitch else None

        away_seconds_int = int(self._looking_away_seconds)

        duration = time.time() - self._start_time if self._start_time else n_ear / 30.0

        away_ratio = self._looking_away_seconds / max(duration, 1.0)

        blink_rate = (
            self._ear_tracker.blink_count / (duration / 60.0)
            if duration > 0
            else 0.0
        )

        score = self._engagement_scorer.compute(
            avg_ear=avg_ear,
            min_ear=min_ear,
            avg_mar=avg_mar,
            looking_away_ratio=away_ratio,
            blink_rate=blink_rate,
        )

        insights = self._prompt_mapper.generate_insights(
            avg_ear=avg_ear,
            min_ear=min_ear,
            blink_count=self._ear_tracker.blink_count,
            blink_rate=blink_rate,
            avg_mar=avg_mar,
            yawn_count=self._mar_tracker.yawn_count,
            looking_away_count=away_seconds_int,
            looking_away_ratio=away_ratio,
            engagement_score=score,
        )

        return SessionSummary(
            total_frames=len(self._results),
            processed_frames=n_ear,
            avg_inference_ms=self._face_mesh.avg_inference_ms,
            duration_seconds=round(duration, 1),
            avg_ear=round(avg_ear, 4),
            min_ear=round(min_ear, 4),
            blink_count=self._ear_tracker.blink_count,
            blink_rate_per_min=round(blink_rate, 2),
            avg_mar=round(avg_mar, 4),
            yawn_count=self._mar_tracker.yawn_count,
            avg_pitch=round(avg_pitch, 2) if avg_pitch is not None else None,
            avg_yaw=round(avg_yaw, 2) if avg_yaw is not None else None,
            avg_roll=round(avg_roll, 2) if avg_roll is not None else None,
            looking_away_count=away_seconds_int,
            looking_away_ratio=round(away_ratio, 4),
            engagement_score=round(score, 1),
            insights=insights,
        )

    def close(self) -> None:
        """Close resources owned by the processor."""
        if getattr(self, "_owns_face_mesh", False) and self._face_mesh:
            try:
                self._face_mesh.close()
            except Exception:
                pass

    def _process_loop(
        self, cap: cv2.VideoCapture, on_frame: Callable = None
    ) -> Generator[FrameResult, None, None]:
        """Shared loop for both camera and video sources."""
        frame_idx = -1

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            timestamp = time.time()

            self._fps_timestamps.append(timestamp)
            if len(self._fps_timestamps) > 1:
                elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
                current_fps = (len(self._fps_timestamps) - 1) / elapsed if elapsed > 0 else 30.0
            else:
                current_fps = 30.0

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
                    is_talking=self._mar_tracker.is_talking,
                )
                continue

            loop_start = time.perf_counter()

            if self._resize_scale < 1.0:
                h, w = frame_bgr.shape[:2]
                new_w, new_h = (
                    int(w * self._resize_scale),
                    int(h * self._resize_scale),
                )
                interp = cv2.INTER_AREA if self._resize_scale <= 0.75 else cv2.INTER_LINEAR
                frame_bgr = cv2.resize(
                    frame_bgr, (new_w, new_h), interpolation=interp
                )

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_bgr.shape[:2]

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
                blink_count=self._ear_tracker.blink_count if self._calibrated else 0,
                yawn_count=self._mar_tracker.yawn_count if self._calibrated else 0,
                inference_ms=self._face_mesh.avg_inference_ms,
                total_latency_ms=0.0,
                fps=current_fps,
                is_talking=False,
                is_distracted=False,
                distraction_type="",
                looking_away_seconds=self._looking_away_seconds,
            )

            landmarks = self._face_mesh.extract_landmarks(results)
            if landmarks is not None:
                left_ear, right_ear, avg_ear = calculate_ear_both(landmarks)
                mar = calculate_mar(landmarks)
                result.left_ear = left_ear
                result.right_ear = right_ear
                if self._calibrated:
                    result.avg_ear = self._ear_tracker.update(avg_ear)
                    result.mar = self._mar_tracker.update(mar)
                    result.blink_count = self._ear_tracker.blink_count
                    result.yawn_count = self._mar_tracker.yawn_count
                    result.is_talking = self._mar_tracker.is_talking
                    self._sum_ear += result.avg_ear
                    self._count_ear += 1
                    self._sum_mar += result.mar
                    self._count_mar += 1
                    if result.avg_ear < self._min_ear:
                        self._min_ear = result.avg_ear
                else:
                    result.avg_ear = avg_ear
                    result.mar = mar
                    result.blink_count = 0
                    result.yawn_count = 0
                    result.is_talking = False

                pose = self._pose_estimator.estimate(landmarks, w, h)
                if pose:
                    result.pose_axes_2d = pose.get("pose_axes_2d")
                    raw_pitch = pose["pitch"]
                    raw_yaw = pose["yaw"]
                    raw_roll = pose["roll"]

                    if not self._calibrated:
                        self._calibration_frames.append((raw_pitch, raw_yaw, raw_roll))
                        if len(self._calibration_frames) >= 15:
                            import numpy as np
                            self._pitch_offset = float(np.mean([f[0] for f in self._calibration_frames]))
                            self._yaw_offset = float(np.mean([f[1] for f in self._calibration_frames]))
                            self._roll_offset = float(np.mean([f[2] for f in self._calibration_frames]))
                            self._calibrated = True
                            self._ear_tracker.reset()
                            self._mar_tracker.reset()
                            self._logger.info(
                                f"Webcam head-pose calibration complete: Pitch offset = {self._pitch_offset:.2f}°, "
                                f"Yaw offset = {self._yaw_offset:.2f}°, Roll offset = {self._roll_offset:.2f}°"
                            )

                    calibrated_pitch = raw_pitch - self._pitch_offset
                    calibrated_yaw = raw_yaw - self._yaw_offset
                    calibrated_roll = raw_roll - self._roll_offset

                    if abs(calibrated_pitch) > 50 or abs(calibrated_yaw) > 50 or abs(calibrated_roll) > 50:
                        if self._last_pose:
                            result.pitch, result.yaw, result.roll = self._last_pose
                        else:
                            result.pitch, result.yaw, result.roll = 0.0, 0.0, 0.0
                    else:
                        if self._last_pose:
                            lp, ly, lr = self._last_pose
                            result.pitch = 0.3 * calibrated_pitch + 0.7 * lp
                            result.yaw = 0.3 * calibrated_yaw + 0.7 * ly
                            result.roll = 0.3 * calibrated_roll + 0.7 * lr
                        else:
                            result.pitch = calibrated_pitch
                            result.yaw = calibrated_yaw
                            result.roll = calibrated_roll

                        self._last_pose = (result.pitch, result.yaw, result.roll)
                        self._sum_pitch += result.pitch
                        self._sum_yaw += result.yaw
                        self._sum_roll += result.roll
                        self._count_pitch += 1

                    head_looking_away = (
                        abs(result.pitch) > settings.HEAD_PITCH_THRESHOLD
                        or abs(result.yaw) > settings.HEAD_YAW_THRESHOLD
                    )
                    eye_looking_away = calculate_gaze_distraction(landmarks)
                    result.is_looking_away = bool(head_looking_away or eye_looking_away)

                current_time = timestamp
                dt = 0.0
                if self._last_frame_time is not None:
                    dt = current_time - self._last_frame_time
                    if result.is_looking_away:
                        self._looking_away_seconds += dt
                self._last_frame_time = current_time
                result.looking_away_seconds = self._looking_away_seconds

                self._consecutive_no_face_seconds = 0.0
                if result.is_looking_away:
                    self._consecutive_looking_away_seconds += dt
                else:
                    self._consecutive_looking_away_seconds = 0.0

                if self._calibrated and result.avg_ear < settings.EAR_THRESHOLD:
                    self._consecutive_closed_eyes_seconds += dt
                else:
                    self._consecutive_closed_eyes_seconds = 0.0

                if self._consecutive_closed_eyes_seconds > 2.0:
                    result.is_distracted = True
                    result.distraction_type = "drowsy"
                elif self._consecutive_looking_away_seconds > 2.0:
                    result.is_distracted = True
                    result.distraction_type = "distracted"
                else:
                    result.is_distracted = False
                    result.distraction_type = ""

                self._processed_frames_total += 1
                if result.is_looking_away:
                    self._looking_away_frames_total += 1

                result.engagement_score = self._engagement_scorer.compute(
                    avg_ear=result.avg_ear,
                    min_ear=self._ear_tracker.avg_ear,
                    avg_mar=result.mar,
                    looking_away_ratio=(
                        self._looking_away_frames_total / self._processed_frames_total
                    ),
                    blink_rate=(
                        self._ear_tracker.blink_count
                        / max((timestamp - (self._start_time or timestamp)), 1)
                        * 60.0
                    ),
                )
                
                p_val = result.pitch if result.pitch is not None else 0.0
                y_val = result.yaw if result.yaw is not None else 0.0
                r_val = result.roll if result.roll is not None else 0.0
                self._logger.debug(
                    "Pose: P=%.1f Y=%.1f R=%.1f | LookingAway=%s",
                    p_val, y_val, r_val, result.is_looking_away
                )

                FaceMeshEngine.draw_mesh(frame_bgr, landmarks)
            else:
                result.is_looking_away = True
                
                current_time = timestamp
                dt = 0.0
                if self._last_frame_time is not None:
                    dt = current_time - self._last_frame_time
                    self._looking_away_seconds += dt
                self._last_frame_time = current_time
                result.looking_away_seconds = self._looking_away_seconds

                self._consecutive_looking_away_seconds = 0.0
                self._consecutive_closed_eyes_seconds = 0.0
                self._consecutive_no_face_seconds += dt

                if self._consecutive_no_face_seconds > 2.0:
                    result.is_distracted = True
                    result.distraction_type = "absent"
                else:
                    result.is_distracted = False
                    result.distraction_type = ""

                self._processed_frames_total += 1
                if result.is_looking_away:
                    self._looking_away_frames_total += 1

                result.engagement_score = self._engagement_scorer.compute(
                    avg_ear=0.0,
                    min_ear=self._ear_tracker.avg_ear,
                    avg_mar=0.0,
                    looking_away_ratio=(
                        self._looking_away_frames_total / self._processed_frames_total
                    ),
                    blink_rate=(
                        self._ear_tracker.blink_count
                        / max((timestamp - (self._start_time or timestamp)), 1)
                        * 60.0
                    ),
                )

            loop_elapsed = (time.perf_counter() - loop_start) * 1000.0
            result.total_latency_ms = round(loop_elapsed, 2)
            self._results.append(result)
            yield result

            if on_frame and callable(on_frame):
                on_frame(frame_bgr)

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
