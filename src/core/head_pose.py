"""
Head Pose Estimation using OpenCV's solvePnP.

Estimates pitch, yaw, and roll from 2D–3D landmark correspondences.

References
----------
- OpenCV calibration tutorial (camera matrix approximation).
- MediaPipe canonical face model for 3D reference points.
"""

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.config import settings, logger
from src.core.face_mesh_engine import HEAD_POSE_LANDMARKS

# ---------------------------------------------------------------------------
# 3D reference points in an approximate canonical face model (metric units).
# These values come from the MediaPipe canonical face model.
# ---------------------------------------------------------------------------
_3D_REF_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],        # 1   — Nose tip
        [-0.225, -0.17, -0.12],  # 33  — Left eye outer corner
        [0.225, -0.17, -0.12],   # 263 — Right eye outer corner
        [-0.15, 0.08, -0.05],    # 61  — Left mouth corner
        [0.15, 0.08, -0.05],     # 291 — Right mouth corner
        [0.0, 0.23, -0.10],      # 199 — Chin
    ],
    dtype=np.float64,
)


class HeadPoseEstimator:
    """
    Estimates head pose (pitch, yaw, roll) from a set of 2D–3D correspondences.

    Uses the standard PnP formulation with an approximated camera matrix.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

        # Camera matrix — approximated for a 640×480 viewport
        # Will be re-calibrated on first call to ``estimate`` based on actual
        # frame dimensions.
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._logger.info("HeadPoseEstimator initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self, landmarks: np.ndarray, frame_w: int, frame_h: int
    ) -> Optional[Dict[str, float]]:
        """
        Compute pitch, yaw, roll (in degrees) from the 468-point landmark array.

        Parameters
        ----------
        landmarks : (468, 3) normalised landmark coordinates.
        frame_w, frame_h : Dimensions of the source frame (pixels).

        Returns
        -------
        dict with keys ``pitch``, ``yaw``, ``roll``, or ``None`` if PnP fails.
        """
        # Lazily build camera matrix
        if self._camera_matrix is None or self._camera_matrix.shape != (3, 3):
            self._build_camera_matrix(frame_w, frame_h)

        # Gather 2D pixel coordinates for the 6 reference landmarks
        img_pts = landmarks[HEAD_POSE_LANDMARKS, :2].copy()
        img_pts[:, 0] *= frame_w   # denormalise x
        img_pts[:, 1] *= frame_h   # denormalise y
        img_pts = img_pts.astype(np.float64)

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            _3D_REF_POINTS,
            img_pts,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            self._logger.warning("solvePnP failed to converge.")
            return None

        # Convert rotation vector to Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        angles = self._rotation_matrix_to_euler(rmat)

        pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
        
        # Adjust roll if it's centered around 180 (due to canonical model orientation)
        if roll > 90:
            roll -= 180
        elif roll < -90:
            roll += 180
            
        # Adjust pitch if it's centered around 180
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
            
        # Adjust yaw if it's centered around 180
        if yaw > 90:
            yaw -= 180
        elif yaw < -90:
            yaw += 180

        print(f"DEBUG: Head pose - Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}")

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_camera_matrix(self, w: int, h: int) -> None:
        """
        Approximate camera intrinsic matrix.

        Assumes focal length ≈ image width (a common approximation for
        typical webcams) and optical centre at image centre.
        """
        f = max(w, h)
        self._camera_matrix = np.array(
            [
                [f, 0, w / 2.0],
                [0, f, h / 2.0],
                [0, 0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_matrix_to_euler(rmat: np.ndarray) -> np.ndarray:
        """
        Convert a 3×3 rotation matrix to Euler angles (pitch, yaw, roll)
        using the XYZ convention.

        Returns
        -------
        np.array of (pitch, yaw, roll) in degrees.
        """
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0.0

        return np.degrees([x, y, z])

    def euler_angles_to_dict(self, pitch, yaw, roll) -> Dict[str, float]:
        """Convenience wrapper."""
        return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}
