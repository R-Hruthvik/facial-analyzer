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

_3D_REF_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -0.08, -0.06],
        [-0.225, -0.17, -0.12],
        [0.225, -0.17, -0.12],
        [-0.08, -0.17, -0.09],
        [0.08, -0.17, -0.09],
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

        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._logger.info("HeadPoseEstimator initialised.")

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
        if self._camera_matrix is None or self._camera_matrix.shape != (3, 3):
            self._build_camera_matrix(frame_w, frame_h)

        img_pts = landmarks[HEAD_POSE_LANDMARKS, :2].copy()
        img_pts[:, 0] *= frame_w
        img_pts[:, 1] *= frame_h
        img_pts = img_pts.astype(np.float64)

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

        rmat, _ = cv2.Rodrigues(rvec)
        angles = self._rotation_matrix_to_euler(rmat)

        pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])

        if roll > 90:
            roll -= 180
        elif roll < -90:
            roll += 180

        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

        if yaw > 90:
            yaw -= 180
        elif yaw < -90:
            yaw += 180

        logger.debug(f"Head pose - Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}")

        axis_len = 0.15
        axis_points = np.array([
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ], dtype=np.float64)

        imgpts, _ = cv2.projectPoints(
            axis_points,
            rvec,
            tvec,
            self._camera_matrix,
            self._dist_coeffs
        )
        imgpts = imgpts.reshape(-1, 2)
        pose_axes_2d = {
            "origin": [float(imgpts[0][0]), float(imgpts[0][1])],
            "x_axis": [float(imgpts[1][0]), float(imgpts[1][1])],
            "y_axis": [float(imgpts[2][0]), float(imgpts[2][1])],
            "z_axis": [float(imgpts[3][0]), float(imgpts[3][1])],
        }

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "pose_axes_2d": pose_axes_2d,
        }

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
