import numpy as np
import pytest
from src.core.head_pose import HeadPoseEstimator


def test_head_pose_estimator_projection():
    estimator = HeadPoseEstimator()

    # Create mock face landmarks of shape (478, 3)
    # Using normalized coordinates centered around 0.5 (face facing straight forward)
    landmarks = np.zeros((478, 3), dtype=np.float32)

    # Approximate 2D canonical indices to face forward:
    # 1: nose tip (0.5, 0.5)
    # 4: nose bridge (0.5, 0.45)
    # 33: left eye outer (0.4, 0.4)
    # 263: right eye outer (0.6, 0.4)
    # 133: left eye inner (0.46, 0.41)
    # 362: right eye inner (0.54, 0.41)
    landmarks[1] = [0.5, 0.5, 0.0]
    landmarks[4] = [0.5, 0.45, -0.05]
    landmarks[33] = [0.4, 0.4, -0.1]
    landmarks[263] = [0.6, 0.4, -0.1]
    landmarks[133] = [0.46, 0.41, -0.08]
    landmarks[362] = [0.54, 0.41, -0.08]

    # Run estimate for 640x480 frame size
    result = estimator.estimate(landmarks, 640, 480)

    assert result is not None
    assert "pitch" in result
    assert "yaw" in result
    assert "roll" in result
    assert "pose_axes_2d" in result

    axes = result["pose_axes_2d"]
    assert "origin" in axes
    assert "x_axis" in axes
    assert "y_axis" in axes
    assert "z_axis" in axes

    # Origin should be close to center (denormalized nose tip is 0.5 * 640 = 320, 0.5 * 480 = 240)
    origin_x, origin_y = axes["origin"]
    assert 300 <= origin_x <= 340
    assert 220 <= origin_y <= 260

    # Verify other axes are lists of two numbers
    for axis in ["x_axis", "y_axis", "z_axis"]:
        coords = axes[axis]
        assert len(coords) == 2
        assert isinstance(coords[0], float)
        assert isinstance(coords[1], float)
