import numpy as np
import pytest
from src.core.head_pose import HeadPoseEstimator
def test_head_pose_estimator_projection():
    estimator = HeadPoseEstimator()
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[1] = [0.5, 0.5, 0.0]
    landmarks[4] = [0.5, 0.45, -0.05]
    landmarks[33] = [0.4, 0.4, -0.1]
    landmarks[263] = [0.6, 0.4, -0.1]
    landmarks[133] = [0.46, 0.41, -0.08]
    landmarks[362] = [0.54, 0.41, -0.08]
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
    origin_x, origin_y = axes["origin"]
    assert 300 <= origin_x <= 340
    assert 220 <= origin_y <= 260
    for axis in ["x_axis", "y_axis", "z_axis"]:
        coords = axes[axis]
        assert len(coords) == 2
        assert isinstance(coords[0], float)
        assert isinstance(coords[1], float)
