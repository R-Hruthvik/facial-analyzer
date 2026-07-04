import pytest
from src.core.metrics import EyeAspectRatioTracker
def test_eye_aspect_ratio_tracker_blink():
    tracker = EyeAspectRatioTracker(threshold=0.20, consecutive_frames=2, window_size=5)
    for _ in range(5):
        tracker.update(0.3)
    assert tracker.blink_count == 0
    assert tracker._below_counter == 0
    tracker.update(0.15)
    assert tracker.blink_count == 0
    assert tracker._below_counter == 1
    tracker.update(0.15)
    assert tracker.blink_count == 1
    assert tracker._below_counter == 2
    assert tracker._blink_in_progress == True
    tracker.update(0.3)
    assert tracker.blink_count == 1
    assert tracker._blink_in_progress == True
    tracker.update(0.3)
    assert tracker.blink_count == 1
    assert tracker._blink_in_progress == False
    assert tracker._below_counter == 0
def test_calculate_mar():
    import numpy as np
    from src.core.metrics import calculate_mar
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[13] = [0.5, 0.45, 0.0]
    landmarks[14] = [0.5, 0.55, 0.0]
    landmarks[61] = [0.4, 0.5, 0.0]
    landmarks[291] = [0.6, 0.5, 0.0]
    mar = calculate_mar(landmarks)
    assert abs(mar - 0.5) < 1e-4
def test_calculate_gaze_distraction():
    import numpy as np
    from src.core.metrics import calculate_gaze_distraction
    landmarks_small = np.zeros((468, 3), dtype=np.float32)
    assert calculate_gaze_distraction(landmarks_small) == False
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[33] = [0.4, 0.5, 0.0]
    landmarks[133] = [0.46, 0.5, 0.0]
    landmarks[468] = [0.43, 0.5, 0.0]
    landmarks[159] = [0.43, 0.48, 0.0]
    landmarks[145] = [0.43, 0.52, 0.0]
    landmarks[263] = [0.6, 0.5, 0.0]
    landmarks[362] = [0.54, 0.5, 0.0]
    landmarks[473] = [0.57, 0.5, 0.0]
    landmarks[386] = [0.57, 0.48, 0.0]
    landmarks[374] = [0.57, 0.52, 0.0]
    assert calculate_gaze_distraction(landmarks) == False
    landmarks[468] = [0.45, 0.5, 0.0]
    landmarks[473] = [0.59, 0.5, 0.0]
    assert calculate_gaze_distraction(landmarks) == True
