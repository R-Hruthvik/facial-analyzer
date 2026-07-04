import pytest
from src.core.metrics import EyeAspectRatioTracker

def test_eye_aspect_ratio_tracker_blink():
    # threshold = 0.20, consecutive_frames = 2, window_size = 5
    tracker = EyeAspectRatioTracker(threshold=0.20, consecutive_frames=2, window_size=5)
    
    # Normal EAR (0.3)
    for _ in range(5):
        tracker.update(0.3)
    assert tracker.blink_count == 0
    assert tracker._below_counter == 0
    
    # Drop below threshold (0.15) - 1st frame
    tracker.update(0.15)
    assert tracker.blink_count == 0
    assert tracker._below_counter == 1
    
    # Drop below threshold (0.15) - 2nd frame
    tracker.update(0.15)
    assert tracker.blink_count == 1
    assert tracker._below_counter == 2
    assert tracker._blink_in_progress == True
    
    # Recover to normal EAR (0.3) - 1st frame
    tracker.update(0.3)
    assert tracker.blink_count == 1
    assert tracker._blink_in_progress == True

    # Recover to normal EAR (0.3) - 2nd frame
    tracker.update(0.3)
    assert tracker.blink_count == 1
    assert tracker._blink_in_progress == False
    assert tracker._below_counter == 0


def test_calculate_mar():
    import numpy as np
    from src.core.metrics import calculate_mar
    # Create mock face landmarks
    landmarks = np.zeros((478, 3), dtype=np.float32)
    # Upper inner lip (13) and lower inner lip (14)
    landmarks[13] = [0.5, 0.45, 0.0]
    landmarks[14] = [0.5, 0.55, 0.0] # height = 0.10
    # Left corner (61) and right corner (291)
    landmarks[61] = [0.4, 0.5, 0.0]
    landmarks[291] = [0.6, 0.5, 0.0] # width = 0.20
    
    mar = calculate_mar(landmarks)
    # MAR = height / width = 0.10 / 0.20 = 0.5
    assert abs(mar - 0.5) < 1e-4


def test_calculate_gaze_distraction():
    import numpy as np
    from src.core.metrics import calculate_gaze_distraction
    
    # 1. Less than 478 landmarks should return False
    landmarks_small = np.zeros((468, 3), dtype=np.float32)
    assert calculate_gaze_distraction(landmarks_small) == False
    
    # 2. Centered gaze should return False
    landmarks = np.zeros((478, 3), dtype=np.float32)
    # Left eye: outer (33), inner (133), iris (468), top (159), bottom (145)
    landmarks[33] = [0.4, 0.5, 0.0]
    landmarks[133] = [0.46, 0.5, 0.0]
    landmarks[468] = [0.43, 0.5, 0.0] # exactly in center
    landmarks[159] = [0.43, 0.48, 0.0]
    landmarks[145] = [0.43, 0.52, 0.0]
    
    # Right eye: outer (263), inner (362), iris (473), top (386), bottom (374)
    landmarks[263] = [0.6, 0.5, 0.0]
    landmarks[362] = [0.54, 0.5, 0.0]
    landmarks[473] = [0.57, 0.5, 0.0] # exactly in center
    landmarks[386] = [0.57, 0.48, 0.0]
    landmarks[374] = [0.57, 0.52, 0.0]
    
    assert calculate_gaze_distraction(landmarks) == False
    
    # 3. Looking away horizontally (iris shifted right)
    landmarks[468] = [0.45, 0.5, 0.0] # offset by 0.02, width is 0.06 -> offset_x = 0.02 / 0.06 = 0.33 > 0.18
    landmarks[473] = [0.59, 0.5, 0.0] # offset by 0.02, width is 0.06 -> offset_x = 0.33 > 0.18
    assert calculate_gaze_distraction(landmarks) == True

