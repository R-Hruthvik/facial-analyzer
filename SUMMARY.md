# 🧠 Facial Engagement Analyzer - Application Summary

The **Facial Engagement Analyzer** tracks real-time facial landmarks to compute user engagement and behavior metrics.

---

## 1. Architecture & Core Metrics

### Architecture Flow
`Video Source (Camera/File) ➔ Frame Processor (MediaPipe Face Mesh) ➔ FastAPI Backend & Streamlit Dashboard`

### Computed Metrics
1. **Eye Aspect Ratio (EAR):** Tracks blinks and drowsiness (Threshold `< 0.20` is closed).
2. **Mouth Aspect Ratio (MAR):** Tracks yawns or speaking (Threshold `> 0.60` is open).
3. **Head Pose (Pitch/Yaw/Roll):** Estimated using OpenCV `solvePnP` (Pitch `> 20°`, Yaw `> 25°`, Roll `> 30°` marks "Looking away").
4. **Engagement Score (0-100%):** Weighted combination:
   $$\text{Score} = 40\% \cdot \text{EAR}_{norm} + 30\% \cdot \text{Focus} + 15\% \cdot \text{Blink}_{norm} + 15\% \cdot \text{MAR}_{norm}$$

---

## 2. Quick Start

### Installation & Run
```bash
# Set up environment
python3 -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run Streamlit Dashboard (default port 8501)
python main.py dashboard

# Run FastAPI Backend (default port 8000)
python main.py api
```

---

## 3. Deployment Summary (AWS EC2 Ubuntu)

For production, run using `systemd` services to auto-restart on failures.

### Systemd Services config
* **API Service (`/etc/systemd/system/facial-analyzer-api.service`):**
  ```ini
  [Service]
  WorkingDirectory=/opt/facial-analyzer
  ExecStart=/opt/facial-analyzer/.venv/bin/python main.py api
  Restart=always
  ```
* **Dashboard Service (`/etc/systemd/system/facial-analyzer-dashboard.service`):**
  ```ini
  [Service]
  WorkingDirectory=/opt/facial-analyzer
  ExecStart=/opt/facial-analyzer/.venv/bin/python main.py dashboard
  Restart=always
  ```

---

## 4. Completed Bug Fixes

The following critical issues were resolved in the Streamlit application:
1. **Camera Feed:** Integrated `display_frame_callback` in `src/dashboard/app.py` to update the video placeholder in real time.
2. **Infinite Logs:** Placed logs in a scrollable, fixed-height UI container with level filtering to prevent layout displacement.
3. **Away Counter:** Fixed looking-away logic to only increment if a face is detected (`result.avg_ear > 0`) to prevent infinite increments.
4. **Stop Button:** Added proper generator cleanup using `frame_generator.close()` and state checks to release camera resources instantly.
5. **Metrics Flashing:** Replaced Plotly charts with native Streamlit charts (`st.line_chart`, `st.metric`) to prevent iframe reloads and visual glitching.
