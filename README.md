# Facial Engagement Analyzer

The Facial Engagement Analyzer tracks user attention in real-time using a standard webcam. Designed for scenarios like online class engagement and personal study sessions, it helps monitor focus without requiring specialized hardware. It uses MediaPipe to extract facial landmarks and calculates engagement metrics, including eye blinks, yawns, and head pose. It provides a FastAPI backend and a web dashboard to visualize this telemetry.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://developers.google.com/mediapipe)

## Features

- **Facial Landmark Detection**: Maps 468 facial points using MediaPipe Face Mesh.
- **Engagement Metrics**: Calculates Eye Aspect Ratio (EAR) for blink detection, Mouth Aspect Ratio (MAR) for yawn detection, and head pitch/yaw/roll.
- **Real-Time Dashboard**: Displays live telemetry and video feed through a built-in web interface.
- **REST API**: Exposes endpoints for data integration via FastAPI.
- **CPU-Optimized**: Runs on standard CPUs without requiring discrete GPUs.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd facial-analyzer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `requirements.txt` installs `opencv-python-headless` for server environments. For local development requiring GUI windows, swap it with `opencv-python`.*

4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to adjust camera settings and calculation thresholds.

## Usage

Control the application via `main.py`.

### Start the Dashboard

Launch the unified web dashboard (this also starts the API):
```bash
python main.py dashboard
```
The application opens `http://localhost:8000` in your default web browser.

### Start the API Only

Run the FastAPI server without launching the browser:
```bash
python main.py api
```

### Run a Quick Test

Test the camera and analysis pipeline without starting the web server. It captures a brief 100-frame sample from the camera and prints the average inference latency:
```bash
python main.py test
```

## Configuration

Adjust the following variables in your `.env` file to tune detection sensitivity:

- `EAR_THRESHOLD`: Eye aspect ratio below which the eye is closed (default: 0.22).
- `MAR_THRESHOLD`: Mouth aspect ratio above which the mouth is open (default: 0.60).
- `HEAD_PITCH_THRESHOLD`, `HEAD_YAW_THRESHOLD`, `HEAD_ROLL_THRESHOLD`: Degrees beyond which the user is looking away.
- `CAMERA_ID`: Index of the camera device (default: 0).
- `FRAME_SKIP`: Number of frames to skip to reduce CPU load (default: 0).
