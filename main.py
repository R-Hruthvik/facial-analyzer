"""
Entry point for the Facial Engagement Analyzer.

Usage:
    # Start the FastAPI backend
    python main.py api

    # Start the Streamlit dashboard
    python main.py dashboard

    # Run a quick test with the default camera
    python main.py test
"""

import argparse
import sys
import logging
import os

# Suppress TensorFlow and MediaPipe warnings before importing anything else
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "2"

from src.config import settings, logger


def start_api():
    """Launch the FastAPI server via uvicorn."""
    import uvicorn
    logger.info("Starting API server on %s:%s", settings.API_HOST, settings.API_PORT)
    uvicorn.run(
        "src.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info",
    )


def start_dashboard():
    """Launch the unified web dashboard."""
    import webbrowser
    import threading
    import time
    import socket
    
    host = settings.API_HOST
    if host == "0.0.0.0":
        host = "localhost"
        
    url = f"http://{host}:{settings.API_PORT}"
    
    def open_browser():
        port = int(settings.API_PORT)
        start_time = time.time()
        # Poll socket until port is open, up to 10 seconds
        while time.time() - start_time < 10:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except (ConnectionRefusedError, socket.timeout):
                time.sleep(0.1)
                
        # Give a small 200ms grace period for the server workers to fully initialize
        time.sleep(0.2)
        logger.info("Opening dashboard in web browser: %s", url)
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning("Failed to open web browser: %s", e)
            
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start the API server which serves the static dashboard
    start_api()


def run_test():
    """
    Quick sanity test — open the default camera, process 50 frames, print
    average latency.
    """
    import cv2
    from src.core.face_mesh_engine import FaceMeshEngine
    from src.core.metrics import calculate_ear_both, calculate_mar

    logger.info("Running quick test on camera %s ...", settings.CAMERA_ID)
    engine = FaceMeshEngine()
    import sys
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(settings.CAMERA_ID, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(settings.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)

    if not cap.isOpened():
        logger.error("Cannot open camera %s", settings.CAMERA_ID)
        sys.exit(1)

    processed = 0
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break

        if _ % (settings.FRAME_SKIP + 1) != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = engine.process(frame_rgb)
        landmarks = engine.extract_landmarks(results)

        if landmarks is not None:
            left_ear, right_ear, avg_ear = calculate_ear_both(landmarks)
            mar = calculate_mar(landmarks)
            processed += 1
            if processed <= 5 or processed % 10 == 0:
                logger.info(
                    "Frame %d | EAR=%.3f | MAR=%.3f | Inf=%.1f ms",
                    processed, avg_ear, mar, engine.avg_inference_ms,
                )

    cap.release()
    engine.close()
    logger.info(
        "Test completed — %d frames processed, avg inference %.1f ms",
        processed, engine.avg_inference_ms,
    )


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Facial Engagement Analyzer"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="dashboard",
        choices=["api", "dashboard", "test"],
        help="Which component to run (default: dashboard).",
    )
    args = parser.parse_args()

    commands = {
        "api": start_api,
        "dashboard": start_dashboard,
        "test": run_test,
    }
    try:
        commands[args.command]()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()

