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
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys as _sys
    cmd = [
        _sys.executable, "-m", "streamlit", "run",
        "src/dashboard/app.py",
        "--server.port", str(settings.DASHBOARD_PORT),
        "--server.address", settings.API_HOST,
    ]
    logger.info("Starting Streamlit dashboard on port %s", settings.DASHBOARD_PORT)
    subprocess.run(cmd)


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

