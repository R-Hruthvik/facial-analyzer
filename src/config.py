"""
Configuration module for the Facial Engagement Analyzer.

All tunable parameters are centralized here.  These can be overridden
via environment variables (loaded from a .env file if present).
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
import logging
from logging.handlers import RotatingFileHandler

# Create logger
logger = logging.getLogger("facial-analyzer")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# Remove existing handlers to avoid duplicate logs
logger.handlers.clear()

class SafeStreamHandler(logging.StreamHandler):
    """A StreamHandler that doesn't crash if the underlying stream (like stdout/stderr) is closed or unavailable on Windows."""
    def emit(self, record):
        try:
            super().emit(record)
        except Exception:
            pass

    def flush(self):
        try:
            super().flush()
        except Exception:
            pass

# Console handler for terminal output
console_handler = SafeStreamHandler()
console_handler.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
console_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler for persistent logs
log_dir = os.getenv("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "app.log"),
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
)
file_handler.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Streamlit dashboard handler (will be added by dashboard app)
# This allows logs to be displayed in the UI

# Prevent propagation to root logger to avoid duplicate logs
logger.propagate = False


class Settings:
    """Application settings — read from environment with sensible defaults."""

    # ---- MediaPipe Face Mesh ------------------------------------------------
    FACEMESH_STATIC_IMAGE_MODE: bool = False
    FACEMESH_MAX_NUM_FACES: int = int(os.getenv("FACEMESH_MAX_NUM_FACES", "1"))
    FACEMESH_REFINE_LANDMARKS: bool = True          # get the iris landmarks
    FACEMESH_MIN_DETECTION_CONFIDENCE: float = float(
        os.getenv("FACEMESH_MIN_DETECTION_CONFIDENCE", "0.5")
    )
    FACEMESH_MIN_TRACKING_CONFIDENCE: float = float(
        os.getenv("FACEMESH_MIN_TRACKING_CONFIDENCE", "0.5")
    )

    # ---- Video / Stream -----------------------------------------------------
    CAMERA_ID: int = int(os.getenv("CAMERA_ID", "0"))
    FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "640"))
    FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "480"))
    FRAME_SKIP: int = int(os.getenv("FRAME_SKIP", "0"))
    """Process every N-th frame to reduce CPU load."""

    RESIZE_SCALE: float = float(os.getenv("RESIZE_SCALE", "1.0"))
    """Downscale factor applied before inference (< 1.0 lowers resolution)."""

    # ---- EAR (Eye Aspect Ratio) ---------------------------------------------
    EAR_THRESHOLD: float = float(os.getenv("EAR_THRESHOLD", "0.22"))
    """Below this value the eye is considered closed."""

    EAR_CONSECUTIVE_FRAMES: int = int(os.getenv("EAR_CONSECUTIVE_FRAMES", "1"))
    """Number of consecutive frames below threshold to count a blink."""

    # ---- MAR (Mouth Aspect Ratio) -------------------------------------------
    MAR_THRESHOLD: float = float(os.getenv("MAR_THRESHOLD", "0.6"))
    """Above this value the mouth is considered open (yawn / speak)."""

    # ---- Head Pose -----------------------------------------------------------
    HEAD_PITCH_THRESHOLD: float = float(
        os.getenv("HEAD_PITCH_THRESHOLD", "35.0")
    )
    """Degrees of pitch beyond which the user is looking away."""

    HEAD_YAW_THRESHOLD: float = float(os.getenv("HEAD_YAW_THRESHOLD", "35.0"))
    HEAD_ROLL_THRESHOLD: float = float(os.getenv("HEAD_ROLL_THRESHOLD", "30.0"))

    # ---- Engagement Score ----------------------------------------------------
    ENGAGEMENT_WINDOW_SECONDS: int = int(
        os.getenv("ENGAGEMENT_WINDOW_SECONDS", "30")
    )

    # ---- API Server ----------------------------------------------------------
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # ---- Streamlit Dashboard -------------------------------------------------
    DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8501"))

    # ---- Paths ---------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"


settings = Settings()
