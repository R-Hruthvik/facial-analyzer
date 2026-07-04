import os
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from logging.handlers import RotatingFileHandler
logger = logging.getLogger("facial-analyzer")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logger.handlers.clear()
class SafeStreamHandler(logging.StreamHandler):
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
console_handler = SafeStreamHandler()
console_handler.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
console_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
log_dir = os.getenv("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "app.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
file_handler.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.propagate = False
class Settings:
    FACEMESH_STATIC_IMAGE_MODE: bool = False
    FACEMESH_MAX_NUM_FACES: int = int(os.getenv("FACEMESH_MAX_NUM_FACES", "1"))
    FACEMESH_REFINE_LANDMARKS: bool = True
    FACEMESH_MIN_DETECTION_CONFIDENCE: float = float(
        os.getenv("FACEMESH_MIN_DETECTION_CONFIDENCE", "0.5")
    )
    FACEMESH_MIN_TRACKING_CONFIDENCE: float = float(
        os.getenv("FACEMESH_MIN_TRACKING_CONFIDENCE", "0.5")
    )
    CAMERA_ID: int = int(os.getenv("CAMERA_ID", "0"))
    FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "640"))
    FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "480"))
    FRAME_SKIP: int = int(os.getenv("FRAME_SKIP", "0"))
    RESIZE_SCALE: float = float(os.getenv("RESIZE_SCALE", "1.0"))
    EAR_THRESHOLD: float = float(os.getenv("EAR_THRESHOLD", "0.22"))
    EAR_CONSECUTIVE_FRAMES: int = int(os.getenv("EAR_CONSECUTIVE_FRAMES", "1"))
    MAR_THRESHOLD: float = float(os.getenv("MAR_THRESHOLD", "0.6"))
    HEAD_PITCH_THRESHOLD: float = float(
        os.getenv("HEAD_PITCH_THRESHOLD", "35.0")
    )
    HEAD_YAW_THRESHOLD: float = float(os.getenv("HEAD_YAW_THRESHOLD", "35.0"))
    HEAD_ROLL_THRESHOLD: float = float(os.getenv("HEAD_ROLL_THRESHOLD", "30.0"))
    ENGAGEMENT_WINDOW_SECONDS: int = int(
        os.getenv("ENGAGEMENT_WINDOW_SECONDS", "30")
    )
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8501"))
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"
settings = Settings()
