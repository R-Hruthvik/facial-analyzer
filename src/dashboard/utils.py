"""
Utility functions for the Streamlit dashboard.

Includes helpers for saving summaries, managing result directories, and
video-streaming callbacks.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Generator, Optional

import cv2
import numpy as np

from src.config import settings, logger
from src.pipeline.frame_processor import FrameProcessor

RESULTS_DIR = settings.BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_session_summary(
    summary_data: dict, session_id: str
) -> Path:
    """
    Save a session summary as a JSON file.

    Parameters
    ----------
    summary_data : dict — the summary to persist.
    session_id : str — used for the filename.

    Returns
    -------
    Path to the saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = RESULTS_DIR / f"session_{session_id}_{timestamp}.json"

    with open(file_path, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    logger.info("Session summary saved to %s", file_path)
    return file_path


def load_session_summaries() -> list[dict]:
    """
    Load all saved session summaries from the results directory.

    Returns a list of dicts sorted by modification time (newest first).
    """
    summaries = []
    for fpath in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(fpath) as f:
                summaries.append(json.load(f))
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Failed to load %s: %s", fpath, exc)
    return summaries


def streaming_video_loop(
    processor: FrameProcessor,
    video_source: str,
    on_frame: Callable[[np.ndarray], None],
) -> Generator:
    """
    Run the frame processor and invoke ``on_frame`` with each annotated BGR
    frame for display.

    Yields ``FrameResult`` objects.
    """
    if video_source.isdigit() or isinstance(video_source, int):
        gen = processor.run_camera(int(video_source), on_frame=on_frame)
    else:
        gen = processor.run_video(video_source, on_frame=on_frame)

    yield from gen
