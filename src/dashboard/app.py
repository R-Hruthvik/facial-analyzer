"""
Streamlit Dashboard — main entry point.

Run with:
    streamlit run src/dashboard/app.py
"""

import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.config import settings, logger
from src.dashboard.components import (
    render_engagement_gauge,
    render_metric_chart,
    render_sidebar,
    render_summary_report,
)
from src.dashboard.utils import RESULTS_DIR
from src.dashboard.logs import get_log_manager, setup_dashboard_logging

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Facial Engagement Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 AI-Powered Facial Engagement Analyzer")
st.markdown("Real-time 468-point face-mesh tracking with engagement telemetry.")

# Hide Streamlit "Running..." status to prevent top-right flickering
st.markdown("""
    <style>
        .stApp [data-testid="stStatusWidget"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Initialise session state
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "running" not in st.session_state:
    st.session_state.running = False
if "ear_history" not in st.session_state:
    st.session_state.ear_history = []
if "mar_history" not in st.session_state:
    st.session_state.mar_history = []
if "engagement_history" not in st.session_state:
    st.session_state.engagement_history = []
if "pose_history" not in st.session_state:
    st.session_state.pose_history = {"pitch": [], "yaw": [], "roll": []}
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []
if "blink_count" not in st.session_state:
    st.session_state.blink_count = 0
if "yawn_count" not in st.session_state:
    st.session_state.yawn_count = 0
if "looking_away_count" not in st.session_state:
    st.session_state.looking_away_count = 0
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False
if "video_source_closed" not in st.session_state:
    st.session_state.video_source_closed = False
if "latest_frame" not in st.session_state:
    st.session_state.latest_frame = None
if "frame_generator" not in st.session_state:
    st.session_state.frame_generator = None

# Initialize log manager
log_manager = get_log_manager()
if "logging_setup" not in st.session_state:
    setup_dashboard_logging()
    st.session_state.logging_setup = True

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
config = render_sidebar()

# ---------------------------------------------------------------------------
# Controls (Moved to top as requested)
# ---------------------------------------------------------------------------
st.markdown("---")
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

with btn_col1:
    start_btn = st.button("▶ Start" if not st.session_state.running else "⏹ Stop", use_container_width=True)
with btn_col2:
    reset_btn = st.button("🔄 Reset Session", use_container_width=True)
with btn_col3:
    gen_summary_btn = st.button("📊 Generate Summary", use_container_width=True)
st.markdown("---")

if start_btn:
    st.session_state.running = not st.session_state.running
    if not st.session_state.running:
        st.session_state.frame_generator = None
        st.session_state.video_source_closed = True
    else:
        st.session_state.frame_generator = None
        st.session_state.video_source_closed = False
        st.session_state.summary_generated = False
    st.rerun()

if reset_btn:
    for key in ["ear_history", "mar_history", "engagement_history", "pose_history", "timestamps"]:
        if key in st.session_state:
            if key == "pose_history":
                st.session_state[key] = {"pitch": [], "yaw": [], "roll": []}
            else:
                st.session_state[key] = []
    st.session_state.blink_count = 0
    st.session_state.yawn_count = 0
    st.session_state.looking_away_count = 0
    st.session_state.summary_generated = False
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.video_source_closed = False
    st.session_state.latest_frame = None
    st.session_state.running = False
    st.session_state.frame_generator = None
    st.rerun()

if gen_summary_btn:
    st.session_state.summary_generated = True
    st.rerun()

# ---------------------------------------------------------------------------
# Main Layout Containers (Horizontal Rows)
# ---------------------------------------------------------------------------
row1_video = st.container()
row2_metrics = st.container()
row3_logs = st.container()

# ---------------------------------------------------------------------------
# 1. Video and Processing Fragment (High FPS: 30+ times per second)
# ---------------------------------------------------------------------------
@st.fragment(run_every=0.03)
def process_and_video():
    if not st.session_state.running:
        if st.session_state.video_source_closed:
            st.info("⏹ Video source closed. Click Start to begin.")
        else:
            st.info("⏸ Waiting to start... Click Start to begin.")
        return

    # Initialize generator if needed
    if st.session_state.frame_generator is None:
        video_source = config.get("video_source", "camera")
        source_value = config.get("camera_id", 0)
        if video_source == "video_file":
            uploaded = config.get("uploaded_video")
            if uploaded is not None:
                temp_path = RESULTS_DIR / "uploaded_temp.mp4"
                with open(temp_path, "wb") as f:
                    f.write(uploaded.read())
                source_value = str(temp_path)
            else:
                st.warning("Please upload a video file.")
                st.session_state.running = False
                st.rerun()

        from src.pipeline.frame_processor import FrameProcessor
        processor = FrameProcessor(
            ear_threshold=config.get("ear_threshold", settings.EAR_THRESHOLD),
            mar_threshold=config.get("mar_threshold", settings.MAR_THRESHOLD),
            frame_skip=config.get("frame_skip", settings.FRAME_SKIP),
            resize_scale=config.get("resize_scale", settings.RESIZE_SCALE),
        )

        def display_frame_callback(frame_bgr: np.ndarray) -> None:
            st.session_state.latest_frame = frame_bgr

        if video_source == "camera":
            st.session_state.frame_generator = processor.run_camera(source_value, on_frame=display_frame_callback)
        else:
            st.session_state.frame_generator = processor.run_video(source_value, on_frame=display_frame_callback)

    tick_start = time.perf_counter()
    gen_next_ms = 0.0
    try:
        # Process multiple frames per UI tick if possible to keep up with camera
        for _ in range(3): 
            t_next = time.perf_counter()
            result = next(st.session_state.frame_generator)
            gen_next_ms += (time.perf_counter() - t_next) * 1000.0
            while getattr(result, 'frame_skipped', False):
                t_skip = time.perf_counter()
                result = next(st.session_state.frame_generator)
                gen_next_ms += (time.perf_counter() - t_skip) * 1000.0
                
            # Update state
            st.session_state.timestamps.append(result.timestamp)
            st.session_state.ear_history.append(result.avg_ear)
            st.session_state.mar_history.append(result.mar)
            st.session_state.engagement_history.append(result.engagement_score)
            st.session_state.pose_history["pitch"].append(result.pitch or 0)
            st.session_state.pose_history["yaw"].append(result.yaw or 0)
            st.session_state.pose_history["roll"].append(result.roll or 0)

            max_len = 200
            for key in ["timestamps", "ear_history", "mar_history", "engagement_history"]:
                if len(st.session_state[key]) > max_len:
                    st.session_state[key] = st.session_state[key][-max_len:]
            for key in ["pitch", "yaw", "roll"]:
                if len(st.session_state.pose_history[key]) > max_len:
                    st.session_state.pose_history[key] = st.session_state.pose_history[key][-max_len:]

            st.session_state.blink_count = result.blink_count
            st.session_state.yawn_count = result.yawn_count
            if result.avg_ear > 0 and result.is_looking_away:
                st.session_state.looking_away_count += 1
            
            # Break after one successful frame update to render it immediately
            break
            
    except StopIteration:
        st.session_state.running = False
        st.session_state.frame_generator = None
        st.session_state.video_source_closed = True
    except Exception as exc:
        logger.exception("Error during frame processing")
        st.session_state.running = False
        st.session_state.frame_generator = None
        st.session_state.video_source_closed = True

    # Render video instantly
    encode_ms = 0.0
    if st.session_state.latest_frame is not None and st.session_state.running:
        t_enc = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', st.session_state.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        encode_ms = (time.perf_counter() - t_enc) * 1000.0
        if ret:
            st.image(buffer.tobytes(), width="stretch")

    if st.session_state.running:
        logger.warning(
            "Fragment tick total: %.1f ms (gen.next=%.1f ms, encode=%.1f ms)",
            (time.perf_counter() - tick_start) * 1000.0, gen_next_ms, encode_ms
        )

# ---------------------------------------------------------------------------
# 2. Metrics Fragment (Low FPS: 2 times per second)
# ---------------------------------------------------------------------------
@st.fragment(run_every=0.5)
def render_metrics():
    if not st.session_state.running:
        return
        
    col_g, col_e, col_m, col_p = st.columns(4)
    if st.session_state.ear_history:
        import pandas as pd
        t_start = st.session_state.timestamps[0]
        
        with col_g:
            score = st.session_state.engagement_history[-1]
            from src.pipeline.engagement_scorer import EngagementScorer
            scorer = EngagementScorer()
            label = scorer.score_to_label(score)
            st.metric(label="Engagement Score", value=f"{score:.1f}%", delta=label)
            st.progress(min(max(score / 100.0, 0.0), 1.0))
            
        with col_e:
            ear_df = pd.DataFrame({
                "EAR": st.session_state.ear_history
            }, index=[round(t - t_start, 1) for t in st.session_state.timestamps])
            st.markdown("**Eye Aspect Ratio (EAR)**")
            st.line_chart(ear_df, height=180)
            
        with col_m:
            mar_df = pd.DataFrame({
                "MAR": st.session_state.mar_history
            }, index=[round(t - t_start, 1) for t in st.session_state.timestamps])
            st.markdown("**Mouth Aspect Ratio (MAR)**")
            st.line_chart(mar_df, height=180)
            
        with col_p:
            if st.session_state.pose_history["pitch"]:
                pose_df = pd.DataFrame({
                    "Pitch": st.session_state.pose_history["pitch"],
                    "Yaw": st.session_state.pose_history["yaw"],
                    "Roll": st.session_state.pose_history["roll"]
                }, index=[round(t - t_start, 1) for t in st.session_state.timestamps])
                st.markdown("**Head Pose (Pitch, Yaw, Roll)**")
                st.line_chart(pose_df, height=180)
        
        st.markdown(
            f"**Session:** `{st.session_state.session_id}` &nbsp;|&nbsp; "
            f"**Blinks:** {st.session_state.blink_count} &nbsp;|&nbsp; "
            f"**Yawns:** {st.session_state.yawn_count} &nbsp;|&nbsp; "
            f"**Looking Away:** {st.session_state.looking_away_count} frames"
        )

# ---------------------------------------------------------------------------
# Render the Layout
# ---------------------------------------------------------------------------
with row1_video:
    st.markdown("### 📹 Video Feed")
    process_and_video()

with row2_metrics:
    st.markdown("### 📊 Metrics")
    render_metrics()

with row3_logs:
    st.markdown("### 📝 Logs")
    recent_logs = log_manager.get_recent_logs(level_filter=None, limit=50)
    try:
        st.markdown(f"**Log Entries:** {log_manager.get_handler().get_log_count()}")
    except Exception:
        st.markdown("**Log Entries:** 0")
        
    log_lines = []
    if recent_logs:
        for log_entry in reversed(recent_logs):
            ts = log_entry.timestamp.strftime("%H:%M:%S")
            level = log_entry.level
            name = log_entry.name
            filename = log_entry.filename
            lineno = log_entry.lineno
            msg = log_entry.message
            
            # Dynamic colors based on severity
            if level == "ERROR":
                color = "#ff4d4d"
            elif level == "WARNING":
                color = "#ffa64d"
            elif level == "INFO":
                color = "#00d4ff"
            else:
                color = "#e0e0e0"
                
            line_html = f"<div style='color: {color}; font-family: monospace; font-size: 13px; line-height: 1.4; margin: 2px 0; border-bottom: 1px solid #2a2a2a; padding-bottom: 2px;'>"
            line_html += f"<code>[{ts}] [{level}] [{name}:{filename}:{lineno}] &mdash; {msg}</code>"
            
            if log_entry.traceback:
                # Escape standard HTML and handle spaces/newlines
                tb_clean = log_entry.traceback.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>").replace(" ", "&nbsp;")
                line_html += f"<details style='margin-left: 20px; color: #ff8080; margin-top: 2px;'><summary style='cursor: pointer; font-size: 11px; outline: none;'>View Stack Trace</summary>"
                line_html += f"<div style='font-family: monospace; font-size: 11px; margin-top: 4px; white-space: pre-wrap; background-color: #2b1111; padding: 8px; border-radius: 4px; border: 1px solid #552222; overflow-x: auto;'>{tb_clean}</div></details>"
            
            line_html += "</div>"
            log_lines.append(line_html)
            
    logs_html = f"""
    <div style='background-color: #121212; border: 1px solid #333; border-radius: 6px; padding: 12px; font-family: monospace; max-height: 250px; overflow-y: auto; box-shadow: inset 0 0 10px rgba(0,0,0,0.8);'>
        {"".join(log_lines) if log_lines else "<span style='color: #707070;'>No logs yet.</span>"}
    </div>
    """
    st.markdown(logs_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
if st.session_state.summary_generated and st.session_state.ear_history:
    render_summary_report(
        ear_history=st.session_state.ear_history,
        mar_history=st.session_state.mar_history,
        engagement_history=st.session_state.engagement_history,
        pose_history=st.session_state.pose_history,
        blink_count=st.session_state.blink_count,
        yawn_count=st.session_state.yawn_count,
        looking_away_count=st.session_state.looking_away_count,
        session_id=st.session_state.session_id,
    )
