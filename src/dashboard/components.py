"""
Reusable UI components for the Streamlit dashboard.

Each function returns a Plotly figure or a Streamlit element that can be
rendered in the main app.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import settings, logger
from src.pipeline.engagement_scorer import EngagementScorer
from src.pipeline.prompt_mapper import PromptMapper

_scorer = EngagementScorer()
_mapper = PromptMapper()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> Dict:
    """
    Render the sidebar configuration panel.

    Returns a dict of user-selected settings.
    """
    with st.sidebar:
        st.header("⚙ Configuration")

        # Video source
        video_source = st.radio(
            "Video Source",
            options=["camera", "video_file"],
            index=0,
            help="Select 'camera' for live webcam or 'video_file' to upload.",
        )

        config = {"video_source": video_source}

        if video_source == "camera":
            config["camera_id"] = st.number_input(
                "Camera Device ID",
                min_value=0,
                max_value=10,
                value=settings.CAMERA_ID,
                help="0 = default webcam.",
            )
        else:
            config["uploaded_video"] = st.file_uploader(
                "Upload Video",
                type=["mp4", "avi", "mov", "mkv"],
                help="Upload a pre-recorded video file.",
            )

        st.markdown("---")
        st.subheader("Sensitivity Thresholds")

        config["ear_threshold"] = st.slider(
            "EAR Threshold (eye closure)",
            min_value=0.05,
            max_value=0.40,
            value=settings.EAR_THRESHOLD,
            step=0.01,
            help="Below this EAR, the eye is considered closed.",
        )

        config["mar_threshold"] = st.slider(
            "MAR Threshold (mouth open)",
            min_value=0.2,
            max_value=1.0,
            value=settings.MAR_THRESHOLD,
            step=0.05,
            help="Above this MAR, the mouth is considered open.",
        )

        st.markdown("---")
        st.subheader("Performance")

        config["frame_skip"] = st.slider(
            "Frame Skip (0 = process all)",
            min_value=0,
            max_value=10,
            value=settings.FRAME_SKIP,
            step=1,
            help="Process every N-th frame. Higher = lower CPU usage.",
        )

        config["resize_scale"] = st.slider(
            "Resolution Scale",
            min_value=0.25,
            max_value=1.0,
            value=settings.RESIZE_SCALE,
            step=0.05,
            help="Downscale factor before inference. 0.5 halves resolution.",
        )

        st.markdown("---")
        st.markdown(
            "**Session ID:** `{}`".format(
                st.session_state.get("session_id", "—")
            )
        )

        st.caption(
            "AI-Powered Facial Engagement Analyzer v1.0.0"
        )

    return config


# ---------------------------------------------------------------------------
# Engagement Gauge
# ---------------------------------------------------------------------------

def render_engagement_gauge(score: float) -> go.Figure:
    """
    Create a Plotly gauge chart for the engagement score (0–100 %).
    """
    label = _scorer.score_to_label(score)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"suffix": "%", "font": {"size": 36}},
            title={
                "text": f"Engagement<br><span style='font-size:18px'>{label}</span>",
                "font": {"size": 20},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "darkblue", "thickness": 0.3},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 30], "color": "#ff4d4d"},
                    {"range": [30, 50], "color": "#ffa64d"},
                    {"range": [50, 70], "color": "#ffd633"},
                    {"range": [70, 90], "color": "#99cc33"},
                    {"range": [90, 100], "color": "#33cc33"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white" if st.get_option("theme.base") == "dark" else "black"},
    )
    return fig


# ---------------------------------------------------------------------------
# Metric Chart
# ---------------------------------------------------------------------------

def render_metric_chart(
    timestamps: List[float],
    series: List[float],
    title: str = "",
    y_range: Optional[Tuple[float, float]] = None,
    threshold: Optional[float] = None,
    secondary_series: Optional[Tuple[List[float], str]] = None,
    tertiary_series: Optional[Tuple[List[float], str]] = None,
) -> go.Figure:
    """
    Plotly line chart for a single metric over time.

    Parameters
    ----------
    timestamps : List of x-axis values (seconds elapsed).
    series : Primary y-axis data.
    title : Chart title.
    y_range : (min, max) for y-axis.
    threshold : Horizontal dashed line indicating a threshold.
    secondary_series : (data, label) for a second trace.
    tertiary_series : (data, label) for a third trace.
    """
    fig = go.Figure()

    # Normalise timestamps to seconds elapsed
    t0 = timestamps[0] if timestamps else 0
    t_sec = [t - t0 for t in timestamps]

    fig.add_trace(
        go.Scatter(
            x=t_sec,
            y=series,
            mode="lines",
            name=title.split("—")[0].strip() or "Value",
            line=dict(width=2, color="#1f77b4"),
        )
    )

    if secondary_series is not None:
        fig.add_trace(
            go.Scatter(
                x=t_sec[: len(secondary_series[0])],
                y=secondary_series[0],
                mode="lines",
                name=secondary_series[1],
                line=dict(width=2, color="#ff7f0e"),
            )
        )

    if tertiary_series is not None:
        fig.add_trace(
            go.Scatter(
                x=t_sec[: len(tertiary_series[0])],
                y=tertiary_series[0],
                mode="lines",
                name=tertiary_series[1],
                line=dict(width=2, color="#2ca02c"),
            )
        )

    # Threshold line
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({threshold})",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        height=180,
        margin=dict(l=10, r=10, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font={"color": "white" if st.get_option("theme.base") == "dark" else "black"},
    )

    if y_range:
        fig.update_yaxes(range=y_range)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    return fig


# ---------------------------------------------------------------------------
# Video Feed Placeholder
# ---------------------------------------------------------------------------

def render_video_feed(frame_bgr: np.ndarray, placeholder) -> None:
    """
    Display an OpenCV BGR frame in a Streamlit placeholder.

    This is called from the processing loop callback.
    """
    import cv2

    # Convert BGR -> RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    placeholder.image(frame_rgb, channels="RGB", width='stretch')


def render_video_feed_simple(frame_bgr: np.ndarray, placeholder) -> None:
    """
    Simplified video feed rendering - optimized for very high FPS using cv2.imencode.
    """
    import cv2
    
    # Fast JPEG encoding via OpenCV avoids Streamlit's slow Pillow fallback
    ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ret:
        placeholder.image(buffer.tobytes(), width='stretch')


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------

def render_summary_report(
    ear_history: List[float],
    mar_history: List[float],
    engagement_history: List[float],
    pose_history: Dict[str, List[float]],
    blink_count: int,
    yawn_count: int,
    looking_away_count: int,
    session_id: str,
) -> None:
    """
    Render a comprehensive session summary report with insights.
    """
    if not ear_history:
        st.info("No data collected for this session.")
        return

    # Compute aggregates
    avg_ear = float(np.mean(ear_history))
    min_ear = float(np.min(ear_history))
    avg_mar = float(np.mean(mar_history))
    avg_engagement = float(np.mean(engagement_history))
    duration_est = len(ear_history) / 30.0  # assume ~30 fps
    blink_rate = blink_count / (duration_est / 60.0) if duration_est > 0 else 0
    looking_away_ratio = looking_away_count / max(len(ear_history), 1)

    avg_pitch = float(np.mean(pose_history["pitch"])) if pose_history["pitch"] else 0
    avg_yaw = float(np.mean(pose_history["yaw"])) if pose_history["yaw"] else 0
    avg_roll = float(np.mean(pose_history["roll"])) if pose_history["roll"] else 0

    # Score
    score = _scorer.compute(
        avg_ear=avg_ear,
        min_ear=min_ear,
        avg_mar=avg_mar,
        looking_away_ratio=looking_away_ratio,
        blink_rate=blink_rate,
    )

    # Insights
    insights = _mapper.generate_insights(
        avg_ear=avg_ear,
        min_ear=min_ear,
        blink_count=blink_count,
        blink_rate=blink_rate,
        avg_mar=avg_mar,
        yawn_count=yawn_count,
        looking_away_count=looking_away_count,
        looking_away_ratio=looking_away_ratio,
        engagement_score=score,
    )

    # --- Render ---
    st.markdown("---")
    st.header("📊 Session Summary Report")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Engagement Score", f"{score:.1f}%")
    col2.metric("Avg EAR", f"{avg_ear:.3f}")
    col3.metric("Avg MAR", f"{avg_mar:.3f}")
    col4.metric("Blinks", str(blink_count))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Yawns", str(yawn_count))
    col6.metric("Looked Away", str(looking_away_count))
    col7.metric("Blink Rate", f"{blink_rate:.1f}/min")
    col8.metric("Avg Pitch/Yaw", f"{avg_pitch:.1f}° / {avg_yaw:.1f}°")

    st.markdown("### 🔍 Insights")
    for insight in insights:
        st.markdown(f"- {insight}")

    # LLM prompt section
    with st.expander("📝 LLM Coaching Prompt (expand to copy)"):
        llm_prompt = insights[-1] if insights else ""
        st.text_area(
            "Copy this prompt into an LLM for a behavioural coaching summary:",
            value=llm_prompt,
            height=250,
        )

    # Final gauge
    st.plotly_chart(
        render_engagement_gauge(score), use_container_width=True, key="summary_gauge"
    )

    st.caption(f"Session ID: `{session_id}` | Estimated duration: {duration_est:.1f}s")
