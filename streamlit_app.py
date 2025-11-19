import time
import math
import threading
from typing import Optional, Dict, Any

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd  # <-- ADDED


@st.cache_resource(show_spinner=True)
def load_movenet():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    return model.signatures['serving_default']


def draw_keypoints_overlay(frame: np.ndarray, keypoints: np.ndarray, threshold: float = 0.3):
    h, w, _ = frame.shape
    kp = keypoints[0, 0, :, :]  # [17, (y, x, score)]

    def to_px(pt):
        y, x, s = pt
        return int(x * w), int(y * h), float(s)

    # Eyes and nose with enhanced markers
    nose = kp[0]
    left_eye = kp[1]
    right_eye = kp[2]

    nx, ny, ns = to_px(nose)
    lex, ley, les = to_px(left_eye)
    rex, rey, res = to_px(right_eye)

    if ns > threshold:
        cv2.circle(frame, (nx, ny), 10, (0, 0, 255), 2)
        cv2.putText(frame, "Nose", (nx + 8, ny - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if les > threshold:
        cv2.circle(frame, (lex, ley), 12, (0, 255, 255), 2)
        cv2.circle(frame, (lex, ley), 4, (0, 0, 255), -1)
        cv2.line(frame, (lex - 8, ley), (lex + 8, ley), (255, 255, 255), 1)
        cv2.line(frame, (lex, ley - 8), (lex, ley + 8), (255, 255, 255), 1)
        cv2.putText(frame, "L-Eye", (lex + 8, ley - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if res > threshold:
        cv2.circle(frame, (rex, rey), 12, (0, 255, 255), 2)
        cv2.circle(frame, (rex, rey), 4, (0, 0, 255), -1)
        cv2.line(frame, (rex - 8, rey), (rex + 8, rey), (255, 255, 255), 1)
        cv2.line(frame, (rex, rey - 8), (rex, rey + 8), (255, 255, 255), 1)
        cv2.putText(frame, "R-Eye", (rex + 8, rey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Other keypoints brighter markers
    for i, pt in enumerate(kp):
        x, y, score = to_px(pt)
        if score > threshold:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 9, (0, 200, 0), 1)


def face_alerts(keypoints: np.ndarray, threshold: float = 0.2):
    kp = keypoints[0, 0, :, :]
    nose, left_eye, right_eye, left_ear, right_ear = kp[0], kp[1], kp[2], kp[3], kp[4]

    alerts = []
    if nose[2] < threshold:
        alerts.append("Face not visible")
    if left_eye[2] < threshold and right_eye[2] < threshold:
        alerts.append("Eyes not visible")
    if left_ear[2] < threshold and right_ear[2] < threshold:
        alerts.append("Possibly looking away")
    return alerts


class PoseTransformer(VideoTransformerBase):
    def __init__(self, movenet, conf_threshold: float, movement_threshold: float, alert_eliminate_threshold: int):
        self.movenet = movenet
        self.conf_threshold = conf_threshold
        self.movement_threshold = movement_threshold
        self.alert_eliminate_threshold = alert_eliminate_threshold

        self.prev_left_eye: Optional[tuple] = None
        self.prev_right_eye: Optional[tuple] = None
        self.eye_movement_total: float = 0.0
        self.window_start_time: float = time.time()

        self.alert_count: int = 0
        self.eliminated: bool = False
        self.movement_alert_active: bool = False

        self.latest_metrics: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def _euclidean(self, pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Run MoveNet
        input_img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
        input_img = tf.cast(input_img, dtype=tf.int32)
        outputs = self.movenet(input_img)
        keypoints = outputs['output_0'].numpy()

        h, w, _ = img.shape
        left_eye = (keypoints[0, 0, 1, 1] * w, keypoints[0, 0, 1, 0] * h)
        right_eye = (keypoints[0, 0, 2, 1] * w, keypoints[0, 0, 2, 0] * h)

        # Eye movement accumulation
        if self.prev_left_eye is not None and self.prev_right_eye is not None:
            self.eye_movement_total += self._euclidean(left_eye, self.prev_left_eye)
            self.eye_movement_total += self._euclidean(right_eye, self.prev_right_eye)

        self.prev_left_eye = left_eye
        self.prev_right_eye = right_eye

        # Alerts from keypoint visibility
        alerts = face_alerts(keypoints, threshold=self.conf_threshold)

        # Movement alert: if movement in window >= threshold over >=3s
        elapsed = time.time() - self.window_start_time
        movement_alert_now = False
        if elapsed >= 3.0:
            if self.eye_movement_total >= self.movement_threshold:
                movement_alert_now = True
                alerts.append("Too much movement")
            # Reset window
            with self.lock:
                self.latest_metrics = {
                    "time_elapsed": int(elapsed),
                    "eye_movement": float(self.eye_movement_total),
                    "alerts": list(alerts),
                    "alert_count": int(self.alert_count),
                    "eliminated": bool(self.eliminated),
                }
            self.eye_movement_total = 0.0
            self.window_start_time = time.time()

        # Update alert count and elimination logic
        if alerts:
            self.alert_count += 1
        if movement_alert_now:
            self.movement_alert_active = True
            self.alert_count += 1
        else:
            self.movement_alert_active = False

        if self.alert_count > self.alert_eliminate_threshold:
            self.eliminated = True

        # Draw overlay
        draw_keypoints_overlay(img, keypoints, threshold=self.conf_threshold)

        # Render alert status text
        y0 = 30
        if alerts:
            for idx, alert in enumerate(alerts):
                cv2.putText(img, alert, (10, y0 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        status = f"Alerts: {self.alert_count} | Movement: {'ON' if self.movement_alert_active else 'OFF'} | Eliminated: {'YES' if self.eliminated else 'NO'}"
        cv2.putText(img, status, (10, y0 + max(1, len(alerts)) * 24 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.set_page_config(page_title="Face & Eye Tracking (MoveNet)", layout="wide")
    st.title("Face & Eye Tracking with MoveNet (Streamlit)")
    st.caption("Real-time keypoint overlay, eye movement graph (3s), alerts, and elimination.")

    # Controls
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05)
    with col_b:
        movement_threshold = st.number_input("Movement threshold per 3s", min_value=0.0, value=300.0, step=10.0)
    with col_c:
        eliminate_threshold = st.slider("Eliminate when alerts exceed", 1, 15, 7)

    movenet = load_movenet()
    webrtc_ctx = webrtc_streamer(
        key="movenet-stream",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: PoseTransformer(movenet, conf_threshold, movement_threshold, eliminate_threshold),
        media_stream_constraints={"video": True, "audio": False},
    )

    # Metrics and chart
    if "chart_data" not in st.session_state:
        st.session_state.chart_data = []

    st.subheader("Metrics")
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()

    if webrtc_ctx and webrtc_ctx.video_transformer:
        vt = webrtc_ctx.video_transformer
        # Read latest metrics snapshot
        latest = {}
        with vt.lock:
            latest = dict(vt.latest_metrics) if vt.latest_metrics else {}

        # Only append if we have a valid metrics snapshot (prevents empty/no-op entries)
        if latest:
            st.session_state.chart_data.append({
                "t": len(st.session_state.chart_data) + 1,
                "movement": latest.get("eye_movement", 0.0)
            })
            # Keep last 10 points
            st.session_state.chart_data = st.session_state.chart_data[-10:]

            # Display metrics
            cols = st.columns(4)
            cols[0].metric("Eye movement (last 3s)", f"{latest.get('eye_movement', 0.0):.2f}")
            cols[1].metric("Alert count", str(latest.get("alert_count", 0)))
            cols[2].metric("Eliminated", "YES" if latest.get("eliminated", False) else "NO")
            cols[3].metric("Alerts", ", ".join(latest.get("alerts", [])) or "None")

            # Chart -> convert to pandas DataFrame and set 't' as index (fixes plotting issue)
            df = pd.DataFrame(st.session_state.chart_data)
            if not df.empty:
                df = df.set_index("t")
                # plot movement series
                chart_placeholder.line_chart(df["movement"])

    st.divider()
    st.info("Tip: Adjust thresholds if you see too many or too few alerts. Use the slider to change elimination criteria.")


if __name__ == "__main__":
    main()
