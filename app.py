"""Streamlit dashboard for the Face Mask Detector.

Run:  streamlit run app.py
"""
import json
import os
import tempfile
import threading
import time
from datetime import datetime

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from config import (ROOT, ALERT_COOLDOWN_SEC, MASK_MODEL_PATH, MASK_ONNX_PATH, MONITOR_THRESHOLD,
                    REPORTS_DIR, SNAPSHOTS_DIR)
from src import database, voice
from src.camera import Camera
from src.detector import FaceTracker, draw_detections, pretty

st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
:root { --teal-900: #0f3d36; --teal: #1f6f63; --teal-500: #2a8a7a; --mint: #3fe5b2;
        --label: #e0913f; --page: #e6ebed; --display: 'Bebas Neue', 'Oswald', Impact, sans-serif; }
@keyframes rise { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }

/* page: grey canvas with the app as one white card */
[data-testid="stAppViewContainer"] { background: var(--page); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMain"] { padding: 1.5rem 2rem 2rem; }
[data-testid="stMainBlockContainer"] { background: #fff; border-radius: 22px; max-width: 1280px !important;
    margin: 0 auto !important; padding: 1.5rem 2.5rem 2.5rem !important; box-shadow: 0 18px 50px rgba(15,61,54,.10); }
[data-testid="stSidebar"] { background: #f5f8f8; }

/* display type */
h1, h2, h3 { font-family: var(--display) !important; font-weight: 400 !important; letter-spacing: .02em;
             color: var(--teal); text-transform: uppercase; }
h2 { font-size: 2.2rem !important; } h3 { font-size: 1.7rem !important; }

/* nav-style tabs */
[role="tablist"] { gap: 1.6rem; }
[data-testid="stTab"] p { font-size: .78rem !important; font-weight: 700; letter-spacing: .12em;
                         text-transform: uppercase; }

/* metric cards like the "Smoke / Fine Dust" readouts */
[data-testid="stMetric"] { animation: rise .5s ease both; }
[data-testid="stMetricLabel"] p { font-size: .7rem !important; font-weight: 700; letter-spacing: .14em;
                                  text-transform: uppercase; color: var(--label); }
[data-testid="stMetricValue"] { font-family: var(--display); color: var(--teal); font-size: 2.6rem !important; }

/* mint call-to-action buttons */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
    background: var(--mint) !important; border-color: var(--mint) !important; color: var(--teal-900) !important;
    font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }

/* hero panel with stepped bottom-left corner */
.hero { position: relative; overflow: hidden; color: #fff; margin: .25rem 0 1.25rem;
        padding: 1.4rem 2.2rem 2rem 6.5rem; min-height: 260px;
        background: linear-gradient(135deg, var(--teal) 0%, var(--teal-900) 100%);
        clip-path: polygon(0 0, 100% 0, 100% 100%, 9% 100%, 9% 82%, 5% 82%, 5% 62%, 0 62%);
        border-radius: 18px; animation: rise .6s ease both; }
.hero .nav { display: flex; justify-content: space-between; align-items: center; font-size: .7rem;
             letter-spacing: .14em; text-transform: uppercase; opacity: .85; }
.hero .pill { background: var(--mint); color: var(--teal-900); padding: .3rem .8rem; border-radius: 6px;
              font-weight: 700; opacity: 1; }
.hero .title { font-family: var(--display); font-size: clamp(2.8rem, 6vw, 5rem); line-height: .95;
               color: var(--mint); margin-top: 1.2rem; animation: rise .7s .1s ease both; }
.hero .sub { max-width: 30rem; opacity: .75; font-size: .9rem; margin-top: .4rem; }
.hero .stats { display: flex; gap: 3rem; margin-top: 1.6rem; position: relative; z-index: 1;
               animation: rise .7s .25s ease both; }
.hero .stat .v { font-family: var(--display); font-size: 2.6rem; line-height: 1; }
.hero .stat .v sup { font-size: .9rem; color: var(--mint); margin-left: .2rem; }
.hero .stat .k { font-size: .65rem; letter-spacing: .14em; text-transform: uppercase; opacity: .7; }
.hero svg { position: absolute; right: 0; bottom: 0; width: 42%; height: 70%; opacity: .45; }

/* live / image verdict banners */
.banner { padding: .8rem 1.1rem; border-radius: 10px; font-weight: 700; font-size: 1rem; letter-spacing: .02em;
          margin-bottom: .75rem; border-left: 6px solid; animation: rise .3s ease both; }
.banner.ok   { background: rgba(63,229,178,.14); border-color: var(--teal-500); color: var(--teal); }
.banner.bad  { background: rgba(211,60,60,.10);  border-color: #d33c3c; color: #b42f2f; }
.banner.idle { background: #f1f4f5; border-color: #b7c3c6; color: #5b6f6b; }

@media (max-width: 700px) {
  [data-testid="stMain"] { padding: 0; }
  [data-testid="stMainBlockContainer"] { padding: 1rem !important; border-radius: 0; }
  .hero { padding: 1.2rem 1.2rem 1.6rem 1.6rem; clip-path: none; }
  .hero .stats { gap: 1.5rem; flex-wrap: wrap; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models...")
def load_detector():
    from src.detector import MaskDetector
    return MaskDetector()


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def decode_upload(file) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def save_violation(frame: np.ndarray) -> str:
    path = SNAPSHOTS_DIR / f"violation_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def log_all(detections, source: str, frame=None) -> None:
    snapshot = save_violation(frame) if frame is not None and any(d.is_violation for d in detections) else None
    database.log_detections([(d.label, d.confidence, d.is_violation, source,
                              snapshot if d.is_violation else None) for d in detections])


def show_frame(box, frame_bgr: np.ndarray) -> None:
    # JPEG is ~10x smaller than the default PNG, so the browser keeps up with the camera.
    box.image(to_rgb(frame_bgr), channels="RGB", output_format="JPEG", width="stretch")


def banner(box, n_faces: int, n_bad: int) -> None:
    """One-line verdict above an image/stream: green, red, or grey."""
    if not n_faces:
        cls, text = "idle", "No faces in view"
    elif n_bad:
        cls, text = "bad", f"⚠️ {n_bad} of {n_faces} {'person' if n_faces == 1 else 'people'} not wearing a mask"
    else:
        cls, text = "ok", f"✅ All {n_faces} {'person' if n_faces == 1 else 'people'} masked"
    box.markdown(f'<div class="banner {cls}">{text}</div>', unsafe_allow_html=True)


def snapshot_caption(path) -> str:
    try:
        return datetime.strptime(path.stem[10:25], "%Y%m%d_%H%M%S").strftime("%d %b, %H:%M:%S")
    except ValueError:
        return path.stem


def detection_table(detections) -> None:
    if not detections:
        return
    st.dataframe(pd.DataFrame([{
        "Face": i + 1, "Prediction": pretty(d.label), "Confidence": d.confidence,
        "Status": "❌ Violation" if d.is_violation else "✅ OK",
    } for i, d in enumerate(detections)]), hide_index=True, width="stretch",
        column_config={"Confidence": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                                     max_value=1)})


# --- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.title("😷 Mask Detector")
    st.caption("CNN (MobileNetV2) · OpenCV DNN face detector · LLM agents (Groq / Claude)")

    st.subheader("Detection")
    face_conf = st.slider("Face detection confidence", 0.2, 0.9, 0.5, 0.05,
                          help="Lower finds more (and smaller) faces but adds false detections.")
    log_enabled = st.toggle("Log detections to database", value=True)

    st.subheader("AI agents")
    voice_alerts = st.toggle("🔊 Voice alerts", value=True,
                             help="Speak 'Please wear a mask' and monitor alerts out loud.")
    monitor_on = st.toggle("🛡️ Auto-monitoring agent", value=True,
                           help="Watches compliance while the camera runs, raises alerts and "
                                "asks the AI for an hourly summary.")
load_dotenv(ROOT / ".env", override=True)  # pick up a key added while the app is running

if "monitor" not in st.session_state:
    from src.monitor import ComplianceMonitor
    st.session_state.monitor = ComplianceMonitor()
monitor = st.session_state.monitor
monitor.voice_alerts = voice_alerts

if not (MASK_MODEL_PATH.exists() or MASK_ONNX_PATH.exists()):
    st.error("No trained model found. Run `python scripts/download_dataset.py` then "
             "`python -m src.train` first.")
    st.stop()

detector = load_detector()
detector.face_confidence = face_conf  # applied per run, no model reload

with st.sidebar:
    st.divider()
    st.caption(f"Classes: {', '.join(pretty(l) for l in detector.labels)}  \n"
               f"Engine: {'⚡ ONNX Runtime' if detector.backend == 'onnx' else 'TensorFlow'}")

# Hero: teal panel with a stepped corner, a skyline and today's headline numbers.
SKYLINE = [38, 62, 48, 84, 55, 100, 70, 44, 90, 60, 118, 74, 50, 92, 66, 40, 80, 58, 96, 52]
buildings = "".join(f'<rect x="{i * 26}" y="{130 - h}" width="22" height="{h}" rx="2"/>'
                    f'<rect x="{i * 26 + 9}" y="{122 - h}" width="4" height="8"/>'
                    for i, h in enumerate(SKYLINE))
today = database.summary(24)
rate = today["compliance_rate"]
whole, frac = (f"{rate:.1f}".split(".") if rate is not None else ("--", ""))
stats = [(whole, f".{frac} %" if frac else "%", "Compliance · 24h"),
         (today["total_detections"], "", "Detections · 24h"),
         (today["violations"], "", "Violations · 24h")]
hero_html = f"""
<div class="hero">
  <div class="nav"><span>Real-time detection · Analytics · MaskGuard AI</span>
    <span class="pill">{'⚡ ONNX' if detector.backend == 'onnx' else 'TensorFlow'} · Ready</span></div>
  <div class="title">Face Mask<br>Detector</div>
  <div class="sub">Real-time mask compliance monitoring with a CNN classifier and an AI agent that
    explains the numbers.</div>
  <div class="stats">{"".join(f'<div class="stat"><div class="v">{v}<sup>{u}</sup></div><div class="k">{k}</div></div>'
                              for v, u, k in stats)}</div>
  <svg viewBox="0 0 520 130" preserveAspectRatio="xMaxYMax meet">
    <defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#3fe5b2"/><stop offset="1" stop-color="#3fe5b2" stop-opacity="0"/>
    </linearGradient></defs>
    <g fill="url(#sky)">{buildings}</g>
  </svg>
</div>"""
# strip indentation so Markdown never treats lines as code blocks
st.markdown(" ".join(line.strip() for line in hero_html.splitlines()), unsafe_allow_html=True)

tab_live, tab_image, tab_video, tab_dash, tab_ai, tab_model = st.tabs(
    ["🎥 Live", "🖼️ Image", "🎞️ Video", "📊 Dashboard", "🤖 AI Assistant", "🧠 Model"])

# --- Live webcam --------------------------------------------------------------
class LiveStats:
    """Shared between the WebRTC video thread (writes) and the page (reads)."""

    def __init__(self, labels):
        self.lock = threading.Lock()
        self.tracker = FaceTracker(labels)
        self.faces = self.bad = 0
        self.fps, self.t_prev, self.last_snap = 0.0, None, 0.0
        self.log_enabled = True


@st.cache_resource
def detect_lock() -> threading.Lock:
    return threading.Lock()   # the OpenCV face net isn't safe to run from several streams at once


def rtc_configuration() -> dict:
    """STUN is enough on most networks; set TURN_URL/TURN_USERNAME/TURN_CREDENTIAL for strict NATs."""
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    if os.getenv("TURN_URL"):
        servers.append({"urls": [os.environ["TURN_URL"]], "username": os.getenv("TURN_USERNAME", ""),
                        "credential": os.getenv("TURN_CREDENTIAL", "")})
    return {"iceServers": servers}


def render_stats(box, faces: int, bad: int, fps: float) -> None:
    with box.container(border=True):
        st.metric("Faces", faces)
        st.metric("Without mask", bad)
        st.metric("FPS", f"{fps:.1f}")


def live_alerts(n_bad: int, agent_box) -> None:
    """Voice + monitor-agent side effects; call from the page thread only."""
    now = time.time()
    if voice_alerts and n_bad and now - st.session_state.get("last_voice", 0.0) >= ALERT_COOLDOWN_SEC:
        st.session_state.last_voice = now
        voice.speak("Please wear a mask.")
    if monitor_on:
        for alert in monitor.tick():
            st.toast(f"🛡️ {alert.message}", icon="⚠️" if alert.kind != "summary" else "📝")
            agent_box.warning(f"**Monitor agent:** {alert.message}")


with tab_live:
    col_a, col_b = st.columns([3, 1], gap="large")
    with col_b:
        cam_mode = st.segmented_control(
            "Camera", ["🌐 Browser", "💻 Server"], default="🌐 Browser",
            help="**Browser** streams your device's camera (works when deployed). "
                 "**Server** opens a webcam plugged into the machine running the app (local only, "
                 "a bit faster).") or "🌐 Browser"
        if cam_mode == "💻 Server":
            run = st.toggle("**Start camera**")
            with st.expander("Camera settings"):
                cam_index = st.number_input("Camera index", 0, 5, 0,
                                            help="0 is the built-in webcam; try 1, 2... for USB cameras.")
        stats_box = st.empty()
        agent_box = st.empty()
    banner_box = col_a.empty()

    if cam_mode == "🌐 Browser":
        if "live" not in st.session_state:
            st.session_state.live = LiveStats(detector.labels)
        live = st.session_state.live
        live.log_enabled = log_enabled
        lock = detect_lock()

        def process(frame: av.VideoFrame) -> av.VideoFrame:
            img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
            with lock:
                dets = detector.detect(img, live.tracker)
            annotated = draw_detections(img.copy(), dets)
            now = time.time()
            with live.lock:
                if live.t_prev is not None:
                    inst = 1.0 / max(now - live.t_prev, 1e-6)
                    live.fps = inst if live.fps == 0.0 else 0.9 * live.fps + 0.1 * inst
                live.t_prev = now
                live.faces, live.bad = len(dets), sum(d.is_violation for d in dets)
                events = [d for d in dets if d.new_event]   # each person logged once
                snap = live.log_enabled and any(d.is_violation for d in events) \
                    and now - live.last_snap >= ALERT_COOLDOWN_SEC
                if snap:
                    live.last_snap = now
            if live.log_enabled and events:
                log_all(events, "webcam", annotated if snap else None)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        with col_a:
            ctx = webrtc_streamer(
                key="live-camera", mode=WebRtcMode.SENDRECV, video_frame_callback=process,
                rtc_configuration=rtc_configuration(), async_processing=True,
                media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                                          "audio": False},
                translations={"start": "▶ Start camera", "stop": "■ Stop camera",
                              "select_device": "Choose camera"})
        if ctx.state.playing:
            while ctx.state.playing:   # refresh the side panel while the stream runs
                with live.lock:
                    faces, bad, fps = live.faces, live.bad, live.fps
                banner(banner_box, faces, bad)
                render_stats(stats_box, faces, bad, fps)
                live_alerts(bad, agent_box)
                time.sleep(0.3)
        else:
            with live.lock:
                live.faces = live.bad = 0
                live.fps, live.t_prev = 0.0, None
                live.tracker.reset()
            banner_box.markdown('<div class="banner idle">Press <b>Start camera</b> and allow camera '
                                'access in your browser.</div>', unsafe_allow_html=True)
            col_b.caption("If the video stays black on a strict network (office / campus Wi-Fi), "
                          "the app needs a TURN server - see the README.")
    elif run:
        frame_box = col_a.empty()
        camera = Camera(int(cam_index))
        if not camera.ok:
            frame_box.error(f"Could not open camera {cam_index} on the server. On a deployed app use "
                            "**🌐 Browser** mode; locally, check no other app is using the webcam.")
        tracker = FaceTracker(detector.labels)
        last_snap = 0.0
        seq, fps = 0, 0.0
        t_prev = time.time()
        while camera.ok:
            seq, frame = camera.read(seq)
            if frame is None:
                break
            frame = cv2.flip(frame, 1)
            dets = detector.detect(frame, tracker)
            annotated = draw_detections(frame.copy(), dets)
            now = time.time()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst   # smoothed FPS readout
            t_prev = now
            n_bad = sum(d.is_violation for d in dets)
            banner(banner_box, len(dets), n_bad)
            show_frame(frame_box, annotated)
            render_stats(stats_box, len(dets), n_bad, fps)
            events = [d for d in dets if d.new_event]   # each person logged once
            if log_enabled and events:
                snap_frame = annotated if any(d.is_violation for d in events) \
                    and now - last_snap >= ALERT_COOLDOWN_SEC else None
                if snap_frame is not None:
                    last_snap = now
                log_all(events, "webcam", snap_frame)
            live_alerts(n_bad, agent_box)
        camera.release()
    else:
        with col_a.container(border=True):
            st.markdown("### 🎥 Server camera is off")
            st.markdown("Switch on **Start camera** to read the webcam attached to this machine. "
                        "People without a mask are highlighted in red, logged, and announced by voice.")
            st.caption("Need more speed? Run `python -m src.realtime` for a standalone OpenCV window.")

# --- Image --------------------------------------------------------------------
with tab_image:
    src = st.segmented_control("Source", ["📁 Upload", "📸 Take a photo"], default="📁 Upload",
                               label_visibility="collapsed")
    file = st.camera_input("Camera") if src == "📸 Take a photo" else \
        st.file_uploader("Drop a photo here", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if file is not None:
        img = decode_upload(file)
        dets = detector.detect(img)
        annotated = draw_detections(img.copy(), dets)
        banner(st, len(dets), sum(d.is_violation for d in dets))
        c1, c2 = st.columns(2)
        c1.image(to_rgb(img), caption="Input", width="stretch")
        c2.image(to_rgb(annotated), caption="Result", width="stretch")
        if not dets:
            st.caption("No face was found. If the photo is a close-up of one face, classify it directly:")
            if st.button("Classify whole image as a face"):
                label, conf = detector.classify_image(img)
                st.metric("Prediction", pretty(label), f"{conf:.1%} confidence", delta_color="off")
        detection_table(dets)
        if log_enabled and dets and st.button("💾 Log these detections"):
            log_all(dets, "image", annotated)
            st.toast("Detections logged to the dashboard.", icon="✅")
    elif src != "📸 Take a photo":
        st.caption("Supports JPG, PNG, BMP and WebP. Group photos work too.")

# --- Video --------------------------------------------------------------------
with tab_video:
    vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    stride = st.select_slider("Speed vs. detail", options=[1, 2, 3, 5, 10, 15, 30], value=5,
                              format_func=lambda n: "every frame" if n == 1 else f"every {n}th frame",
                              help="Analysing fewer frames is faster; every frame catches brief appearances.")
    if st.button("▶️ Run analysis", type="primary", disabled=vid is None):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vid.name)[1]) as tmp:
            tmp.write(vid.getvalue())
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        progress, preview = st.progress(0.0, text="Analysing..."), st.empty()
        tracker = FaceTracker(detector.labels, max_missing=max(2, 15 // stride))
        rows, events, idx = [], [], 0
        while True:
            ok = cap.grab()   # grab() skips decoding frames we won't analyse
            if not ok:
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                dets = detector.detect(frame, tracker)
                for d in dets:
                    rows.append({"time_s": round(idx / fps, 1), "label": pretty(d.label),
                                 "violation": d.is_violation, "confidence": d.confidence})
                events += [d for d in dets if d.new_event]
                if idx % (stride * 5) == 0:   # refresh the preview occasionally, not every frame
                    show_frame(preview, draw_detections(frame, dets))
                    progress.progress(min(idx / total, 1.0),
                                      text=f"Analysing... {idx / fps:.0f}s / {total / fps:.0f}s")
            idx += 1
        progress.progress(1.0, text="Analysis complete")
        if log_enabled and events:
            log_all(events, "video")
        cap.release()
        os.unlink(tmp.name)
        if rows:
            df = pd.DataFrame(rows)
            v = int(df["violation"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Face observations", len(df), f"{len(events)} unique people", delta_color="off",
                      border=True)
            m2.metric("Violations", v, border=True)
            m3.metric("Compliance", f"{100 * (len(df) - v) / len(df):.1f}%", border=True)
            st.markdown("**Violations over time**")
            st.area_chart(df.groupby("time_s")["violation"].sum(), x_label="time (s)",
                          y_label="violations", color="#d33c3c")
            st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False),
                               file_name=f"{os.path.splitext(vid.name)[0]}_masks.csv", mime="text/csv")
        else:
            st.info("No faces found in the video.")

# --- Dashboard ----------------------------------------------------------------
with tab_dash:
    head, ctrl = st.columns([3, 2], vertical_alignment="bottom")
    head.subheader("Compliance analytics")
    window = ctrl.segmented_control("Time window", ["1h", "24h", "7d", "All"], default="24h",
                                    label_visibility="collapsed") or "24h"
    hours = {"1h": 1, "24h": 24, "7d": 168, "All": None}[window]
    s = database.summary(hours)
    rate = s["compliance_rate"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Detections", s["total_detections"], border=True)
    m2.metric("Violations", s["violations"], border=True)
    m3.metric("Compliance rate", f"{rate}%" if rate is not None else "--",
              f"{rate - MONITOR_THRESHOLD:+.1f} pts vs {MONITOR_THRESHOLD}% target" if rate is not None else None,
              border=True)

    recent = database.recent(100)
    if not recent:
        with st.container(border=True):
            st.markdown("### 📭 No detections logged yet")
            st.markdown("Start the camera on the **🎥 Live** tab, or analyse a photo or video. "
                        "Charts and events will appear here.")

    alerts = database.recent_alerts(20)
    if alerts:
        with st.expander(f"🛡️ Monitoring agent alerts ({len(alerts)})", expanded=False):
            st.dataframe(pd.DataFrame(alerts)[["ts", "kind", "message"]], hide_index=True,
                         width="stretch", column_config={"ts": "Time", "kind": "Type", "message": "Alert"})

    trend = database.hourly(hours)
    if trend:
        df = pd.DataFrame(trend)
        df["compliant"] = df["total"] - df["violations"]
        st.markdown("**Detections per hour**")
        st.bar_chart(df.set_index("hour")[["compliant", "violations"]], color=["#2a8a7a", "#d33c3c"])

    if recent:
        events_df = pd.DataFrame(recent).drop(columns=["snapshot"])
        events_df["label"] = events_df["label"].map(pretty)
        st.markdown("**Recent events**")
        st.dataframe(events_df, hide_index=True, width="stretch", height=300, column_config={
            "id": None,
            "ts": st.column_config.TextColumn("Time"),
            "label": "Prediction",
            "confidence": st.column_config.ProgressColumn("Confidence", format="percent",
                                                          min_value=0, max_value=1),
            "is_violation": st.column_config.CheckboxColumn("Violation"),
            "source": "Source",
        })

    snaps = sorted(SNAPSHOTS_DIR.glob("*.jpg"), reverse=True)[:8]
    if snaps:
        st.markdown("**Latest violation snapshots**")
        cols = st.columns(4)
        for i, p in enumerate(snaps):
            cols[i % 4].image(str(p), caption=snapshot_caption(p), width="stretch")

    if recent:
        a1, a2, _ = st.columns([1, 1, 3])
        a1.download_button("⬇️ Export CSV", pd.DataFrame(recent).to_csv(index=False),
                           file_name="detections.csv", mime="text/csv", width="stretch")
        with a2.popover("🗑️ Clear log", width="stretch"):
            st.markdown("Delete **all** detections and alerts? This can't be undone.")
            if st.button("Yes, clear everything", type="primary"):
                database.clear()
                st.rerun()

# --- AI assistant -------------------------------------------------------------
with tab_ai:
    from src.agent import llm_available
    if not llm_available():
        st.subheader("🤖 Set up MaskGuard AI")
        st.markdown("The assistant answers questions about your compliance data and writes reports. "
                    "It needs one API key:")
        k1, k2 = st.columns(2)
        with k1.container(border=True):
            st.markdown("**Groq** · free  \n[console.groq.com/keys](https://console.groq.com/keys) → "
                        "**Create API Key** (starts with `gsk_`). Uses `openai/gpt-oss-120b`.")
        with k2.container(border=True):
            st.markdown("**Anthropic** · Claude  \n[console.anthropic.com](https://console.anthropic.com/settings/keys) "
                        "→ add billing credit → **Create Key** (starts with `sk-ant-`).")
        with st.form("api_key_form"):
            key = st.text_input("API key", type="password", placeholder="gsk_... or sk-ant-...")
            st.caption("🔒 Saved only to the local `.env` file on this computer.")
            if st.form_submit_button("Save key", type="primary"):
                key = key.strip()
                kinds = {"gsk_": ("GROQ_API_KEY", "groq"), "sk-ant-": ("ANTHROPIC_API_KEY", "claude")}
                match = next((v for prefix, v in kinds.items() if key.startswith(prefix)), None)
                if not match:
                    st.error("Unrecognised key - Groq keys start with `gsk_`, Anthropic keys with `sk-ant-`.")
                else:
                    var, prov = match
                    env_path = ROOT / ".env"
                    lines = [l for l in (env_path.read_text().splitlines() if env_path.exists() else [])
                             if not l.startswith((f"{var}=", "LLM_PROVIDER="))]
                    env_path.write_text("\n".join(lines + [f"{var}={key}", f"LLM_PROVIDER={prov}"]) + "\n")
                    os.environ[var], os.environ["LLM_PROVIDER"] = key, prov
                    st.rerun()
    else:
        from src.agent import MaskGuardAgent, describe_api_error

        if "agent" not in st.session_state:
            st.session_state.agent = MaskGuardAgent()
            st.session_state.chat = []

        h1, h2, h3 = st.columns([3, 1.3, 1], vertical_alignment="center")
        h1.subheader("MaskGuard AI")
        h1.caption(f"Ask about your compliance data · {st.session_state.agent.label}")
        speak_answers = h2.toggle("🔊 Read answers aloud", value=True)
        if h3.button("↺ New chat", width="stretch", disabled=not st.session_state.chat):
            st.session_state.agent.reset()
            st.session_state.chat = []
            st.rerun()

        presets = {
            "📝 Daily compliance report": "Write today's compliance report and save it.",
            "⏰ Peak violation times": "When do most violations happen? Use the hourly trend.",
            "🔍 Check for false positives": "Inspect the latest few violation snapshots and tell me "
                                            "if any look like false positives.",
        }
        def use_preset():   # act like a button: send once, then clear the selection
            st.session_state.preset = presets[st.session_state.preset_pill]
            st.session_state.preset_pill = None

        st.pills("Suggestions", list(presets), label_visibility="collapsed", key="preset_pill",
                 on_change=use_preset)
        preset = st.session_state.pop("preset", None)

        recording = st.audio_input("🎤 Ask by voice")
        voice_prompt = None
        if recording is not None:
            clip = recording.getvalue()
            clip_id = hash(clip)
            if st.session_state.get("last_clip") != clip_id:   # process each recording once
                st.session_state.last_clip = clip_id
                try:
                    voice_prompt = voice.transcribe(clip)
                except Exception as exc:
                    st.error(str(exc))

        if not st.session_state.chat:
            st.chat_message("assistant").markdown(
                "Hi! I'm **MaskGuard**. I can summarise compliance, find peak violation hours, "
                "check snapshots for false positives and write reports. Pick a suggestion above "
                "or ask me anything.")
        for role, text in st.session_state.chat:
            st.chat_message(role).markdown(text)

        prompt = st.chat_input("Ask anything about mask compliance...") or preset or voice_prompt
        if prompt:
            st.session_state.chat.append(("user", f"🎤 {prompt}" if prompt == voice_prompt else prompt))
            st.chat_message("user").markdown(prompt)
            with st.chat_message("assistant"):
                with st.status("Thinking...", expanded=False) as status:
                    try:
                        answer = st.session_state.agent.chat(
                            prompt, on_tool=lambda n, a: status.write(f"🔧 `{n}` {json.dumps(a)[:80]}"))
                    except Exception as exc:
                        answer = describe_api_error(exc)
                    status.update(label="Done", state="complete")
                st.markdown(answer)
                if speak_answers:
                    wav = voice.synthesize_wav(answer)
                    if wav:
                        st.audio(wav, format="audio/wav", autoplay=True)
            st.session_state.chat.append(("assistant", answer))

# --- Model --------------------------------------------------------------------
with tab_model:
    st.subheader("Model performance (validation set)")
    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        rep = m["report"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Architecture", m["arch"].replace("mobilenet", "MobileNetV2"), border=True)
        k2.metric("Accuracy", f"{rep['accuracy']:.1%}", border=True)
        k3.metric("Macro F1", f"{rep['macro avg']['f1-score']:.1%}", border=True)
        k4.metric("Validation images", int(rep["macro avg"]["support"]), border=True)
        with st.expander("Per-class report"):
            report = pd.DataFrame({k: v for k, v in rep.items() if isinstance(v, dict)}).T.round(4)
            report.index = [pretty(i) if i in m["classes"] else i for i in report.index]
            st.dataframe(report, width="stretch")
    else:
        st.info("No metrics yet - train a model with `python -m src.train`.")
    c1, c2 = st.columns(2)
    for col, name, title in ((c1, "training_curves.png", "Training curves"),
                             (c2, "confusion_matrix.png", "Confusion matrix")):
        if (REPORTS_DIR / name).exists():
            col.markdown(f"**{title}**")
            col.image(str(REPORTS_DIR / name), width="stretch")
    saved_reports = sorted(REPORTS_DIR.glob("*.md"), reverse=True)
    if saved_reports:
        st.divider()
        st.subheader("AI-generated reports")
        pick = st.selectbox("Report", saved_reports, format_func=lambda p: p.name)
        with st.container(border=True):
            st.markdown(pick.read_text(encoding="utf-8"))
