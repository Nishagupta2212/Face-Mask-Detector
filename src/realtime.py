"""Real-time mask detection from a webcam or video file.

Usage:
    python -m src.realtime                     # default webcam
    python -m src.realtime --source 1          # another camera
    python -m src.realtime --source clip.mp4 --save out.mp4
    python -m src.realtime --voice --monitor   # spoken alerts + auto-monitoring agent

Keys:  q = quit   p = pause   s = save snapshot
"""
import argparse
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2

from config import ALERT_COOLDOWN_SEC, SNAPSHOTS_DIR
from src import database, voice
from src.camera import Camera
from src.detector import FaceTracker, MaskDetector, draw_detections


def beep() -> None:
    if sys.platform == "win32":
        import winsound
        threading.Thread(target=winsound.Beep, args=(1200, 300), daemon=True).start()
    else:
        print("\a", end="", flush=True)


def save_snapshot(frame, prefix: str) -> str:
    path = SNAPSHOTS_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def draw_hud(frame, fps: float, n_faces: int, n_violations: int, session: dict,
             agent_msg: str = "") -> None:
    total = session["total"]
    rate = f"{100 * (total - session['violations']) / total:.0f}%" if total else "--"
    lines = [
        f"FPS: {fps:.1f}",
        f"Faces: {n_faces}  Violations: {n_violations}",
        f"Session compliance: {rate}",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (290, 26 * len(lines) + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 26 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
    if n_violations:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        cv2.putText(frame, "PLEASE WEAR A MASK", (w // 2 - 160, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    if agent_msg:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 90), (w, h - 55), (0, 90, 200), -1)
        cv2.putText(frame, f"AGENT: {agent_msg[:70]}", (10, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time face mask detection")
    parser.add_argument("--source", default="0", help="camera index or video file path")
    parser.add_argument("--save", help="write annotated video to this path")
    parser.add_argument("--no-log", action="store_true", help="don't log to the database")
    parser.add_argument("--no-sound", action="store_true", help="disable alert beep / voice")
    parser.add_argument("--voice", action="store_true", help="spoken alerts instead of a beep")
    parser.add_argument("--monitor", action="store_true", help="run the auto-monitoring agent")
    args = parser.parse_args()

    monitor = None
    if args.monitor:
        from src.monitor import ComplianceMonitor
        monitor = ComplianceMonitor(voice_alerts=args.voice and not args.no_sound)
    agent_msg, agent_msg_until = "", 0.0

    detector = MaskDetector()
    source = int(args.source) if args.source.isdigit() else args.source
    src_kind = "webcam" if isinstance(source, int) else "video"
    tracker = FaceTracker(detector.labels)
    if src_kind == "webcam":
        camera, cap = Camera(source), None   # threaded: always process the newest frame
        if not camera.ok:
            raise SystemExit(f"Could not open camera {source}")
    else:
        camera, cap = None, cv2.VideoCapture(source)  # files: read every frame in order
        if not cap.isOpened():
            raise SystemExit(f"Could not open video source: {args.source}")
    print(f"Mask classifier backend: {detector.backend}")

    writer = None
    frame_times: deque[float] = deque(maxlen=30)
    session = {"total": 0, "violations": 0}
    last_alert = 0.0
    seq = 0
    paused = False
    frame = None
    print("Running - press 'q' to quit, 'p' to pause, 's' to save a snapshot.")

    while True:
        if not paused:
            if camera:
                seq, raw = camera.read(seq)
                if raw is None:
                    break
                raw = cv2.flip(raw, 1)  # mirror view
            else:
                ok, raw = cap.read()
                if not ok:
                    break
            detections = detector.detect(raw, tracker)
            frame = draw_detections(raw.copy(), detections)

            now = time.time()
            frame_times.append(now)
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) \
                if len(frame_times) > 1 else 0.0
            violations = [d for d in detections if d.is_violation]
            if monitor:
                for alert in monitor.tick():
                    agent_msg, agent_msg_until = alert.message, now + 8
            draw_hud(frame, fps, len(detections), len(violations), session,
                     agent_msg if now < agent_msg_until else "")

            snapshot = None
            if violations and now - last_alert >= ALERT_COOLDOWN_SEC:
                last_alert = now
                snapshot = save_snapshot(frame, "violation")
                if args.no_sound:
                    pass
                elif args.voice:
                    voice.speak("Please wear a mask.")
                else:
                    beep()

            # The tracker flags each person once (and again only if their mask status changes).
            events = [d for d in detections if d.new_event]
            if events:
                session["total"] += len(events)
                session["violations"] += sum(d.is_violation for d in events)
                if not args.no_log:
                    database.log_detections([
                        (d.label, d.confidence, d.is_violation, src_kind,
                         snapshot if d.is_violation else None) for d in events])

            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             (cap.get(cv2.CAP_PROP_FPS) if cap else 0) or 20, (w, h))
                writer.write(frame)

        cv2.imshow("Face Mask Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        window_closed = cv2.getWindowProperty("Face Mask Detector", cv2.WND_PROP_VISIBLE) < 1
        if key == ord("q") or window_closed:
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("s") and frame is not None:
            print(f"Saved {save_snapshot(frame, 'manual')}")

    (camera or cap).release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Session: {session['total']} people logged, {session['violations']} violations.")


if __name__ == "__main__":
    main()
