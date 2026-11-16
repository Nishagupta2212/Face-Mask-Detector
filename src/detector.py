"""Face detection (OpenCV DNN SSD) + mask classification (CNN) pipeline.

The classifier runs on ONNX Runtime when models/mask_detector.onnx exists (python -m src.export),
otherwise on TensorFlow/Keras.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from config import (
    FACE_CONFIDENCE, FACE_MODEL_PATH, FACE_MODEL_URL, FACE_PROTO_PATH, FACE_PROTO_URL,
    IMG_SIZE, LABELS_PATH, MASK_MODEL_PATH, MASK_ONNX_PATH, MASK_THRESHOLD,
)


@dataclass
class Detection:
    box: tuple[int, int, int, int]       # x1, y1, x2, y2
    label: str
    confidence: float                    # classifier probability for `label`
    face_confidence: float
    track_id: int | None = None
    new_event: bool = True               # False for repeat sightings of an already-logged track

    @property
    def is_violation(self) -> bool:
        return is_violation(self.label)


def is_violation(label: str) -> bool:
    """Anything other than a properly worn mask counts as a violation."""
    name = label.lower()
    return "without" in name or "no_mask" in name or "incorrect" in name


def label_color(label: str) -> tuple[int, int, int]:
    name = label.lower()
    if "incorrect" in name:
        return (0, 165, 255)             # orange (BGR)
    return (0, 0, 255) if is_violation(label) else (0, 200, 0)


def pretty(label: str) -> str:
    return label.replace("_", " ").title()


def ensure_face_detector() -> None:
    for url, path in ((FACE_PROTO_URL, FACE_PROTO_PATH), (FACE_MODEL_URL, FACE_MODEL_PATH)):
        if not path.exists():
            print(f"Downloading face detector: {path.name} ...")
            urllib.request.urlretrieve(url, path)


def _load_classifier():
    """Prefer the ONNX model (fast, no TensorFlow); fall back to Keras."""
    if MASK_ONNX_PATH.exists() and (not MASK_MODEL_PATH.exists()
                                    or MASK_ONNX_PATH.stat().st_mtime >= MASK_MODEL_PATH.stat().st_mtime):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(MASK_ONNX_PATH), opts, providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            return (lambda batch: sess.run(None, {name: batch})[0]), "onnx"
        except ImportError:
            pass
    if not MASK_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MASK_MODEL_PATH}. Train first: python -m src.train")
    import tensorflow as tf  # slow, memory-hungry import - only when ONNX is unavailable
    model = tf.keras.models.load_model(MASK_MODEL_PATH)
    return (lambda batch: model(batch, training=False).numpy()), "keras"


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


class FaceTracker:
    """Lightweight IoU tracker: stable IDs, smoothed predictions, one event per person.

    - Smoothing (EMA of class probabilities) stops labels flickering between frames.
    - A track emits `new_event=True` once it has been seen `min_hits` frames, and again only
      if its smoothed label changes - so each person is logged once, not 15 times a second.
    """

    def __init__(self, labels: list[str], alpha: float = 0.35, iou_threshold: float = 0.3,
                 max_missing: int = 15, min_hits: int = 3):
        self.labels, self.alpha, self.iou_threshold = labels, alpha, iou_threshold
        self.max_missing, self.min_hits = max_missing, min_hits
        self.tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(self, faces, probs) -> list[Detection]:
        pairs = sorted(((_iou(t["box"], f[0]), tid, i) for tid, t in self.tracks.items()
                        for i, f in enumerate(faces)), reverse=True)
        matched_tracks, matched_faces = set(), set()
        for iou, tid, i in pairs:
            if iou < self.iou_threshold:
                break
            if tid in matched_tracks or i in matched_faces:
                continue
            t = self.tracks[tid]
            t["box"], t["face_conf"] = faces[i][0], faces[i][1]
            t["probs"] = self.alpha * probs[i] + (1 - self.alpha) * t["probs"]
            t["hits"] += 1
            t["missing"] = 0
            matched_tracks.add(tid)
            matched_faces.add(i)
        for i, (box, face_conf) in enumerate(faces):
            if i not in matched_faces:
                self.tracks[self._next_id] = {"box": box, "face_conf": face_conf, "probs": probs[i].copy(),
                                              "hits": 1, "missing": 0, "logged_label": None}
                matched_tracks.add(self._next_id)
                self._next_id += 1
        for tid in list(self.tracks):
            if tid not in matched_tracks:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]

        detections = []
        for tid in matched_tracks:
            t = self.tracks[tid]
            idx = int(np.argmax(t["probs"]))
            if t["probs"][idx] < MASK_THRESHOLD:
                continue
            label = self.labels[idx]
            new_event = t["hits"] >= self.min_hits and label != t["logged_label"]
            if new_event:
                t["logged_label"] = label
            detections.append(Detection(t["box"], label, float(t["probs"][idx]), t["face_conf"],
                                        track_id=tid, new_event=new_event))
        return detections

    def reset(self) -> None:
        self.tracks.clear()


class MaskDetector:
    def __init__(self, face_confidence: float = FACE_CONFIDENCE):
        ensure_face_detector()
        self.face_net = cv2.dnn.readNetFromCaffe(str(FACE_PROTO_PATH), str(FACE_MODEL_PATH))
        self._predict, self.backend = _load_classifier()
        self.labels: list[str] = json.loads(LABELS_PATH.read_text())
        self.face_confidence = face_confidence

    def detect_faces(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        out = self.face_net.forward()
        faces = []
        for i in range(out.shape[2]):
            conf = float(out[0, 0, i, 2])
            if conf < self.face_confidence:
                continue
            x1, y1, x2, y2 = (out[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            faces.append(((x1, y1, x2, y2), conf))
        return faces

    def classify(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
            for f in faces_bgr
        ]).astype("float32")
        return self._predict(batch)

    def detect(self, frame: np.ndarray, tracker: FaceTracker | None = None) -> list[Detection]:
        """Detect + classify faces. Pass a FaceTracker for video to get smoothed, tracked results."""
        faces = self.detect_faces(frame)
        crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2), _ in faces]
        probs = self.classify(crops) if crops else np.empty((0, len(self.labels)), "float32")
        if tracker is not None:
            return tracker.update(faces, probs)
        detections = []
        for (box, face_conf), p in zip(faces, probs):
            idx = int(np.argmax(p))
            if p[idx] < MASK_THRESHOLD:
                continue
            detections.append(Detection(box, self.labels[idx], float(p[idx]), face_conf))
        return detections

    def classify_image(self, image_bgr: np.ndarray) -> tuple[str, float]:
        """Classify a whole image (e.g. an already-cropped face) without face detection."""
        p = self.classify([image_bgr])[0]
        idx = int(np.argmax(p))
        return self.labels[idx], float(p[idx])


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for d in detections:
        x1, y1, x2, y2 = d.box
        color = label_color(d.label)
        text = f"{pretty(d.label)}: {d.confidence * 100:.1f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1, th + 8)
        cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 6, ty), color, -1)
        cv2.putText(frame, text, (x1 + 3, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
    return frame
