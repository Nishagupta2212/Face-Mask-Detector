"""Central configuration for the Face Mask Detector project."""
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- Paths -------------------------------------------------------------------
DATA_DIR = ROOT / "dataset"              # dataset/<class_name>/*.jpg
MODELS_DIR = ROOT / "models"
FACE_DETECTOR_DIR = MODELS_DIR / "face_detector"
MASK_MODEL_PATH = MODELS_DIR / "mask_detector.keras"
MASK_ONNX_PATH = MODELS_DIR / "mask_detector.onnx"   # fast runtime copy (python -m src.export)
LABELS_PATH = MODELS_DIR / "labels.json"
REPORTS_DIR = ROOT / "reports"           # training plots, metrics, AI reports
SNAPSHOTS_DIR = ROOT / "snapshots"       # saved frames of people without masks
DB_PATH = ROOT / "detections.db"

for _d in (MODELS_DIR, FACE_DETECTOR_DIR, REPORTS_DIR, SNAPSHOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Face detector (OpenCV DNN, ResNet-10 SSD) ---------------------------------
FACE_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
FACE_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)
FACE_PROTO_PATH = FACE_DETECTOR_DIR / "deploy.prototxt"
FACE_MODEL_PATH = FACE_DETECTOR_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
FACE_CONFIDENCE = 0.5

# --- Mask classifier (CNN) ----------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
MASK_THRESHOLD = 0.5   # minimum class probability to accept a prediction

# --- Real-time behaviour ------------------------------------------------------
ALERT_COOLDOWN_SEC = 5      # min seconds between alerts / snapshots

# --- Auto-monitoring agent ------------------------------------------------------
MONITOR_CHECK_SEC = 30            # how often the monitor evaluates the log
MONITOR_THRESHOLD = 80            # alert when compliance (%) drops below this...
MONITOR_WINDOW_MIN = 10           # ...over this rolling window
MONITOR_MIN_SAMPLES = 5           # ignore windows with fewer detections than this
PERSIST_WINDOW_MIN = 2            # "persistent violations": many violations in a short window
PERSIST_COUNT = 3                 # (unique people - the tracker logs each person once)
ALERT_REPEAT_MIN = 5              # don't repeat the same alert type more often than this
SUMMARY_EVERY_MIN = 60            # Claude writes an automatic summary this often

# --- LLM agent (Claude) -------------------------------------------------------
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")

# --- LLM agent (Groq alternative) -----------------------------------------------
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
