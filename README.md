# 😷 Face Mask Detector — Deep Learning + AI Agent

Real-time face mask detection with a CNN, plus an LLM agent (Claude) that analyses
compliance data and writes reports.

## Architecture

```
Webcam / video / image
        │
        ▼
OpenCV DNN face detector (ResNet-10 SSD)  ──► face crops
        │
        ▼
CNN mask classifier (MobileNetV2 transfer learning, or custom CNN)
        │
        ├──► Live overlay: boxes, confidence, FPS, compliance %, "PLEASE WEAR A MASK" banner
        ├──► Alerts: beep + snapshot of violators (with cooldown)
        └──► SQLite event log (detections.db)
                    │
                    ├──► Streamlit dashboard: charts, recent events, snapshot gallery
                    └──► MaskGuard AI (Claude agent with tools):
                           Q&A over the logs, peak-hour analysis, visual
                           false-positive checks on snapshots, Markdown reports
```

| Component | Tech |
|---|---|
| Face detection | OpenCV DNN, ResNet-10 SSD (auto-downloaded, works on masked faces better than Haar) |
| Mask classifier | TensorFlow/Keras CNN — MobileNetV2 (ImageNet) + fine-tuning, or a custom 8-layer CNN |
| Training extras | In-model data augmentation, class weighting, early stopping, LR scheduling, confusion matrix & curves |
| AI agent | Claude (`claude-opus-5-5`) with tool use + vision |
| UI | OpenCV window (fastest) and Streamlit web dashboard |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Get the dataset

```powershell
python scripts/download_dataset.py
```

Downloads ~1,376 labelled images into `dataset/with_mask` and `dataset/without_mask`.
Any dataset laid out as `dataset/<class_name>/*.jpg` works — e.g. add a
`mask_weared_incorrect` folder for 3-class detection (shown in orange).
For better accuracy, use a bigger set such as Kaggle's *Face Mask Dataset* (7.5k images).

## 2. Train the CNN

```powershell
python -m src.train                    # MobileNetV2 transfer learning + 5 fine-tuning epochs
python -m src.train --arch cnn --epochs 30   # custom CNN from scratch, for comparison
```

Outputs: `models/mask_detector.keras`, `models/labels.json`,
`reports/training_curves.png`, `reports/confusion_matrix.png`, `reports/metrics.json`.

### Performance (ONNX Runtime)

Training also exports `models/mask_detector.onnx` (or run `python -m src.export`). The apps use it
automatically, so they don't need TensorFlow at runtime. Measured on a laptop CPU (640×480 frames,
face detection + classification):

| | TensorFlow/Keras | ONNX Runtime |
|---|---|---|
| Model load time | 9.1 s | 0.4 s |
| RAM | 851 MB peak | 180 MB peak |
| Per frame | 185 ms (~5 FPS) | 36 ms (~28 FPS) |
| Predictions | — | identical (100% agreement) |

Other efficiency features: threaded camera capture (always processes the newest frame),
an IoU face tracker with smoothed predictions (no label flicker, each person logged once
instead of every frame), batched SQLite writes on a reused WAL-mode connection, and JPEG
frame streaming in the dashboard.

## 3. Real-time detection

```powershell
python -m src.realtime                       # webcam
python -m src.realtime --source video.mp4 --save annotated.mp4
```

Keys: `q` quit · `p` pause · `s` snapshot. Violations beep and save a snapshot to `snapshots/`;
detections are sampled every 2 s into `detections.db`.

## 4. Dashboard

```powershell
streamlit run app.py
```

Tabs: **Live** webcam · **Image** (upload or take a photo) · **Video** file analysis ·
**Dashboard** (compliance rate, hourly chart, events, snapshots) · **AI Assistant** · **Model** metrics.

## 5. MaskGuard AI (LLM agent)

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. Then use the dashboard's
**AI Assistant** tab or the terminal:

```powershell
python -m src.agent
python -m src.agent "Write today's compliance report"
```

Example questions:
- "What was the compliance rate over the last hour?"
- "When do most violations happen?"
- "Check the latest violation snapshots for false positives."
- "Write a weekly compliance report with recommendations."

The agent's tools: `get_compliance_summary`, `get_hourly_trend`, `get_recent_detections`,
`inspect_snapshot` (Claude looks at the image), `get_model_metrics`, `save_report`
(saves Markdown into `reports/`).

## 6. Voice assistant

- **Spoken alerts:** "Please wear a mask" when someone has no mask (dashboard sidebar toggle,
  or `python -m src.realtime --voice`). Uses the built-in Windows voice, offline.
- **Ask by voice:** 🎤 recorder in the AI Assistant tab, transcribed by SpeechRecognition (needs internet).
- **Spoken answers:** "Read answers aloud" toggle plays the agent's reply.

## 7. Auto-monitoring agent

Watches the detection log without being asked and raises alerts (shown on screen, spoken,
stored in the `alerts` table, visible in the Dashboard tab and to the chat agent):

| Alert | Rule (edit in `config.py`) |
|---|---|
| Low compliance | compliance < 80% over the last 10 min (min 5 detections) |
| Persistent violations | ≥ 5 violations within 2 min |
| Automatic summary | every 60 min Claude writes a report into `reports/` and speaks a 2-line summary |

It runs while the dashboard camera is on (sidebar toggle), with `python -m src.realtime --monitor`,
or standalone 24/7: `python -m src.monitor --voice`.

## Project layout

```
config.py                 settings & paths
app.py                    Streamlit dashboard
scripts/download_dataset.py
src/model.py              CNN architectures (MobileNetV2, custom CNN)
src/train.py              training + evaluation
src/detector.py           face detection + mask classification pipeline
src/realtime.py           OpenCV real-time app with alerts
src/database.py           SQLite event log
src/agent.py              Claude agent with tools
src/monitor.py            auto-monitoring agent (alerts + hourly Claude summaries)
src/voice.py              text-to-speech / speech-to-text
```

## Ideas to extend
- Add `mask_weared_incorrect` class (dataset folder) — no code changes needed.
- Object tracking (e.g. centroid/SORT) to count unique people instead of samples.
- Email/Slack alerts from the agent when compliance drops below a threshold.
- Export the model to TensorFlow Lite for Raspberry Pi / mobile.
