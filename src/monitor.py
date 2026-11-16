"""Auto-monitoring agent: watches the detection log and acts on its own.

- Low compliance:        compliance over the rolling window drops below the threshold
- Persistent violations: many violations within a couple of minutes
- Hourly summary:        the LLM agent (MaskGuard AI, Claude or Groq) writes and saves a short report

Alerts are stored in the `alerts` table, printed, and optionally spoken.
Runs inside the real-time app / dashboard (call tick() often), or standalone:

    python -m src.monitor --voice
"""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass

from config import (
    ALERT_REPEAT_MIN, MONITOR_CHECK_SEC, MONITOR_MIN_SAMPLES, MONITOR_THRESHOLD,
    MONITOR_WINDOW_MIN, PERSIST_COUNT, PERSIST_WINDOW_MIN, SUMMARY_EVERY_MIN,
)
from src import database, voice


@dataclass
class Alert:
    kind: str
    message: str
    spoken: str


def llm_available() -> bool:
    from src.agent import llm_available as _available
    return _available()


class ComplianceMonitor:
    def __init__(self, voice_alerts: bool = False, use_llm: bool = True,
                 threshold: float = MONITOR_THRESHOLD, check_every: float = MONITOR_CHECK_SEC):
        self.voice_alerts = voice_alerts
        self.use_llm = use_llm
        self.threshold = threshold
        self.check_every = check_every
        self._last_check = 0.0
        self._last_alert: dict[str, float] = {}
        self._last_summary = time.time()
        self._summary_thread: threading.Thread | None = None
        self.pending: list[Alert] = []   # alerts produced by background work (summaries)

    def _cooldown_ok(self, kind: str, now: float) -> bool:
        return now - self._last_alert.get(kind, 0.0) >= ALERT_REPEAT_MIN * 60

    def _emit(self, alert: Alert, now: float) -> Alert:
        self._last_alert[alert.kind] = now
        database.log_alert(alert.kind, alert.message)
        print(f"[monitor] {alert.kind}: {alert.message}")
        if self.voice_alerts and alert.spoken:
            voice.speak(alert.spoken, interrupt=True)
        return alert

    def tick(self) -> list[Alert]:
        """Cheap to call every frame - only evaluates every `check_every` seconds."""
        now = time.time()
        alerts, self.pending = self.pending, []
        if now - self._last_check < self.check_every:
            return alerts
        self._last_check = now

        window = database.summary(MONITOR_WINDOW_MIN / 60)
        rate = window["compliance_rate"]
        if (window["total_detections"] >= MONITOR_MIN_SAMPLES and rate is not None
                and rate < self.threshold and self._cooldown_ok("low_compliance", now)):
            alerts.append(self._emit(Alert(
                "low_compliance",
                f"Compliance dropped to {rate}% over the last {MONITOR_WINDOW_MIN} min "
                f"({window['violations']} of {window['total_detections']} detections without a mask).",
                f"Attention. Mask compliance has dropped to {rate:.0f} percent. Please wear your masks.",
            ), now))

        recent = database.summary(PERSIST_WINDOW_MIN / 60)
        if recent["violations"] >= PERSIST_COUNT and self._cooldown_ok("persistent_violations", now):
            alerts.append(self._emit(Alert(
                "persistent_violations",
                f"{recent['violations']} violations in the last {PERSIST_WINDOW_MIN} minutes - "
                "someone is repeatedly not wearing a mask.",
                "Repeated mask violations detected. Please put on your mask now.",
            ), now))

        if (self.use_llm and llm_available() and now - self._last_summary >= SUMMARY_EVERY_MIN * 60
                and not (self._summary_thread and self._summary_thread.is_alive())):
            self._last_summary = now
            if database.summary(SUMMARY_EVERY_MIN / 60)["total_detections"]:
                self._summary_thread = threading.Thread(target=self._write_summary, daemon=True)
                self._summary_thread.start()
        return alerts

    def _write_summary(self) -> None:
        """Ask the LLM agent for an automatic report. Runs in a background thread."""
        from src.agent import MaskGuardAgent, describe_api_error
        try:
            text = MaskGuardAgent().chat(
                f"Automatic check-in: summarise mask compliance for the last {SUMMARY_EVERY_MIN} "
                "minutes (compliance rate, peak violation times, recent alerts) with 2-3 practical "
                "recommendations. Save it with save_report titled 'Automatic summary', then reply "
                "with a 2-sentence spoken-style summary only.")
        except Exception as exc:
            text = describe_api_error(exc)
            self.pending.append(Alert("summary_error", text, ""))
            database.log_alert("summary_error", text)
            return
        alert = Alert("summary", text, text)
        database.log_alert(alert.kind, alert.message)
        if self.voice_alerts:
            voice.speak(alert.spoken)
        self.pending.append(alert)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-monitoring agent for mask compliance")
    parser.add_argument("--voice", action="store_true", help="speak alerts out loud")
    parser.add_argument("--no-llm", action="store_true", help="disable Claude hourly summaries")
    parser.add_argument("--threshold", type=float, default=MONITOR_THRESHOLD)
    args = parser.parse_args()

    monitor = ComplianceMonitor(voice_alerts=args.voice, use_llm=not args.no_llm,
                                threshold=args.threshold)
    print(f"Monitoring detections.db (threshold {args.threshold}%, AI summaries "
          f"{'on' if monitor.use_llm and llm_available() else 'off'}). Ctrl+C to stop.")
    try:
        while True:
            monitor.tick()
            time.sleep(2)
    except KeyboardInterrupt:
        print("Monitor stopped.")


if __name__ == "__main__":
    main()
