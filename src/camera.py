"""Threaded video capture: grabs frames in the background so processing never waits on the
camera and always works on the newest frame (no growing lag when inference is slower than FPS)."""
from __future__ import annotations

import sys
import threading

import cv2


def open_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)   # opens much faster than MSMF on Windows
    else:
        cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)          # 640x480 is plenty for face detection
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class Camera:
    """Live camera reader. Video files should be read directly (every frame matters there)."""

    def __init__(self, index: int = 0):
        self.cap = open_capture(index)
        self.ok = self.cap.isOpened()
        self._frame = None
        self._seq = 0
        self._cond = threading.Condition()
        self._running = self.ok
        if self.ok:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            with self._cond:
                if not ok:
                    self._running = False
                else:
                    self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def read(self, last_seq: int = 0, timeout: float = 2.0):
        """Return (seq, frame) for a frame newer than last_seq, or (last_seq, None) on failure."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq or not self._running, timeout)
            if self._seq > last_seq:
                return self._seq, self._frame
            return last_seq, None

    def release(self) -> None:
        self._running = False
        self.cap.release()
