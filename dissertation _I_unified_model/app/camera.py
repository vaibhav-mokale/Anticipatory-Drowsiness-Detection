from __future__ import annotations

import cv2
import numpy as np


class VideoSource:
    """Thin OpenCV VideoCapture wrapper. No GUI imports."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self._cap: cv2.VideoCapture | None = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self, index: int | None = None) -> None:
        if index is not None:
            self.index = index
        self.release()
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(
                f"Could not open webcam at index {self.index}."
            )

    def read(self) -> np.ndarray | None:
        if not self.is_open:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()
