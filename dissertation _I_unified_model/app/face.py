from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def as_xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class FaceDetector:
    """OpenCV Haar face detector with an upper-center ROI fallback."""

    def __init__(
        self,
        scale_factor: float = 1.08,
        min_neighbors: int = 4,
        min_size: int = 48,
    ) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load face cascade: {cascade_path}")
        alt_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self._cascade_alt = cv2.CascadeClassifier(alt_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect_largest(self, bgr_frame: np.ndarray) -> FaceBox | None:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._detect(gray, self._cascade)
        if len(faces) == 0 and not self._cascade_alt.empty():
            faces = self._detect(gray, self._cascade_alt)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
        return FaceBox(int(x), int(y), int(w), int(h))

    def resolve_roi(self, bgr_frame: np.ndarray) -> tuple[FaceBox, bool]:
        """Return (box, face_detected). Falls back to upper-center crop."""
        box = self.detect_largest(bgr_frame)
        if box is not None:
            return box, True
        return default_upper_center_box(bgr_frame.shape), False

    def _detect(self, gray: np.ndarray, cascade: cv2.CascadeClassifier):
        return cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )


def default_upper_center_box(frame_shape: tuple[int, ...]) -> FaceBox:
    """Driver-style ROI when Haar misses: upper-middle square."""
    h_img, w_img = frame_shape[:2]
    side = int(min(h_img, w_img) * 0.55)
    x = max(0, (w_img - side) // 2)
    y = max(0, int(h_img * 0.08))
    if y + side > h_img:
        y = max(0, h_img - side)
    if x + side > w_img:
        x = max(0, w_img - side)
    return FaceBox(x, y, side, side)


def expand_box(
    box: FaceBox,
    frame_shape: tuple[int, ...],
    pad_ratio: float = 0.35,
) -> FaceBox:
    """Pad around the face so the crop closer matches NTHU-style framing."""
    h_img, w_img = frame_shape[:2]
    pad_w = int(box.w * pad_ratio)
    pad_h = int(box.h * pad_ratio)
    x1 = max(0, box.x - pad_w)
    y1 = max(0, box.y - pad_h)
    x2 = min(w_img, box.x + box.w + pad_w)
    y2 = min(h_img, box.y + box.h + pad_h)
    return FaceBox(x1, y1, x2 - x1, y2 - y1)


def crop_face(bgr_frame: np.ndarray, box: FaceBox) -> np.ndarray:
    return bgr_frame[box.y : box.y + box.h, box.x : box.x + box.w].copy()
