from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torchvision import transforms

from app.config import (
    DEFAULT,
    AppConfig,
    checkpoint_best_path,
    checkpoint_last_path,
    meta_path,
)
from app.face import FaceBox, FaceDetector, crop_face, expand_box
from app.model import CNN_LSTM_ViT

_NORMALIZE = transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)


@dataclass
class DetectionResult:
    label: str
    score: float
    is_drowsy: bool
    annotated_bgr: np.ndarray
    face_found: bool
    face_bbox: tuple[int, int, int, int] | None = None
    score_alert: float = 0.0
    infer_ms: float = 0.0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_state_dict(
    cfg: AppConfig = DEFAULT,
    device: torch.device | None = None,
) -> dict:
    device = device or pick_device()
    best = checkpoint_best_path(cfg)
    last = checkpoint_last_path(cfg)
    meta = meta_path(cfg)

    if os.path.exists(best):
        if os.path.exists(meta):
            with open(meta, encoding="utf-8") as f:
                _ = json.load(f)
        return torch.load(best, map_location=device, weights_only=True)

    if os.path.exists(last):
        ckpt = torch.load(last, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            return ckpt["model_state"]
        return ckpt

    raise FileNotFoundError(
        f"No trained model found. Expected {best} or {last}. "
        "Run 01_Train.ipynb first."
    )


def load_model(
    cfg: AppConfig = DEFAULT,
    device: torch.device | None = None,
) -> tuple[CNN_LSTM_ViT, torch.device]:
    device = device or pick_device()
    state_dict = load_state_dict(cfg, device)
    model = CNN_LSTM_ViT().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def preprocess_frame(
    bgr_frame: np.ndarray,
    cfg: AppConfig = DEFAULT,
) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size))
    img = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)
    img = _NORMALIZE(img)
    return torch.stack([img] * cfg.seq_len).unsqueeze(0)


def annotate_frame(
    bgr_frame: np.ndarray,
    label: str,
    score: float,
    is_drowsy: bool,
    face_box: FaceBox,
    face_found: bool,
) -> np.ndarray:
    out = bgr_frame.copy()
    if is_drowsy:
        color = (0, 0, 255)
    elif face_found:
        color = (0, 255, 0)
    else:
        color = (0, 165, 255)

    x, y, w, h = face_box.as_xywh
    thickness = 2
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)

    tick = max(14, min(w, h) // 5)
    for cx, cy in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
        dx = tick if cx == x else -tick
        dy = tick if cy == y else -tick
        cv2.line(out, (cx, cy), (cx + dx, cy), color, thickness)
        cv2.line(out, (cx, cy), (cx, cy + dy), color, thickness)

    tag = label if face_found else f"{label} [ROI]"
    text = f"{tag} ({score:.2f})"
    text_y = max(28, y - 10)
    cv2.putText(
        out,
        text,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


@torch.inference_mode()
def predict_frame(
    model: CNN_LSTM_ViT,
    bgr_frame: np.ndarray,
    device: torch.device,
    threshold: float | None = None,
    cfg: AppConfig = DEFAULT,
    face_detector: FaceDetector | None = None,
) -> DetectionResult:
    """Crop face (or upper-center ROI), classify, draw outline on the frame."""
    thresh = cfg.drowsy_threshold if threshold is None else threshold
    detector = face_detector or FaceDetector()

    raw_box, face_found = detector.resolve_roi(bgr_frame)
    pad = cfg.face_pad_ratio if face_found else 0.05
    padded = expand_box(raw_box, bgr_frame.shape, pad)
    face_img = crop_face(bgr_frame, padded)

    if face_img.size == 0:
        face_img = bgr_frame
        raw_box, face_found = detector.resolve_roi(bgr_frame)

    seq = preprocess_frame(face_img, cfg).to(device)
    t0 = time.perf_counter()
    output = model(seq)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    probs = torch.softmax(output, dim=1)[0]
    score_alert = float(probs[0].item())
    score = float(probs[1].item())
    is_drowsy = score > thresh
    label = "DROWSY" if is_drowsy else "ALERT"
    annotated = annotate_frame(
        bgr_frame, label, score, is_drowsy, raw_box, face_found
    )
    return DetectionResult(
        label=label,
        score=score,
        is_drowsy=is_drowsy,
        annotated_bgr=annotated,
        face_found=face_found,
        face_bbox=raw_box.as_xywh,
        score_alert=score_alert,
        infer_ms=infer_ms,
    )
