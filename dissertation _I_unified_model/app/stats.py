from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from app import __version__ as APP_VERSION
from app.config import AppConfig, checkpoint_best_path, meta_path
from app.inference import DetectionResult
from app.resources import resource_path


@dataclass
class RuntimeStats:
    frames: int = 0
    infer_ms: float = 0.0
    fps: float = 0.0
    score_min: float | None = None
    score_max: float | None = None
    score_sum: float = 0.0
    score_ema: float | None = None
    drowsy_frames: int = 0
    face_hit_frames: int = 0
    streak: int = 0
    alert_on: bool = False
    started_at: str | None = None
    last_result: DetectionResult | None = None
    _fps_t0: float = field(default_factory=time.perf_counter, repr=False)
    _fps_n: int = 0

    def observe(self, result: DetectionResult, infer_ms: float, streak: int, alert_on: bool) -> None:
        self.frames += 1
        self.infer_ms = infer_ms
        self.streak = streak
        self.alert_on = alert_on
        self.last_result = result
        self._fps_n += 1
        now = time.perf_counter()
        dt = now - self._fps_t0
        if dt >= 0.5:
            self.fps = self._fps_n / dt
            self._fps_t0 = now
            self._fps_n = 0

        s = result.score
        self.score_sum += s
        self.score_min = s if self.score_min is None else min(self.score_min, s)
        self.score_max = s if self.score_max is None else max(self.score_max, s)
        self.score_ema = (
            s if self.score_ema is None else (0.85 * self.score_ema + 0.15 * s)
        )
        if result.is_drowsy:
            self.drowsy_frames += 1
        if result.face_found:
            self.face_hit_frames += 1

    def reset_session(self) -> None:
        self.frames = 0
        self.infer_ms = 0.0
        self.fps = 0.0
        self.score_min = None
        self.score_max = None
        self.score_sum = 0.0
        self.score_ema = None
        self.drowsy_frames = 0
        self.face_hit_frames = 0
        self.streak = 0
        self.alert_on = False
        self.started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        self.last_result = None
        self._fps_t0 = time.perf_counter()
        self._fps_n = 0


def _safe_ver(mod_name: str) -> str:
    try:
        mod = __import__(mod_name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        return f"n/a ({exc})"


def _pkg_ver_alt(names: list[str]) -> str:
    for name in names:
        v = _safe_ver(name)
        if not v.startswith("n/a"):
            return f"{name} {v}"
    return "n/a"


def collect_dump(
    cfg: AppConfig,
    runtime: RuntimeStats,
    *,
    threshold: float,
    camera_index: int,
    device: Any = None,
    model: Any = None,
) -> dict[str, Any]:
    """Kitchen-sink diagnostics dict for UI + export panel."""
    result = runtime.last_result
    meta: dict[str, Any] = {}
    try:
        with open(meta_path(cfg), encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        meta = {"error": str(exc)}

    best = checkpoint_best_path(cfg)
    ckpt_stat: dict[str, Any] = {}
    try:
        st = os.stat(best)
        ckpt_stat = {
            "path": best,
            "exists": True,
            "bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        }
    except Exception as exc:
        ckpt_stat = {"path": best, "exists": False, "error": str(exc)}

    param_count = None
    if model is not None:
        try:
            param_count = int(sum(p.numel() for p in model.parameters()))
        except Exception:
            param_count = None

    score_avg = (
        runtime.score_sum / runtime.frames if runtime.frames else None
    )
    face_rate = (
        runtime.face_hit_frames / runtime.frames if runtime.frames else None
    )
    drowsy_rate = (
        runtime.drowsy_frames / runtime.frames if runtime.frames else None
    )

    import torch

    dump: dict[str, Any] = {
        "exported_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "app": {
            "name": "drowsiness-detection",
            "version": APP_VERSION,
            "entry": "python -m app",
            "frozen": bool(getattr(sys, "frozen", False)),
            "meipass": getattr(sys, "_MEIPASS", None),
        },
        "host": {
            "python": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "cwd": os.getcwd(),
        },
        "versions": {
            "torch": _safe_ver("torch"),
            "torchvision": _safe_ver("torchvision"),
            "timm": _safe_ver("timm"),
            "opencv": _safe_ver("cv2"),
            "numpy": _safe_ver("numpy"),
            "customtkinter": _safe_ver("customtkinter"),
            "PIL": _pkg_ver_alt(["PIL", "Pillow"]),
            "pygame": _safe_ver("pygame"),
        },
        "torch_runtime": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cudnn_enabled": bool(getattr(torch.backends, "cudnn", None) and torch.backends.cudnn.enabled),
            "device": str(device) if device is not None else None,
        },
        "model": {
            "class": type(model).__name__ if model is not None else None,
            "architecture": "CNN_LSTM_ViT (MobileNetV3-Large + BiLSTM + ViT-Tiny)",
            "num_parameters": param_count,
            "img_size": cfg.img_size,
            "seq_len": cfg.seq_len,
            "classes": {"0": "ALERT/not_drowsy", "1": "DROWSY"},
            "checkpoint": ckpt_stat,
            "training_meta": meta,
        },
        "config": asdict(cfg),
        "session": {
            "started_at": runtime.started_at,
            "camera_index": camera_index,
            "threshold_live": threshold,
            "frames": runtime.frames,
            "fps": round(runtime.fps, 3),
            "infer_ms": round(runtime.infer_ms, 3),
            "streak": runtime.streak,
            "alert_on": runtime.alert_on,
            "score_min": runtime.score_min,
            "score_max": runtime.score_max,
            "score_avg": score_avg,
            "score_ema": runtime.score_ema,
            "face_hit_frames": runtime.face_hit_frames,
            "face_hit_rate": face_rate,
            "drowsy_frames": runtime.drowsy_frames,
            "drowsy_rate": drowsy_rate,
        },
        "paths": {
            "resource_root_sample": resource_path("."),
            "alert_sound": resource_path(cfg.alert_sound),
            "icon": resource_path(cfg.icon_file),
            "meta": meta_path(cfg),
        },
        "frame": None,
        "detection": None,
    }

    if result is not None:
        frame = result.annotated_bgr
        dump["frame"] = {
            "height": int(frame.shape[0]),
            "width": int(frame.shape[1]),
            "channels": int(frame.shape[2]) if frame.ndim == 3 else 1,
            "dtype": str(frame.dtype),
            "nbytes": int(frame.nbytes),
        }
        dump["detection"] = {
            "label": result.label,
            "score_drowsy": result.score,
            "score_alert": result.score_alert,
            "is_drowsy": result.is_drowsy,
            "face_found": result.face_found,
            "face_bbox_xywh": result.face_bbox,
            "threshold_used": threshold,
            "margin_above_threshold": result.score - threshold,
        }

    return dump


def dump_to_lines(dump: dict[str, Any], mode: str = "all") -> list[str]:
    """Flatten dump. mode='important' keeps a short curated view."""
    if mode == "important":
        return format_important_lines(dump)
    return format_all_lines(dump)


def format_important_lines(dump: dict[str, Any]) -> list[str]:
    det = dump.get("detection") or {}
    ses = dump.get("session") or {}
    model = dump.get("model") or {}
    torch_rt = dump.get("torch_runtime") or {}
    vers = dump.get("versions") or {}
    host = dump.get("host") or {}
    cfg = dump.get("config") or {}
    ckpt = model.get("checkpoint") or {}

    def fnum(v: Any, digits: int = 3) -> str:
        if v is None:
            return "--"
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    bbox = det.get("face_bbox_xywh")
    if not det:
        face = "--"
    elif det.get("face_found"):
        face = f"yes  {bbox}" if bbox else "yes"
    else:
        face = f"roi/fallback  {bbox}" if bbox else "roi/fallback"

    lines = [
        "DETECTION",
        f"  label       {det.get('label', '--')}",
        f"  drowsy      {fnum(det.get('score_drowsy'), 4)}",
        f"  alert       {fnum(det.get('score_alert'), 4)}",
        f"  threshold   {fnum(det.get('threshold_used'), 2)}",
        f"  margin      {fnum(det.get('margin_above_threshold'), 4)}",
        f"  face        {face}",
        "",
        "SESSION",
        f"  fps         {fnum(ses.get('fps'), 2)}",
        f"  infer_ms    {fnum(ses.get('infer_ms'), 1)}",
        f"  frames      {ses.get('frames', 0)}",
        f"  streak      {ses.get('streak', 0)}  alert={ses.get('alert_on', False)}",
        f"  score ema   {fnum(ses.get('score_ema'), 4)}",
        f"  score range {fnum(ses.get('score_min'), 3)} .. {fnum(ses.get('score_max'), 3)}",
        f"  face hit    {fnum(ses.get('face_hit_rate'), 3)}",
        f"  drowsy rate {fnum(ses.get('drowsy_rate'), 3)}",
        "",
        "MODEL",
        f"  arch        {model.get('architecture', '--')}",
        f"  params      {model.get('num_parameters', '--')}",
        f"  device      {torch_rt.get('device', '--')}",
        f"  checkpoint  {os.path.basename(str(ckpt.get('path', ''))) or '--'}",
        f"  best_acc    {(model.get('training_meta') or {}).get('best_acc', '--')}",
        "",
        "STACK",
        f"  python      {str(host.get('python', '--')).split()[0]}",
        f"  torch       {vers.get('torch', '--')}",
        f"  opencv      {vers.get('opencv', '--')}",
        f"  timm        {vers.get('timm', '--')}",
        f"  img/seq     {cfg.get('img_size')} / {cfg.get('seq_len')}",
    ]
    return lines


def format_all_lines(dump: dict[str, Any]) -> list[str]:
    """Full kitchen-sink flatten for UI + export."""
    lines: list[str] = []

    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            if prefix:
                lines.append("")
                lines.append(f"[{prefix}]")
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, (dict, list)) and not _is_short_list(v):
                    walk(v, key)
                else:
                    lines.append(f"  {k}: {_fmt(v)}" if prefix else f"{k}: {_fmt(v)}")
            return
        if isinstance(obj, list):
            lines.append(f"{prefix}: {_fmt(obj)}")
            return
        lines.append(f"{prefix}: {_fmt(obj)}")

    walk(dump)
    return lines


def format_dump_text(dump: dict[str, Any], mode: str = "important") -> str:
    return "\n".join(dump_to_lines(dump, mode=mode))


def important_sections(dump: dict[str, Any]) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    title = ""
    body: list[str] = []
    for line in format_important_lines(dump):
        stripped = line.strip()
        if not stripped:
            continue
        if line == line.upper() and not line.startswith(" "):
            if title:
                sections.append((title, body))
            title = stripped
            body = []
            continue
        body.append(stripped)
    if title:
        sections.append((title, body))
    return sections


SECTION_DISPLAY_NAMES = {
    "DETECTION": "Detection",
    "SESSION": "Session metrics",
    "MODEL": "Model specification",
    "STACK": "Software stack",
}


def format_tree_lines(dump: dict[str, Any], mode: str = "important") -> list[str]:
    if mode == "all":
        return _full_tree_lines(dump)
    return _important_tree_lines(dump)


def format_tree_text(dump: dict[str, Any], mode: str = "important") -> str:
    lines = format_tree_lines(dump, mode=mode)
    return "\n".join(lines) if lines else "Awaiting session data."


def _parse_kv(line: str) -> tuple[str, str]:
    stripped = line.strip()
    if not stripped:
        return "", ""
    parts = re.split(r"\s{2,}", stripped, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return stripped, ""


def _important_tree_lines(dump: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    sections = important_sections(dump)
    for si, (title, body) in enumerate(sections):
        display = SECTION_DISPLAY_NAMES.get(title, title.title())
        lines.append(f"▸ {display}")
        for bi, item in enumerate(body):
            branch = "└─" if bi == len(body) - 1 else "├─"
            key, val = _parse_kv(item)
            if val:
                lines.append(f"  {branch} {key}: {val}")
            else:
                lines.append(f"  {branch} {key}")
        if si < len(sections) - 1:
            lines.append("")
    return lines


def _full_tree_lines(dump: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    def walk(obj: Any, prefix: str) -> None:
        if not isinstance(obj, dict):
            lines.append(f"{prefix}└─ {_fmt(obj)}")
            return
        keys = list(obj.keys())
        for i, key in enumerate(keys):
            last = i == len(keys) - 1
            branch = "└─" if last else "├─"
            value = obj[key]
            if isinstance(value, dict) and value:
                lines.append(f"{prefix}{branch} {key}")
                child_prefix = prefix + ("   " if last else "│  ")
                walk(value, child_prefix)
            else:
                lines.append(f"{prefix}{branch} {key}: {_fmt(value)}")

    top_keys = list(dump.keys())
    for i, key in enumerate(top_keys):
        lines.append(f"▸ {key}")
        value = dump[key]
        if isinstance(value, dict) and value:
            walk(value, "  ")
        else:
            lines.append(f"  └─ {_fmt(value)}")
        if i < len(top_keys) - 1:
            lines.append("")
    return lines


def diagnostics_summary_lines(dump: dict[str, Any]) -> list[str]:
    det = dump.get("detection") or {}
    ses = dump.get("session") or {}
    model = dump.get("model") or {}
    torch_rt = dump.get("torch_runtime") or {}

    def fnum(v: Any, digits: int = 2) -> str:
        if v is None:
            return "--"
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    label = det.get("label", "--")
    face = "face tracked" if det.get("face_found") else "ROI fallback"
    line1 = (
        f"{label} state · CDS {fnum(det.get('score_drowsy'), 2)} · "
        f"alert {fnum(det.get('score_alert'), 2)} · {face}"
    )

    if ses.get("frames", 0):
        line2 = (
            f"{fnum(ses.get('fps'), 1)} FPS · {fnum(ses.get('infer_ms'), 0)} ms · "
            f"{ses.get('frames', 0)} frames · "
            f"drowsy {fnum((ses.get('drowsy_rate') or 0) * 100, 1)}%"
        )
    else:
        line2 = "Session not started."

    arch = str(model.get("architecture", "CNN-LSTM-ViT")).split("(")[0].strip()
    device = torch_rt.get("device", "cpu")
    line3 = f"{arch} · {device} inference"

    return [line1, line2, line3]


def _is_short_list(v: Any) -> bool:
    return isinstance(v, list) and len(v) <= 8 and all(
        not isinstance(x, (dict, list)) for x in v
    )


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=True)
        except Exception:
            return str(v)
    return str(v)


def render_stats_panel(
    lines: list[str],
    *,
    width: int = 520,
    line_height: int = 18,
    pad: int = 16,
    bg: tuple[int, int, int] = (18, 18, 22),
    fg: tuple[int, int, int] = (220, 220, 220),
    accent: tuple[int, int, int] = (120, 180, 255),
) -> np.ndarray:
    """Rasterize stats lines into a BGR panel image."""
    # Measure with a monospace-ish OpenCV Hershey font
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    header = "DIAGNOSTICS DUMP"
    body = lines
    max_w = width - 2 * pad
    wrapped: list[tuple[str, bool]] = [(header, True)]
    for line in body:
        wrapped.extend((w, False) for w in _wrap_line(line, max_w, font, scale, thickness))

    height = pad * 2 + line_height * len(wrapped) + 8
    height = max(height, 200)
    panel = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (50, 50, 60), 1)

    y = pad + line_height - 4
    for text, is_header in wrapped:
        color = accent if is_header or text.startswith("[") else fg
        weight = 2 if is_header else thickness
        sz = 0.55 if is_header else scale
        cv2.putText(panel, text, (pad, y), font, sz, color, weight, cv2.LINE_AA)
        y += line_height + (4 if is_header else 0)
    return panel


def _wrap_line(
    text: str,
    max_width: int,
    font: int,
    scale: float,
    thickness: int,
) -> list[str]:
    if not text:
        return [""]
    words = text.split(" ")
    rows: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        (tw, _), _ = cv2.getTextSize(trial, font, scale, thickness)
        if tw <= max_width:
            cur = trial
            continue
        if cur:
            rows.append(cur)
        # hard-split long tokens
        if cv2.getTextSize(word, font, scale, thickness)[0][0] > max_width:
            chunk = ""
            for ch in word:
                t2 = chunk + ch
                if cv2.getTextSize(t2, font, scale, thickness)[0][0] > max_width and chunk:
                    rows.append(chunk)
                    chunk = ch
                else:
                    chunk = t2
            cur = chunk
        else:
            cur = word
    if cur:
        rows.append(cur)
    return rows or [text]


def compose_export(
    annotated_bgr: np.ndarray,
    lines: list[str],
    *,
    panel_width: int = 560,
) -> np.ndarray:
    """Side-by-side: camera frame | stats panel. Panel stays readable; frame scales."""
    frame = annotated_bgr
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    panel = render_stats_panel(lines, width=panel_width)
    ph = panel.shape[0]
    fh, fw = frame.shape[:2]
    if fh != ph:
        new_w = max(1, int(round(fw * (ph / fh))))
        frame = cv2.resize(frame, (new_w, ph), interpolation=cv2.INTER_AREA)
    gap = np.full((ph, 10, 3), (30, 30, 36), dtype=np.uint8)
    return np.hstack([frame, gap, panel])


def default_export_dir() -> str:
    path = os.path.join(os.getcwd(), "exports")
    os.makedirs(path, exist_ok=True)
    return path
