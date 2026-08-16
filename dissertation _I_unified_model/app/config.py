from __future__ import annotations

from dataclasses import dataclass

from app.resources import resource_path


@dataclass(frozen=True)
class AppConfig:
    img_size: int = 224
    seq_len: int = 4
    # Face-crop inference is closer to training; lower default than full-frame 0.65
    drowsy_threshold: float = 0.45
    webcam_index: int = 0
    consecutive_drowsy_frames: int = 5
    face_pad_ratio: float = 0.35
    preview_width: int = 800
    preview_height: int = 600
    checkpoint_best: str = "checkpoints/best.pth"
    checkpoint_last: str = "checkpoints/last.pth"
    meta_file: str = "checkpoints/meta.json"
    alert_sound: str = "assets/alert.mp3"
    icon_file: str = "assets/icon.ico"


DEFAULT = AppConfig()


def checkpoint_best_path(cfg: AppConfig = DEFAULT) -> str:
    return resource_path(cfg.checkpoint_best)


def checkpoint_last_path(cfg: AppConfig = DEFAULT) -> str:
    return resource_path(cfg.checkpoint_last)


def meta_path(cfg: AppConfig = DEFAULT) -> str:
    return resource_path(cfg.meta_file)


def alert_sound_path(cfg: AppConfig = DEFAULT) -> str:
    return resource_path(cfg.alert_sound)


def icon_path(cfg: AppConfig = DEFAULT) -> str:
    return resource_path(cfg.icon_file)


def paper_figure_path(filename: str, cfg: AppConfig = DEFAULT) -> str:
    return resource_path(f"assets/figures/{filename}")
