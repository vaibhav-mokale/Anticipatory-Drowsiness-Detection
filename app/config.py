from __future__ import annotations

from dataclasses import dataclass

from app.resources import resource_path


@dataclass(frozen=True)
class AppConfig:
    webcam_index: int = 0
    ear_threshold: float = 0.25
    mouth_threshold: float = 0.60
    tilt_threshold: int = 40
    eye_consec_frames: int = 20
    calibration_frames: int = 50
    history_len: int = 50
    preview_width: int = 800
    preview_height: int = 600
    landmark_file: str = "shape_predictor_68_face_landmarks.dat"
    alert_sound: str = "alert.mp3"
    icon_file: str = "icon.ico"


DEFAULT = AppConfig()


def landmark_path(cfg: AppConfig = DEFAULT) -> str:
    return resource_path(cfg.landmark_file)


def alert_sound_path(cfg: AppConfig = DEFAULT) -> str:
    return resource_path(cfg.alert_sound)


def icon_path(cfg: AppConfig = DEFAULT) -> str:
    return resource_path(cfg.icon_file)


def paper_figure_path(filename: str, cfg: AppConfig = DEFAULT) -> str:
    return resource_path(f"assets/figures/{filename}")
