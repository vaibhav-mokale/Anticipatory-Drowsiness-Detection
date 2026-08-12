from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from app.audio import bootstrap_mixer
    from app.config import DEFAULT, AppConfig
    from app.gui.window import run_app

    bootstrap_mixer()

    cfg = AppConfig(
        webcam_index=DEFAULT.webcam_index,
        ear_threshold=DEFAULT.ear_threshold,
        mouth_threshold=DEFAULT.mouth_threshold,
        tilt_threshold=DEFAULT.tilt_threshold,
        eye_consec_frames=DEFAULT.eye_consec_frames,
        calibration_frames=DEFAULT.calibration_frames,
        history_len=DEFAULT.history_len,
        preview_width=DEFAULT.preview_width,
        preview_height=DEFAULT.preview_height,
        landmark_file=DEFAULT.landmark_file,
        alert_sound=DEFAULT.alert_sound,
        icon_file=DEFAULT.icon_file,
    )

    try:
        run_app(cfg)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        if "tkinter" in missing or "customtkinter" in missing:
            print(
                "GUI requires Tk + customtkinter.\n"
                "  Debian/Ubuntu: sudo apt install python3-tk\n"
                "  then: pip install -r requirements.txt",
                file=sys.stderr,
            )
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
