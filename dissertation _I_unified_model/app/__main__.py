from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drowsiness detection GUI")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Load model and exit (smoke check, no GUI)",
    )
    parser.add_argument(
        "--webcam",
        type=int,
        default=None,
        help="Webcam index override",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Drowsy probability threshold override",
    )
    args = parser.parse_args(argv)

    from app.config import DEFAULT, AppConfig

    cfg = AppConfig(
        img_size=DEFAULT.img_size,
        seq_len=DEFAULT.seq_len,
        drowsy_threshold=(
            args.threshold
            if args.threshold is not None
            else DEFAULT.drowsy_threshold
        ),
        webcam_index=(
            args.webcam if args.webcam is not None else DEFAULT.webcam_index
        ),
        consecutive_drowsy_frames=DEFAULT.consecutive_drowsy_frames,
        face_pad_ratio=DEFAULT.face_pad_ratio,
        preview_width=DEFAULT.preview_width,
        preview_height=DEFAULT.preview_height,
        checkpoint_best=DEFAULT.checkpoint_best,
        checkpoint_last=DEFAULT.checkpoint_last,
        meta_file=DEFAULT.meta_file,
        alert_sound=DEFAULT.alert_sound,
        icon_file=DEFAULT.icon_file,
    )

    if args.headless:
        from app.inference import load_model

        model, device = load_model(cfg)
        print(f"Model loaded on {device}. params={sum(p.numel() for p in model.parameters())}")
        return 0

    try:
        from app.gui.window import run_app
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        if "tkinter" in missing or "customtkinter" in missing:
            print(
                "GUI requires Tk + customtkinter.\n"
                "  Debian/Ubuntu: sudo apt install python3-tk\n"
                "  then: pip install customtkinter\n"
                "Or smoke-check the model with: python -m app --headless",
                file=sys.stderr,
            )
            return 1
        raise

    run_app(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
