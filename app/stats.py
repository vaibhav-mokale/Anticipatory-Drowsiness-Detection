from __future__ import annotations

from app.idas_core import FrameResult, IdasEngine


def diagnostics_tree(result: FrameResult | None, engine: IdasEngine) -> str:
    if result is None:
        return "Awaiting session data."
    lines = [
        "▸ Detection",
        f"  ├─ status: {result.status_text}",
        f"  ├─ ear: {result.ear:.3f}",
        f"  ├─ mar: {result.mar:.3f}",
        f"  ├─ perclos: {result.perclos:.1f}%",
        f"  └─ face: {'tracked' if result.face_found else 'not found'}",
        "",
        "▸ Session",
        f"  ├─ blinks: {result.total_blinks}",
        f"  ├─ yawns: {result.total_yawns}",
        f"  ├─ pitch: {result.pitch:.1f}°",
        f"  └─ yaw: {result.yaw:.1f}°",
        "",
        "▸ Thresholds",
        f"  ├─ ear: {engine.ear_threshold:.2f}",
        f"  ├─ mar: {engine.mouth_threshold:.2f}",
        f"  ├─ tilt: {engine.tilt_threshold}",
        f"  └─ eye frames: {engine.eye_consec_frames}",
        "",
        "▸ Calibration",
        f"  └─ {'complete' if result.calibrated else f'in progress {result.calibration_progress * 100:.0f}%'}",
    ]
    return "\n".join(lines)
