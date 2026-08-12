from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

from app.config import AppConfig, DEFAULT, landmark_path


def eye_aspect_ratio(eye: np.ndarray) -> float:
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c)


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    a = dist.euclidean(mouth[2], mouth[10])
    b = dist.euclidean(mouth[4], mouth[8])
    c = dist.euclidean(mouth[0], mouth[6])
    return (a + b) / (2.0 * c)


def get_head_pose(
    shape: np.ndarray, w: int, h: int
) -> tuple[tuple[int, int], tuple[int, int], float, float]:
    image_points = np.array(
        [shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]],
        dtype="double",
    )
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ]
    )
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype="double",
    )
    dist_coeffs = np.zeros((4, 1))
    (_, rot_vec, trans_vec) = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    (nose_end, _) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]),
        rot_vec,
        trans_vec,
        camera_matrix,
        dist_coeffs,
    )
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end[0][0][0]), int(nose_end[0][0][1]))
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0])
    yaw = float(angles[1])
    return p1, p2, pitch, yaw


def geometric_head_angles(shape: np.ndarray) -> tuple[float, float]:
    nose = shape[30]
    left_eye = shape[36:42].mean(axis=0)
    right_eye = shape[42:48].mean(axis=0)
    eye_mid = (left_eye + right_eye) / 2.0
    eye_span = max(float(np.linalg.norm(right_eye - left_eye)), 1e-6)
    yaw = float((nose[0] - eye_mid[0]) / eye_span * 90.0)
    pitch = float((nose[1] - eye_mid[1]) / eye_span * 70.0)
    return pitch, yaw


def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


@dataclass
class FrameResult:
    annotated_bgr: np.ndarray
    ear: float = 0.0
    mar: float = 0.0
    perclos: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    total_blinks: int = 0
    total_yawns: int = 0
    is_drowsy: bool = False
    is_yawning: bool = False
    is_distracted: bool = False
    distraction_reason: str = ""
    status_text: str = "SYSTEM STATUS: NORMAL"
    label: str = "NORMAL"
    calibrated: bool = False
    calibration_progress: float = 0.0
    face_found: bool = False
    show_alert_flash: bool = False


@dataclass
class IdasEngine:
    cfg: AppConfig = field(default_factory=lambda: DEFAULT)
    ear_threshold: float = DEFAULT.ear_threshold
    mouth_threshold: float = DEFAULT.mouth_threshold
    tilt_threshold: int = DEFAULT.tilt_threshold
    eye_consec_frames: int = DEFAULT.eye_consec_frames
    counter_eye: int = 0
    counter_yawn: int = 0
    total_blinks: int = 0
    total_yawns: int = 0
    pitch_offset: float = 0.0
    yaw_offset: float = 0.0
    pitch_geo_offset: float = 0.0
    yaw_geo_offset: float = 0.0
    frame_count: int = 0
    calib_pitch_sum: float = 0.0
    calib_yaw_sum: float = 0.0
    calib_pitch_geo_sum: float = 0.0
    calib_yaw_geo_sum: float = 0.0
    is_calibrated: bool = False
    ear_history: collections.deque = field(default_factory=collections.deque)
    mar_history: collections.deque = field(default_factory=collections.deque)
    _detector: object | None = None
    _predictor: object | None = None
    _last_pitch: float = 0.0
    _last_yaw: float = 0.0
    _last_pitch_geo: float = 0.0
    _last_yaw_geo: float = 0.0

    def __post_init__(self) -> None:
        self.ear_history = collections.deque(
            [0.3] * self.cfg.history_len, maxlen=self.cfg.history_len
        )
        self.mar_history = collections.deque(
            [0.0] * self.cfg.history_len, maxlen=self.cfg.history_len
        )

    def load(self) -> None:
        if self._detector is not None:
            return
        path = landmark_path(self.cfg)
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(path)

    def reset_session(self) -> None:
        self.counter_eye = 0
        self.counter_yawn = 0
        self.total_blinks = 0
        self.total_yawns = 0
        self.recalibrate()

    def recalibrate(self) -> None:
        self.pitch_offset = 0.0
        self.yaw_offset = 0.0
        self.pitch_geo_offset = 0.0
        self.yaw_geo_offset = 0.0
        self.calib_pitch_sum = 0.0
        self.calib_yaw_sum = 0.0
        self.calib_pitch_geo_sum = 0.0
        self.calib_yaw_geo_sum = 0.0
        self.is_calibrated = False
        self.frame_count = 0

    def snap_calibrate(self) -> None:
        self.pitch_offset += self._last_pitch
        self.yaw_offset += self._last_yaw
        self.pitch_geo_offset += self._last_pitch_geo
        self.yaw_geo_offset += self._last_yaw_geo

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        self.load()
        frame = cv2.flip(frame_bgr, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._detector(gray, 0)

        ear = 0.0
        mar = 0.0
        pitch = 0.0
        yaw = 0.0
        is_drowsy = False
        is_yawning = False
        is_distracted = False
        distraction_reason = ""
        face_found = len(rects) > 0

        if not self.is_calibrated:
            self.frame_count += 1

        for rect in rects:
            shape = self._predictor(gray, rect)
            shape_np = face_utils.shape_to_np(shape)
            left_eye = shape_np[36:42]
            right_eye = shape_np[42:48]
            mouth = shape_np[48:68]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            p1, p2, raw_pitch, raw_yaw = get_head_pose(
                shape_np, frame.shape[1], frame.shape[0]
            )
            raw_pitch = normalize_angle(raw_pitch)
            raw_yaw = normalize_angle(raw_yaw)
            raw_pitch_geo, raw_yaw_geo = geometric_head_angles(shape_np)

            if not self.is_calibrated:
                self.calib_pitch_sum += raw_pitch
                self.calib_yaw_sum += raw_yaw
                self.calib_pitch_geo_sum += raw_pitch_geo
                self.calib_yaw_geo_sum += raw_yaw_geo
                if self.frame_count >= self.cfg.calibration_frames:
                    n = max(self.cfg.calibration_frames, 1)
                    self.pitch_offset = self.calib_pitch_sum / n
                    self.yaw_offset = self.calib_yaw_sum / n
                    self.pitch_geo_offset = self.calib_pitch_geo_sum / n
                    self.yaw_geo_offset = self.calib_yaw_geo_sum / n
                    self.is_calibrated = True

            pitch = raw_pitch - self.pitch_offset
            yaw = normalize_angle(raw_yaw - self.yaw_offset)
            pitch_geo = raw_pitch_geo - self.pitch_geo_offset
            yaw_geo = raw_yaw_geo - self.yaw_geo_offset
            self._last_pitch = pitch
            self._last_yaw = yaw
            self._last_pitch_geo = pitch_geo
            self._last_yaw_geo = yaw_geo
            pitch_dev = max(abs(pitch), abs(pitch_geo))
            yaw_dev = max(abs(yaw), abs(yaw_geo))
            pitch = pitch_dev
            yaw = yaw_dev

            cv2.polylines(frame, [left_eye], True, (200, 100, 0), 1)
            cv2.polylines(frame, [right_eye], True, (200, 100, 0), 1)
            cv2.polylines(frame, [mouth], True, (0, 0, 200), 1)

            if ear < self.ear_threshold:
                self.counter_eye += 1
                if self.counter_eye >= self.eye_consec_frames:
                    is_drowsy = True
            else:
                if 3 <= self.counter_eye < self.eye_consec_frames:
                    self.total_blinks += 1
                self.counter_eye = 0

            if mar > self.mouth_threshold:
                self.counter_yawn += 1
                if self.counter_yawn > 15:
                    is_yawning = True
            else:
                if self.counter_yawn > 15:
                    self.total_yawns += 1
                self.counter_yawn = 0

            if self.is_calibrated and (
                pitch_dev > self.tilt_threshold or yaw_dev > self.tilt_threshold
            ):
                is_distracted = True
                if pitch_dev >= yaw_dev:
                    distraction_reason = f"Pitch ({int(pitch_dev)})"
                else:
                    distraction_reason = f"Yaw ({int(yaw_dev)})"

            pose_color = (0, 0, 220) if is_distracted else (0, 160, 60)
            if not self.is_calibrated:
                pose_color = (0, 140, 220)
            cv2.line(frame, p1, p2, pose_color, 3)
            cv2.circle(frame, p1, 5, pose_color, -1)
            cv2.circle(frame, p2, 4, pose_color, 2)
            cv2.putText(
                frame,
                f"P{int(pitch_dev)} Y{int(yaw_dev)}",
                (p1[0] + 10, max(p1[1] - 12, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                pose_color,
                2,
                cv2.LINE_AA,
            )
            if is_distracted:
                cv2.putText(
                    frame,
                    "HEAD TILT",
                    (rect.left(), max(rect.top() - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 220),
                    2,
                    cv2.LINE_AA,
                )

        self.ear_history.append(ear)
        self.mar_history.append(mar)
        closed = sum(1 for e in self.ear_history if e < self.ear_threshold)
        perclos = (closed / len(self.ear_history) * 100) if self.ear_history else 0.0

        alerts: list[str] = []
        if is_drowsy:
            alerts.append("DROWSY")
        if is_yawning:
            alerts.append("YAWNING")
        if is_distracted:
            alerts.append(f"DISTRACTED {distraction_reason}".strip())

        if alerts:
            status_text = "ALERT: " + ", ".join(alerts)
            label = "ALERT"
        else:
            status_text = "SYSTEM STATUS: NORMAL"
            label = "NORMAL"

        progress = min(self.frame_count / self.cfg.calibration_frames, 1.0)
        if self.is_calibrated:
            progress = 1.0

        if not self.is_calibrated:
            self._draw_calibration_overlay(frame, progress)

        show_alert_flash = bool(alerts) and int(time.time() * 5) % 2 == 0
        if show_alert_flash:
            cv2.putText(
                frame,
                "DISTRACTED / FATIGUE DETECTED",
                (40, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        return FrameResult(
            annotated_bgr=frame,
            ear=ear,
            mar=mar,
            perclos=perclos,
            pitch=pitch,
            yaw=yaw,
            total_blinks=self.total_blinks,
            total_yawns=self.total_yawns,
            is_drowsy=is_drowsy,
            is_yawning=is_yawning,
            is_distracted=is_distracted,
            distraction_reason=distraction_reason,
            status_text=status_text,
            label=label,
            calibrated=self.is_calibrated,
            calibration_progress=progress,
            face_found=face_found,
            show_alert_flash=show_alert_flash,
        )

    @staticmethod
    def _draw_calibration_overlay(frame: np.ndarray, progress: float) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = 40, h // 2 - 40, w - 40, h // 2 + 40
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "CALIBRATING... LOOK STRAIGHT",
            (x1 + 20, h // 2),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        bar_w = int((x2 - x1 - 20) * progress)
        cv2.rectangle(frame, (x1 + 10, y2 - 12), (x1 + 10 + bar_w, y2 - 4), (0, 255, 0), -1)
