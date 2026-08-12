from __future__ import annotations

import os
import threading
import time

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from app.audio import AlertPlayer
from app.camera import VideoSource
from app.config import DEFAULT, AppConfig, alert_sound_path, icon_path, paper_figure_path
from app.dissertation import DISSERTATION, PAPER_FIGURES, REFERENCE_METRICS
from app.graphs import LiveSignalGraphs
from app.gui.theme import (
    BORDER_W,
    FONT_BODY,
    FONT_SECTION,
    FONT_SMALL,
    FONT_STATUS,
    FONT_TITLE,
    Theme,
    card_header,
    flat_card,
    highlight_metric_tile,
    tnr,
)
from app.idas_core import FrameResult, IdasEngine
from app.stats import diagnostics_tree


class MainWindow(ctk.CTk):
    def __init__(self, cfg: AppConfig = DEFAULT) -> None:
        super().__init__()
        self.cfg = cfg
        self.title(DISSERTATION.window_title)
        self.geometry("1440x920")
        self.minsize(1200, 800)
        self.configure(fg_color=Theme.BG)

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("green")

        self._running = False
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: FrameResult | None = None
        self._engine = IdasEngine(cfg)
        self._graphs = LiveSignalGraphs(cfg.history_len)
        self._camera = VideoSource(cfg.webcam_index)
        self._audio = AlertPlayer(alert_sound_path(cfg))
        self._last_frame: FrameResult | None = None
        self._stats_text_cache = ""
        self._figure_images: list[ctk.CTkImage] = []
        self._preview_dims = (cfg.preview_width, cfg.preview_height)
        self._graph_dims = (240, 180)
        self._preview_ctk: ctk.CTkImage | None = None
        self._graph_ctk: ctk.CTkImage | None = None
        self._graph_tick = 0
        self._resize_after_id: str | None = None

        self._build_ui()
        self._bind_keys()
        self._try_set_icon()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._audio.preload()
        self._sync_audio_status()
        self.after(250, self._cache_display_dims)
        self._graph_host.bind("<Configure>", self._on_display_resize)
        self._preview_host.bind("<Configure>", self._on_display_resize)
        self.after(50, self._poll_ui)

    def _try_set_icon(self) -> None:
        try:
            self.iconbitmap(icon_path(self.cfg))
        except Exception:
            pass

    @staticmethod
    def _fit_box(box_w: int, box_h: int, aspect: float) -> tuple[int, int]:
        box_w = max(box_w, 1)
        box_h = max(box_h, 1)
        if box_w / box_h > aspect:
            h = box_h
            w = max(int(h * aspect), 1)
        else:
            w = box_w
            h = max(int(w / aspect), 1)
        return w, h

    def _cache_display_dims(self) -> None:
        pw = self._preview_host.winfo_width()
        ph = self._preview_host.winfo_height()
        if pw > 160 and ph > 120:
            self._preview_dims = self._fit_box(pw - 8, ph - 8, 4 / 3)
        gw = self._graph_host.winfo_width()
        gh = self._graph_host.winfo_height()
        if gw > 120 and gh > 80:
            self._graph_dims = (max(gw - 8, 120), max(gh - 8, 80))

    def _on_display_resize(self, _event=None) -> None:
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(300, self._cache_display_dims)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=0, minsize=196)

        ctk.CTkLabel(
            self,
            text=DISSERTATION.title_line,
            font=tnr(FONT_TITLE, "bold"),
            text_color=Theme.TEXT,
            wraplength=1400,
            justify="center",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 0))
        ctk.CTkLabel(
            self,
            text=DISSERTATION.subtitle,
            font=tnr(FONT_SMALL + 1),
            text_color=Theme.TEXT_MUTED,
            justify="center",
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 10))

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 8))
        main.grid_columnconfigure(0, weight=2, minsize=280)
        main.grid_columnconfigure(1, weight=5, minsize=560)
        main.grid_columnconfigure(2, weight=2, minsize=280)
        main.grid_rowconfigure(0, weight=1)

        left_card = flat_card(main)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left_card.grid_rowconfigure(0, weight=1)
        left_card.grid_columnconfigure(0, weight=1)

        left_tabs = ctk.CTkTabview(
            left_card,
            fg_color=Theme.SURFACE,
            segmented_button_fg_color=Theme.SURFACE_ALT,
            segmented_button_selected_color=Theme.ACCENT,
            segmented_button_selected_hover_color=Theme.ACCENT_DARK,
            segmented_button_unselected_color=Theme.SURFACE_ALT,
            segmented_button_unselected_hover_color=Theme.SURFACE,
            text_color=Theme.TEXT_BODY,
        )
        left_tabs.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        tab_live = left_tabs.add("Live")
        tab_eval = left_tabs.add("Evaluation")
        tab_live.grid_rowconfigure(0, weight=1)
        tab_live.grid_columnconfigure(0, weight=1)
        tab_eval.grid_rowconfigure(0, weight=1)
        tab_eval.grid_columnconfigure(0, weight=1)

        signals_scroll = ctk.CTkScrollableFrame(
            tab_live,
            fg_color=Theme.SURFACE,
            border_width=0,
            scrollbar_button_color=Theme.ACCENT_SOFT,
            scrollbar_button_hover_color=Theme.ACCENT,
        )
        signals_scroll.grid(row=0, column=0, sticky="nsew")
        signals_scroll.grid_columnconfigure(0, weight=1)

        eval_scroll = ctk.CTkScrollableFrame(
            tab_eval,
            fg_color=Theme.SURFACE,
            border_width=0,
            scrollbar_button_color=Theme.ACCENT_SOFT,
            scrollbar_button_hover_color=Theme.ACCENT,
        )
        eval_scroll.grid(row=0, column=0, sticky="nsew")
        eval_scroll.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            eval_scroll,
            text="Model evaluation results",
            font=tnr(FONT_SECTION, "bold"),
            text_color=Theme.ACCENT,
            anchor="w",
        ).pack(anchor="w", padx=4, pady=(0, 6))
        for fig in PAPER_FIGURES:
            self._add_paper_figure(eval_scroll, fig.file, fig.caption, width=248)

        self._graph_host = ctk.CTkFrame(signals_scroll, fg_color=Theme.SURFACE_ALT, height=200)
        self._graph_host.pack(fill="x", padx=4, pady=(0, 6))
        self._graph_host.pack_propagate(False)
        self.graph_label = ctk.CTkLabel(
            self._graph_host,
            text="EAR / MAR",
            font=tnr(FONT_SMALL),
            text_color=Theme.TEXT_DIM,
        )
        self.graph_label.place(relx=0.5, rely=0.5, anchor="center")

        diag_block = ctk.CTkFrame(signals_scroll, fg_color="transparent")
        diag_block.pack(fill="both", expand=True, padx=2, pady=(0, 4))
        diag_block.grid_columnconfigure(0, weight=1)
        diag_block.grid_rowconfigure(1, weight=1)
        diag_head = ctk.CTkFrame(diag_block, fg_color="transparent")
        diag_head.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ctk.CTkLabel(
            diag_head,
            text="Real-time diagnostics",
            font=tnr(FONT_SECTION, "bold"),
            text_color=Theme.ACCENT,
        ).pack(side="left")
        ctk.CTkButton(
            diag_head,
            text="Copy",
            width=56,
            height=26,
            font=tnr(FONT_SMALL),
            fg_color=Theme.SURFACE,
            hover_color=Theme.SURFACE_ALT,
            text_color=Theme.ACCENT,
            border_width=BORDER_W,
            border_color=Theme.BORDER,
            command=self._copy_stats,
        ).pack(side="right")
        self.stats_box = ctk.CTkTextbox(
            diag_block,
            height=140,
            font=ctk.CTkFont(family="Courier New", size=10),
            fg_color=Theme.SURFACE,
            border_width=0,
            text_color=Theme.TEXT_BODY,
            wrap="none",
            activate_scrollbars=True,
        )
        self.stats_box.grid(row=1, column=0, sticky="nsew", padx=2, pady=(0, 2))
        self._bind_stats_textbox()

        cam_card = flat_card(main)
        cam_card.grid(row=0, column=1, sticky="nsew", padx=6)
        cam_card.grid_rowconfigure(1, weight=1)
        cam_card.grid_columnconfigure(0, weight=1)
        card_header(cam_card, "Live camera", row=0)

        self._preview_host = ctk.CTkFrame(cam_card, fg_color=Theme.SURFACE_ALT)
        self._preview_host.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._preview_host.grid_propagate(False)
        self.preview = ctk.CTkLabel(
            self._preview_host,
            text="Camera idle. Press Start.",
            font=tnr(FONT_BODY),
            text_color=Theme.TEXT_DIM,
        )
        self.preview.place(relx=0.5, rely=0.5, anchor="center")

        actions_card = flat_card(main)
        actions_card.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        actions_card.grid_columnconfigure(0, weight=1)
        actions_card.grid_rowconfigure(1, weight=1)
        card_header(actions_card, "Detection controls", row=0)

        actions_scroll = ctk.CTkScrollableFrame(
            actions_card,
            fg_color=Theme.SURFACE,
            border_width=0,
            scrollbar_button_color=Theme.ACCENT_SOFT,
            scrollbar_button_hover_color=Theme.ACCENT,
        )
        actions_scroll.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 8))
        actions_scroll.grid_columnconfigure(0, weight=1)

        status_inner = ctk.CTkFrame(actions_scroll, fg_color="transparent")
        status_inner.pack(fill="x", padx=6, pady=(0, 6))
        self.status_label = ctk.CTkLabel(
            status_inner,
            text="READY",
            font=tnr(FONT_STATUS, "bold"),
            text_color=Theme.SUCCESS,
            anchor="w",
        )
        self.status_label.pack(anchor="w", fill="x")
        self.metrics_line = ctk.CTkLabel(
            status_inner,
            text="EAR: --  |  MAR: --  |  PERCLOS: --%",
            font=tnr(FONT_BODY),
            text_color=Theme.TEXT_BODY,
            anchor="w",
        )
        self.metrics_line.pack(anchor="w", pady=(4, 0))
        self.session_line = ctk.CTkLabel(
            status_inner,
            text="Blinks: 0  |  Yawns: 0  |  Pitch: --  |  Yaw: --",
            font=tnr(FONT_SMALL),
            text_color=Theme.TEXT_MUTED,
            anchor="w",
        )
        self.session_line.pack(anchor="w", pady=(2, 0))
        self.info_label = ctk.CTkLabel(
            status_inner,
            text="Press Start to begin monitoring.",
            font=tnr(FONT_SMALL),
            text_color=Theme.TEXT_MUTED,
            wraplength=260,
            justify="left",
            anchor="w",
        )
        self.info_label.pack(anchor="w", pady=(4, 0))
        self.audio_label = ctk.CTkLabel(
            status_inner,
            text="",
            font=tnr(FONT_SMALL),
            text_color=Theme.TEXT_MUTED,
            anchor="w",
        )
        self.audio_label.pack(anchor="w", pady=(2, 0))
        self._sync_audio_status()

        controls = ctk.CTkFrame(actions_scroll, fg_color="transparent")
        controls.pack(fill="x", padx=6, pady=(0, 4))
        self._add_slider(controls, "EAR threshold", self.cfg.ear_threshold, 0.10, 0.50, self._on_ear)
        self._add_slider(controls, "MAR threshold", self.cfg.mouth_threshold, 0.30, 1.00, self._on_mar)
        self._add_slider(controls, "Tilt threshold", float(self.cfg.tilt_threshold), 10, 90, self._on_tilt, steps=80)
        self._add_slider(
            controls,
            "Eye closure frames",
            float(self.cfg.eye_consec_frames),
            5,
            60,
            self._on_eye_frames,
            steps=55,
            as_int=True,
        )

        btn_col = ctk.CTkFrame(controls, fg_color="transparent")
        btn_col.pack(fill="x", pady=(6, 0))
        self.start_btn = ctk.CTkButton(
            btn_col, text="Start", command=self.start, height=38,
            font=tnr(FONT_BODY + 1, "bold"), fg_color=Theme.ACCENT,
            hover_color=Theme.ACCENT_DARK, text_color="#FFFFFF",
        )
        self.start_btn.pack(fill="x", pady=(0, 6))
        self.stop_btn = ctk.CTkButton(
            btn_col, text="Stop", command=self.stop, height=38,
            font=tnr(FONT_BODY + 1, "bold"), state="disabled",
            fg_color=Theme.NEUTRAL_BTN, hover_color=Theme.NEUTRAL_BTN_HOVER,
            text_color=Theme.NEUTRAL_TEXT, text_color_disabled=Theme.TEXT_MUTED,
        )
        self.stop_btn.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(
            btn_col, text="Test alert sound", command=self._test_alert_sound, height=32,
            font=tnr(FONT_BODY), fg_color=Theme.SURFACE, border_width=BORDER_W,
            border_color=Theme.BORDER, hover_color=Theme.SURFACE_ALT, text_color=Theme.ACCENT,
        ).pack(fill="x", pady=(0, 6))
        ctk.CTkButton(
            btn_col, text="Calibrate pose (C)", command=self._snap_calibrate, height=32,
            font=tnr(FONT_BODY), fg_color=Theme.SURFACE, border_width=BORDER_W,
            border_color=Theme.BORDER, hover_color=Theme.SURFACE_ALT, text_color=Theme.ACCENT,
        ).pack(fill="x", pady=(0, 6))
        ctk.CTkButton(
            btn_col, text="Reset calibration", command=self._reset_calibration, height=32,
            font=tnr(FONT_BODY), fg_color=Theme.NEUTRAL_BTN,
            hover_color=Theme.NEUTRAL_BTN_HOVER, text_color=Theme.TEXT_MUTED,
        ).pack(fill="x", pady=(0, 6))
        ctk.CTkButton(
            btn_col, text="Quit", command=self._on_close, height=32,
            font=tnr(FONT_BODY), fg_color=Theme.NEUTRAL_BTN,
            hover_color=Theme.NEUTRAL_BTN_HOVER, text_color=Theme.TEXT_MUTED,
        ).pack(fill="x")

        acc_card = flat_card(self, fg_color=Theme.ACCENT_SOFT, border_color=Theme.ACCENT)
        acc_card.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 8))
        for col in range(4):
            acc_card.grid_columnconfigure(col, weight=1)
        ctk.CTkLabel(
            acc_card,
            text="Empirical benchmarks",
            font=tnr(FONT_SECTION, "bold"),
            text_color=Theme.ACCENT,
            anchor="w",
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=12, pady=(8, 2))
        m = REFERENCE_METRICS
        ctk.CTkLabel(
            acc_card,
            text=f"{m.model}  |  {m.dataset}",
            font=tnr(FONT_SMALL),
            text_color=Theme.TEXT_MUTED,
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=12, pady=(0, 4))
        tiles = [
            (f"{m.accuracy:.1f}%", "Accuracy"),
            (f"{m.precision:.1f}%", "Precision"),
            (f"{m.recall:.1f}%", "Recall"),
            (f"{m.f1:.1f}%", "F1-score"),
        ]
        for i, (val, lbl) in enumerate(tiles):
            highlight_metric_tile(acc_card, val, lbl).grid(
                row=2, column=i, sticky="ew", padx=6, pady=(0, 6)
            )
        ctk.CTkLabel(
            acc_card,
            text=(
                f"{m.latency_ms} ms alert latency  ·  ROC-AUC {m.roc_auc:.3f}  ·  "
                f"FAR {m.false_alarm_rate:.1f}%  ·  {m.fps_embedded} FPS embedded "
                f"(~{m.embedded_latency_ms} ms)  ·  {m.robustness_band} robustness"
            ),
            font=tnr(FONT_SMALL),
            text_color=Theme.TEXT_BODY,
            anchor="w",
            justify="left",
        ).grid(row=3, column=0, columnspan=4, sticky="w", padx=12, pady=(0, 8))

    def _add_paper_figure(
        self,
        parent: ctk.CTkScrollableFrame,
        filename: str,
        caption: str,
        *,
        width: int,
    ) -> None:
        path = paper_figure_path(filename)
        block = ctk.CTkFrame(parent, fg_color="transparent")
        block.pack(fill="x", pady=(0, 10))
        if not os.path.exists(path):
            ctk.CTkLabel(
                block,
                text=f"{caption}\n(missing {filename})",
                font=tnr(FONT_SMALL),
                text_color=Theme.TEXT_MUTED,
                wraplength=width,
                justify="left",
            ).pack(anchor="w", padx=4)
            return
        image = Image.open(path).convert("RGB")
        ratio = width / max(image.width, 1)
        height = max(int(image.height * ratio), 1)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        ctk_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(width, height),
        )
        self._figure_images.append(ctk_image)
        ctk.CTkLabel(block, image=ctk_image, text="").pack(anchor="w", padx=2)
        ctk.CTkLabel(
            block,
            text=caption,
            font=tnr(FONT_SMALL),
            text_color=Theme.TEXT_MUTED,
            wraplength=width,
            justify="left",
            anchor="w",
        ).pack(anchor="w", padx=4, pady=(4, 0))

    def _sync_audio_status(self) -> None:
        if self._audio.ready:
            state = "looping" if self._audio.is_playing else "ready"
            backend = self._audio.backend or "unknown"
            self.audio_label.configure(
                text=f"Alert audio: {state} via {backend}",
                text_color=Theme.SUCCESS,
            )
        else:
            err = self._audio.last_error or "unavailable"
            self.audio_label.configure(
                text=f"Alert audio: {err}",
                text_color=Theme.DANGER,
            )

    def _test_alert_sound(self) -> None:
        if self._audio.test():
            self.info_label.configure(text="Playing test alert.")
            self._sync_audio_status()
            return
        self.info_label.configure(text=f"Alert sound failed: {self._audio.last_error}")
        self._sync_audio_status()

    def _bind_keys(self) -> None:
        self.bind("<c>", lambda _e: self._snap_calibrate())
        self.bind("<C>", lambda _e: self._snap_calibrate())
        self.bind("<q>", lambda _e: self._on_close())
        self.bind("<Q>", lambda _e: self._on_close())

    def _add_slider(
        self, parent, label, initial, low, high, command, *, steps=100, as_int=False
    ) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            row, text=label, font=tnr(FONT_SMALL), text_color=Theme.TEXT_BODY
        ).pack(side="left")
        fmt = f"{int(initial)}" if as_int else f"{initial:.2f}"
        val_lbl = ctk.CTkLabel(
            row, text=fmt, font=tnr(FONT_SMALL, "bold"), text_color=Theme.ACCENT
        )
        val_lbl.pack(side="right")
        slider = ctk.CTkSlider(
            parent, from_=low, to=high, number_of_steps=steps,
            command=lambda v, lbl=val_lbl, fn=command, ai=as_int: fn(v, lbl, ai),
            button_color=Theme.ACCENT, progress_color=Theme.ACCENT, fg_color=Theme.SLIDER_TRACK,
            height=14,
        )
        slider.set(initial)
        slider.pack(fill="x", pady=(0, 2))

    def _bind_stats_textbox(self) -> None:
        widget = self.stats_box._textbox

        def block_keys(event):
            if event.state & 0x4 and event.keysym.lower() in {"c", "a"}:
                return None
            if event.keysym in {
                "Left", "Right", "Up", "Down", "Prior", "Next",
                "Home", "End", "Shift_L", "Shift_R", "Control_L", "Control_R",
            }:
                return None
            return "break"

        widget.bind("<Key>", block_keys)

    def _on_ear(self, value, lbl, _as_int=False) -> None:
        self._engine.ear_threshold = float(value)
        lbl.configure(text=f"{self._engine.ear_threshold:.2f}")

    def _on_mar(self, value, lbl, _as_int=False) -> None:
        self._engine.mouth_threshold = float(value)
        lbl.configure(text=f"{self._engine.mouth_threshold:.2f}")

    def _on_tilt(self, value, lbl, _as_int=False) -> None:
        self._engine.tilt_threshold = int(value)
        lbl.configure(text=str(self._engine.tilt_threshold))

    def _on_eye_frames(self, value, lbl, as_int=False) -> None:
        self._engine.eye_consec_frames = int(value)
        lbl.configure(text=str(self._engine.eye_consec_frames))

    def _snap_calibrate(self) -> None:
        if not self._last_frame or not self._last_frame.face_found:
            self.info_label.configure(text="Face required for pose calibration.")
            return
        self._engine.snap_calibrate()
        self.info_label.configure(text="Neutral pose calibrated.")

    def _reset_calibration(self) -> None:
        self._engine.recalibrate()
        self.info_label.configure(text="Auto-calibration restarted. Look straight ahead.")

    def _copy_stats(self) -> None:
        text = diagnostics_tree(self._last_frame, self._engine)
        if not text.strip():
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.info_label.configure(text="Diagnostics copied to clipboard.")

    def start(self) -> None:
        if self._running:
            return
        self.info_label.configure(text="Loading landmark model...")
        self.update_idletasks()
        try:
            self._engine.load()
            self._engine.reset_session()
            self._camera.open(self.cfg.webcam_index)
        except Exception as exc:
            self.info_label.configure(text=str(exc))
            return
        self._running = True
        self._graph_tick = 0
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(
            state="normal", fg_color=Theme.DANGER,
            hover_color=Theme.DANGER_HOVER, text_color="#FFFFFF",
        )
        self.info_label.configure(text="Monitoring active.")
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        self._camera.release()
        self._audio.stop()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(
            state="disabled", fg_color=Theme.NEUTRAL_BTN,
            hover_color=Theme.NEUTRAL_BTN_HOVER, text_color=Theme.NEUTRAL_TEXT,
        )
        self.status_label.configure(text="STOPPED", text_color=Theme.TEXT_MUTED)
        self.preview.configure(image=None, text="Camera idle")
        self.graph_label.configure(image=None, text="EAR / MAR")
        self.info_label.configure(text="Stopped.")

    def _loop(self) -> None:
        while self._running:
            frame = self._camera.read()
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                result = self._engine.process_frame(frame)
            except Exception as exc:
                self.after(0, lambda e=str(exc): self.info_label.configure(text=e))
                time.sleep(0.2)
                continue
            with self._lock:
                self._latest = result
            time.sleep(0.01)

    def _poll_ui(self) -> None:
        with self._lock:
            result = self._latest
            self._latest = None
        if result is not None:
            self._show_result(result)
        self._audio.pump()
        if self.winfo_exists():
            self.after(33, self._poll_ui)

    def _set_ctk_image(
        self, label: ctk.CTkLabel, bgr: np.ndarray, dims: tuple[int, int], attr: str
    ) -> None:
        pw, ph = dims
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(rgb)
        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(pw, ph))
        setattr(self, attr, ctk_image)
        label.configure(image=ctk_image, text="")
        label.image = ctk_image

    def _show_result(self, result: FrameResult) -> None:
        self._last_frame = result
        if result.label == "ALERT":
            color = Theme.DANGER
        elif result.face_found:
            color = Theme.SUCCESS
        else:
            color = Theme.WARNING
        self.status_label.configure(text=result.status_text, text_color=color)

        self.metrics_line.configure(
            text=(
                f"EAR: {result.ear:.2f}  |  MAR: {result.mar:.2f}  |  "
                f"PERCLOS: {result.perclos:.1f}%"
            )
        )
        self.session_line.configure(
            text=(
                f"Blinks: {result.total_blinks}  |  Yawns: {result.total_yawns}  |  "
                f"Pitch: {int(result.pitch)}°  |  Yaw: {int(result.yaw)}° "
                f"(±{self._engine.tilt_threshold})"
            )
        )

        alert_on = result.is_drowsy or result.is_yawning or result.is_distracted
        self._audio.set_active(alert_on)
        self._sync_audio_status()
        if not result.calibrated:
            self.info_label.configure(
                text=f"Calibrating pose... {result.calibration_progress * 100:.0f}%"
            )

        text = diagnostics_tree(result, self._engine)
        if text != self._stats_text_cache:
            self.stats_box.delete("1.0", "end")
            self.stats_box.insert("1.0", text)
            self._stats_text_cache = text

        self._set_ctk_image(
            self.preview, result.annotated_bgr, self._preview_dims, "_preview_ctk"
        )

        self._graph_tick += 1
        if self._graph_tick % 5 == 0:
            graph_bgr = self._graphs.render(
                self._engine.ear_history,
                self._engine.mar_history,
                self._engine.ear_threshold,
                self._engine.mouth_threshold,
            )
            self._set_ctk_image(self.graph_label, graph_bgr, self._graph_dims, "_graph_ctk")

    def _on_close(self) -> None:
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self._camera.release()
        self._audio.stop()
        self.destroy()


def run_app(cfg: AppConfig = DEFAULT) -> None:
    app = MainWindow(cfg)
    app.mainloop()
