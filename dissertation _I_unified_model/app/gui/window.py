from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
from PIL import Image

from app.audio import AlertPlayer
from app.camera import VideoSource
from app.config import (
    DEFAULT,
    AppConfig,
    alert_sound_path,
    icon_path,
    paper_figure_path,
)
from app.dissertation import (
    DISSERTATION,
    PAPER_FIGURES,
    REFERENCE_METRICS,
)
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
from app.face import FaceDetector
from app.inference import DetectionResult, load_model, predict_frame
from app.stats import (
    RuntimeStats,
    collect_dump,
    compose_export,
    default_export_dir,
    dump_to_lines,
    format_dump_text,
    format_tree_text,
)


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
        self._latest: DetectionResult | None = None
        self._drowsy_streak = 0
        self._threshold = cfg.drowsy_threshold
        self._camera_index = cfg.webcam_index

        self._model = None
        self._device = None
        self._face_detector = FaceDetector()
        self._camera = VideoSource(cfg.webcam_index)
        self._audio = AlertPlayer(alert_sound_path(cfg))
        self._runtime = RuntimeStats()
        self._last_frame: DetectionResult | None = None
        self._stats_tick = 0
        self._stats_mode = "important"
        self._stats_text_cache = ""
        self._figure_images: list[ctk.CTkImage] = []

        self._build_ui()
        self._try_set_icon()
        if not self._audio.preload():
            err = self._audio.last_error or "unknown error"
            self.info_label.configure(text=f"Alert sound unavailable: {err}")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._poll_ui)
        self._refresh_stats_panel(force=True)

    def _try_set_icon(self) -> None:
        try:
            path = icon_path(self.cfg)
            self.iconbitmap(path)
        except Exception:
            pass

    def _preview_size(self) -> tuple[int, int]:
        self.update_idletasks()
        if hasattr(self, "_preview_host"):
            w = self._preview_host.winfo_width() - 28
            h = self._preview_host.winfo_height() - 36
            if w > 120 and h > 90:
                return w, h
        return self.cfg.preview_width, self.cfg.preview_height

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
        main.grid_columnconfigure(0, weight=2, minsize=260)
        main.grid_columnconfigure(1, weight=4, minsize=420)
        main.grid_columnconfigure(2, weight=2, minsize=260)
        main.grid_rowconfigure(0, weight=1)

        paper_card = flat_card(main)
        paper_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        paper_card.grid_rowconfigure(1, weight=1)
        paper_card.grid_columnconfigure(0, weight=1)
        card_header(paper_card, "Model evaluation results", row=0)
        paper_scroll = ctk.CTkScrollableFrame(
            paper_card,
            fg_color=Theme.SURFACE,
            border_width=0,
            scrollbar_button_color=Theme.ACCENT_SOFT,
            scrollbar_button_hover_color=Theme.ACCENT,
        )
        paper_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        for fig in PAPER_FIGURES:
            self._add_paper_figure(paper_scroll, fig.file, fig.caption, width=248)

        cam_card = flat_card(main)
        cam_card.grid(row=0, column=1, sticky="nsew", padx=6)
        cam_card.grid_rowconfigure(1, weight=1)
        cam_card.grid_columnconfigure(0, weight=1)
        card_header(cam_card, "Live camera")
        self._preview_host = ctk.CTkFrame(cam_card, fg_color=Theme.SURFACE_ALT)
        self._preview_host.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._preview_host.grid_rowconfigure(0, weight=1)
        self._preview_host.grid_columnconfigure(0, weight=1)
        self.preview = ctk.CTkLabel(
            self._preview_host,
            text="Camera idle. Press Start.",
            font=tnr(FONT_BODY),
            text_color=Theme.TEXT_DIM,
        )
        self.preview.grid(row=0, column=0, sticky="nsew")

        actions_card = flat_card(main)
        actions_card.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        actions_card.grid_columnconfigure(0, weight=1)
        actions_card.grid_rowconfigure(3, weight=1)
        card_header(actions_card, "Experimental controls")

        status_inner = ctk.CTkFrame(actions_card, fg_color="transparent")
        status_inner.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        self.status_label = ctk.CTkLabel(
            status_inner,
            text="READY",
            font=tnr(FONT_STATUS, "bold"),
            text_color=Theme.SUCCESS,
        )
        self.status_label.pack(anchor="w")
        self.score_label = ctk.CTkLabel(
            status_inner,
            text="CDS: --  |  alert: --  |  infer: -- ms",
            font=tnr(FONT_BODY),
            text_color=Theme.TEXT_BODY,
        )
        self.score_label.pack(anchor="w", pady=(4, 0))
        self.info_label = ctk.CTkLabel(
            status_inner,
            text="Press Start to commence detection.",
            font=tnr(FONT_BODY),
            text_color=Theme.TEXT_MUTED,
            wraplength=280,
            justify="left",
            anchor="w",
        )
        self.info_label.pack(anchor="w", pady=(4, 0))

        controls = ctk.CTkFrame(actions_card, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        thresh_row = ctk.CTkFrame(controls, fg_color="transparent")
        thresh_row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(
            thresh_row,
            text="Drowsiness threshold",
            font=tnr(FONT_BODY),
            text_color=Theme.TEXT_BODY,
        ).pack(side="left")
        self.thresh_label = ctk.CTkLabel(
            thresh_row,
            text=f"{self._threshold:.2f}",
            font=tnr(FONT_BODY, "bold"),
            text_color=Theme.ACCENT,
        )
        self.thresh_label.pack(side="right")
        self.thresh_slider = ctk.CTkSlider(
            controls,
            from_=0.1,
            to=0.95,
            number_of_steps=85,
            command=self._on_threshold,
            button_color=Theme.ACCENT,
            progress_color=Theme.ACCENT,
            fg_color=Theme.SLIDER_TRACK,
        )
        self.thresh_slider.set(self._threshold)
        self.thresh_slider.pack(fill="x", pady=(0, 12))

        btn_col = ctk.CTkFrame(controls, fg_color="transparent")
        btn_col.pack(fill="x")
        self.start_btn = ctk.CTkButton(
            btn_col,
            text="Start",
            command=self.start,
            height=42,
            font=tnr(FONT_BODY + 1, "bold"),
            fg_color=Theme.ACCENT,
            hover_color=Theme.ACCENT_DARK,
            text_color="#FFFFFF",
            corner_radius=6,
        )
        self.start_btn.pack(fill="x", pady=(0, 6))
        self.stop_btn = ctk.CTkButton(
            btn_col,
            text="Stop",
            command=self.stop,
            height=42,
            font=tnr(FONT_BODY + 1, "bold"),
            state="disabled",
            fg_color=Theme.NEUTRAL_BTN,
            hover_color=Theme.NEUTRAL_BTN_HOVER,
            text_color=Theme.NEUTRAL_TEXT,
            text_color_disabled=Theme.TEXT_MUTED,
            corner_radius=6,
        )
        self.stop_btn.pack(fill="x", pady=(0, 6))
        self.export_btn = ctk.CTkButton(
            btn_col,
            text="Export",
            command=self.export_snapshot,
            height=42,
            font=tnr(FONT_BODY + 1, "bold"),
            fg_color=Theme.SURFACE,
            border_width=BORDER_W,
            border_color=Theme.ACCENT,
            hover_color=Theme.SURFACE_ALT,
            text_color=Theme.ACCENT,
            corner_radius=6,
        )
        self.export_btn.pack(fill="x", pady=(0, 6))
        self.quit_btn = ctk.CTkButton(
            btn_col,
            text="Quit",
            command=self._on_close,
            height=34,
            font=tnr(FONT_BODY),
            fg_color=Theme.NEUTRAL_BTN,
            hover_color=Theme.NEUTRAL_BTN_HOVER,
            text_color=Theme.TEXT_MUTED,
            corner_radius=6,
        )
        self.quit_btn.pack(fill="x")

        diag_block = ctk.CTkFrame(actions_card, fg_color="transparent")
        diag_block.grid(row=3, column=0, sticky="nsew", padx=8, pady=(4, 10))
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
        self.copy_stats_btn = ctk.CTkButton(
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
        )
        self.copy_stats_btn.pack(side="right")
        self.full_details_switch = ctk.CTkSwitch(
            diag_head,
            text="Full details",
            command=self._on_full_details_switch,
            font=tnr(FONT_SMALL),
            text_color=Theme.TEXT_BODY,
            fg_color=Theme.SLIDER_TRACK,
            progress_color=Theme.ACCENT,
            button_color=Theme.SURFACE,
            button_hover_color=Theme.NEUTRAL_BTN_HOVER,
        )
        self.full_details_switch.pack(side="right", padx=(0, 8))

        self.stats_box = ctk.CTkTextbox(
            diag_block,
            height=128,
            font=ctk.CTkFont(family="Courier New", size=10),
            fg_color=Theme.SURFACE,
            border_width=0,
            text_color=Theme.TEXT_BODY,
            wrap="none",
            activate_scrollbars=True,
        )
        self.stats_box.grid(row=1, column=0, sticky="nsew", padx=2, pady=(0, 4))
        self._bind_stats_textbox()

        acc_card = flat_card(
            self,
            fg_color=Theme.ACCENT_SOFT,
            border_color=Theme.ACCENT,
        )
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

    def _bind_stats_textbox(self) -> None:
        widget = self.stats_box._textbox

        def block_keys(event):
            if event.state & 0x4:
                if event.keysym.lower() in {"c", "a"}:
                    return None
            if event.keysym in {
                "Left",
                "Right",
                "Up",
                "Down",
                "Prior",
                "Next",
                "Home",
                "End",
                "Shift_L",
                "Shift_R",
                "Control_L",
                "Control_R",
            }:
                return None
            return "break"

        widget.bind("<Key>", block_keys)
        for seq in (
            "<Button-1>",
            "<B1-Motion>",
            "<MouseWheel>",
            "<Button-4>",
            "<Button-5>",
        ):
            widget.bind(seq, self._on_stats_interact, add="+")
            self.stats_box.bind(seq, self._on_stats_interact, add="+")

    def _on_stats_interact(self, _event=None):
        pass

    def _resume_stats_autoscroll(self) -> None:
        pass

    def _on_full_details_switch(self) -> None:
        self._stats_mode = "all" if self.full_details_switch.get() else "important"
        self._stats_text_cache = ""
        self._refresh_stats_panel(force=True)

    def _copy_stats(self) -> None:
        dump = collect_dump(
            self.cfg,
            self._runtime,
            threshold=self._threshold,
            camera_index=self._camera_index,
            device=self._device,
            model=self._model,
        )
        text = format_dump_text(dump, mode=self._stats_mode)
        if not text.strip():
            self.info_label.configure(text="No diagnostics to copy yet.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.info_label.configure(text="Diagnostics copied to clipboard.")

    def _on_threshold(self, value: float) -> None:
        self._threshold = float(value)
        self.thresh_label.configure(text=f"{self._threshold:.2f}")

    def start(self) -> None:
        if self._running:
            return

        self.info_label.configure(text="Loading model...")
        self.update_idletasks()

        try:
            if self._model is None:
                self._model, self._device = load_model(self.cfg)
            self._camera.open(self._camera_index)
        except Exception as exc:
            self.info_label.configure(text=str(exc))
            return

        self._running = True
        self._drowsy_streak = 0
        self._runtime.reset_session()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(
            state="normal",
            fg_color=Theme.DANGER,
            hover_color=Theme.DANGER_HOVER,
            text_color="#FFFFFF",
        )
        self.info_label.configure(text="Running. Export anytime for a dump.")
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        self._camera.release()
        self._audio.stop()
        self._drowsy_streak = 0
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(
            state="disabled",
            fg_color=Theme.NEUTRAL_BTN,
            hover_color=Theme.NEUTRAL_BTN_HOVER,
            text_color=Theme.NEUTRAL_TEXT,
        )
        self.status_label.configure(text="STOPPED", text_color=Theme.TEXT_MUTED)
        self.score_label.configure(text="score: --")
        self.preview.configure(image=None, text="Camera idle")
        self.info_label.configure(text="Stopped. Last frame still exportable.")
        self._refresh_stats_panel(force=True)

    def export_snapshot(self) -> None:
        result = self._last_frame
        if result is None:
            messagebox.showinfo(
                "Export",
                "No frame yet. Press Start and wait for one preview frame.",
            )
            return

        dump = collect_dump(
            self.cfg,
            self._runtime,
            threshold=self._threshold,
            camera_index=self._camera_index,
            device=self._device,
            model=self._model,
        )
        lines = dump_to_lines(dump, mode="all")
        composite = compose_export(result.annotated_bgr, lines)

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        initial = os.path.join(default_export_dir(), f"drowsiness-{stamp}.png")
        path = filedialog.asksaveasfilename(
            title="Export frame + diagnostics",
            initialfile=os.path.basename(initial),
            initialdir=default_export_dir(),
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        ok = cv2.imwrite(path, composite)
        if not ok:
            messagebox.showerror("Export", f"Failed to write {path}")
            return

        sidecar = os.path.splitext(path)[0] + ".json"
        try:
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(dump, f, indent=2, default=str)
        except Exception as exc:
            self.info_label.configure(
                text=f"Saved image, JSON failed: {exc}"
            )
            return

        self.info_label.configure(
            text=f"Exported:\n{os.path.basename(path)}\n+ {os.path.basename(sidecar)}"
        )

    def _loop(self) -> None:
        while self._running:
            frame = self._camera.read()
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                result = predict_frame(
                    self._model,
                    frame,
                    self._device,
                    threshold=self._threshold,
                    cfg=self.cfg,
                    face_detector=self._face_detector,
                )
            except Exception as exc:
                with self._lock:
                    self._latest = None
                self.after(
                    0,
                    lambda e=str(exc): self.info_label.configure(text=e),
                )
                time.sleep(0.2)
                continue

            if result.is_drowsy:
                self._drowsy_streak += 1
            else:
                self._drowsy_streak = 0

            alert_on = self._drowsy_streak >= self.cfg.consecutive_drowsy_frames
            self._runtime.observe(
                result, result.infer_ms, self._drowsy_streak, alert_on
            )

            with self._lock:
                self._latest = result

            time.sleep(0.01)

    def _poll_ui(self) -> None:
        result: DetectionResult | None
        with self._lock:
            result = self._latest
            self._latest = None

        if result is not None:
            self._show_result(result)

        self._stats_tick += 1
        # ~1s refresh; don't fight the user while they scroll/select
        if self._stats_tick % 30 == 0:
            self._refresh_stats_panel()

        if self.winfo_exists():
            self.after(33, self._poll_ui)

    def _show_result(self, result: DetectionResult) -> None:
        self._last_frame = result
        if result.is_drowsy:
            color = Theme.DANGER
        elif result.face_found:
            color = Theme.SUCCESS
        else:
            color = Theme.WARNING
        self.status_label.configure(text=result.label, text_color=color)
        face_tag = "" if result.face_found else " · roi"
        self.score_label.configure(
            text=(
                f"CDS drowsy {result.score:.2f}  |  alert {result.score_alert:.2f}"
                f"{face_tag}  |  {result.infer_ms:.0f} ms"
            )
        )
        self._audio.set_active(self._runtime.alert_on)

        pw, ph = self._preview_size()
        rgb = cv2.cvtColor(result.annotated_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (pw, ph))
        image = Image.fromarray(rgb)
        ctk_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(pw, ph),
        )
        self.preview.configure(image=ctk_image, text="")
        self.preview.image = ctk_image

    def _refresh_stats_panel(self, force: bool = False) -> None:
        dump = collect_dump(
            self.cfg,
            self._runtime,
            threshold=self._threshold,
            camera_index=self._camera_index,
            device=self._device,
            model=self._model,
        )
        text = format_tree_text(dump, mode=self._stats_mode)
        if text == self._stats_text_cache and not force:
            return
        self.stats_box.delete("1.0", "end")
        self.stats_box.insert("1.0", text)
        self._stats_text_cache = text

    def _on_close(self) -> None:
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        self._camera.release()
        self._audio.stop()
        self.destroy()


def run_app(cfg: AppConfig = DEFAULT) -> None:
    app = MainWindow(cfg)
    app.mainloop()
