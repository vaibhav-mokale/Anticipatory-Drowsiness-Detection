from __future__ import annotations

import collections

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class LiveSignalGraphs:
    def __init__(self, history_len: int = 50) -> None:
        self.history_len = history_len
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(3.0, 2.5), dpi=110)
        self.fig.patch.set_facecolor("#F7F3F3")
        self.line_ear, = self.ax1.plot([], [], color="#5E81AC", linewidth=2)
        self.line_mar, = self.ax2.plot([], [], color="#BF616A", linewidth=2)
        self._ear_thresh_line = None
        self._mar_thresh_line = None
        self._setup_ax(self.ax1, "Eye Aspect Ratio (EAR)", 0.5)
        self._setup_ax(self.ax2, "Mouth Aspect Ratio (MAR)", 1.0)
        self.fig.tight_layout()

    @staticmethod
    def _setup_ax(ax, title: str, ylim: float) -> None:
        ax.set_title(title, color="#2E3440", fontsize=9, fontname="serif", weight="bold")
        ax.set_ylim(0, ylim)
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(colors="#2E3440", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#C45C5C")
        ax.grid(True, color="#D8DEE9", linewidth=0.5, linestyle="--")

    def render(
        self,
        ear_history: collections.deque,
        mar_history: collections.deque,
        ear_threshold: float,
        mar_threshold: float,
    ) -> np.ndarray:
        x_vals = range(len(ear_history))
        self.line_ear.set_data(x_vals, list(ear_history))
        self.ax1.set_xlim(0, self.history_len)
        if self._ear_thresh_line is not None:
            self._ear_thresh_line.remove()
        self._ear_thresh_line = self.ax1.axhline(
            y=ear_threshold, color="#B42318", linestyle="--", alpha=0.85
        )

        self.line_mar.set_data(x_vals, list(mar_history))
        self.ax2.set_xlim(0, self.history_len)
        if self._mar_thresh_line is not None:
            self._mar_thresh_line.remove()
        self._mar_thresh_line = self.ax2.axhline(
            y=mar_threshold, color="#B42318", linestyle="--", alpha=0.85
        )

        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
