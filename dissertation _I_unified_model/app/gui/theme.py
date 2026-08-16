from __future__ import annotations

import customtkinter as ctk

ICON_DOT = "\u25cb"
ICON_DIAMOND = "\u25c7"

BORDER_W = 1

FONT_BODY = 12
FONT_SMALL = 11
FONT_SECTION = 14
FONT_TITLE = 22
FONT_METRIC = 19
FONT_COMPACT = 10
FONT_BOTTOM = 11
FONT_STATUS = 24


class Theme:
    BG = "#FCFAFA"
    SURFACE = "#FFFFFF"
    SURFACE_ALT = "#F7F3F3"
    BORDER = "#C45C5C"
    TEXT = "#12161C"
    TEXT_BODY = "#1E2430"
    TEXT_MUTED = "#3A4250"
    TEXT_DIM = "#4E5868"
    ACCENT = "#B42318"
    ACCENT_SOFT = "#FCE8E8"
    ACCENT_DARK = "#8F1A12"
    SUCCESS = "#166534"
    SUCCESS_HOVER = "#14532D"
    DANGER = "#B42318"
    DANGER_HOVER = "#8F1A12"
    WARNING = "#9A3412"
    NEUTRAL_BTN = "#ECECEC"
    NEUTRAL_BTN_HOVER = "#DDDDDD"
    NEUTRAL_TEXT = "#1A1F24"
    SLIDER_TRACK = "#F0E4E4"

    PAD_TIGHT = 3
    PAD_ROW = 6
    PAD_SECTION = 10


def tnr(size: int, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family="Times New Roman", size=size, weight=weight)


def section_label(parent, title: str) -> ctk.CTkLabel:
    return ctk.CTkLabel(
        parent,
        text=title,
        font=tnr(FONT_SECTION, "bold"),
        text_color=Theme.ACCENT,
        anchor="w",
    )


def flat_card(parent, **kwargs) -> ctk.CTkFrame:
    opts = {
        "fg_color": Theme.SURFACE,
        "corner_radius": 8,
        "border_width": BORDER_W,
        "border_color": Theme.BORDER,
    }
    opts.update(kwargs)
    return ctk.CTkFrame(parent, **opts)


def card_header(parent, title: str, row: int = 0) -> ctk.CTkLabel:
    lbl = ctk.CTkLabel(
        parent,
        text=title,
        font=tnr(FONT_SECTION, "bold"),
        text_color=Theme.ACCENT,
        anchor="w",
    )
    lbl.grid(row=row, column=0, sticky="w", padx=12, pady=(12, 6))
    return lbl


def info_line(parent, text: str, *, wrap: int = 300) -> None:
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=12, pady=Theme.PAD_TIGHT)
    ctk.CTkLabel(
        row,
        text=ICON_DOT,
        width=14,
        font=tnr(FONT_BODY),
        text_color=Theme.ACCENT,
    ).pack(side="left", anchor="n", padx=(0, 6))
    ctk.CTkLabel(
        row,
        text=text,
        font=tnr(FONT_BODY),
        text_color=Theme.TEXT_BODY,
        wraplength=wrap,
        justify="left",
        anchor="w",
    ).pack(side="left", fill="x", expand=True)


def bullet_line(parent, text: str, *, wrap: int = 300) -> None:
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=12, pady=Theme.PAD_ROW)
    ctk.CTkLabel(
        row,
        text=ICON_DIAMOND,
        width=14,
        font=tnr(FONT_BODY),
        text_color=Theme.ACCENT,
    ).pack(side="left", anchor="n", padx=(0, 6))
    ctk.CTkLabel(
        row,
        text=text,
        font=tnr(FONT_BODY),
        text_color=Theme.TEXT_BODY,
        wraplength=wrap,
        justify="left",
        anchor="w",
    ).pack(side="left", fill="x", expand=True)


def metric_tile(parent, value: str, label: str) -> ctk.CTkFrame:
    tile = ctk.CTkFrame(
        parent,
        fg_color=Theme.SURFACE_ALT,
        corner_radius=6,
        border_width=BORDER_W,
        border_color=Theme.BORDER,
    )
    ctk.CTkLabel(
        tile,
        text=value,
        font=tnr(FONT_METRIC, "bold"),
        text_color=Theme.ACCENT,
    ).pack(pady=(10, 0))
    ctk.CTkLabel(
        tile,
        text=label,
        font=tnr(FONT_SMALL),
        text_color=Theme.TEXT_BODY,
    ).pack(pady=(0, 10))
    return tile


def bottom_header(parent, title: str, row: int = 0) -> ctk.CTkLabel:
    lbl = ctk.CTkLabel(
        parent,
        text=title,
        font=tnr(FONT_BOTTOM, "bold"),
        text_color=Theme.ACCENT,
        anchor="w",
    )
    lbl.grid(row=row, column=0, sticky="w", padx=10, pady=(6, 2))
    return lbl


def ghost_button(parent, text: str, command) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        width=48,
        height=22,
        font=tnr(FONT_COMPACT),
        fg_color="transparent",
        hover_color=Theme.SURFACE_ALT,
        text_color=Theme.TEXT_MUTED,
        border_width=0,
        corner_radius=4,
    )


def highlight_metric_tile(parent, value: str, label: str) -> ctk.CTkFrame:
    tile = ctk.CTkFrame(
        parent,
        fg_color=Theme.SURFACE,
        corner_radius=8,
        border_width=BORDER_W,
        border_color=Theme.ACCENT,
    )
    ctk.CTkLabel(
        tile,
        text=value,
        font=tnr(18, "bold"),
        text_color=Theme.ACCENT,
    ).pack(pady=(8, 0))
    ctk.CTkLabel(
        tile,
        text=label,
        font=tnr(FONT_SMALL),
        text_color=Theme.TEXT_BODY,
    ).pack(pady=(0, 8))
    return tile
