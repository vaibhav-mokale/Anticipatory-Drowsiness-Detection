from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DissertationInfo:
    title_line: str
    subtitle: str
    window_title: str


@dataclass(frozen=True)
class PaperFigure:
    file: str
    caption: str


@dataclass(frozen=True)
class ReferenceMetrics:
    dataset: str
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    latency_ms: int
    false_alarm_rate: float
    fps_embedded: int
    embedded_latency_ms: int
    robustness_band: str


DISSERTATION = DissertationInfo(
    title_line=(
        "Anticipatory Driver Drowsiness Detection Using a Unified Deep Learning "
        "Framework Leveraging the Composite Drowsiness Score"
    ),
    subtitle="ADAS  |  Composite Drowsiness Score (CDS)  |  M.Tech CSE",
    window_title="ADAS | Anticipatory Drowsiness Detection",
)

REFERENCE_METRICS = ReferenceMetrics(
    dataset="NTHU-DDD test set",
    model="CNN-LSTM-ViT hybrid",
    accuracy=98.9,
    precision=96.8,
    recall=91.0,
    f1=98.8,
    roc_auc=0.991,
    latency_ms=192,
    false_alarm_rate=1.8,
    fps_embedded=22,
    embedded_latency_ms=38,
    robustness_band="97-99%",
)

PAPER_FIGURES: tuple[PaperFigure, ...] = (
    PaperFigure(
        "Training_Validation_Accuracy.png",
        "Training and validation accuracy",
    ),
    PaperFigure("Training_Loss.png", "Training loss convergence"),
    PaperFigure("Confusion_Matrix.png", "Confusion matrix on test set"),
    PaperFigure("Precision_Recall_Curve.png", "Precision-recall curve"),
)
