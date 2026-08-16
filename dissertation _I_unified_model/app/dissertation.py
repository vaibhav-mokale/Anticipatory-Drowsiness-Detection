from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DissertationInfo:
    title_line: str
    subtitle: str
    window_title: str
    framework: str
    score_name: str
    degree: str
    student: str
    student_id: str
    guide: str
    hod: str
    month_year: str


@dataclass(frozen=True)
class HighlightBullet:
    text: str


@dataclass(frozen=True)
class AttributionRow:
    label: str
    value: str


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
    framework="ADAS",
    score_name="Composite Drowsiness Score (CDS)",
    degree="M.Tech Computer Science and Engineering",
    student="Vaibhav Maroti Mokale",
    student_id="MT24F05F005",
    guide="Dr. Vikul J. Pawar",
    hod="Dr. Vikul J. Pawar",
    month_year="July 2026",
)

HIGHLIGHT_BULLETS: tuple[HighlightBullet, ...] = (
    HighlightBullet(
        "Unified CNN-LSTM-ViT framework fuses spatial, temporal and "
        "global-context cues into one anticipatory pipeline."
    ),
    HighlightBullet(
        "Achieves 98.9% accuracy, 96.8% precision, 91.0% recall, "
        "outperforming classical ML and standalone deep models."
    ),
    HighlightBullet(
        "Composite Drowsiness Score (CDS) and RARI enable continuous, "
        "interpretable, proactive risk quantification."
    ),
    HighlightBullet("Real-time feasible: 192 ms alert latency on embedded GPUs."),
    HighlightBullet(
        "Robust across lighting, occlusion and pose variation "
        "(97-99% accuracy band)."
    ),
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
    PaperFigure("ROC_Curve.png", "ROC curve"),
    PaperFigure("Precision_Recall_Curve.png", "Precision-recall curve"),
)

SCHOLARLY_CREDIT_ROWS: tuple[AttributionRow, ...] = (
    AttributionRow(
        "Candidate",
        f"{DISSERTATION.student} ({DISSERTATION.student_id})",
    ),
    AttributionRow("Research Supervisor", DISSERTATION.guide),
    AttributionRow("Head of Department", DISSERTATION.hod),
    AttributionRow(
        "Degree Programme",
        f"{DISSERTATION.degree}, {DISSERTATION.month_year}",
    ),
)

ATTRIBUTION_ROWS = SCHOLARLY_CREDIT_ROWS
