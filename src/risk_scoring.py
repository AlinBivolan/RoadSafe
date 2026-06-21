from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreWeights:
    """Ponderi interpretetype pentru scoruri de risc."""

    frequency: float = 0.35
    severity: float = 0.40
    severe_share: float = 0.25

    def validate(self) -> None:
        total = self.frequency + self.severity + self.severe_share
        if not np.isclose(total, 1.0):
            raise ValueError(f"Ponderile trebuie să însumeze 1.0, dar însumează {total:.3f}")


GRID_WEIGHTS = ScoreWeights(0.35, 0.40, 0.25)
PREVENTION_WEIGHTS = ScoreWeights(0.35, 0.40, 0.25)
ROUTE_WEIGHTS = ScoreWeights(0.15, 0.60, 0.25)

SEVERITY_SCORE_MAP = {
    1: 8,  # Fatal
    2: 4,  # Serious
    3: 1,  # Slight
}


def minmax(series: pd.Series) -> pd.Series:
    """Normalizează o serie numerică în intervalul [0, 1]."""
    if series.empty:
        return series.astype(float)
    smin = series.min()
    smax = series.max()
    if smax == smin:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - smin) / (smax - smin)


def weighted_composite_score(
    frequency_norm: pd.Series | float,
    severity_norm: pd.Series | float,
    severe_share_norm: pd.Series | float,
    weights: ScoreWeights = GRID_WEIGHTS,
) -> pd.Series | float:
    """Calculează scor compozit 0–100 din trei componente normalizate."""
    weights.validate()
    return (
        weights.frequency * frequency_norm
        + weights.severity * severity_norm
        + weights.severe_share * severe_share_norm
    ) * 100.0


def risk_level(score: float) -> str:
    """Etichetă ușor de afișat pentru un scor 0–100."""
    try:
        value = float(score)
    except Exception:
        return "Necunoscut"
    if value < 30:
        return "Scăzut"
    if value < 60:
        return "Mediu"
    return "Ridicat"


def severity_interpretation_from_weighted_score(value: float) -> str:
    """Interpretare pentru scorul ponderat 1=Slight, 4=Serious, 8=Fatal."""
    try:
        v = float(value)
    except Exception:
        return "Necunoscut"
    if v < 2.0:
        return "scăzută / preponderent ușoară"
    if v < 5.5:
        return "medie / influență serious"
    return "ridicată / influență fatală"
