"""Motor quantitativo: devig, modelo de gols, ajuste ao vivo e valor."""

from .devig import margin, remove_vig
from .live import LiveConfig, live_matrix, remaining_fraction
from .poisson import ScoreMatrix, calibrate
from .value import edge, evaluate, expected_value, filter_value, kelly

__all__ = [
    "remove_vig",
    "margin",
    "ScoreMatrix",
    "calibrate",
    "live_matrix",
    "remaining_fraction",
    "LiveConfig",
    "expected_value",
    "edge",
    "kelly",
    "evaluate",
    "filter_value",
]
