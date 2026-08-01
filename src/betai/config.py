"""Configuração via variáveis de ambiente, com defaults sensatos."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    # Provedor
    provider: str = os.environ.get("BETAI_PROVIDER", "mock")
    odds_api_key: str = os.environ.get("THE_ODDS_API_KEY", "")
    sport: str = os.environ.get("BETAI_SPORT", "soccer_brazil_campeonato")
    regions: str = os.environ.get("BETAI_REGIONS", "eu")
    bookmaker: str | None = os.environ.get("BETAI_BOOKMAKER") or None
    field_map_path: str | None = os.environ.get("BETAI_FIELD_MAP") or None

    # Motor
    devig_method: str = os.environ.get("BETAI_DEVIG", "power")
    prior_total_goals: float = _env_float("BETAI_PRIOR_TOTAL", 2.7)
    min_edge: float = _env_float("BETAI_MIN_EDGE", 0.03)
    kelly_fraction: float = _env_float("BETAI_KELLY_FRACTION", 0.25)

    # IA
    anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY") or None
    model: str = os.environ.get("BETAI_MODEL", "claude-opus-5")
    effort: str = os.environ.get("BETAI_EFFORT", "medium")
    max_ai_shift: float = _env_float("BETAI_MAX_AI_SHIFT", 0.05)

    # Persistência
    db_path: str = os.environ.get("BETAI_DB", str(Path.cwd() / "bet-ai.db"))


def load() -> Settings:
    return Settings()
