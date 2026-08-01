"""Configuração via variáveis de ambiente, com defaults sensatos."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Carrega um arquivo .env para o ambiente, sem sobrescrever o que já existe.

    Uma variável exportada na mão vence o arquivo — é o comportamento que as
    pessoas esperam quando querem testar algo pontualmente sem editar o .env.

    Implementado aqui em vez de usar python-dotenv: são vinte linhas, e cada
    dependência a mais é mais uma chance de a instalação falhar.
    """
    arquivo = Path(path)
    if not arquivo.is_file():
        return {}

    carregadas: dict[str, str] = {}
    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, _, valor = linha.partition("=")
        chave = chave.strip()
        valor = valor.strip().strip('"').strip("'")
        if chave and chave not in os.environ:
            os.environ[chave] = valor
            carregadas[chave] = valor
    return carregadas


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env(name: str, default: str = "") -> Callable[[], str]:
    return lambda: os.environ.get(name, default)


def _env_opt(name: str) -> Callable[[], str | None]:
    return lambda: os.environ.get(name) or None


@dataclass
class Settings:
    """Configuração lida do ambiente.

    Todos os campos usam `default_factory` de propósito: um valor padrão
    comum seria avaliado quando o módulo é importado, antes do `.env` ser
    carregado, e o arquivo nunca teria efeito.
    """

    # Provedor
    provider: str = field(default_factory=_env("BETAI_PROVIDER", "mock"))
    odds_api_key: str = field(default_factory=_env("THE_ODDS_API_KEY"))
    sport: str = field(default_factory=_env("BETAI_SPORT", "soccer_brazil_campeonato"))
    regions: str = field(default_factory=_env("BETAI_REGIONS", "eu"))
    bookmaker: str | None = field(default_factory=_env_opt("BETAI_BOOKMAKER"))
    field_map_path: str | None = field(default_factory=_env_opt("BETAI_FIELD_MAP"))

    # Motor
    devig_method: str = field(default_factory=_env("BETAI_DEVIG", "power"))
    prior_total_goals: float = field(
        default_factory=lambda: _env_float("BETAI_PRIOR_TOTAL", 2.7)
    )
    min_edge: float = field(default_factory=lambda: _env_float("BETAI_MIN_EDGE", 0.03))
    kelly_fraction: float = field(
        default_factory=lambda: _env_float("BETAI_KELLY_FRACTION", 0.25)
    )

    # IA
    anthropic_api_key: str | None = field(default_factory=_env_opt("ANTHROPIC_API_KEY"))
    model: str = field(default_factory=_env("BETAI_MODEL", "claude-opus-5"))
    effort: str = field(default_factory=_env("BETAI_EFFORT", "medium"))
    max_ai_shift: float = field(
        default_factory=lambda: _env_float("BETAI_MAX_AI_SHIFT", 0.05)
    )

    # Persistência
    db_path: str = field(
        default_factory=lambda: os.environ.get("BETAI_DB", str(Path.cwd() / "bet-ai.db"))
    )


def load(dotenv: str | Path = ".env") -> Settings:
    """Configuração efetiva: .env primeiro, variáveis de ambiente por cima."""
    load_dotenv(dotenv)
    return Settings()
