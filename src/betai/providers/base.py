"""Contrato dos provedores de dados.

Toda fonte de odds — API oficial, endpoint JSON de uma casa, arquivo de
fixture — implementa a mesma interface. O resto do sistema não sabe de onde
os dados vieram.
"""

from __future__ import annotations

import abc
import time
from typing import Iterable

from ..models import Event, Market


class RateLimiter:
    """Espaçamento mínimo entre requisições.

    Provedor nenhum gosta de rajada. Além de educação, é o que mantém a
    chave de API viva.
    """

    def __init__(self, min_interval: float = 1.0) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


class Provider(abc.ABC):
    """Fonte de eventos e odds."""

    name: str

    @abc.abstractmethod
    def fetch_live(self) -> Iterable[Event]:
        """Eventos ao vivo com o snapshot de odds do momento."""

    def fetch_upcoming(self) -> Iterable[Event]:
        """Eventos que ainda vão começar. Opcional."""
        return []

    @property
    def supports_details(self) -> bool:
        """A fonte tem um endpoint por jogo com mercados além do principal?"""
        return False

    def fetch_details(self, event_id: str) -> list[Market]:
        """Mercados extras de um jogo. Opcional — custa uma requisição cada."""
        return []

    def close(self) -> None:
        """Libera conexões. Sobrescreva se o provedor mantiver um cliente HTTP."""


class ProviderError(RuntimeError):
    """Falha ao consultar um provedor."""
