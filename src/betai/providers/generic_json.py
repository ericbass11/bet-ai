"""Provedor genérico para endpoints JSON de casas de apostas.

Quase todo site de apostas moderno é um front-end que consome o próprio
endpoint JSON interno. Este provedor lê esse endpoint a partir de um mapa de
campos declarado em arquivo, sem precisar de código novo por casa.

Antes de apontar isto para um site:

* leia os Termos de Uso — a maioria das casas proíbe coleta automatizada, e
  isso é uma decisão sua, não do código;
* o `robots.txt` é verificado e respeitado por padrão (`respect_robots`);
* há rate limit obrigatório entre requisições;
* prefira uma API licenciada (veja `the_odds_api.py`) sempre que existir uma.
"""

from __future__ import annotations

import urllib.robotparser
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx

from ..models import (
    Event,
    Market,
    MarketKey,
    MatchState,
    Selection,
)
from .base import Provider, ProviderError, RateLimiter


def dig(data: Any, path: str, default: Any = None) -> Any:
    """Navega um JSON aninhado com um caminho pontilhado.

    Suporta índices de lista: `"events.0.markets.name"`.
    """
    current = data
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return default
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return default
    return current if current is not None else default


class FieldMap:
    """Declara onde cada campo vive no JSON do provedor.

    Exemplo de mapa (JSON no disco):

    ```json
    {
      "url": "https://exemplo.com/api/inplay",
      "events_path": "data.events",
      "fields": {
        "event_id": "id",
        "league": "competition.name",
        "home_team": "teams.0.name",
        "away_team": "teams.1.name",
        "minute": "clock.minute",
        "score_home": "score.home",
        "score_away": "score.away"
      },
      "markets": [
        {
          "key": "1x2",
          "path": "markets.match_odds.runners",
          "outcome_field": "name",
          "odds_field": "price",
          "outcome_map": {"1": "home", "X": "draw", "2": "away"}
        }
      ]
    }
    ```
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        self.url: str = spec["url"]
        self.events_path: str = spec.get("events_path", "")
        self.fields: dict[str, str] = spec.get("fields", {})
        self.markets: list[dict[str, Any]] = spec.get("markets", [])
        self.headers: dict[str, str] = spec.get("headers", {})
        self.params: dict[str, str] = spec.get("params", {})


class GenericJsonProvider(Provider):
    """Lê odds de um endpoint JSON descrito por um `FieldMap`."""

    name = "generic_json"

    def __init__(
        self,
        field_map: FieldMap,
        *,
        min_interval: float = 2.0,
        respect_robots: bool = True,
        timeout: float = 15.0,
        user_agent: str = "bet-ai/0.1 (analise pessoal)",
    ) -> None:
        self.map = field_map
        self.limiter = RateLimiter(min_interval)
        self.respect_robots = respect_robots
        self.user_agent = user_agent
        self.client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent, **field_map.headers},
        )
        self._robots_ok: bool | None = None

    def close(self) -> None:
        self.client.close()

    def _check_robots(self) -> None:
        if not self.respect_robots or self._robots_ok is not None:
            if self._robots_ok is False:
                raise ProviderError(f"robots.txt proíbe a coleta de {self.map.url}")
            return

        parsed = urlparse(self.map.url)
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        try:
            parser.read()
        except Exception:
            # Sem robots.txt acessível, seguimos — mas com rate limit.
            self._robots_ok = True
            return

        self._robots_ok = parser.can_fetch(self.user_agent, self.map.url)
        if not self._robots_ok:
            raise ProviderError(f"robots.txt proíbe a coleta de {self.map.url}")

    def _parse_market(self, raw_event: Any, spec: dict[str, Any]) -> Market | None:
        runners = dig(raw_event, spec["path"], [])
        if not isinstance(runners, list) or not runners:
            return None

        # Muitas casas achatam todos os mercados num array só, distinguindo-os
        # por um campo interno (`marketName`, `marketId`). O filtro recorta o
        # pedaço que pertence a este mercado.
        filter_field = spec.get("filter_field")
        if filter_field is not None:
            alvos = spec.get("filter_values") or [spec.get("filter_value")]
            alvos = {str(v) for v in alvos if v is not None}
            runners = [r for r in runners if str(dig(r, filter_field)) in alvos]
            if not runners:
                return None

        outcome_map: dict[str, str] = spec.get("outcome_map", {})
        selections: list[Selection] = []
        for runner in runners:
            raw_outcome = str(dig(runner, spec.get("outcome_field", "name"), ""))
            odds = dig(runner, spec.get("odds_field", "price"))
            if odds is None:
                continue
            try:
                odds = float(odds)
            except (TypeError, ValueError):
                continue
            if odds <= 1.0:
                continue
            line_field = spec.get("line_field")
            line = dig(runner, line_field) if line_field else spec.get("line")
            selections.append(
                Selection(
                    outcome=outcome_map.get(raw_outcome, raw_outcome.lower()),
                    odds=odds,
                    line=float(line) if line is not None else None,
                )
            )

        if not selections:
            return None
        line = spec.get("line")
        return Market(
            key=MarketKey(spec["key"]),
            selections=selections,
            line=float(line) if line is not None else None,
        )

    def _field(self, raw: Any, spec: Any, default: Any = None) -> Any:
        """Lê um campo do evento.

        O spec normalmente é um caminho pontilhado. Também aceita a forma
        `{"path": ..., "split": "·", "index": 0}`, para casas que só publicam
        os dois times num texto único ("Casa·Fora", "Casa - Fora").
        """
        if spec is None:
            return default
        if isinstance(spec, str):
            return dig(raw, spec, default)

        value = dig(raw, spec.get("path", ""), None)
        separator = spec.get("split")
        if value is None or separator is None:
            return value if value is not None else default
        partes = [p.strip() for p in str(value).split(separator)]
        index = int(spec.get("index", 0))
        return partes[index] if 0 <= index < len(partes) else default

    def _parse_event(self, raw: Any) -> Event | None:
        f = self.map.fields
        event_id = self._field(raw, f.get("event_id", "id"))
        home = self._field(raw, f.get("home_team", "home"))
        away = self._field(raw, f.get("away_team", "away"))
        if not event_id or not home or not away:
            return None

        minute = _as_int(self._field(raw, f.get("minute", "minute")))
        state = MatchState(
            minute=minute,
            period=_period_for(minute),
            score_home=_as_int(self._field(raw, f.get("score_home", "score_home"))),
            score_away=_as_int(self._field(raw, f.get("score_away", "score_away"))),
            red_cards_home=_as_int(self._field(raw, f.get("red_cards_home"))),
            red_cards_away=_as_int(self._field(raw, f.get("red_cards_away"))),
        )

        markets = [m for spec in self.map.markets if (m := self._parse_market(raw, spec))]
        if not markets:
            return None

        return Event(
            event_id=str(event_id),
            league=str(self._field(raw, f.get("league", "league"), "desconhecida")),
            home_team=str(home),
            away_team=str(away),
            starts_at=datetime.now(timezone.utc),
            source=self.name,
            state=state,
            markets=markets,
        )

    def fetch_live(self) -> Iterable[Event]:
        self._check_robots()
        self.limiter.wait()
        try:
            resp = self.client.get(self.map.url, params=self.map.params)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"falha ao ler {self.map.url}: {exc}") from exc
        except ValueError as exc:
            raise ProviderError(f"resposta de {self.map.url} não é JSON válido") from exc

        raw_events = dig(payload, self.map.events_path, payload) if self.map.events_path else payload
        if not isinstance(raw_events, list):
            raise ProviderError(
                f"events_path '{self.map.events_path}' não aponta para uma lista"
            )

        return [e for raw in raw_events if (e := self._parse_event(raw))]


def _as_int(value: Any) -> int:
    """Converte para inteiro tolerando texto e ausência.

    Várias casas publicam placar e minuto como string ("59", "2"), e alguns
    campos vêm nulos em jogos que ainda não começaram.
    """
    if value is None:
        return 0
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _period_for(minute: int) -> str:
    if minute <= 0:
        return "pre_match"
    if minute <= 45:
        return "1h"
    if minute < 90:
        return "2h"
    return "encerrado"
