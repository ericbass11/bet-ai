"""Provedor The Odds API — agregador comercial de odds com chave de API.

É o caminho recomendado para dados reais: acesso licenciado, formato estável
e sem risco de violar termos de uso de um site de apostas. Cadastro em
https://the-odds-api.com (há plano gratuito com cota mensal).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import httpx

from ..models import Event, Market, MarketKey, Selection
from .base import Provider, ProviderError, RateLimiter

BASE_URL = "https://api.the-odds-api.com/v4"

# Mapeia o vocabulário da API para o do domínio.
_MARKET_MAP = {
    "h2h": MarketKey.MATCH_ODDS,
    "totals": MarketKey.OVER_UNDER,
    "spreads": MarketKey.ASIAN_HANDICAP,
}


class TheOddsApiProvider(Provider):
    """Lê eventos e odds da The Odds API."""

    name = "the_odds_api"

    def __init__(
        self,
        api_key: str,
        sport: str = "soccer_brazil_campeonato",
        regions: str = "eu",
        bookmaker: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        if not api_key:
            raise ProviderError("THE_ODDS_API_KEY não configurada")
        self.api_key = api_key
        self.sport = sport
        self.regions = regions
        self.bookmaker = bookmaker
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": "bet-ai/0.1"})
        self.limiter = RateLimiter(min_interval=1.0)

    def close(self) -> None:
        self.client.close()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        self.limiter.wait()
        params = {**params, "apiKey": self.api_key}
        try:
            resp = self.client.get(f"{BASE_URL}{path}", params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"The Odds API retornou {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"falha de rede ao consultar The Odds API: {exc}") from exc
        return resp.json()

    def _parse_event(self, raw: dict[str, Any]) -> Event | None:
        books = raw.get("bookmakers") or []
        if self.bookmaker:
            books = [b for b in books if b.get("key") == self.bookmaker]
        if not books:
            return None

        book = books[0]
        home = raw["home_team"]
        away = raw["away_team"]
        markets: list[Market] = []

        for raw_market in book.get("markets", []):
            key = _MARKET_MAP.get(raw_market.get("key"))
            if key is None:
                continue

            selections: list[Selection] = []
            line: float | None = None
            for out in raw_market.get("outcomes", []):
                price = out.get("price")
                if not price or price <= 1.0:
                    continue
                point = out.get("point")
                if point is not None:
                    line = float(point)
                selections.append(
                    Selection(
                        outcome=_normalize_outcome(out.get("name", ""), home, away),
                        odds=float(price),
                        line=float(point) if point is not None else None,
                    )
                )

            if selections:
                markets.append(Market(key=key, selections=selections, line=line))

        if not markets:
            return None

        return Event(
            event_id=raw["id"],
            league=raw.get("sport_title", self.sport),
            home_team=home,
            away_team=away,
            starts_at=datetime.fromisoformat(raw["commence_time"].replace("Z", "+00:00")),
            source=self.name,
            markets=markets,
        )

    def _fetch(self) -> Iterable[Event]:
        data = self._get(
            f"/sports/{self.sport}/odds",
            {
                "regions": self.regions,
                "markets": "h2h,totals,spreads",
                "oddsFormat": "decimal",
            },
        )
        events = []
        for raw in data:
            parsed = self._parse_event(raw)
            if parsed:
                events.append(parsed)
        return events

    def fetch_live(self) -> Iterable[Event]:
        """A The Odds API não separa ao vivo; filtramos pelo horário de início.

        Sem feed de placar, o estado da partida fica zerado — o motor então
        trata o jogo como pré-jogo. Para análise ao vivo de verdade, combine
        com um provedor de placar.
        """
        now = datetime.now().astimezone()
        return [e for e in self._fetch() if e.starts_at <= now]

    def fetch_upcoming(self) -> Iterable[Event]:
        now = datetime.now().astimezone()
        return [e for e in self._fetch() if e.starts_at > now]


def _normalize_outcome(name: str, home: str, away: str) -> str:
    """Traduz o rótulo da API para o vocabulário interno."""
    lowered = name.strip().lower()
    if lowered == home.lower():
        return "home"
    if lowered == away.lower():
        return "away"
    if lowered in {"draw", "tie", "empate"}:
        return "draw"
    if lowered in {"over", "mais"}:
        return "over"
    if lowered in {"under", "menos"}:
        return "under"
    return lowered
