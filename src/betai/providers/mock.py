"""Provedor de fixtures: jogos sintéticos para desenvolvimento e testes.

Permite rodar o pipeline inteiro sem chave de API e sem rede — o que também
torna os testes determinísticos.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Iterable

from ..models import (
    Event,
    MarketKey,
    Market,
    MatchState,
    MatchStats,
    Selection,
)
from .base import Provider

_FIXTURES = [
    ("Brasileirão Série A", "Palmeiras", "Flamengo"),
    ("Brasileirão Série A", "Corinthians", "São Paulo"),
    ("Premier League", "Arsenal", "Manchester City"),
    ("La Liga", "Real Madrid", "Sevilla"),
    ("Serie A", "Inter", "Napoli"),
]


def _with_margin(probs: list[float], margin: float) -> list[float]:
    """Converte probabilidades justas em odds com a margem da casa embutida.

    O overround resultante é `1 + margin` — se a soma das implícitas der menos
    que 1, a casa estaria pagando para receber apostas.

    As odds são limitadas a [1.01, 500]: nenhuma casa oferece odd abaixo de 1
    nem paga 10.000 num placar improvável.
    """
    return [
        round(min(max(1.0 / (max(p, 1e-6) * (1 + margin)), 1.01), 500.0), 2) for p in probs
    ]


class MockProvider(Provider):
    """Gera jogos plausíveis e internamente coerentes a partir de uma semente.

    As odds saem todas da mesma matriz de placares, então os mercados não se
    contradizem — sem isso o pipeline "acharia valor" em over e under ao mesmo
    tempo, que é artefato da fixture e não do modelo.

    Nas versões ao vivo, a casa usa um decaimento **linear** do tempo restante,
    ignorando placar e cartões. É de propósito: reproduz o tipo de simplificação
    que gera divergência real contra um modelo mais completo.
    """

    name = "mock"

    def __init__(self, seed: int = 42, count: int = 3, margin: float = 0.06) -> None:
        self.rng = random.Random(seed)
        self.count = min(count, len(_FIXTURES))
        self.margin = margin

    def _build(self, index: int, live: bool) -> Event:
        from ..engine.poisson import ScoreMatrix

        league, home, away = _FIXTURES[index]
        rng = random.Random(index)  # mesmo jogo pré e ao vivo: mesmo λ base

        lam_home = rng.uniform(1.1, 2.0)
        lam_away = rng.uniform(0.7, 1.6)

        state = MatchState()
        if live:
            minute = rng.randint(12, 82)
            sh = rng.choice([0, 0, 1, 1, 2])
            sa = rng.choice([0, 0, 1, 1, 2])
            state = MatchState(
                minute=minute,
                period="1h" if minute <= 45 else "2h",
                score_home=sh,
                score_away=sa,
                red_cards_home=1 if rng.random() < 0.08 else 0,
                red_cards_away=1 if rng.random() < 0.08 else 0,
                stats=MatchStats(
                    shots_on_target_home=rng.randint(0, 2 + minute // 20),
                    shots_on_target_away=rng.randint(0, 2 + minute // 20),
                    corners_home=rng.randint(0, 2 + minute // 15),
                    corners_away=rng.randint(0, 2 + minute // 15),
                    # xG proporcional ao tempo jogado e à força do time, com
                    # ruído — não um número solto entre 0 e 2.2 aos 20 minutos.
                    xg_home=round(lam_home * (minute / 90) * rng.uniform(0.5, 1.6), 2),
                    xg_away=round(lam_away * (minute / 90) * rng.uniform(0.5, 1.6), 2),
                    possession_home=round(rng.uniform(35, 65), 1),
                ),
            )

        # Modelo da "casa": ao vivo, só encolhe as taxas linearmente pelo tempo
        # restante e soma o placar. Sem efeito de placar, sem cartão vermelho.
        if live:
            remaining = max(1.0 - state.minute / 90.0, 0.02)
            book = ScoreMatrix(lam_home * remaining, lam_away * remaining).shifted(
                state.score_home, state.score_away
            )
        else:
            book = ScoreMatrix(lam_home, lam_away)

        mo = book.match_odds()
        odds_1x2 = _with_margin([mo["home"], mo["draw"], mo["away"]], self.margin)
        ou = book.over_under(2.5)
        odds_ou = _with_margin([ou["over"], ou["under"]], self.margin)
        bt = book.btts()
        odds_btts = _with_margin([bt["yes"], bt["no"]], self.margin)

        now = datetime.now(timezone.utc)
        return Event(
            event_id=f"mock-{index}",
            league=league,
            home_team=home,
            away_team=away,
            starts_at=now - timedelta(minutes=state.minute) if live else now + timedelta(hours=6),
            source=self.name,
            state=state,
            markets=[
                Market(
                    key=MarketKey.MATCH_ODDS,
                    selections=[
                        Selection(outcome="home", odds=odds_1x2[0]),
                        Selection(outcome="draw", odds=odds_1x2[1]),
                        Selection(outcome="away", odds=odds_1x2[2]),
                    ],
                ),
                Market(
                    key=MarketKey.OVER_UNDER,
                    line=2.5,
                    selections=[
                        Selection(outcome="over", odds=odds_ou[0], line=2.5),
                        Selection(outcome="under", odds=odds_ou[1], line=2.5),
                    ],
                ),
                Market(
                    key=MarketKey.BTTS,
                    selections=[
                        Selection(outcome="yes", odds=odds_btts[0]),
                        Selection(outcome="no", odds=odds_btts[1]),
                    ],
                ),
            ],
        )

    def fetch_live(self) -> Iterable[Event]:
        return [self._build(i, live=True) for i in range(self.count)]

    def fetch_upcoming(self) -> Iterable[Event]:
        return [self._build(i, live=False) for i in range(self.count)]
