"""Modelo de domínio: eventos, mercados e odds capturados dos provedores."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class MarketKey(str, Enum):
    """Mercados suportados pelo motor de probabilidade."""

    MATCH_ODDS = "1x2"
    DOUBLE_CHANCE = "double_chance"
    OVER_UNDER = "over_under"
    BTTS = "btts"
    ASIAN_HANDICAP = "asian_handicap"
    CORRECT_SCORE = "correct_score"


class Outcome(str, Enum):
    HOME = "home"
    DRAW = "draw"
    AWAY = "away"
    OVER = "over"
    UNDER = "under"
    YES = "yes"
    NO = "no"


class Selection(BaseModel):
    """Uma seleção apostável dentro de um mercado."""

    outcome: str
    odds: float = Field(gt=1.0, description="Odd decimal oferecida pela casa")
    line: float | None = Field(
        default=None,
        description="Linha do mercado (2.5 para over/under, -0.75 para handicap...)",
    )

    @property
    def implied(self) -> float:
        """Probabilidade implícita bruta, ainda com a margem da casa embutida."""
        return 1.0 / self.odds


class Market(BaseModel):
    key: MarketKey
    selections: list[Selection]
    line: float | None = None

    def selection(self, outcome: str) -> Selection | None:
        for sel in self.selections:
            if sel.outcome == outcome:
                return sel
        return None

    @property
    def coverage(self) -> float:
        """Quanta probabilidade as seleções deste mercado cobrem somadas.

        Quase todo mercado particiona os resultados possíveis e cobre 1.
        Dupla chance é a exceção: cada resultado do jogo aparece em dois dos
        três pares, então um livro saudável soma ~2.
        """
        return 2.0 if self.key == MarketKey.DOUBLE_CHANCE else 1.0

    @property
    def overround(self) -> float:
        """Soma das probabilidades implícitas. 1.06 significa 6% de margem."""
        return sum(sel.implied for sel in self.selections) / self.coverage


class MatchStats(BaseModel):
    """Estatísticas ao vivo, quando o provedor as expõe.

    Hoje só `xg_*` alimenta o modelo. Os demais campos são coletados e
    guardados sem serem usados, de propósito: sem histórico avaliado não há
    como saber se acrescentam informação além do que as odds já embutem, e
    ligar um coeficiente no chute pioraria o que já funciona.
    """

    shots_on_target_home: int = 0
    shots_on_target_away: int = 0
    corners_home: int = 0
    corners_away: int = 0
    yellow_cards_home: int = 0
    yellow_cards_away: int = 0
    dangerous_attacks_home: int = 0
    dangerous_attacks_away: int = 0
    xg_home: float | None = None
    xg_away: float | None = None
    possession_home: float | None = None


class MatchState(BaseModel):
    """Estado instantâneo da partida."""

    minute: int = 0
    period: str = "pre_match"  # pre_match | 1h | intervalo | 2h | encerrado
    score_home: int = 0
    score_away: int = 0
    red_cards_home: int = 0
    red_cards_away: int = 0
    stats: MatchStats = Field(default_factory=MatchStats)

    @property
    def is_live(self) -> bool:
        return self.period in {"1h", "intervalo", "2h"}


class Event(BaseModel):
    """Um jogo com o snapshot de odds capturado num instante."""

    event_id: str
    league: str
    home_team: str
    away_team: str
    starts_at: datetime
    source: str
    state: MatchState = Field(default_factory=MatchState)
    markets: list[Market] = Field(default_factory=list)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def market(self, key: MarketKey, line: float | None = None) -> Market | None:
        for mkt in self.markets:
            if mkt.key == key and (line is None or mkt.line == line):
                return mkt
        return None

    def markets_of(self, key: MarketKey) -> list[Market]:
        return [m for m in self.markets if m.key == key]

    def with_markets(self, extras: list[Market]) -> "Event":
        """Cópia do evento com mercados adicionais.

        Um extra de mesma chave e linha substitui o que veio da listagem: o
        endpoint por jogo é a fonte mais completa das duas.
        """
        if not extras:
            return self
        por_chave = {(m.key, m.line): m for m in self.markets}
        for mkt in extras:
            por_chave[(mkt.key, mkt.line)] = mkt
        return self.model_copy(update={"markets": list(por_chave.values())})

    @property
    def label(self) -> str:
        return f"{self.home_team} x {self.away_team}"


class ValueBet(BaseModel):
    """Uma seleção onde o modelo discorda do mercado a favor do apostador."""

    market: MarketKey
    outcome: str
    line: float | None
    odds: float
    model_probability: float
    market_probability: float
    edge: float = Field(description="p_modelo * odd - 1")
    expected_value: float = Field(description="Retorno esperado por unidade apostada")
    kelly_fraction: float = Field(description="Fração do bankroll pelo critério de Kelly")

    @property
    def fair_odds(self) -> float:
        return 1.0 / self.model_probability if self.model_probability > 0 else float("inf")


class Analysis(BaseModel):
    """Resultado completo da análise de um evento."""

    event: Event
    lambda_home: float
    lambda_away: float
    baseline_source: str = Field(
        default="unknown",
        description=(
            "De onde vieram as taxas base: 'pre_match' (odds antes do jogo, o caso "
            "bom), 'live_inverted' (invertidas das odds ao vivo — o modelo então "
            "concorda com o mercado por construção) ou 'prior' (prior da liga)."
        ),
    )
    probabilities: dict[str, float]
    value_bets: list[ValueBet] = Field(default_factory=list)
    ai_summary: str | None = None
    ai_adjustments: dict[str, float] = Field(default_factory=dict)
