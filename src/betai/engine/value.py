"""Detecção de valor: EV, edge e dimensionamento por Kelly."""

from __future__ import annotations

from ..models import MarketKey, ValueBet


def expected_value(probability: float, odds: float) -> float:
    """Retorno esperado por unidade apostada.

    Positivo significa que, repetida infinitas vezes, a aposta lucra.
    """
    return probability * (odds - 1.0) - (1.0 - probability)


def edge(probability: float, odds: float) -> float:
    """`p * odd - 1`. Idêntico ao EV, exposto por ser a métrica mais citada."""
    return probability * odds - 1.0


def kelly(probability: float, odds: float, fraction: float = 1.0) -> float:
    """Fração do bankroll pelo critério de Kelly.

    Kelly cheio maximiza o crescimento logarítmico mas é violento: qualquer
    erro na estimativa de probabilidade vira aposta grande demais. Como as
    probabilidades aqui são estimadas e não conhecidas, o padrão de uso é
    Kelly fracionado (1/4 ou 1/8) — daí o parâmetro `fraction`.
    """
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (probability * odds - 1.0) / b
    return max(0.0, f * fraction)


def evaluate(
    market: MarketKey,
    outcome: str,
    odds: float,
    model_probability: float,
    market_probability: float,
    line: float | None = None,
    kelly_fraction: float = 0.25,
) -> ValueBet:
    """Monta o registro de valor de uma seleção."""
    return ValueBet(
        market=market,
        outcome=outcome,
        line=line,
        odds=odds,
        model_probability=model_probability,
        market_probability=market_probability,
        edge=edge(model_probability, odds),
        expected_value=expected_value(model_probability, odds),
        kelly_fraction=kelly(model_probability, odds, kelly_fraction),
    )


def filter_value(
    bets: list[ValueBet],
    min_edge: float = 0.02,
    min_probability: float = 0.02,
    max_odds: float = 30.0,
) -> list[ValueBet]:
    """Descarta ruído e devolve as apostas com valor, das melhores para as piores.

    Os três filtros existem por motivos distintos: `min_edge` porque uma
    vantagem de 0.5% não sobrevive ao erro do modelo; `min_probability`
    porque a estimativa da cauda é a menos confiável; `max_odds` porque em
    odds muito altas o erro relativo do modelo explode.
    """
    keep = [
        b
        for b in bets
        if b.edge >= min_edge
        and b.model_probability >= min_probability
        and b.odds <= max_odds
    ]
    keep.sort(key=lambda b: b.edge, reverse=True)
    return keep
