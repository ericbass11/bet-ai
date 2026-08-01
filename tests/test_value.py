import pytest

from betai.engine.value import edge, evaluate, expected_value, filter_value, kelly
from betai.models import MarketKey


def test_aposta_justa_tem_ev_zero():
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert edge(0.5, 2.0) == pytest.approx(0.0)
    assert kelly(0.5, 2.0) == pytest.approx(0.0)


def test_aposta_com_valor():
    # Modelo diz 55%, odd paga como se fosse 50%.
    assert expected_value(0.55, 2.0) == pytest.approx(0.10)
    assert edge(0.55, 2.0) == pytest.approx(0.10)
    assert kelly(0.55, 2.0) == pytest.approx(0.10)


def test_kelly_nunca_negativo():
    """Sem valor não se aposta — Kelly não sugere apostar contra."""
    assert kelly(0.40, 2.0) == 0.0


def test_kelly_fracionado_escala_linear():
    cheio = kelly(0.60, 2.0, fraction=1.0)
    quarto = kelly(0.60, 2.0, fraction=0.25)
    assert quarto == pytest.approx(cheio * 0.25)


def test_odd_invalida_nao_gera_stake():
    assert kelly(0.9, 1.0) == 0.0


def test_evaluate_monta_o_registro():
    bet = evaluate(
        market=MarketKey.MATCH_ODDS,
        outcome="home",
        odds=2.20,
        model_probability=0.50,
        market_probability=0.45,
        kelly_fraction=0.25,
    )
    assert bet.edge == pytest.approx(0.10)
    assert bet.fair_odds == pytest.approx(2.0)
    assert bet.kelly_fraction > 0


def _bet(edge_target: float, prob: float = 0.5, odds: float | None = None):
    odds = odds if odds is not None else (1 + edge_target) / prob
    return evaluate(MarketKey.MATCH_ODDS, "home", odds, prob, prob)


def test_filtro_descarta_edge_pequeno():
    bets = [_bet(0.005), _bet(0.08)]
    kept = filter_value(bets, min_edge=0.02)
    assert len(kept) == 1
    assert kept[0].edge == pytest.approx(0.08)


def test_filtro_descarta_cauda_improvavel():
    """Probabilidade minúscula é onde o modelo erra mais — não vale a pena."""
    bets = [_bet(0.20, prob=0.005)]
    assert filter_value(bets, min_probability=0.02) == []


def test_filtro_descarta_odds_extremas():
    bets = [evaluate(MarketKey.CORRECT_SCORE, "5-4", 80.0, 0.02, 0.01)]
    assert filter_value(bets, max_odds=30.0) == []


def test_filtro_ordena_por_edge():
    bets = [_bet(0.05), _bet(0.15), _bet(0.09)]
    kept = filter_value(bets, min_edge=0.02)
    assert [round(b.edge, 2) for b in kept] == [0.15, 0.09, 0.05]
