import math

import pytest

from betai.engine.poisson import ScoreMatrix, calibrate


def test_matriz_normalizada():
    m = ScoreMatrix(1.5, 1.1)
    total = sum(sum(row) for row in m.grid)
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_1x2_soma_um():
    probs = ScoreMatrix(1.6, 1.2).match_odds()
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    assert probs["home"] > probs["away"]  # mandante com lambda maior


def test_over_under_meia_linha_sem_push():
    ou = ScoreMatrix(1.4, 1.3).over_under(2.5)
    assert ou["push"] == pytest.approx(0.0, abs=1e-12)
    assert math.isclose(ou["over"] + ou["under"], 1.0, abs_tol=1e-9)


def test_over_under_linha_inteira_tem_push():
    ou = ScoreMatrix(1.4, 1.3).over_under(2.0)
    assert ou["push"] > 0.15  # P(exatamente 2 gols) é substancial
    assert math.isclose(ou["over"] + ou["under"] + ou["push"], 1.0, abs_tol=1e-9)


def test_linha_de_quarto_fica_entre_as_adjacentes():
    m = ScoreMatrix(1.4, 1.3)
    low = m.over_under(2.0)["over"]
    high = m.over_under(2.5)["over"]
    quarter = m.over_under(2.25)["over"]
    assert min(low, high) <= quarter <= max(low, high)


def test_over_cresce_com_o_total_de_gols():
    baixo = ScoreMatrix(0.8, 0.7).over_under(2.5)["over"]
    alto = ScoreMatrix(2.0, 1.8).over_under(2.5)["over"]
    assert alto > baixo


def test_handicap_asiatico_soma_um():
    ah = ScoreMatrix(1.6, 1.1).asian_handicap(-0.5)
    assert math.isclose(ah["home"] + ah["away"] + ah["push"], 1.0, abs_tol=1e-9)


def test_handicap_inteiro_tem_push():
    ah = ScoreMatrix(1.5, 1.5).asian_handicap(0.0)
    assert ah["push"] > 0.2  # empate devolve a aposta


def test_dixon_coles_eleva_placares_baixos():
    """A correção existe para consertar exatamente estes quatro placares."""
    sem = ScoreMatrix(1.3, 1.1, rho=0.0)
    com = ScoreMatrix(1.3, 1.1, rho=-0.05)
    assert com.prob(0, 0) > sem.prob(0, 0)
    assert com.prob(1, 1) > sem.prob(1, 1)
    # E rebaixa 1-0 / 0-1 em compensação.
    assert com.prob(1, 0) < sem.prob(1, 0)


def test_shift_desloca_o_placar():
    base = ScoreMatrix(1.0, 0.8)
    shifted = base.shifted(2, 1)
    # O que era 0-0 na matriz de gols restantes vira 2-1 no placar final.
    assert shifted.prob(2, 1) == pytest.approx(base.prob(0, 0))
    assert shifted.prob(0, 0) == pytest.approx(0.0)
    assert math.isclose(sum(sum(r) for r in shifted.grid), 1.0, abs_tol=1e-9)


def test_btts_coerente_com_a_matriz():
    m = ScoreMatrix(1.5, 1.2)
    btts = m.btts()
    manual = 1.0 - sum(m.prob(0, j) for j in range(13)) - sum(
        m.prob(i, 0) for i in range(13)
    ) + m.prob(0, 0)
    assert btts["yes"] == pytest.approx(manual, abs=1e-9)


def test_calibracao_reproduz_o_mercado():
    """Ida e volta: probabilidades → lambdas → probabilidades."""
    alvo_home, alvo_away, alvo_over = 0.48, 0.25, 0.55

    lh, la = calibrate(p_home=alvo_home, p_away=alvo_away, p_over=alvo_over)
    m = ScoreMatrix(lh, la)

    assert m.match_odds()["home"] - m.match_odds()["away"] == pytest.approx(
        alvo_home - alvo_away, abs=0.005
    )
    assert m.over_under(2.5)["over"] == pytest.approx(alvo_over, abs=0.005)


def test_calibracao_sem_over_under_usa_o_prior():
    lh, la = calibrate(p_home=0.45, p_away=0.30, prior_total=2.7)
    assert lh + la == pytest.approx(2.7, abs=0.01)
    assert lh > la


def test_calibracao_ao_vivo_devolve_gols_restantes():
    """Com 2-0 no placar, as taxas calibradas são só do que falta."""
    lh, la = calibrate(
        p_home=0.85, p_away=0.05, p_over=0.60, over_line=2.5, base_home=2, base_away=0
    )
    # O jogo já tem 2 gols; sobra pouca coisa para chegar a over 2.5.
    assert 0 < lh + la < 2.5
    shifted = ScoreMatrix(lh, la).shifted(2, 0)
    assert shifted.over_under(2.5)["over"] == pytest.approx(0.60, abs=0.01)


def test_gols_esperados_batem_com_os_lambdas():
    lh, la = 1.7, 1.1
    eh, ea = ScoreMatrix(lh, la, rho=0.0).expected_goals()
    assert eh == pytest.approx(lh, abs=0.02)
    assert ea == pytest.approx(la, abs=0.02)
