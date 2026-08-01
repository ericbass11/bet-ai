"""Testes das travas contra o excesso de confiança do modelo.

Com dados reais, a tela mostrou: Vasco x Minas Brasília, 90', 2-2, empate com
"chance 100.0%" e "quanto apostar 25%". Três defeitos somados.
"""

import pytest

from betai.engine.live import MATCH_MINUTES, live_matrix, remaining_fraction
from betai.engine.value import MAX_STAKE, evaluate, filter_value, kelly
from betai.models import MarketKey, MatchState


def estado(minute, sh=0, sa=0):
    return MatchState(minute=minute, period="2h", score_home=sh, score_away=sa)


# ---------- acréscimos ----------


def test_aos_90_ainda_sobra_jogo():
    """O relógio da casa congela em 90' enquanto a bola rola. Zerar aqui faz
    o modelo declarar o placar decidido."""
    resta = remaining_fraction(90)
    assert 0.02 < resta < 0.12, resta


def test_so_acaba_no_fim_dos_acrescimos():
    assert remaining_fraction(MATCH_MINUTES) == pytest.approx(0.0, abs=1e-9)
    assert remaining_fraction(120) == pytest.approx(0.0, abs=1e-9)


def test_gols_esperados_nos_acrescimos_sao_realistas():
    """Empiricamente sai ~0.14 gol por partida depois dos 90."""
    esperado = 2.7 * remaining_fraction(90)
    assert 0.05 < esperado < 0.30, esperado


def test_modelo_nao_declara_certeza_aos_90():
    """Era 99.9998%. Um gol nos acréscimos existe e o modelo tem que saber."""
    m = live_matrix(1.43, 1.27, estado(90, 2, 2))
    empate = m.match_odds()["draw"]
    assert empate < 0.98, f"modelo ainda declara quase certeza: {empate:.4%}"
    assert empate > 0.85, "mas o empate segue sendo de longe o mais provável"


def test_fracao_restante_continua_decrescente():
    valores = [remaining_fraction(m) for m in range(0, 95, 10)]
    assert all(a > b for a, b in zip(valores, valores[1:]))


# ---------- teto do stake ----------


def test_kelly_puro_manda_um_quarto_da_banca_numa_quase_certeza():
    """O comportamento que motiva o teto: matematicamente correto, na prática
    imprudente, porque a probabilidade é estimada e não conhecida."""
    assert kelly(1.0, 1.05, 0.25) == pytest.approx(0.25)


def test_stake_sugerido_tem_teto():
    bet = evaluate(MarketKey.MATCH_ODDS, "draw", 1.05, 1.0, 0.95)
    assert bet.kelly_fraction <= MAX_STAKE


def test_teto_nao_afeta_aposta_de_tamanho_normal():
    bet = evaluate(MarketKey.MATCH_ODDS, "home", 3.00, 0.38, 0.33)
    assert 0 < bet.kelly_fraction < MAX_STAKE


# ---------- quase-certezas fora da lista ----------


def _bet(prob, odd):
    return evaluate(MarketKey.MATCH_ODDS, "draw", odd, prob, prob * 0.95)


def test_quase_certeza_nao_e_recomendada():
    """Vantagem percentual parece ótima, mas um único azar apaga dezenas de
    acertos — e é onde o modelo não distingue 97% de 99.9%."""
    assert filter_value([_bet(0.999, 1.05)]) == []
    assert filter_value([_bet(0.97, 1.10)]) == []


def test_probabilidade_normal_continua_passando():
    mantidas = filter_value([_bet(0.40, 3.00)])
    assert len(mantidas) == 1


def test_os_dois_extremos_sao_cortados():
    bets = [_bet(0.001, 500.0), _bet(0.999, 1.02), _bet(0.45, 2.60)]
    mantidas = filter_value(bets)
    assert len(mantidas) == 1
    assert mantidas[0].model_probability == 0.45
