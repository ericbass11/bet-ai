import pytest

from betai.engine.live import (
    MATCH_MINUTES,
    LiveConfig,
    live_matrix,
    remaining_fraction,
)
from betai.models import MatchState, MatchStats


def state(minute=0, sh=0, sa=0, rh=0, ra=0, period=None, **stats):
    return MatchState(
        minute=minute,
        period=period or ("pre_match" if minute == 0 else ("1h" if minute <= 45 else "2h")),
        score_home=sh,
        score_away=sa,
        red_cards_home=rh,
        red_cards_away=ra,
        stats=MatchStats(**stats),
    )


def test_fracao_restante_nos_extremos():
    assert remaining_fraction(0) == pytest.approx(1.0)
    # Aos 90' ainda restam os acréscimos — só zera no fim deles.
    assert remaining_fraction(90) > 0.0
    assert remaining_fraction(MATCH_MINUTES) == pytest.approx(0.0, abs=1e-9)
    assert remaining_fraction(120) == pytest.approx(0.0, abs=1e-9)


def test_fracao_restante_decresce():
    valores = [remaining_fraction(m) for m in range(0, 91, 10)]
    assert all(a > b for a, b in zip(valores, valores[1:]))


def test_intervalo_deixa_mais_da_metade_dos_gols_para_tras():
    """Gols são mais frequentes no segundo tempo — no intervalo ainda sobra >50%."""
    assert remaining_fraction(45) > 0.5


def test_pre_jogo_nao_altera_o_modelo():
    m = live_matrix(1.5, 1.1, state())
    assert m.lambda_home == pytest.approx(1.5)
    assert m.lambda_away == pytest.approx(1.1)


def test_placar_atual_entra_na_matriz():
    m = live_matrix(1.5, 1.1, state(minute=60, sh=2, sa=0))
    # Impossível terminar com menos gols do que já foram feitos.
    assert m.prob(1, 0) == pytest.approx(0.0)
    assert m.prob(0, 0) == pytest.approx(0.0)
    assert m.match_odds()["home"] > 0.85


def test_vantagem_diminui_conforme_o_tempo_passa():
    inicio = live_matrix(1.4, 1.4, state(minute=15, sh=1, sa=0)).match_odds()["home"]
    fim = live_matrix(1.4, 1.4, state(minute=85, sh=1, sa=0)).match_odds()["home"]
    assert fim > inicio


def test_cartao_vermelho_prejudica_quem_levou():
    normal = live_matrix(1.4, 1.4, state(minute=30)).match_odds()
    com_vermelho = live_matrix(1.4, 1.4, state(minute=30, rh=1)).match_odds()
    assert com_vermelho["home"] < normal["home"]
    assert com_vermelho["away"] > normal["away"]


def test_time_perdendo_ataca_mais():
    """O efeito de placar deve elevar a taxa de gols de quem está atrás."""
    cfg = LiveConfig()
    perdendo = live_matrix(1.4, 1.4, state(minute=60, sh=0, sa=1), cfg)
    # Total de gols esperado é maior do que seria com o efeito desligado.
    sem_efeito = LiveConfig(chasing_boost=0.0, leading_damp=0.0)
    neutro = live_matrix(1.4, 1.4, state(minute=60, sh=0, sa=1), sem_efeito)
    assert perdendo.over_under(2.5)["over"] > neutro.over_under(2.5)["over"]


def test_efeito_de_placar_tem_teto():
    """Uma goleada não pode multiplicar o modelo indefinidamente."""
    cfg = LiveConfig()
    m5 = live_matrix(1.4, 1.4, state(minute=60, sh=0, sa=5), cfg)
    m9 = live_matrix(1.4, 1.4, state(minute=60, sh=0, sa=9), cfg)
    # O multiplicador satura, então as taxas restantes são iguais.
    assert m5.lambda_home == pytest.approx(m9.lambda_home)


def test_xg_puxa_o_modelo_quando_ha_jogo_suficiente():
    base = live_matrix(1.3, 1.3, state(minute=60, xg_home=0.5, xg_away=0.5))
    dominando = live_matrix(1.3, 1.3, state(minute=60, xg_home=2.5, xg_away=0.2))
    assert dominando.match_odds()["home"] > base.match_odds()["home"]


def test_xg_ignorado_no_comeco_do_jogo():
    """Cinco minutos de xG não são informação, são ruído."""
    cedo = live_matrix(1.3, 1.3, state(minute=5, xg_home=1.0, xg_away=0.0))
    sem_xg = live_matrix(1.3, 1.3, state(minute=5))
    assert cedo.lambda_home == pytest.approx(sem_xg.lambda_home)


def test_xg_pesa_mais_no_fim_do_que_no_comeco():
    """A mesma taxa de xG é evidência mais forte quanto mais jogo houve."""
    # Mesma taxa por minuto (2x a expectativa) em dois momentos diferentes.
    cedo = live_matrix(1.0, 1.0, state(minute=25, xg_home=0.55, xg_away=0.28))
    tarde = live_matrix(1.0, 1.0, state(minute=75, xg_home=1.67, xg_away=0.83))
    razao_cedo = cedo.lambda_home / cedo.lambda_away
    razao_tarde = tarde.lambda_home / tarde.lambda_away
    assert razao_tarde > razao_cedo


def test_xg_extremo_nao_estoura_o_modelo():
    """Um xG absurdo não pode multiplicar a expectativa sem limite."""
    m = live_matrix(1.2, 1.2, state(minute=25, xg_home=6.0, xg_away=0.0))
    remaining = remaining_fraction(25)
    # Trava: no máximo o dobro da expectativa pré-jogo, antes do tempo restante.
    assert m.lambda_home <= 1.2 * 2.0 * remaining + 1e-9
    assert m.over_under(2.5)["over"] < 0.95


def test_matriz_ao_vivo_continua_normalizada():
    m = live_matrix(1.6, 1.2, state(minute=70, sh=1, sa=2, rh=1))
    assert sum(sum(r) for r in m.grid) == pytest.approx(1.0, abs=1e-9)
