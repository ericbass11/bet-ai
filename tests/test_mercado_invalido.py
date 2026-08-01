"""Testes das duas causas do valor falso visto com dados reais da Superbet.

A tela mostrava jogos com os dois times apontados como oportunidade ao mesmo
tempo (+731% e +297%) e cabeçalho "ainda não começou · 0-1". Ambos os
sintomas vinham de dados que o código aceitava sem questionar.
"""

from datetime import datetime, timezone

import pytest

from betai.engine.devig import MAX_OVERROUND, is_plausible, margin
from betai.models import Event, Market, MarketKey, MatchState, Selection
from betai.pipeline import Pipeline, market_probabilities, usable_markets
from betai.providers.generic_json import _period_for


def evento(odds, minute=0, sh=0, sa=0, event_id="e"):
    return Event(
        event_id=event_id,
        league="Teste",
        home_team="Casa",
        away_team="Fora",
        starts_at=datetime.now(timezone.utc),
        source="test",
        state=MatchState(
            minute=minute,
            period="pre_match" if minute == 0 else "2h",
            score_home=sh,
            score_away=sa,
        ),
        markets=[
            Market(
                key=MarketKey.MATCH_ODDS,
                selections=[
                    Selection(outcome="home", odds=odds[0]),
                    Selection(outcome="draw", odds=odds[1]),
                    Selection(outcome="away", odds=odds[2]),
                ],
            )
        ],
    )


# ---------- livro que não fecha ----------


def test_livro_normal_e_aceito():
    assert is_plausible([2.10, 3.40, 3.60])
    assert is_plausible([1.90, 1.90])


def test_livro_somando_bem_menos_que_um_e_recusado():
    """O caso real: odds somando 40% de probabilidade total.

    Nenhuma casa opera com margem negativa de 60%. São odds suspensas ou
    incompletas, e normalizá-las inventa probabilidades.
    """
    odds = [6.25, 5.77, 15.00]
    assert margin(odds) < -0.5
    assert not is_plausible(odds)


def test_margem_absurda_e_recusada():
    assert not is_plausible([1.05, 1.05, 1.05])  # overround ~2.86


def test_mercado_de_uma_selecao_so_e_recusado():
    assert not is_plausible([2.0])


def test_mercado_invalido_sai_da_analise():
    ruim = evento([6.25, 5.77, 15.00])
    assert usable_markets(ruim) == []
    assert market_probabilities(ruim) == {}


def test_mercado_invalido_nao_gera_aposta():
    """O sintoma que apareceu na tela: os dois lados como oportunidade."""
    analise = Pipeline(min_edge=0.03).analyze(evento([6.25, 5.77, 15.00]))
    assert analise.value_bets == []


def test_mercado_valido_continua_funcionando():
    """A validação não pode matar o caminho normal."""
    pipeline = Pipeline(min_edge=0.03)
    pipeline.register_baseline(evento([2.10, 3.40, 3.60], event_id="j"))
    ao_vivo = evento([1.06, 11.00, 34.00], minute=80, sh=2, sa=0, event_id="j")
    assert usable_markets(ao_vivo)
    assert pipeline.analyze(ao_vivo).value_bets


def test_limite_superior_permite_margem_alta_de_verdade():
    """Casas menores chegam a 15% em mercados exóticos; isso é real."""
    odds = [1.80, 3.60, 4.20]
    assert 1.0 < sum(1 / o for o in odds) < MAX_OVERROUND
    assert is_plausible(odds)


# ---------- jogo em andamento sem minuto ----------


def test_placar_sem_minuto_conta_como_jogo_em_andamento():
    """Gol antes do apito inicial não existe. Tratar como pré-jogo faria o
    modelo calcular sem os gols que as odds já embutem."""
    assert _period_for(0, placar=1) == "1h"
    assert _period_for(0, placar=0) == "pre_match"


def test_status_da_casa_manda_mais_que_o_minuto():
    assert _period_for(0, status="STARTED") == "1h"
    assert _period_for(0, status="2H") == "2h"
    assert _period_for(0, status="HT") == "intervalo"
    assert _period_for(70, status="Ended") == "encerrado"


def test_status_desconhecido_cai_para_o_minuto():
    assert _period_for(60, status="ALGO_ESTRANHO") == "2h"
    assert _period_for(0, status="NotStarted") == "pre_match"


def test_status_tolera_variacoes_de_escrita():
    for texto in ("in_play", "InPlay", "IN PLAY", "live"):
        assert _period_for(30, status=texto) == "1h", texto


def test_provedor_marca_como_ao_vivo_pelo_status(tmp_path):
    """Ponta a ponta: o mapa da Superbet lê periodStatus."""
    import json
    from pathlib import Path

    from betai.providers import FieldMap, GenericJsonProvider

    spec = json.loads(
        (Path(__file__).resolve().parents[1] / "examples" / "superbet.json").read_text(
            encoding="utf-8"
        )
    )
    provider = GenericJsonProvider(FieldMap(spec), respect_robots=False)
    cru = {
        "eventId": 1,
        "tournamentId": 3,
        "matchName": "A·B",
        # Minuto ausente, mas o jogo está no segundo tempo com 0-1.
        "metadata": {
            "homeTeamScore": "0", "awayTeamScore": "1", "periodStatus": "2H",
        },
        "odds": [
            {"marketName": "Resultado Final", "code": "1", "price": 6.25},
            {"marketName": "Resultado Final", "code": "0", "price": 3.40},
            {"marketName": "Resultado Final", "code": "2", "price": 1.55},
        ],
    }
    ev = provider._parse_event(cru)
    assert ev.state.is_live, "jogo com placar 0-1 não pode constar como pré-jogo"
    assert ev.state.period == "2h"
