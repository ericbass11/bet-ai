import json
from pathlib import Path

import pytest

from betai.engine.devig import margin
from betai.models import MarketKey
from betai.providers import FieldMap, GenericJsonProvider, MockProvider
from betai.providers.generic_json import dig

FIELD_MAP = Path(__file__).resolve().parents[1] / "examples" / "field_map.json"


# ---------- mock ----------


def test_mock_gera_odds_com_margem_positiva():
    """Overround abaixo de 1 significaria a casa pagando para receber aposta."""
    for event in list(MockProvider().fetch_live()) + list(MockProvider().fetch_upcoming()):
        for mkt in event.markets:
            m = margin([s.odds for s in mkt.selections])
            assert 0.0 < m < 0.20, f"{event.label} {mkt.key}: margem {m:.3f}"


def test_mock_e_coerente_entre_mercados():
    """Todos os mercados saem da mesma matriz, então não podem se contradizer.

    Sem isso o pipeline acharia valor em over e under ao mesmo tempo — que é
    defeito da fixture, não do modelo.
    """
    from betai.pipeline import Pipeline

    provider = MockProvider()
    pipeline = Pipeline(min_edge=0.03)
    for event in provider.fetch_upcoming():
        pipeline.register_baseline(event)
        analysis = pipeline.analyze(event)
        # Um mercado com os dois lados apontados como valor é contradição.
        mercados = [(b.market, b.line) for b in analysis.value_bets]
        assert len(set(mercados)) == len(mercados), f"{event.label}: {mercados}"
        # E, calibrado pelas próprias odds, não deveria haver valor nenhum.
        assert not analysis.value_bets, f"{event.label}: {analysis.value_bets}"


def test_mock_reusa_o_mesmo_id_entre_pre_jogo_e_ao_vivo():
    """É o que permite o baseline pré-jogo alcançar o evento ao vivo."""
    pre = {e.event_id for e in MockProvider().fetch_upcoming()}
    live = {e.event_id for e in MockProvider().fetch_live()}
    assert pre == live


def test_mock_e_deterministico():
    a = list(MockProvider().fetch_live())
    b = list(MockProvider().fetch_live())
    assert [e.state.minute for e in a] == [e.state.minute for e in b]
    assert [s.odds for s in a[0].markets[0].selections] == [
        s.odds for s in b[0].markets[0].selections
    ]


# ---------- dig ----------


def test_dig_navega_dicionarios_e_listas():
    data = {"a": {"b": [{"c": 42}]}}
    assert dig(data, "a.b.0.c") == 42


def test_dig_devolve_default_em_caminho_ausente():
    assert dig({"a": 1}, "a.b.c", "vazio") == "vazio"
    assert dig({"a": [1]}, "a.9", None) is None
    assert dig(None, "qualquer", 0) == 0


# ---------- generic_json ----------


@pytest.fixture
def provider():
    spec = json.loads(FIELD_MAP.read_text(encoding="utf-8"))
    return GenericJsonProvider(FieldMap(spec), respect_robots=False)


def _payload_event():
    """Simula o JSON que o mapa de exemplo espera receber."""
    return {
        "id": "evt-991",
        "competition": {"name": "Brasileirão Série A"},
        "participants": [{"name": "Grêmio"}, {"name": "Internacional"}],
        "clock": {"minutes": 58},
        "score": {"home": 1, "away": 1},
        "cards": {"red": {"home": 0, "away": 1}},
        "markets": {
            "match_result": {
                "selections": [
                    {"label": "1", "decimal_odds": 2.05},
                    {"label": "X", "decimal_odds": 3.50},
                    {"label": "2", "decimal_odds": 4.10},
                ]
            },
            "total_goals_2_5": {
                "selections": [
                    {"label": "Mais de 2.5", "decimal_odds": 1.85},
                    {"label": "Menos de 2.5", "decimal_odds": 2.00},
                ]
            },
            "both_teams_to_score": {
                "selections": [
                    {"label": "Sim", "decimal_odds": 1.70},
                    {"label": "Não", "decimal_odds": 2.15},
                ]
            },
            "asian_handicap": {
                "selections": [
                    {"side": "home", "decimal_odds": 1.95, "handicap": -0.5},
                    {"side": "away", "decimal_odds": 1.95, "handicap": -0.5},
                ]
            },
        },
    }


def test_mapa_de_exemplo_parseia_o_payload(provider):
    """O mapa documentado no README precisa funcionar de verdade."""
    event = provider._parse_event(_payload_event())

    assert event is not None
    assert event.event_id == "evt-991"
    assert event.label == "Grêmio x Internacional"
    assert event.league == "Brasileirão Série A"
    assert event.state.minute == 58
    assert event.state.period == "2h"
    assert (event.state.score_home, event.state.score_away) == (1, 1)
    assert event.state.red_cards_away == 1


def test_mapa_traduz_os_rotulos_da_casa(provider):
    event = provider._parse_event(_payload_event())
    mo = event.market(MarketKey.MATCH_ODDS)
    assert [s.outcome for s in mo.selections] == ["home", "draw", "away"]
    assert event.market(MarketKey.BTTS).selection("yes").odds == 1.70
    assert event.market(MarketKey.OVER_UNDER).line == 2.5


def test_handicap_le_a_linha_do_proprio_runner(provider):
    event = provider._parse_event(_payload_event())
    ah = event.market(MarketKey.ASIAN_HANDICAP)
    assert ah.selection("home").line == -0.5


def test_evento_sem_mercado_util_e_descartado(provider):
    payload = _payload_event()
    payload["markets"] = {}
    assert provider._parse_event(payload) is None


def test_evento_sem_identificacao_e_descartado(provider):
    payload = _payload_event()
    payload["participants"] = [{"name": "Grêmio"}]
    assert provider._parse_event(payload) is None


def test_odds_invalidas_sao_ignoradas(provider):
    payload = _payload_event()
    payload["markets"]["match_result"]["selections"] = [
        {"label": "1", "decimal_odds": 1.0},  # odd inválida
        {"label": "X", "decimal_odds": "n/d"},  # não numérica
        {"label": "2", "decimal_odds": 4.10},
    ]
    event = provider._parse_event(payload)
    mo = event.market(MarketKey.MATCH_ODDS)
    assert [s.outcome for s in mo.selections] == ["away"]
