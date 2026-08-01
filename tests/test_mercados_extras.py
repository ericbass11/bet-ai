"""Testes dos mercados que só existem no endpoint por jogo.

A tela mostrava um mercado só (Resultado Final) porque é isso que a listagem
da casa devolve — os outros vinte vivem em `/events/{id}`. Três coisas
precisavam existir para eles entrarem na análise: buscar o detalhe, separar as
linhas de over/under que vêm misturadas, e não tratar dupla chance como se
fosse uma partição.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from betai.engine.devig import is_plausible, margin, remove_vig
from betai.models import Event, Market, MarketKey, MatchState, Selection
from betai.pipeline import Pipeline, market_probabilities, usable_markets
from betai.providers import FieldMap, GenericJsonProvider

MAPA = json.loads(
    (Path(__file__).resolve().parents[1] / "examples" / "superbet.json").read_text(
        encoding="utf-8"
    )
)


# Resposta do endpoint por jogo, no formato real da casa: todos os mercados
# achatados num array `odds`, distinguidos por `marketName`, e as linhas de
# Total de Gols misturadas entre si.
DETALHE = {
    "data": [
        {
            "eventId": 14045808,
            "matchName": "Casa·Fora",
            "odds": [
                {"marketName": "Resultado Final", "code": "1", "price": 2.10},
                {"marketName": "Resultado Final", "code": "0", "price": 3.40},
                {"marketName": "Resultado Final", "code": "2", "price": 3.60},
                {"marketName": "Total de Gols", "code": None, "name": "Mais de 1.5",
                 "price": 1.35, "showSpecialBetValue": "1.5"},
                {"marketName": "Total de Gols", "code": None, "name": "Menos de 1.5",
                 "price": 3.10, "showSpecialBetValue": "1.5"},
                {"marketName": "Total de Gols", "code": None, "name": "Mais de 2.5",
                 "price": 2.05, "showSpecialBetValue": "2.5"},
                {"marketName": "Total de Gols", "code": None, "name": "Menos de 2.5",
                 "price": 1.75, "showSpecialBetValue": "2.5"},
                {"marketName": "Ambas as Equipes Marcam", "code": "1", "name": "Sim",
                 "price": 1.80},
                {"marketName": "Ambas as Equipes Marcam", "code": "2", "name": "Não",
                 "price": 1.95},
                {"marketName": "Dupla Chance", "code": "10", "price": 1.30},
                {"marketName": "Dupla Chance", "code": "12", "price": 1.32},
                {"marketName": "Dupla Chance", "code": "02", "price": 1.72},
            ],
        }
    ]
}


def provedor(resposta=DETALHE) -> GenericJsonProvider:
    """Provedor com a Superbet mapeada e o transporte HTTP dublado."""
    p = GenericJsonProvider(FieldMap(MAPA), min_interval=0.0, respect_robots=False)
    p.client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=resposta))
    )
    return p


# ---------- busca do detalhe ----------


def test_mapa_da_superbet_declara_o_endpoint_por_jogo():
    p = provedor()
    assert p.supports_details
    p.close()


def test_detalhe_traz_os_quatro_mercados():
    p = provedor()
    mercados = p.fetch_details("14045808")
    p.close()

    chaves = {m.key for m in mercados}
    assert MarketKey.MATCH_ODDS in chaves
    assert MarketKey.OVER_UNDER in chaves
    assert MarketKey.BTTS in chaves
    assert MarketKey.DOUBLE_CHANCE in chaves


def test_linhas_de_over_under_saem_separadas():
    """Vêm misturadas no mesmo `marketName`; juntas, o livro somaria ~2."""
    p = provedor()
    ou = [m for m in p.fetch_details("1") if m.key == MarketKey.OVER_UNDER]
    p.close()

    assert {m.line for m in ou} == {1.5, 2.5}
    for m in ou:
        assert {s.outcome for s in m.selections} == {"over", "under"}
        assert is_plausible([s.odds for s in m.selections]), f"linha {m.line} não fecha"


def test_rotulo_com_a_linha_dentro_vira_over_e_under():
    """'Mais de 2.5' muda a cada linha — mapa fixo não daria conta."""
    p = provedor()
    ou = next(m for m in p.fetch_details("1") if m.key == MarketKey.OVER_UNDER and m.line == 2.5)
    p.close()

    assert ou.selection("over").odds == 2.05
    assert ou.selection("under").odds == 1.75


def test_dupla_chance_traduz_os_pares_de_codigo():
    p = provedor()
    dc = next(m for m in p.fetch_details("1") if m.key == MarketKey.DOUBLE_CHANCE)
    p.close()

    assert {s.outcome for s in dc.selections} == {"home_draw", "home_away", "draw_away"}


def test_ambas_marcam_traduz_sim_e_nao():
    p = provedor()
    btts = next(m for m in p.fetch_details("1") if m.key == MarketKey.BTTS)
    p.close()

    assert btts.selection("yes").odds == 1.80
    assert btts.selection("no").odds == 1.95


def test_provedor_sem_endpoint_por_jogo_nao_promete_detalhe():
    from betai.providers import MockProvider

    m = MockProvider()
    assert not m.supports_details
    assert m.fetch_details("qualquer") == []


# ---------- dupla chance não é uma partição ----------


def _dupla_chance(odds=(1.30, 1.32, 1.72)) -> Market:
    return Market(
        key=MarketKey.DOUBLE_CHANCE,
        selections=[
            Selection(outcome="home_draw", odds=odds[0]),
            Selection(outcome="home_away", odds=odds[1]),
            Selection(outcome="draw_away", odds=odds[2]),
        ],
    )


def test_dupla_chance_cobre_duas_unidades_de_probabilidade():
    assert _dupla_chance().coverage == 2.0
    assert Market(
        key=MarketKey.MATCH_ODDS,
        selections=[
            Selection(outcome="home", odds=2.1),
            Selection(outcome="draw", odds=3.4),
            Selection(outcome="away", odds=3.6),
        ],
    ).coverage == 1.0


def test_livro_saudavel_de_dupla_chance_seria_recusado_como_particao():
    """Soma ~2.09: sem a cobertura, o filtro de coerência mataria o mercado."""
    odds = [s.odds for s in _dupla_chance().selections]
    assert sum(1 / o for o in odds) > 2.0
    assert not is_plausible(odds)
    assert is_plausible(odds, coverage=2.0)
    assert 0.0 < margin(odds, coverage=2.0) < 0.10


def test_devig_de_dupla_chance_soma_dois():
    justas = remove_vig([s.odds for s in _dupla_chance().selections], coverage=2.0)
    assert sum(justas) == pytest.approx(2.0)
    # E cada par continua sendo mais provável que qualquer resultado isolado.
    assert all(0.5 < p < 1.0 for p in justas)


def test_devig_sem_cobertura_continua_igual():
    odds = [2.10, 3.40, 3.60]
    assert remove_vig(odds) == remove_vig(odds, coverage=1.0)


# ---------- ponta a ponta ----------


def _evento(mercados: list[Market], minute=0, sh=0, sa=0, event_id="j") -> Event:
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
        markets=mercados,
    )


def test_mercados_extras_entram_no_evento_sem_duplicar():
    p = provedor()
    extras = p.fetch_details("1")
    p.close()

    base = _evento([m for m in extras if m.key == MarketKey.MATCH_ODDS])
    completo = base.with_markets(extras)

    assert len(completo.markets_of(MarketKey.MATCH_ODDS)) == 1, "1X2 duplicou"
    assert len(completo.markets_of(MarketKey.OVER_UNDER)) == 2
    # 1X2, over/under 1.5, over/under 2.5, ambas marcam, dupla chance.
    assert len(completo.markets) == 5


def test_evento_sem_extras_nao_e_copiado():
    ev = _evento([])
    assert ev.with_markets([]) is ev


def test_todos_os_mercados_extras_sobrevivem_a_analise():
    """O teste que importa: eles aparecem na tela, e com probabilidade."""
    p = provedor()
    extras = p.fetch_details("1")
    p.close()

    evento = _evento(extras)
    assert len(usable_markets(evento)) == len(extras), "algum mercado foi descartado"

    mercado = market_probabilities(evento)
    for chave in (
        "1x2.home",
        "over_under@2.5.over",
        "over_under@1.5.under",
        "btts.yes",
        "double_chance.home_draw",
    ):
        assert chave in mercado, chave
        assert 0.0 < mercado[chave] < 1.0

    analise = Pipeline(min_edge=0.03).analyze(evento)
    for chave in mercado:
        assert chave in analise.probabilities, f"o modelo não opina sobre {chave}"


def test_dupla_chance_do_modelo_bate_com_a_do_mercado():
    """Prova de que a cobertura está certa: sem ela o mercado sairia pela
    metade e a dupla chance viraria uma sequência de valor inventado."""
    p = provedor()
    extras = p.fetch_details("1")
    p.close()

    evento = _evento(extras)
    pipeline = Pipeline(min_edge=0.03)
    pipeline.register_baseline(evento)
    analise = pipeline.analyze(evento)

    mercado = market_probabilities(evento)
    for chave in ("double_chance.home_draw", "double_chance.draw_away"):
        assert abs(analise.probabilities[chave] - mercado[chave]) < 0.06, chave

    assert not [b for b in analise.value_bets if b.market == MarketKey.DOUBLE_CHANCE]


# ---------- quantas requisições o ciclo gasta ----------


class ProvedorComDetalhe(GenericJsonProvider):
    """Provedor de teste que conta as buscas de detalhe."""

    def __init__(self, eventos: list[Event]) -> None:
        self.eventos = eventos
        self.pedidos: list[str] = []
        self.extras = [
            Market(
                key=MarketKey.BTTS,
                selections=[
                    Selection(outcome="yes", odds=1.80),
                    Selection(outcome="no", odds=1.95),
                ],
            )
        ]

    name = "teste"

    @property
    def supports_details(self) -> bool:
        return True

    def fetch_live(self):
        return self.eventos

    def fetch_upcoming(self):
        return []

    def fetch_details(self, event_id: str) -> list[Market]:
        self.pedidos.append(event_id)
        return self.extras

    def close(self) -> None:
        pass


def _mercado_1x2() -> Market:
    return Market(
        key=MarketKey.MATCH_ODDS,
        selections=[
            Selection(outcome="home", odds=2.10),
            Selection(outcome="draw", odds=3.40),
            Selection(outcome="away", odds=3.60),
        ],
    )


def test_ciclo_so_gasta_requisicao_com_jogo_que_tem_referencia():
    from betai.cli import collect_live_analyses

    cedo = _evento([_mercado_1x2()], minute=3, event_id="cedo")
    tarde = _evento([_mercado_1x2()], minute=70, event_id="tarde")
    provider = ProvedorComDetalhe([cedo, tarde])

    analises = collect_live_analyses(provider, None, Pipeline(min_edge=0.03))

    assert provider.pedidos == ["cedo"], "gastou requisição em jogo sem referência"
    assert len(analises) == 2


def test_ciclo_tem_teto_de_requisicoes_por_rodada():
    """Com 190 jogos ao vivo, buscar detalhe de todos levaria mais que o ciclo."""
    from betai.cli import MAX_DETALHES_POR_CICLO, collect_live_analyses

    quantidade = MAX_DETALHES_POR_CICLO + 5
    eventos = [
        _evento([_mercado_1x2()], minute=2, event_id=f"j{i}") for i in range(quantidade)
    ]
    provider = ProvedorComDetalhe(eventos)

    collect_live_analyses(provider, None, Pipeline(min_edge=0.03))

    assert len(provider.pedidos) == MAX_DETALHES_POR_CICLO


def test_detalhe_recebido_entra_na_analise():
    from betai.cli import collect_live_analyses

    provider = ProvedorComDetalhe([_evento([_mercado_1x2()], minute=3, event_id="j")])
    analise = collect_live_analyses(provider, None, Pipeline(min_edge=0.03))[0]

    assert analise.event.market(MarketKey.BTTS) is not None
    assert "btts.yes" in analise.probabilities
