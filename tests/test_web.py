"""Testes da interface web."""

import json
import threading
import urllib.request
from datetime import datetime, timezone

import pytest

from betai.models import Event, Market, MarketKey, MatchState, Selection
from betai.pipeline import Pipeline
from betai.web import Estado, loop_de_coleta, servir


def make_event(minute=70, sh=1, sa=0, event_id="e1"):
    return Event(
        event_id=event_id,
        league="Brasileirão",
        home_team="Palmeiras",
        away_team="Flamengo",
        starts_at=datetime.now(timezone.utc),
        source="test",
        state=MatchState(minute=minute, period="2h", score_home=sh, score_away=sa),
        markets=[
            Market(
                key=MarketKey.MATCH_ODDS,
                selections=[
                    Selection(outcome="home", odds=1.60),
                    Selection(outcome="draw", odds=6.00),
                    Selection(outcome="away", odds=15.0),
                ],
            )
        ],
    )


def _analise_com_valor():
    pipeline = Pipeline(min_edge=0.03)
    pipeline.register_baseline(make_event(minute=0, sh=0, event_id="j"))
    return pipeline.analyze(make_event(minute=80, sh=2, sa=0, event_id="j"))


# ---------- estado ----------


def test_estado_serializa_a_analise():
    estado = Estado()
    estado.atualizar([_analise_com_valor()])
    dados = estado.snapshot()

    assert dados["resumo"]["jogos"] == 1
    jogo = dados["analises"][0]
    assert jogo["casa"] == "Palmeiras"
    assert jogo["placar"] == "2-0"
    assert jogo["minuto"] == 80
    assert jogo["apostas"], "a análise tinha valor e ele sumiu na serialização"
    assert jogo["apostas"][0]["selecao"] == "home"


def test_snapshot_e_json_valido():
    """A página consome isto: qualquer objeto não serializável quebra a tela."""
    estado = Estado()
    estado.atualizar([_analise_com_valor()])
    json.dumps(estado.snapshot(), ensure_ascii=False)


def test_resumo_conta_jogos_sem_referencia():
    estado = Estado()
    sem_base = Pipeline().analyze(make_event(minute=70, event_id="sem"))
    estado.atualizar([sem_base, _analise_com_valor()])
    resumo = estado.snapshot()["resumo"]
    assert resumo["jogos"] == 2
    assert resumo["sem_baseline"] == 1
    assert resumo["com_valor"] == 1


def test_erro_registrado_nao_apaga_os_dados_anteriores():
    """Uma falha de rede não pode esvaziar a tela."""
    estado = Estado()
    estado.atualizar([_analise_com_valor()])
    estado.falhar("ConnectError: sem rede")
    dados = estado.snapshot()
    assert dados["erro"] == "ConnectError: sem rede"
    assert len(dados["analises"]) == 1


def test_coleta_bem_sucedida_limpa_o_erro():
    estado = Estado()
    estado.falhar("falha antiga")
    estado.atualizar([_analise_com_valor()])
    assert estado.snapshot()["erro"] is None


# ---------- loop ----------


def test_loop_sobrevive_a_uma_falha():
    """Provedor fora do ar não pode derrubar a interface."""
    estado = Estado()
    parar = threading.Event()
    chamadas = []

    def coletar():
        chamadas.append(1)
        if len(chamadas) == 1:
            raise RuntimeError("provedor fora do ar")
        parar.set()
        return [_analise_com_valor()]

    loop_de_coleta(estado, coletar, intervalo=0, parar=parar)
    assert len(chamadas) == 2
    assert estado.snapshot()["erro"] is None


def test_loop_para_no_sinal():
    estado = Estado()
    parar = threading.Event()
    parar.set()
    loop_de_coleta(estado, lambda: [], intervalo=0, parar=parar)
    assert estado.snapshot()["ciclos"] == 0


# ---------- servidor ----------


@pytest.fixture
def servidor():
    estado = Estado()
    estado.atualizar([_analise_com_valor()])
    srv = servir(estado, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_pagina_carrega(servidor):
    with urllib.request.urlopen(f"{servidor}/") as r:
        assert r.status == 200
        html = r.read().decode()
    assert "bet-ai" in html
    assert "/api/analises" in html


def test_api_devolve_os_dados(servidor):
    with urllib.request.urlopen(f"{servidor}/api/analises") as r:
        dados = json.loads(r.read())
    assert dados["resumo"]["jogos"] == 1
    assert dados["analises"][0]["casa"] == "Palmeiras"


def test_caminho_desconhecido_da_404(servidor):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{servidor}/nao-existe")
    assert exc.value.code == 404


def test_pagina_nao_busca_nada_externo():
    """A página tem que funcionar sem internet — tudo embutido."""
    from betai.web import PAGINA

    for proibido in ("http://", "https://", "cdn.", "<script src"):
        assert proibido not in PAGINA, f"a página referencia recurso externo: {proibido}"


def test_pagina_traduz_o_vocabulario_interno():
    """A tela é para quem aposta, não para quem programa: 'btts — yes' não
    diz nada, 'Ambas marcam' diz."""
    from betai.web import PAGINA

    for termo in ("vence", "Empate", "Ambas marcam", "Mais de", "Menos de", "handicap"):
        assert termo in PAGINA, f"falta a tradução de {termo}"


def test_pagina_avisa_sobre_o_risco():
    """A tela mostra vantagem estimada — não pode passar por garantia."""
    from betai.web import PAGINA

    assert "não recomendação de aposta" in PAGINA
    assert "não é lucro garantido" in PAGINA
