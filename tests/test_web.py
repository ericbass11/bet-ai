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
                    Selection(outcome="home", odds=1.10),
                    Selection(outcome="draw", odds=11.00),
                    Selection(outcome="away", odds=34.0),
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


# ---------- banco entre threads ----------


def test_banco_funciona_a_partir_de_outra_thread(tmp_path):
    """O defeito que quebrou a interface web na primeira execução.

    O sqlite3 recusa uma conexão usada fora da thread que a criou. A web abre
    o banco no processo principal e coleta numa thread separada.
    """
    from betai.storage import Store

    with Store(tmp_path / "t.db") as store:
        erro: list[Exception] = []

        def gravar():
            try:
                store.save_snapshot(make_event(event_id="de-outra-thread"))
                store.save_analysis(_analise_com_valor())
                store.tracked_events()
                store.value_bet_history()
            except Exception as exc:
                erro.append(exc)

        t = threading.Thread(target=gravar)
        t.start()
        t.join()

        assert not erro, f"o banco recusou o acesso: {erro[0]}"
        assert store.snapshots_for("de-outra-thread")


def test_escritas_concorrentes_nao_se_atropelam(tmp_path):
    """O lock serializa o acesso; sem ele, escritas simultâneas corrompem."""
    from betai.storage import Store

    with Store(tmp_path / "t.db") as store:
        erros: list[Exception] = []

        def gravar(n):
            try:
                for i in range(10):
                    store.save_snapshot(make_event(event_id=f"j{n}-{i}"))
            except Exception as exc:
                erros.append(exc)

        threads = [threading.Thread(target=gravar, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not erros, erros
        assert len(store.tracked_events()) == 40


def test_leitura_nao_segura_o_lock_durante_o_laco(tmp_path):
    """value_bet_history devolvendo gerador travaria quem consultasse o banco
    dentro do próprio laço."""
    from betai.storage import Store

    with Store(tmp_path / "t.db") as store:
        store.save_analysis(_analise_com_valor())
        store.save_snapshot(make_event(event_id="x"))
        for _ in store.value_bet_history():
            store.tracked_events()  # travaria se o lock ainda estivesse preso


def test_porta_ocupada_da_mensagem_util(capsys, tmp_path, monkeypatch):
    """Sem isto o processo morre com traceback de socket — e, em segundo
    plano com a saída redirecionada, morre sem deixar rastro."""
    import argparse

    from betai import cli
    from betai.config import Settings

    ocupado = servir(Estado(), "127.0.0.1", 0)
    porta = ocupado.server_address[1]
    try:
        args = argparse.Namespace(
            port=porta, host="127.0.0.1", interval=60, sem_navegador=True,
            ai=False, context=None, no_store=True, verbose=False,
        )
        monkeypatch.setattr(cli, "build_provider", lambda s: __import__(
            "betai.providers", fromlist=["MockProvider"]).MockProvider())

        assert cli.cmd_web(args, Settings()) == 1
        erro = capsys.readouterr().err
        assert "pkill -f 'bet-ai web'" in erro
        assert f"--port {porta + 1}" in erro
    finally:
        ocupado.server_close()


def test_interface_informa_mercado_descartado():
    """Mercado sumir da tela sem explicação parece defeito; com o aviso, o
    usuário entende que a casa mandou odds suspensas."""
    from betai.models import Event, Market, MarketKey, MatchState, Selection
    from betai.pipeline import Pipeline
    from betai.web import Estado

    ruim = Event(
        event_id="r", league="L", home_team="A", away_team="B",
        starts_at=datetime.now(timezone.utc), source="t",
        state=MatchState(),
        markets=[
            Market(
                key=MarketKey.MATCH_ODDS,
                selections=[
                    Selection(outcome="home", odds=6.25),
                    Selection(outcome="draw", odds=5.77),
                    Selection(outcome="away", odds=15.00),
                ],
            )
        ],
    )
    estado = Estado()
    estado.atualizar([Pipeline().analyze(ruim)])
    jogo = estado.snapshot()["analises"][0]
    assert jogo["mercados_descartados"] == 1
    assert jogo["apostas"] == []

    from betai.web import PAGINA
    assert "mercados_descartados" in PAGINA
