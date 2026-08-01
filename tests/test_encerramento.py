"""Testes do fechamento automático do ciclo.

O programa via o placar final de todo jogo e jogava fora. Existia um comando
`settle` manual, um jogo por vez — com 190 partidas ao vivo, nunca usado. Sem
resultado registrado não há como medir se o modelo acerta, que é a única
pergunta que importa no fim.
"""

from datetime import datetime, timedelta, timezone

from betai.models import Event, Market, MarketKey, MatchState, Selection
from betai.storage import Store


def evento(minute=0, periodo="pre_match", sh=0, sa=0, event_id="j", quando=None) -> Event:
    return Event(
        event_id=event_id,
        league="Teste",
        home_team="Casa",
        away_team="Fora",
        starts_at=datetime.now(timezone.utc),
        source="test",
        captured_at=quando or datetime.now(timezone.utc),
        state=MatchState(minute=minute, period=periodo, score_home=sh, score_away=sa),
        markets=[
            Market(
                key=MarketKey.MATCH_ODDS,
                selections=[
                    Selection(outcome="home", odds=2.10),
                    Selection(outcome="draw", odds=3.40),
                    Selection(outcome="away", odds=3.60),
                ],
            )
        ],
    )


def _resultado(store: Store, event_id: str):
    return store.result_for(event_id)


def test_jogo_marcado_como_encerrado_e_anotado(tmp_path):
    with Store(tmp_path / "t.db") as store:
        store.save_snapshot(evento(minute=45, periodo="2h", sh=1, sa=0))
        store.save_snapshot(evento(minute=90, periodo="encerrado", sh=2, sa=1))

        assert store.settle_finished() == [("j", 2, 1)]


def test_o_placar_anotado_e_o_do_ultimo_estado_visto(tmp_path):
    """Não adianta pegar qualquer snapshot: o de 45' diria 1-0."""
    with Store(tmp_path / "t.db") as store:
        store.save_snapshot(evento(minute=45, periodo="2h", sh=1, sa=0))
        store.save_snapshot(evento(minute=90, periodo="encerrado", sh=3, sa=2))
        store.settle_finished()

        assert _resultado(store, "j") == (3, 2)


def test_jogo_em_andamento_nao_e_encerrado(tmp_path):
    with Store(tmp_path / "t.db") as store:
        store.save_snapshot(evento(minute=70, periodo="2h", sh=1, sa=0))
        assert store.settle_finished() == []


def test_jogo_que_nem_comecou_nao_e_encerrado(tmp_path):
    """Um pré-jogo capturado ontem e cancelado não tem placar para anotar."""
    ontem = datetime.now(timezone.utc) - timedelta(days=1)
    with Store(tmp_path / "t.db") as store:
        store.save_snapshot(evento(minute=0, periodo="pre_match", quando=ontem))
        assert store.settle_finished() == []


def test_jogo_que_sumiu_do_feed_e_dado_por_encerrado(tmp_path):
    """Nem toda casa publica o status final — o jogo só desaparece da lista."""
    velho = datetime.now(timezone.utc) - timedelta(hours=Store.ABANDONO_HORAS + 1)
    with Store(tmp_path / "t.db") as store:
        store.save_snapshot(evento(minute=88, periodo="2h", sh=2, sa=2, quando=velho))
        assert store.settle_finished() == [("j", 2, 2)]


def test_jogo_visto_ha_pouco_ainda_nao_conta_como_sumido(tmp_path):
    recente = datetime.now(timezone.utc) - timedelta(hours=1)
    with Store(tmp_path / "t.db") as store:
        store.save_snapshot(evento(minute=88, periodo="2h", sh=2, sa=2, quando=recente))
        assert store.settle_finished() == []


def test_nao_reanota_o_que_ja_tem_resultado(tmp_path):
    with Store(tmp_path / "t.db") as store:
        store.save_snapshot(evento(minute=90, periodo="encerrado", sh=2, sa=1))
        assert len(store.settle_finished()) == 1
        assert store.settle_finished() == []


def test_placar_anotado_a_mao_nao_e_sobrescrito(tmp_path):
    """O `settle` manual existe para corrigir; a automação não pode desfazer."""
    with Store(tmp_path / "t.db") as store:
        store.save_snapshot(evento(minute=90, periodo="encerrado", sh=2, sa=1))
        store.save_result("j", 9, 9)
        assert store.settle_finished() == []
        assert _resultado(store, "j") == (9, 9)


def test_varios_jogos_de_uma_vez(tmp_path):
    with Store(tmp_path / "t.db") as store:
        store.save_snapshot(evento(minute=90, periodo="encerrado", sh=1, sa=0, event_id="a"))
        store.save_snapshot(evento(minute=90, periodo="encerrado", sh=0, sa=2, event_id="b"))
        store.save_snapshot(evento(minute=30, periodo="1h", event_id="c"))

        assert sorted(store.settle_finished()) == [("a", 1, 0), ("b", 0, 2)]


# ---------- integração com o ciclo ----------


class ProvedorFalso:
    name = "falso"
    supports_details = False

    def __init__(self, eventos):
        self.eventos = eventos

    def fetch_live(self):
        return self.eventos

    def fetch_upcoming(self):
        return []

    def fetch_details(self, event_id):
        return []

    def close(self):
        pass


def test_o_ciclo_anota_sozinho_quem_terminou(tmp_path):
    from betai.cli import collect_live_analyses
    from betai.pipeline import Pipeline

    pipeline = Pipeline(min_edge=0.03)
    with Store(tmp_path / "t.db") as store:
        collect_live_analyses(ProvedorFalso([evento(minute=70, periodo="2h", sh=1)]), store, pipeline)
        assert _resultado(store, "j") is None, "jogo em andamento não pode ser encerrado"

        collect_live_analyses(
            ProvedorFalso([evento(minute=90, periodo="encerrado", sh=1, sa=0)]), store, pipeline
        )
        assert _resultado(store, "j") == (1, 0)


def test_ciclo_sem_banco_nao_quebra(tmp_path):
    from betai.cli import collect_live_analyses
    from betai.pipeline import Pipeline

    analises = collect_live_analyses(
        ProvedorFalso([evento(minute=70, periodo="2h", sh=1)]), None, Pipeline()
    )
    assert len(analises) == 1
