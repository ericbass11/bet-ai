import argparse

from betai.cli import run_live_cycle
from betai.pipeline import Pipeline
from betai.providers import MockProvider
from betai.storage import Store


def _args():
    return argparse.Namespace(verbose=False, no_store=False, ai=False, context=None)


def test_ciclo_ao_vivo_grava_snapshot_e_analise(tmp_path):
    with Store(tmp_path / "t.db") as store:
        pipeline = Pipeline(store=store)
        run_live_cycle(MockProvider(), store, pipeline, None, _args())

        eventos = store.tracked_events()
        assert len(eventos) == 3
        # Um snapshot pré-jogo + um ao vivo por evento.
        assert all(row["snapshots"] == 2 for row in eventos)


def test_ciclos_repetidos_nao_duplicam_o_pre_jogo(tmp_path):
    """Em modo watch isso rodaria a cada minuto — regravar o pré-jogo engorda
    o banco sem adicionar informação."""
    with Store(tmp_path / "t.db") as store:
        provider = MockProvider()
        pipeline = Pipeline(store=store)
        for _ in range(4):
            run_live_cycle(provider, store, pipeline, None, _args())

        for event_id in ("mock-0", "mock-1", "mock-2"):
            snaps = store.snapshots_for(event_id)
            pre = [s for s in snaps if not s.state.is_live]
            assert len(pre) == 1, f"{event_id}: {len(pre)} snapshots pré-jogo"
            assert len(snaps) == 5  # 1 pré-jogo + 4 ao vivo


def test_ciclo_funciona_sem_banco():
    """`--no-store` não pode quebrar o fluxo."""
    pipeline = Pipeline()
    run_live_cycle(MockProvider(), None, pipeline, None, _args())
