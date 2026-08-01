from datetime import datetime, timezone

import pytest

from betai.ai.analyst import AiVerdict, SelectionAdjustment, apply_adjustments
from betai.models import (
    Event,
    Market,
    MarketKey,
    MatchState,
    Selection,
)
from betai.pipeline import Pipeline, derive_baseline, market_probabilities
from betai.providers import MockProvider
from betai.storage import Store


def make_event(minute=0, sh=0, sa=0, event_id="e1", odds_1x2=(2.10, 3.40, 3.60)):
    period = "pre_match" if minute == 0 else ("1h" if minute <= 45 else "2h")
    return Event(
        event_id=event_id,
        league="Teste",
        home_team="Casa",
        away_team="Fora",
        starts_at=datetime.now(timezone.utc),
        source="test",
        state=MatchState(minute=minute, period=period, score_home=sh, score_away=sa),
        markets=[
            Market(
                key=MarketKey.MATCH_ODDS,
                selections=[
                    Selection(outcome="home", odds=odds_1x2[0]),
                    Selection(outcome="draw", odds=odds_1x2[1]),
                    Selection(outcome="away", odds=odds_1x2[2]),
                ],
            ),
            Market(
                key=MarketKey.OVER_UNDER,
                line=2.5,
                selections=[
                    Selection(outcome="over", odds=1.95, line=2.5),
                    Selection(outcome="under", odds=1.95, line=2.5),
                ],
            ),
        ],
    )


def test_probabilidades_de_mercado_por_familia():
    probs = market_probabilities(make_event())
    assert pytest.approx(1.0, abs=1e-9) == sum(
        v for k, v in probs.items() if k.startswith("1x2.")
    )
    assert pytest.approx(1.0, abs=1e-9) == sum(
        v for k, v in probs.items() if k.startswith("over_under@2.5.")
    )


def test_baseline_reproduz_o_mercado_pre_jogo():
    """Sem informação nova, o modelo tem que concordar com as odds pré-jogo."""
    event = make_event()
    pipeline = Pipeline()
    analysis = pipeline.analyze(event)

    assert analysis.baseline_source == "pre_match"
    market = market_probabilities(event)
    for key in ("1x2.home", "1x2.away", "over_under@2.5.over"):
        assert analysis.probabilities[key] == pytest.approx(market[key], abs=0.01)


def test_sem_valor_no_pre_jogo():
    """Se o modelo é calibrado pelo mercado, não pode achar valor em si mesmo."""
    analysis = Pipeline(min_edge=0.03).analyze(make_event())
    assert analysis.value_bets == []


def test_baseline_pre_jogo_e_reaproveitado_ao_vivo():
    pipeline = Pipeline()
    pre = make_event(event_id="jogo")
    pipeline.register_baseline(pre)

    ao_vivo = make_event(minute=70, sh=1, sa=0, event_id="jogo")
    analysis = pipeline.analyze(ao_vivo)

    assert analysis.baseline_source == "pre_match"
    assert analysis.lambda_home == pytest.approx(pipeline._baselines["jogo"][0])


def test_sem_baseline_o_pipeline_sinaliza_inversao():
    ao_vivo = make_event(minute=70, sh=1, sa=0, event_id="sem-historico")
    analysis = Pipeline().analyze(ao_vivo)
    assert analysis.baseline_source == "live_inverted"


def test_baseline_recuperado_do_banco(tmp_path):
    db = tmp_path / "t.db"
    with Store(db) as store:
        store.save_snapshot(make_event(event_id="jogo"))
        pipeline = Pipeline(store=store)
        analysis = pipeline.analyze(make_event(minute=70, sh=1, sa=0, event_id="jogo"))
        assert analysis.baseline_source == "pre_match"


def test_divergencia_ao_vivo_gera_valor():
    """Odds ao vivo desalinhadas com o modelo têm que virar aposta de valor."""
    pipeline = Pipeline(min_edge=0.03)
    pipeline.register_baseline(make_event(event_id="jogo"))

    # 80 minutos, mandante 2-0: o mandante é quase certo, mas a casa ainda
    # paga 1.60 nele.
    ao_vivo = make_event(minute=80, sh=2, sa=0, event_id="jogo", odds_1x2=(1.10, 11.0, 34.0))
    analysis = pipeline.analyze(ao_vivo)

    assert any(b.outcome == "home" for b in analysis.value_bets)
    top = analysis.value_bets[0]
    assert top.edge > 0.03
    assert top.kelly_fraction > 0


def test_pipeline_roda_ponta_a_ponta_com_o_mock():
    provider = MockProvider()
    pipeline = Pipeline()
    for event in provider.fetch_upcoming():
        pipeline.register_baseline(event)
    for event in provider.fetch_live():
        analysis = pipeline.analyze(event)
        assert analysis.lambda_home > 0
        assert analysis.probabilities
        assert 0.99 < sum(
            v for k, v in analysis.probabilities.items() if k.startswith("1x2.")
        ) < 1.01


def test_storage_guarda_e_le_snapshots(tmp_path):
    with Store(tmp_path / "t.db") as store:
        store.save_snapshot(make_event(event_id="x"))
        store.save_snapshot(make_event(minute=40, sh=1, event_id="x"))
        snaps = store.snapshots_for("x")
        assert [s.state.minute for s in snaps] == [0, 40]
        assert store.tracked_events()[0]["snapshots"] == 2


def test_storage_persiste_analise_e_resultado(tmp_path):
    with Store(tmp_path / "t.db") as store:
        pipeline = Pipeline(min_edge=0.0, store=store)
        analysis = pipeline.analyze(make_event(event_id="x"))
        store.save_analysis(analysis)
        store.save_result("x", 2, 1)
        historico = list(store.value_bet_history("x"))
        for row in historico:
            assert row["final_score"] == (2, 1)


# ---------- camada de IA (sem chamar a API) ----------


def test_ajuste_da_ia_respeita_o_teto():
    probs = {"1x2.home": 0.50, "1x2.draw": 0.25, "1x2.away": 0.25}
    verdict = AiVerdict(
        summary="",
        confidence=0.5,
        adjustments=[SelectionAdjustment(key="1x2.home", delta=0.50, rationale="exagero")],
    )
    out = apply_adjustments(probs, verdict, max_shift=0.05)
    # Delta cortado em 0.05, depois renormalizado — o aumento é menor que 0.05.
    assert out["1x2.home"] < 0.56
    assert out["1x2.home"] > probs["1x2.home"]


def test_ajuste_renormaliza_a_familia():
    probs = {"1x2.home": 0.50, "1x2.draw": 0.25, "1x2.away": 0.25}
    verdict = AiVerdict(
        summary="",
        confidence=0.8,
        adjustments=[SelectionAdjustment(key="1x2.home", delta=0.04, rationale="desfalque")],
    )
    out = apply_adjustments(probs, verdict)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)


def test_ajuste_ignora_chave_desconhecida():
    probs = {"1x2.home": 0.5, "1x2.draw": 0.25, "1x2.away": 0.25}
    verdict = AiVerdict(
        summary="",
        confidence=0.1,
        adjustments=[SelectionAdjustment(key="mercado.inexistente", delta=0.9, rationale="")],
    )
    assert apply_adjustments(probs, verdict) == pytest.approx(probs)


def test_sem_ajustes_nada_muda():
    probs = {"1x2.home": 0.5, "1x2.draw": 0.25, "1x2.away": 0.25}
    verdict = AiVerdict(summary="contexto insuficiente", confidence=0.0)
    assert apply_adjustments(probs, verdict) == pytest.approx(probs)


# ---------- sem baseline pré-jogo ----------


def test_sem_baseline_nao_inventa_valor():
    """O caso que quebrou com dados reais da Superbet.

    Sem odds pré-jogo, as taxas vêm das próprias odds ao vivo — então não
    existe opinião independente e não pode existir vantagem. Antes, os
    efeitos de placar e cartão eram aplicados por cima de odds que já os
    embutiam, e a divergência artificial virava dezenas de apostas falsas.
    """
    for minuto, sh, sa in [(20, 0, 0), (45, 1, 0), (73, 1, 1), (87, 2, 4)]:
        ao_vivo = make_event(
            minute=minuto, sh=sh, sa=sa, event_id=f"sem-base-{minuto}",
            odds_1x2=(2.50, 3.20, 3.00),
        )
        analysis = Pipeline(min_edge=0.03).analyze(ao_vivo)
        assert analysis.baseline_source == "live_inverted"
        assert analysis.value_bets == [], f"{minuto}': {analysis.value_bets}"


def test_sem_baseline_o_modelo_reproduz_o_mercado():
    """Se o modelo veio do mercado, ele tem que concordar com o mercado."""
    ao_vivo = make_event(minute=70, sh=1, sa=0, event_id="espelho")
    analysis = Pipeline().analyze(ao_vivo)
    market = market_probabilities(ao_vivo)

    for chave in ("1x2.home", "1x2.away", "over_under@2.5.over"):
        assert analysis.probabilities[chave] == pytest.approx(market[chave], abs=0.02)


def test_lambdas_nao_explodem_no_fim_do_jogo():
    """Aos 87 minutos a fração restante é ~0.04. Reescalar por ela produzia
    lambdas de 50 gols."""
    ao_vivo = make_event(minute=87, sh=2, sa=4, event_id="fim")
    analysis = Pipeline().analyze(ao_vivo)
    assert analysis.lambda_home < 4.0, analysis.lambda_home
    assert analysis.lambda_away < 4.0, analysis.lambda_away


def test_lambda_nunca_e_zero():
    """Lambda 0.00 significaria 'impossível marcar' — nenhuma odd justifica."""
    lopsided = make_event(event_id="desequilibrado", odds_1x2=(1.02, 25.0, 60.0))
    analysis = Pipeline().analyze(lopsided)
    assert analysis.lambda_home > 0
    assert analysis.lambda_away > 0


def test_com_baseline_o_valor_volta_a_aparecer():
    """A supressão vale só para o caso sem baseline — não pode matar o
    funcionamento normal."""
    pipeline = Pipeline(min_edge=0.03)
    pipeline.register_baseline(make_event(event_id="com-base"))
    ao_vivo = make_event(
        minute=80, sh=2, sa=0, event_id="com-base", odds_1x2=(1.10, 11.0, 34.0)
    )
    analysis = pipeline.analyze(ao_vivo)
    assert analysis.baseline_source == "pre_match"
    assert analysis.value_bets


# ---------- baseline capturado no início do jogo ----------


def test_captura_baseline_nos_primeiros_minutos():
    """Aos 3 minutos as odds ao vivo ainda são praticamente as de pré-jogo.

    É o que permite o `watch` funcionar sozinho quando a casa não expõe um
    endpoint de pré-jogo — caso da Superbet.
    """
    cedo = make_event(minute=3, event_id="cedo")
    analysis = Pipeline().analyze(cedo)
    assert analysis.baseline_source == "pre_match"
    assert 1.0 < analysis.lambda_home + analysis.lambda_away < 5.0


def test_baseline_cedo_bate_com_o_pre_jogo():
    """A referência tirada aos 3 minutos tem que ser próxima da real."""
    pre = Pipeline().analyze(make_event(event_id="a"))
    cedo = Pipeline().analyze(make_event(minute=3, event_id="b"))
    assert cedo.lambda_home == pytest.approx(pre.lambda_home, rel=0.15)
    assert cedo.lambda_away == pytest.approx(pre.lambda_away, rel=0.15)


def test_jogo_avancado_nao_serve_de_baseline():
    """Aos 40 minutos as odds já embutem tempo e placar — não é referência."""
    analysis = Pipeline().analyze(make_event(minute=40, sh=1, event_id="tarde"))
    assert analysis.baseline_source == "live_inverted"


def test_baseline_cedo_persiste_e_gera_valor_depois(tmp_path):
    """O ciclo completo do `watch`: pega cedo, guarda, usa mais tarde."""
    with Store(tmp_path / "t.db") as store:
        pipeline = Pipeline(min_edge=0.03, store=store)

        cedo = make_event(minute=2, event_id="jogo")
        store.save_snapshot(cedo)
        pipeline.analyze(cedo)

        # Novo processo: baseline recuperado do banco.
        outro = Pipeline(min_edge=0.03, store=store)
        tarde = make_event(
            minute=80, sh=2, sa=0, event_id="jogo", odds_1x2=(1.10, 11.0, 34.0)
        )
        analysis = outro.analyze(tarde)

        assert analysis.baseline_source == "pre_match"
        assert analysis.value_bets


def test_o_primeiro_baseline_visto_e_o_que_vale():
    """Registrar de novo mais tarde não pode sobrescrever a referência boa."""
    pipeline = Pipeline()
    pipeline.register_baseline(make_event(event_id="j"))
    original = pipeline._baselines["j"]
    pipeline.register_baseline(make_event(minute=4, sh=2, event_id="j"))
    assert pipeline._baselines["j"] == original


def test_provedor_generico_busca_jogos_futuros():
    """Sem isto, `bet-ai upcoming` devolvia vazio e o baseline nunca vinha."""
    from betai.providers import FieldMap, GenericJsonProvider

    spec = {
        "url": "https://casa/api",
        "params": {"state": "live"},
        "params_upcoming": {"state": "prematch"},
        "events_path": "data",
        "fields": {"event_id": "id", "home_team": "home", "away_team": "away"},
        "markets": [
            {"key": "1x2", "path": "odds", "outcome_field": "code", "odds_field": "price"}
        ],
    }
    provider = GenericJsonProvider(FieldMap(spec), respect_robots=False, min_interval=0.0)

    chamadas: list[dict] = []

    def falso(url, params):
        chamadas.append(params)
        return {
            "data": [
                {
                    "id": "1", "home": "A", "away": "B",
                    "odds": [
                        {"code": "home", "price": 2.0},
                        {"code": "draw", "price": 3.4},
                        {"code": "away", "price": 3.6},
                    ],
                }
            ]
        }

    provider._get_json = falso
    eventos = list(provider.fetch_upcoming())
    assert chamadas == [{"state": "prematch"}]
    assert len(eventos) == 1


def test_sem_params_upcoming_nao_quebra():
    from betai.providers import FieldMap, GenericJsonProvider

    spec = {
        "url": "https://casa/api",
        "params": {"state": "live"},
        "events_path": "data",
        "fields": {"event_id": "id", "home_team": "home", "away_team": "away"},
        "markets": [{"key": "1x2", "path": "odds", "outcome_field": "c", "odds_field": "p"}],
    }
    provider = GenericJsonProvider(FieldMap(spec), respect_robots=False)
    assert list(provider.fetch_upcoming()) == []
