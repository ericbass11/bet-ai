"""Testes da medição — a parte que diz se o modelo presta.

Um erro aqui é pior que um erro no modelo: um modelo ruim com medição honesta
se conserta; um modelo ruim que se declara bom faz perder dinheiro com
confiança.
"""

import pytest

from betai.report import (
    Aposta,
    apostas_resolvidas,
    calibracao,
    desempenho,
    resolve,
)
from betai.storage import Store

# ---------- resolução de aposta ----------


@pytest.mark.parametrize(
    "outcome,placar,esperado",
    [
        ("home", (2, 1), True),
        ("home", (1, 1), False),
        ("draw", (1, 1), True),
        ("draw", (0, 1), False),
        ("away", (0, 1), True),
    ],
)
def test_resolve_1x2(outcome, placar, esperado):
    assert resolve("1x2", outcome, None, placar) is esperado


@pytest.mark.parametrize(
    "outcome,placar,esperado",
    [
        ("home_draw", (1, 1), True),
        ("home_draw", (0, 1), False),
        ("home_away", (1, 1), False),
        ("home_away", (2, 1), True),
        ("draw_away", (1, 1), True),
    ],
)
def test_resolve_dupla_chance(outcome, placar, esperado):
    assert resolve("double_chance", outcome, None, placar) is esperado


def test_resolve_over_under():
    assert resolve("over_under", "over", 2.5, (2, 1)) is True
    assert resolve("over_under", "under", 2.5, (2, 1)) is False
    assert resolve("over_under", "under", 2.5, (1, 1)) is True


def test_linha_inteira_batida_na_mosca_anula():
    """Total 3 na linha 3.0: a casa devolve o dinheiro."""
    assert resolve("over_under", "over", 3.0, (2, 1)) is None


def test_resolve_ambas_marcam():
    assert resolve("btts", "yes", None, (1, 1)) is True
    assert resolve("btts", "yes", None, (3, 0)) is False
    assert resolve("btts", "no", None, (3, 0)) is True


def test_resolve_handicap_asiatico():
    assert resolve("asian_handicap", "home", -1.5, (3, 1)) is True
    assert resolve("asian_handicap", "home", -1.5, (2, 1)) is False
    assert resolve("asian_handicap", "away", 0.5, (1, 1)) is False


def test_handicap_que_empata_anula():
    assert resolve("asian_handicap", "home", -1.0, (2, 1)) is None


def test_resolve_placar_exato():
    assert resolve("correct_score", "2-1", None, (2, 1)) is True
    assert resolve("correct_score", "2-1", None, (1, 2)) is False


def test_selecao_desconhecida_nao_e_chutada():
    assert resolve("1x2", "qualquer_coisa", None, (1, 0)) is None
    assert resolve("mercado_novo", "home", None, (1, 0)) is None


# ---------- lucro ----------


def _aposta(odds=2.0, ganhou=True, minute=60, market="1x2", outcome="home", p=0.6, eid="j"):
    return Aposta(
        event_id=eid,
        minute=minute,
        market=market,
        outcome=outcome,
        line=None,
        odds=odds,
        model_probability=p,
        ganhou=ganhou,
    )


def test_lucro_de_uma_unidade():
    assert _aposta(odds=2.5, ganhou=True).lucro == pytest.approx(1.5)
    assert _aposta(odds=2.5, ganhou=False).lucro == pytest.approx(-1.0)
    assert _aposta(ganhou=None).lucro == 0.0


# ---------- uma aposta por seleção ----------


def _analise(bets, eid="j", minute=60, score=(1, 0), probs=None):
    return {
        "event_id": eid,
        "created_at": f"2026-08-01T00:{minute:02d}:00",
        "minute": minute,
        "probabilities": probs or {},
        "value_bets": bets,
        "score": score,
    }


def _bet(market="1x2", outcome="home", odds=2.0, p=0.6, line=None):
    return {
        "market": market,
        "outcome": outcome,
        "odds": odds,
        "model_probability": p,
        "line": line,
    }


def test_mesma_selecao_repetida_conta_uma_vez():
    """O sinal reaparece a cada ciclo; na prática você aposta uma vez só."""
    analises = [
        _analise([_bet(odds=2.0)], minute=60),
        _analise([_bet(odds=1.9)], minute=61),
        _analise([_bet(odds=1.8)], minute=62),
    ]
    apostas = apostas_resolvidas(analises)
    assert len(apostas) == 1
    assert apostas[0].odds == 2.0, "deve valer a primeira aparição"
    assert apostas[0].minute == 60


def test_selecoes_diferentes_contam_separado():
    analises = [_analise([_bet(outcome="home"), _bet(market="btts", outcome="yes")])]
    assert len(apostas_resolvidas(analises)) == 2


def test_linhas_diferentes_do_mesmo_mercado_contam_separado():
    analises = [
        _analise(
            [
                _bet(market="over_under", outcome="over", line=1.5),
                _bet(market="over_under", outcome="over", line=2.5),
            ]
        )
    ]
    assert len(apostas_resolvidas(analises)) == 2


def test_mesma_selecao_em_jogos_diferentes_conta_separado():
    analises = [_analise([_bet()], eid="a"), _analise([_bet()], eid="b")]
    assert len(apostas_resolvidas(analises)) == 2


# ---------- calibração ----------


def test_calibracao_separa_promessa_de_entrega():
    """Modelo que diz 80% e acerta metade das vezes tem que aparecer como tal."""
    analises = []
    for i in range(50):
        venceu = i < 25  # metade ganha
        analises.append(
            _analise([], eid=f"j{i}", score=(1, 0) if venceu else (0, 1),
                     probs={"1x2.home": 0.85})
        )

    faixa = next(f for f in calibracao(analises) if f.de == 0.8)
    assert faixa.n == 50
    assert faixa.previsto == pytest.approx(0.85)
    assert faixa.observado == pytest.approx(0.5)
    assert faixa.desvio == pytest.approx(0.35), "otimista em 35 pontos"


def test_calibracao_de_modelo_honesto_da_desvio_perto_de_zero():
    analises = [
        _analise([], eid=f"j{i}", score=(1, 0) if i < 70 else (0, 1),
                 probs={"1x2.home": 0.70})
        for i in range(100)
    ]
    faixa = next(f for f in calibracao(analises) if f.de == 0.7)
    assert abs(faixa.desvio) < 0.01


def test_faixa_com_poucos_casos_e_omitida():
    """Com 3 observações qualquer número é ruído."""
    analises = [_analise([], eid=f"j{i}", probs={"1x2.home": 0.55}) for i in range(3)]
    assert calibracao(analises) == []


def test_calibracao_ignora_chave_que_nao_sabe_resolver():
    analises = [
        _analise([], eid=f"j{i}", probs={"mercado_novo.algo": 0.5}) for i in range(50)
    ]
    assert calibracao(analises) == []


def test_calibracao_le_a_linha_da_chave():
    """`over_under@2.5.over` precisa virar mercado, linha e seleção."""
    analises = [
        _analise([], eid=f"j{i}", score=(2, 1), probs={"over_under@2.5.over": 0.9})
        for i in range(30)
    ]
    faixa = next(f for f in calibracao(analises) if f.de == 0.9)
    assert faixa.observado == 1.0, "2-1 dá 3 gols, over 2.5 ganhou"


# ---------- relatório completo ----------


def _gravar(store: Store, event_id, minute, bets, probs, score):
    import json

    from datetime import datetime, timezone

    with store._lock:
        cur = store.conn.cursor()
        cur.execute(
            """INSERT INTO analyses (event_id, created_at, minute, lambda_home,
               lambda_away, probabilities, value_bets, ai_summary)
               VALUES (?, ?, ?, 1.5, 1.2, ?, ?, NULL)""",
            (
                event_id,
                datetime.now(timezone.utc).isoformat(),
                minute,
                json.dumps(probs),
                json.dumps(bets),
            ),
        )
        store.conn.commit()
    store.save_result(event_id, *score)


def test_relatorio_de_banco_vazio_nao_quebra(tmp_path):
    with Store(tmp_path / "t.db") as store:
        rel = desempenho(store)
    assert rel.jogos == 0
    assert rel.geral.n == 0
    assert rel.geral.roi == 0.0


def test_relatorio_soma_o_retorno(tmp_path):
    with Store(tmp_path / "t.db") as store:
        # Ganha: home a 2.00 num 1-0.
        _gravar(store, "a", 60, [_bet(odds=2.0)], {"1x2.home": 0.6}, (1, 0))
        # Perde: home a 3.00 num 0-2.
        _gravar(store, "b", 60, [_bet(odds=3.0)], {"1x2.home": 0.4}, (0, 2))
        rel = desempenho(store)

    assert rel.jogos == 2
    assert rel.geral.n == 2
    assert rel.geral.ganhas == 1
    assert rel.geral.lucro == pytest.approx(0.0), "+1.00 e -1.00"
    assert rel.geral.roi == pytest.approx(0.0)


def test_relatorio_separa_por_mercado_e_por_minuto(tmp_path):
    with Store(tmp_path / "t.db") as store:
        _gravar(store, "a", 20, [_bet(odds=2.0)], {}, (1, 0))
        _gravar(store, "b", 80, [_bet(market="btts", outcome="yes", odds=2.0)], {}, (1, 1))
        rel = desempenho(store)

    assert set(rel.por_mercado) == {"1x2", "btts"}
    assert set(rel.por_minuto) == {"16-30'", "76-94'"}
    assert rel.por_mercado["btts"].ganhas == 1


def test_jogo_sem_resultado_fica_de_fora(tmp_path):
    import json

    from datetime import datetime, timezone

    with Store(tmp_path / "t.db") as store:
        with store._lock:
            store.conn.execute(
                """INSERT INTO analyses (event_id, created_at, minute, lambda_home,
                   lambda_away, probabilities, value_bets, ai_summary)
                   VALUES ('sem-resultado', ?, 60, 1.5, 1.2, '{}', ?, NULL)""",
                (datetime.now(timezone.utc).isoformat(), json.dumps([_bet()])),
            )
            store.conn.commit()
        rel = desempenho(store)

    assert rel.analises == 0, "não dá para pontuar jogo que não terminou"


def test_banca_cresce_quando_o_modelo_acerta(tmp_path):
    with Store(tmp_path / "t.db") as store:
        for i in range(10):
            _gravar(store, f"j{i}", 60, [_bet(odds=2.0, p=0.6)], {}, (1, 0))
        rel = desempenho(store)

    assert rel.geral.ganhas == 10
    assert rel.banca_final > 1.0


def test_banca_encolhe_quando_o_modelo_erra(tmp_path):
    with Store(tmp_path / "t.db") as store:
        for i in range(10):
            _gravar(store, f"j{i}", 60, [_bet(odds=2.0, p=0.6)], {}, (0, 1))
        rel = desempenho(store)

    assert rel.geral.ganhas == 0
    assert 0.0 < rel.banca_final < 1.0, "encolhe, mas o teto de stake evita a ruína"


def test_banca_nunca_fica_negativa(tmp_path):
    with Store(tmp_path / "t.db") as store:
        for i in range(500):
            _gravar(store, f"j{i}", 60, [_bet(odds=2.0, p=0.9)], {}, (0, 1))
        rel = desempenho(store)

    assert rel.banca_final >= 0.0
