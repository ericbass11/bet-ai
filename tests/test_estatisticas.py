"""Testes da busca por estatísticas ao vivo numa captura.

O `inspect` procura eventos e odds. Um painel de estatísticas não tem nem uma
coisa nem outra — tem chutes, posse de bola e ataques perigosos — e por isso
passava despercebido na captura.
"""

import json

from betai.discover import TERMOS_DE_ESTATISTICA, _normalizar, buscar_campos

# Formato 1: a casa embute alguns números no próprio payload de odds.
SUPERBET = {
    "data": [
        {
            "eventId": 14068216,
            "metadata": {
                "homeTeamScore": "2",
                "awayTeamScore": "3",
                "homeTeamCorners": 1,
                "awayTeamCorners": 6,
                "homeTeamYellowCards": 0,
                "minutes": "59",
            },
        }
    ]
}

# Formato 2: provedor de dados ao vivo. O nome da métrica é um *valor*, não um
# campo — procurar só por nome de campo passa direto.
SPORTRADAR = {
    "doc": [
        {
            "data": {
                "values": {
                    "110": {"name": "Ball possession", "value": {"home": 58, "away": 42}},
                    "125": {"name": "Shots on target", "value": {"home": 6, "away": 2}},
                    "1029": {"name": "Dangerous attacks", "value": {"home": 41, "away": 22}},
                }
            }
        }
    ]
}


def _caminhos(achados) -> list[str]:
    return [c for c, _ in achados]


def test_acha_escanteio_no_payload_da_casa():
    achados = buscar_campos(SUPERBET, TERMOS_DE_ESTATISTICA)
    caminhos = _caminhos(achados)
    assert "data.0.metadata.homeTeamCorners" in caminhos
    assert "data.0.metadata.awayTeamCorners" in caminhos


def test_nao_confunde_placar_com_estatistica():
    caminhos = _caminhos(buscar_campos(SUPERBET, TERMOS_DE_ESTATISTICA))
    assert not any("Score" in c or "minutes" in c for c in caminhos)


def test_acha_metrica_cujo_nome_e_valor_e_nao_campo():
    """O formato dos provedores de dados ao vivo."""
    achados = buscar_campos(SPORTRADAR, TERMOS_DE_ESTATISTICA)
    rotulos = " ".join(_caminhos(achados))
    assert "Shots on target" in rotulos
    assert "Ball possession" in rotulos
    assert "Dangerous attacks" in rotulos


def test_metrica_encontrada_vem_com_os_dois_lados():
    achados = buscar_campos(SPORTRADAR, TERMOS_DE_ESTATISTICA)
    _, valor = next(a for a in achados if "Shots on target" in a[0])
    assert valor["value"] == {"home": 6, "away": 2}


def test_termo_proprio_restringe_a_busca():
    achados = buscar_campos(SPORTRADAR, [_normalizar("possession")])
    assert len(achados) == 1
    assert "Ball possession" in achados[0][0]


def test_captura_sem_estatistica_nao_inventa_nada():
    assert buscar_campos({"data": [{"eventId": 1, "odds": []}]}, TERMOS_DE_ESTATISTICA) == []


def test_busca_ignora_acento_e_caixa():
    payload = {"jogo": {"Escanteios_Casa": 4, "POSSE": 61}}
    caminhos = _caminhos(buscar_campos(payload, TERMOS_DE_ESTATISTICA))
    assert "jogo.Escanteios_Casa" in caminhos
    assert "jogo.POSSE" in caminhos


def test_busca_nao_percorre_lista_inteira():
    """195 jogos na lista; o formato do primeiro já diz tudo."""
    payload = {"data": [{"corners": i} for i in range(195)]}
    assert len(buscar_campos(payload, TERMOS_DE_ESTATISTICA)) <= 2


# ---------- comando ----------


def _har(*respostas) -> dict:
    return {
        "log": {
            "entries": [
                {
                    "request": {"url": url},
                    "response": {
                        "content": {"mimeType": "application/json", "text": json.dumps(corpo)}
                    },
                }
                for url, corpo in respostas
            ]
        }
    }


def _rodar(tmp_path, har, **kwargs) -> tuple[int, str]:
    import argparse

    from betai.cli import cmd_estatisticas
    from betai.config import Settings

    caminho = tmp_path / "captura.har"
    caminho.write_text(json.dumps(har), encoding="utf-8")
    args = argparse.Namespace(
        captura=str(caminho), termo=None, limite=8, **{"listar": False, **kwargs}
    )
    import contextlib
    import io

    saida = io.StringIO()
    with contextlib.redirect_stdout(saida), contextlib.redirect_stderr(saida):
        codigo = cmd_estatisticas(args, Settings())
    return codigo, saida.getvalue()


def test_comando_lista_a_resposta_mais_promissora_primeiro(tmp_path):
    har = _har(
        ("https://casa/api/odds", SUPERBET),
        ("https://dados-ao-vivo/api/stats/72575500", SPORTRADAR),
    )
    codigo, saida = _rodar(tmp_path, har)

    assert codigo == 0
    assert saida.index("dados-ao-vivo") < saida.index("casa/api/odds")


def test_comando_ignora_resposta_que_nao_e_json(tmp_path):
    har = _har(("https://casa/api/odds", SUPERBET))
    har["log"]["entries"].append(
        {
            "request": {"url": "https://cdn/logo.png"},
            "response": {"content": {"mimeType": "image/png", "text": "binario"}},
        }
    )
    codigo, saida = _rodar(tmp_path, har)
    assert codigo == 0
    assert "logo.png" not in saida


def test_comando_explica_o_que_fazer_quando_nao_acha(tmp_path):
    codigo, saida = _rodar(tmp_path, _har(("https://casa/api/odds", {"data": [{"id": 1}]})))
    assert codigo == 1
    assert "Fetch/XHR" in saida


def test_comando_avisa_quando_o_arquivo_nao_existe(tmp_path):
    import argparse

    from betai.cli import cmd_estatisticas
    from betai.config import Settings

    args = argparse.Namespace(
        captura=str(tmp_path / "nao-existe.har"), termo=None, limite=8, listar=False
    )
    import contextlib
    import io

    saida = io.StringIO()
    with contextlib.redirect_stderr(saida):
        codigo = cmd_estatisticas(args, Settings())
    assert codigo == 1
    assert "não encontrado" in saida.getvalue()


# ---------- coleta das estatísticas pelo provedor ----------


def _provedor_superbet():
    from pathlib import Path

    from betai.providers import FieldMap, GenericJsonProvider

    spec = json.loads(
        (Path(__file__).resolve().parents[1] / "examples" / "superbet.json").read_text(
            encoding="utf-8"
        )
    )
    return GenericJsonProvider(FieldMap(spec), respect_robots=False)


EVENTO_CRU = {
    "eventId": 14068216,
    "tournamentId": 3,
    "matchName": "Gandzasar Kapan·Ararat Armenia",
    "metadata": {
        "homeTeamScore": "2",
        "awayTeamScore": "3",
        "homeTeamCorners": 1,
        "awayTeamCorners": 6,
        "homeTeamYellowCards": 0,
        "awayTeamYellowCards": 3,
        "minutes": "59",
        "periodStatus": "2H",
    },
    "odds": [
        {"marketName": "Resultado Final", "code": "1", "price": 2.10},
        {"marketName": "Resultado Final", "code": "0", "price": 3.40},
        {"marketName": "Resultado Final", "code": "2", "price": 3.60},
    ],
}


def test_provedor_coleta_escanteio_e_amarelo():
    p = _provedor_superbet()
    stats = p._parse_event(EVENTO_CRU).state.stats
    p.close()

    assert (stats.corners_home, stats.corners_away) == (1, 6)
    assert (stats.yellow_cards_home, stats.yellow_cards_away) == (0, 3)


def test_estatistica_ausente_nao_vira_zero_inventado():
    """A casa não publica xG. Zero significaria 'nenhum xG gerado', que é
    diferente de 'não sei' — e o modelo se comporta diferente nos dois casos."""
    p = _provedor_superbet()
    stats = p._parse_event(EVENTO_CRU).state.stats
    p.close()

    assert stats.xg_home is None
    assert stats.shots_on_target_home == 0


def test_estatistica_coletada_vai_para_o_banco(tmp_path):
    from betai.storage import Store

    p = _provedor_superbet()
    evento = p._parse_event(EVENTO_CRU)
    p.close()

    with Store(tmp_path / "t.db") as store:
        store.save_snapshot(evento)
        guardado = store.snapshots_for(evento.event_id)[0]

    assert guardado.state.stats.corners_away == 6
    assert guardado.state.stats.yellow_cards_away == 3


def test_mapa_sem_secao_stats_continua_funcionando():
    from betai.providers import FieldMap, GenericJsonProvider

    p = GenericJsonProvider(
        FieldMap(
            {
                "url": "https://exemplo/api",
                "fields": {"event_id": "id", "home_team": "a", "away_team": "b"},
                "markets": [
                    {
                        "key": "1x2",
                        "path": "odds",
                        "outcome_field": "code",
                        "outcome_map": {"1": "home", "0": "draw", "2": "away"},
                    }
                ],
            }
        ),
        respect_robots=False,
    )
    ev = p._parse_event(
        {
            "id": 1,
            "a": "Casa",
            "b": "Fora",
            "odds": [
                {"code": "1", "price": 2.0},
                {"code": "0", "price": 3.4},
                {"code": "2", "price": 3.6},
            ],
        }
    )
    p.close()
    assert ev.state.stats.corners_home == 0


def test_escanteio_nao_mexe_na_probabilidade():
    """Coletar não é usar. Ligar um coeficiente sem medição pioraria o que
    já funciona, então o modelo tem que ignorar isso por enquanto."""
    from betai.engine.live import live_matrix
    from betai.models import MatchState, MatchStats

    def prob(**stats):
        estado = MatchState(
            minute=60, period="2h", score_home=1, score_away=0, stats=MatchStats(**stats)
        )
        return live_matrix(1.57, 1.13, estado).match_odds()["home"]

    assert prob() == prob(corners_home=11, corners_away=0, yellow_cards_away=4)


# ---------- WebSocket ----------


def _har_ws(url: str, *mensagens) -> dict:
    return {
        "log": {
            "entries": [
                {
                    "request": {"url": url},
                    "response": {"content": {}},
                    "_webSocketMessages": list(mensagens),
                }
            ]
        }
    }


def test_le_estatistica_de_mensagem_de_websocket():
    from betai.discover import extract_websocket_from_har

    har = _har_ws(
        "wss://lmt.exemplo/stream",
        {"type": "receive", "data": json.dumps(SPORTRADAR)},
    )
    achados = extract_websocket_from_har(har)
    assert len(achados) == 1
    assert "Shots on target" in json.dumps(achados[0][1])


def test_ignora_mensagem_enviada_pelo_navegador():
    """As enviadas são inscrição em canal, não dado."""
    from betai.discover import extract_websocket_from_har

    har = _har_ws(
        "wss://lmt.exemplo/stream",
        {"type": "send", "data": json.dumps({"subscribe": "match.72575500"})},
        {"type": "receive", "data": json.dumps(SPORTRADAR)},
    )
    assert len(extract_websocket_from_har(har)) == 1


def test_desembrulha_json_dentro_do_protocolo():
    """socket.io e semelhantes prefixam a mensagem com um código."""
    from betai.discover import extract_websocket_from_har

    har = _har_ws(
        "wss://lmt.exemplo/stream",
        {"type": "receive", "data": '42["stats",' + json.dumps(SPORTRADAR) + "]"},
    )
    achados = extract_websocket_from_har(har)
    assert achados and "Ball possession" in json.dumps(achados[0][1])


def test_mensagem_que_nao_e_json_nao_derruba():
    from betai.discover import extract_websocket_from_har

    har = _har_ws(
        "wss://lmt.exemplo/stream",
        {"type": "receive", "data": "ping"},
        {"type": "receive", "data": "\x00\x01binario ilegivel aqui"},
    )
    assert extract_websocket_from_har(har) == []


def test_comando_encontra_estatistica_so_no_websocket(tmp_path):
    har = _har(("https://casa/api/odds", {"data": [{"eventId": 1}]}))
    har["log"]["entries"].append(
        {
            "request": {"url": "wss://lmt.exemplo/stream"},
            "response": {"content": {}},
            "_webSocketMessages": [{"type": "receive", "data": json.dumps(SPORTRADAR)}],
        }
    )
    codigo, saida = _rodar(tmp_path, har)

    assert codigo == 0
    assert "WebSocket" in saida
    assert "Shots on target" in saida


def test_har_sem_websocket_nao_menciona_websocket(tmp_path):
    codigo, saida = _rodar(tmp_path, _har(("https://casa/api/odds", SUPERBET)))
    assert codigo == 0
    assert "WebSocket" not in saida


# ---------- inventário da captura ----------


def test_listar_agrupa_por_dominio(tmp_path):
    har = _har(
        ("https://casa.example/v2/events/by-date", SUPERBET),
        ("https://casa.example/v2/events/13332095", SUPERBET),
        ("https://widgets.terceiro.example/config", {"ok": 1}),
    )
    codigo, saida = _rodar(tmp_path, har, listar=True)

    assert codigo == 0
    assert "casa.example" in saida and "widgets.terceiro.example" in saida
    assert "/v2/events/by-date" in saida
    # Domínio com mais endereços vem primeiro.
    assert saida.index("casa.example") < saida.index("widgets.terceiro.example")


def test_listar_nao_imprime_a_query_string(tmp_path):
    """É na query que viajam tokens de sessão."""
    har = _har(("https://casa.example/api?token=SEGREDO&startDate=x", SUPERBET))
    _, saida = _rodar(tmp_path, har, listar=True)
    assert "SEGREDO" not in saida
    assert "/api" in saida


def test_listar_conta_websocket(tmp_path):
    har = _har(("https://casa.example/api", SUPERBET))
    har["log"]["entries"].append(
        {
            "request": {"url": "wss://lmt.example/stream"},
            "response": {"content": {}},
            "_webSocketMessages": [{"type": "receive", "data": json.dumps(SPORTRADAR)}],
        }
    )
    _, saida = _rodar(tmp_path, har, listar=True)
    assert "1 conexão(ões) WebSocket" in saida


def test_listar_explica_a_ausencia_de_websocket(tmp_path):
    _, saida = _rodar(tmp_path, _har(("https://casa.example/api", SUPERBET)), listar=True)
    assert "0 conexão(ões) WebSocket" in saida
    assert "F5" in saida
