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
    args = argparse.Namespace(captura=str(caminho), termo=None, limite=8, **kwargs)
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

    args = argparse.Namespace(captura=str(tmp_path / "nao-existe.har"), termo=None, limite=8)
    import contextlib
    import io

    saida = io.StringIO()
    with contextlib.redirect_stderr(saida):
        codigo = cmd_estatisticas(args, Settings())
    assert codigo == 1
    assert "não encontrado" in saida.getvalue()
