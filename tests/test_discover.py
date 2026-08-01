"""Testes da inferência de mapeamento.

Cada payload aqui imita um formato diferente de casa de apostas, porque o
ponto da ferramenta é justamente não depender de conhecer o formato de
antemão.
"""

import json

from betai.discover import (
    build_field_map,
    discover,
    extract_from_har,
    infer_fields,
    infer_markets,
    walk_arrays,
)
from betai.models import MarketKey
from betai.providers import FieldMap, GenericJsonProvider


# Estilo A: times numa lista de participantes, mercados aninhados por chave.
PAYLOAD_PARTICIPANTES = {
    "data": {
        "events": [
            {
                "id": 8812345,
                "tournament": {"name": "Brasileirão Série A"},
                "participants": [{"name": "Santos"}, {"name": "Vasco"}],
                "matchTime": 37,
                "scoreHome": 1,
                "scoreAway": 0,
                "odds": [
                    {"marketName": "Resultado Final", "code": "1", "price": 1.85},
                    {"marketName": "Resultado Final", "code": "X", "price": 3.60},
                    {"marketName": "Resultado Final", "code": "2", "price": 4.20},
                ],
            },
            {
                "id": 8812346,
                "tournament": {"name": "Brasileirão Série A"},
                "participants": [{"name": "Bahia"}, {"name": "Fortaleza"}],
                "matchTime": 12,
                "scoreHome": 0,
                "scoreAway": 0,
                "odds": [
                    {"marketName": "Resultado Final", "code": "1", "price": 2.30},
                    {"marketName": "Resultado Final", "code": "X", "price": 3.10},
                    {"marketName": "Resultado Final", "code": "2", "price": 3.40},
                ],
            },
        ]
    }
}

# Estilo B: campos nomeados, mercados como lista de listas.
PAYLOAD_NOMEADO = {
    "matches": [
        {
            "eventId": "ev-77",
            "league": "Premier League",
            "homeTeam": "Chelsea",
            "awayTeam": "Everton",
            "minute": 64,
            "score_home": 2,
            "score_away": 1,
            "markets": [
                {
                    "name": "Total de Gols 2.5",
                    "selections": [
                        {"label": "Mais de 2.5", "odds": 1.72, "line": 2.5},
                        {"label": "Menos de 2.5", "odds": 2.10, "line": 2.5},
                    ],
                }
            ],
        }
    ]
}


def test_encontra_a_lista_de_eventos_aninhada():
    fmap = build_field_map("https://x/api", PAYLOAD_PARTICIPANTES)
    assert fmap is not None
    assert fmap["events_path"] == "data.events"


def test_encontra_a_lista_de_eventos_na_raiz_do_objeto():
    fmap = build_field_map("https://x/api", PAYLOAD_NOMEADO)
    assert fmap["events_path"] == "matches"


def test_deduz_times_a_partir_de_lista_de_participantes():
    fields = infer_fields(PAYLOAD_PARTICIPANTES["data"]["events"][0])
    assert fields["home_team"] == "participants.0.name"
    assert fields["away_team"] == "participants.1.name"


def test_deduz_times_a_partir_de_campos_nomeados():
    fields = infer_fields(PAYLOAD_NOMEADO["matches"][0])
    assert fields["home_team"] == "homeTeam"
    assert fields["away_team"] == "awayTeam"


def test_deduz_placar_minuto_e_liga():
    fields = infer_fields(PAYLOAD_NOMEADO["matches"][0])
    assert fields["minute"] == "minute"
    assert fields["score_home"] == "score_home"
    assert fields["score_away"] == "score_away"
    assert fields["league"] == "league"


def test_desce_ate_o_escalar_quando_o_campo_e_um_objeto():
    """`tournament: {name: ...}` tem que virar `tournament.name`.

    Parar em `tournament` faria o provedor ler o dicionário inteiro e gravar
    "{'name': 'Brasileirão Série A'}" como nome da liga.
    """
    fields = infer_fields(PAYLOAD_PARTICIPANTES["data"]["events"][0])
    assert fields["league"] == "tournament.name"


def test_liga_inferida_le_um_texto_de_verdade():
    """Fecha o ciclo: o caminho gerado, aplicado ao payload, dá uma string."""
    fmap = build_field_map("https://x/api", PAYLOAD_PARTICIPANTES)
    fmap["markets"][0]["outcome_map"] = {"1": "home", "X": "draw", "2": "away"}
    provider = GenericJsonProvider(FieldMap(fmap), respect_robots=False)
    event = provider._parse_event(PAYLOAD_PARTICIPANTES["data"]["events"][0])
    assert event.league == "Brasileirão Série A"


def test_participantes_em_lista_contam_para_a_confianca():
    fmap = build_field_map("https://x/api", PAYLOAD_PARTICIPANTES)
    assert any("times em lista" in r for r in fmap["_como_foi_inferido"])
    assert fmap["_confianca"] >= 0.8


def test_prefere_o_caminho_mais_curto_para_o_id():
    """`id` do evento tem que ganhar de `markets.0.selections.0.id`."""
    sample = {
        "id": 1,
        "homeTeam": "A",
        "awayTeam": "B",
        "markets": [{"id": 99, "selections": [{"id": 7, "odds": 2.0}]}],
    }
    assert infer_fields(sample)["event_id"] == "id"


def test_identifica_o_campo_de_odds_e_o_rotulo():
    markets = infer_markets(PAYLOAD_PARTICIPANTES["data"]["events"][0])
    assert len(markets) == 1
    assert markets[0]["odds_field"] == "price"
    assert markets[0]["outcome_field"] == "code"
    assert markets[0]["path"] == "odds"


def test_identifica_a_linha_quando_existe():
    markets = infer_markets(PAYLOAD_NOMEADO["matches"][0])
    assert markets[0]["line_field"] == "line"


def test_chuta_over_under_pelos_rotulos():
    markets = infer_markets(PAYLOAD_NOMEADO["matches"][0])
    assert markets[0]["key"] == "over_under"


def test_chuta_1x2_por_tres_selecoes():
    markets = infer_markets(PAYLOAD_PARTICIPANTES["data"]["events"][0])
    assert markets[0]["key"] == "1x2"


def test_nao_confunde_id_com_odd():
    """IDs grandes não podem ser lidos como odds decimais."""
    payload = {"events": [{"id": 88123, "homeTeam": "A", "awayTeam": "B", "price": 99999}]}
    assert build_field_map("u", payload) is None


def test_lista_de_ligas_nao_vira_lista_de_eventos():
    """Sem odds abaixo, é catálogo, não jogo."""
    payload = {"competitions": [{"id": 1, "name": "Série A"}, {"id": 2, "name": "Série B"}]}
    assert build_field_map("u", payload) is None


def test_payload_vazio_nao_gera_mapa():
    assert build_field_map("u", {}) is None
    assert build_field_map("u", []) is None


def test_marca_o_rascunho_como_precisando_de_revisao():
    """A ferramenta não pode passar a impressão de que o mapa está pronto."""
    fmap = build_field_map("https://x/api", PAYLOAD_NOMEADO)
    assert "RASCUNHO" in fmap["_gerado_por"]
    assert fmap["_revisar"]
    assert all(m["_precisa_revisao"] for m in fmap["markets"])


def test_confianca_entre_zero_e_um():
    for payload in (PAYLOAD_PARTICIPANTES, PAYLOAD_NOMEADO):
        assert 0.0 < build_field_map("u", payload)["_confianca"] <= 1.0


# ---------- HAR ----------


def _har(*entradas):
    return {
        "log": {
            "entries": [
                {
                    "request": {"url": url},
                    "response": {
                        "content": {"mimeType": "application/json", "text": json.dumps(body)}
                    },
                }
                for url, body in entradas
            ]
        }
    }


def test_har_extrai_apenas_respostas_json():
    har = _har(("https://x/api/live", PAYLOAD_NOMEADO))
    har["log"]["entries"].append(
        {
            "request": {"url": "https://x/logo.png"},
            "response": {"content": {"mimeType": "image/png", "text": "binario"}},
        }
    )
    payloads = extract_from_har(har)
    assert len(payloads) == 1
    assert payloads[0][0] == "https://x/api/live"


def test_har_ignora_json_malformado():
    har = _har(("https://x/a", PAYLOAD_NOMEADO))
    har["log"]["entries"].append(
        {
            "request": {"url": "https://x/b"},
            "response": {"content": {"mimeType": "application/json", "text": "{quebrado"}},
        }
    )
    assert len(extract_from_har(har)) == 1


def test_discover_ordena_por_confianca():
    har = _har(
        ("https://x/telemetria", {"eventos": [{"tipo": "clique"}]}),
        ("https://x/api/live", PAYLOAD_PARTICIPANTES),
    )
    maps = discover(extract_from_har(har))
    assert len(maps) == 1  # telemetria descartada
    assert maps[0]["url"] == "https://x/api/live"


# ---------- integração: o mapa gerado tem que funcionar de verdade ----------


def test_mapa_inferido_parseia_o_payload_de_origem():
    """O teste que importa: rodar o mapa gerado contra o JSON que o gerou."""
    fmap = build_field_map("https://x/api", PAYLOAD_NOMEADO)
    fmap["markets"][0]["outcome_map"] = {"Mais de 2.5": "over", "Menos de 2.5": "under"}
    fmap["markets"][0]["line"] = 2.5

    provider = GenericJsonProvider(FieldMap(fmap), respect_robots=False)
    event = provider._parse_event(PAYLOAD_NOMEADO["matches"][0])

    assert event is not None
    assert event.label == "Chelsea x Everton"
    assert event.state.minute == 64
    assert (event.state.score_home, event.state.score_away) == (2, 1)
    sel = event.markets[0].selection("over")
    assert sel is not None and sel.odds == 1.72


def test_mapa_inferido_funciona_com_participantes_em_lista():
    fmap = build_field_map("https://x/api", PAYLOAD_PARTICIPANTES)
    fmap["markets"][0]["outcome_map"] = {"1": "home", "X": "draw", "2": "away"}

    provider = GenericJsonProvider(FieldMap(fmap), respect_robots=False)
    eventos = [
        provider._parse_event(e) for e in PAYLOAD_PARTICIPANTES["data"]["events"]
    ]

    assert [e.label for e in eventos] == ["Santos x Vasco", "Bahia x Fortaleza"]
    assert eventos[0].markets[0].selection("home").odds == 1.85
    assert eventos[0].state.minute == 37


# ---------- inspeção de mercados ----------

PAYLOAD_ACHATADO = {
    "data": [
        {
            "eventId": 8812345,
            "team1Name": "Santos",
            "team2Name": "Vasco",
            "tournamentName": "Brasileirão",
            "odds": [
                {"marketId": 1, "marketName": "Resultado Final", "code": "1", "price": 1.85},
                {"marketId": 1, "marketName": "Resultado Final", "code": "0", "price": 3.60},
                {"marketId": 1, "marketName": "Resultado Final", "code": "2", "price": 4.20},
                {"marketId": 8, "marketName": "Total de Gols", "code": "over",
                 "price": 1.72, "specialBetValue": "2.5"},
                {"marketId": 8, "marketName": "Total de Gols", "code": "under",
                 "price": 2.05, "specialBetValue": "2.5"},
            ],
        }
    ]
}


def test_desagrupa_mercados_achatados_num_unico_array():
    """Casas costumam achatar todos os mercados num array só."""
    from betai.discover import market_labels

    mercados = market_labels(PAYLOAD_ACHATADO["data"][0])
    nomes = {m.name for m in mercados}
    assert nomes == {"Resultado Final", "Total de Gols"}


def test_usa_o_nome_do_mercado_e_nao_o_id():
    """`marketId: 8` identifica mas não informa; `marketName` é o que orienta."""
    from betai.discover import market_labels

    mercados = market_labels(PAYLOAD_ACHATADO["data"][0])
    assert all(not m.name.isdigit() for m in mercados)


def test_lista_os_rotulos_de_cada_mercado():
    from betai.discover import market_labels

    por_nome = {m.name: m for m in market_labels(PAYLOAD_ACHATADO["data"][0])}
    assert por_nome["Resultado Final"].labels == ["0", "1", "2"]
    assert por_nome["Total de Gols"].labels == ["over", "under"]


def test_captura_a_linha_do_mercado():
    from betai.discover import market_labels

    por_nome = {m.name: m for m in market_labels(PAYLOAD_ACHATADO["data"][0])}
    assert por_nome["Total de Gols"].line == "2.5"
    assert por_nome["Resultado Final"].line is None


def test_summarize_corta_listas_longas_e_preserva_a_forma():
    from betai.discover import summarize

    out = summarize({"itens": list(range(50)), "texto": "x" * 300, "n": 7})
    assert out["itens"][:3] == [0, 1, 2]
    assert "+47 itens" in out["itens"][-1]
    assert out["texto"].endswith("...") and len(out["texto"]) < 130
    assert out["n"] == 7


def test_mercados_achatados_viram_entradas_separadas_com_filtro():
    """O caso Superbet: um array `odds` com todos os mercados dentro.

    Sem `filter_field`, o provedor leria 1X2 e over/under como se fossem o
    mesmo mercado e a remoção de margem sairia sem sentido.
    """
    markets = infer_markets(PAYLOAD_ACHATADO["data"][0])
    assert len(markets) == 2
    por_nome = {m["_nome_na_casa"]: m for m in markets}

    assert por_nome["Resultado Final"]["filter_field"] == "marketName"
    assert por_nome["Resultado Final"]["filter_value"] == "Resultado Final"
    assert por_nome["Resultado Final"]["key"] == "1x2"

    assert por_nome["Total de Gols"]["key"] == "over_under"
    assert por_nome["Total de Gols"]["line_field"] == "specialBetValue"


def test_provedor_recorta_o_mercado_pelo_filtro():
    """Fecha o ciclo: aplicar o mapa gerado tem que produzir dois mercados
    distintos, cada um com as suas próprias seleções."""
    fmap = build_field_map("https://x/api", PAYLOAD_ACHATADO)
    por_nome = {m["_nome_na_casa"]: m for m in fmap["markets"]}
    por_nome["Resultado Final"]["outcome_map"] = {"1": "home", "0": "draw", "2": "away"}
    por_nome["Total de Gols"]["outcome_map"] = {"over": "over", "under": "under"}
    por_nome["Total de Gols"]["line"] = 2.5

    provider = GenericJsonProvider(FieldMap(fmap), respect_robots=False)
    event = provider._parse_event(PAYLOAD_ACHATADO["data"][0])

    mo = event.market(MarketKey.MATCH_ODDS)
    assert [s.outcome for s in mo.selections] == ["home", "draw", "away"]
    assert mo.selection("home").odds == 1.85

    ou = event.market(MarketKey.OVER_UNDER)
    assert [s.outcome for s in ou.selections] == ["over", "under"]
    assert ou.selection("over").odds == 1.72


def test_filtro_que_nao_casa_descarta_o_mercado():
    from betai.providers.generic_json import FieldMap as FM

    spec = {
        "url": "u",
        "fields": {"event_id": "eventId", "home_team": "team1Name", "away_team": "team2Name"},
        "markets": [
            {
                "key": "1x2",
                "path": "odds",
                "outcome_field": "code",
                "odds_field": "price",
                "filter_field": "marketName",
                "filter_value": "Mercado Inexistente",
            }
        ],
    }
    provider = GenericJsonProvider(FM(spec), respect_robots=False)
    assert provider._parse_event(PAYLOAD_ACHATADO["data"][0]) is None
