"""Descoberta automática do mapeamento de campos de um provedor.

Escrever um `field_map.json` na mão exige abrir o JSON da casa de apostas e
rastrear onde cada campo mora — trabalhoso e frágil. Este módulo faz isso por
inferência: recebe uma resposta capturada (ou um HAR inteiro do DevTools),
encontra a lista de eventos, e deduz quais chaves são time, placar, minuto e
odds.

O resultado é um rascunho de mapa com um nível de confiança por campo. Ele não
é infalível — é um ponto de partida que economiza a parte chata, e que você
confere contra o payload real.

Fluxo pretendido:

1. No navegador, abra a página de jogos ao vivo da casa.
2. DevTools → Network → filtre por Fetch/XHR → clique com o botão direito →
   "Save all as HAR with content".
3. `bet-ai discover captura.har -o examples/minha_casa.json`
4. Confira o mapa gerado e ajuste o que a inferência errou.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Padrões de nome de chave, em português, inglês e romeno (várias casas
# brasileiras rodam plataformas de origem europeia).
PATTERNS: dict[str, list[str]] = {
    "event_id": [r"^id$", r"event_?id", r"match_?id", r"fixture_?id", r"^uuid$"],
    "league": [r"league", r"competition", r"tournament", r"championship", r"campeonato", r"categor"],
    "home_team": [r"home_?team", r"^home$", r"team_?1", r"^localteam", r"mandante", r"gazde"],
    "away_team": [r"away_?team", r"^away$", r"team_?2", r"^visitorteam", r"visitante", r"oaspeti"],
    "minute": [r"^minute", r"^clock", r"elapsed", r"match_?time", r"^minuto", r"^tempo"],
    # `home.*score` cobre tanto `home_score` quanto `homeTeamScore`.
    "score_home": [r"score_?home", r"home.*score", r"goals_?home", r"placar_?casa"],
    "score_away": [r"score_?away", r"away.*score", r"goals_?away", r"placar_?fora"],
    "red_cards_home": [r"red_?cards?_?home", r"home.*red"],
    "red_cards_away": [r"red_?cards?_?away", r"away.*red"],
}

ODDS_KEYS = [r"^odds?$", r"^price$", r"coefficient", r"^decimal", r"^rate$", r"^cota$", r"^valor$"]
LABEL_KEYS = [r"^name$", r"^label$", r"^title$", r"^outcome", r"^selection", r"^type$", r"^code$", r"^sign$"]
LINE_KEYS = [r"^line$", r"^handicap", r"^point$", r"^total$", r"^spread", r"^param"]

# Faixa plausível de uma odd decimal. Serve para não confundir odds com IDs,
# minutos ou percentuais.
ODDS_MIN, ODDS_MAX = 1.01, 1000.0


def _matches(key: str, patterns: list[str]) -> bool:
    lowered = key.lower()
    return any(re.search(p, lowered) for p in patterns)


def _plausible_odds(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and ODDS_MIN <= value <= ODDS_MAX


@dataclass
class Candidate:
    """Uma lista dentro do JSON que pode ser a lista de eventos."""

    path: str
    items: list[dict]
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def walk_arrays(data: Any, path: str = "", depth: int = 0, max_depth: int = 6) -> list[Candidate]:
    """Encontra todas as listas de dicionários no JSON, com seus caminhos."""
    found: list[Candidate] = []
    if depth > max_depth:
        return found

    if isinstance(data, list):
        dicts = [x for x in data if isinstance(x, dict)]
        if dicts:
            found.append(Candidate(path=path, items=dicts))
        for i, item in enumerate(data[:3]):  # amostra: listas são homogêneas
            found.extend(walk_arrays(item, f"{path}.{i}" if path else str(i), depth + 1, max_depth))
    elif isinstance(data, dict):
        for key, value in data.items():
            found.extend(walk_arrays(value, f"{path}.{key}" if path else key, depth + 1, max_depth))

    return found


def score_candidate(cand: Candidate) -> Candidate:
    """Pontua o quanto uma lista parece ser a lista de eventos.

    O sinal mais forte é a presença simultânea de identificação de times e de
    algo que pareça odds em algum lugar abaixo — uma lista de ligas tem nome,
    mas não tem odds; uma lista de seleções tem odds, mas não tem dois times.
    """
    sample = cand.items[0]
    keys = _all_keys(sample)

    for field_name in ("event_id", "home_team", "away_team", "league"):
        if any(_matches(k, PATTERNS[field_name]) for k in keys):
            cand.score += 2.0 if "team" in field_name else 1.0
            cand.reasons.append(f"tem {field_name}")

    # Times numa lista de participantes não casam com os padrões acima, mas
    # identificam um evento tão bem quanto campos nomeados.
    if not any("team" in r for r in cand.reasons):
        for key in keys:
            if key.rsplit(".", 1)[-1].lower() in {"participants", "competitors", "teams", "times"}:
                value = _dig(sample, key)
                if isinstance(value, list) and len(value) >= 2:
                    cand.score += 4.0
                    cand.reasons.append("tem os dois times em lista")
                    break

    if _count_odds(sample) >= 2:
        cand.score += 3.0
        cand.reasons.append("contém odds")

    if len(cand.items) > 1:
        cand.score += 0.5

    # Uma lista de participantes tem exatamente 2 itens e nada mais — não é a
    # lista de eventos.
    if len(cand.items) == 2 and len(keys) <= 3:
        cand.score -= 2.0

    return cand


def _all_keys(data: Any, prefix: str = "", depth: int = 0, max_depth: int = 4) -> list[str]:
    """Todas as chaves alcançáveis, em caminho pontilhado."""
    out: list[str] = []
    if depth > max_depth:
        return out
    if isinstance(data, dict):
        for key, value in data.items():
            full = f"{prefix}.{key}" if prefix else key
            out.append(full)
            out.extend(_all_keys(value, full, depth + 1, max_depth))
    elif isinstance(data, list) and data:
        out.extend(_all_keys(data[0], f"{prefix}.0", depth + 1, max_depth))
    return out


def _count_odds(data: Any, depth: int = 0) -> int:
    if depth > 5:
        return 0
    total = 0
    if isinstance(data, dict):
        for key, value in data.items():
            if _matches(key, ODDS_KEYS) and _plausible_odds(value):
                total += 1
            else:
                total += _count_odds(value, depth + 1)
    elif isinstance(data, list):
        for item in data[:6]:
            total += _count_odds(item, depth + 1)
    return total


def infer_fields(sample: dict) -> dict[str, str]:
    """Deduz o caminho de cada campo do evento dentro de um item de exemplo."""
    keys = _all_keys(sample)
    fields: dict[str, str] = {}

    for name, patterns in PATTERNS.items():
        matches = [k for k in keys if _matches(k.rsplit(".", 1)[-1], patterns)]
        # Nome de time tem que ser texto não-numérico. `homeTeamId: "28148"` e
        # `homeTeamScore: "2"` casam o padrão `home_?team` mas produziriam
        # "28148 x 172622" ou "2 x 3" na tela.
        if "team" in name:
            matches = [k for k in matches if _looks_like_a_name(_dig(sample, k))]
        if matches:
            # O caminho mais curto é quase sempre o certo: `id` ganha de
            # `markets.0.id`.
            path = min(matches, key=lambda k: (k.count("."), len(k)))
            resolved = _descend_to_leaf(sample, path)
            if resolved:
                fields[name] = resolved

    # Alguns provedores só publicam os dois times num texto só — a Superbet
    # usa `matchName: "Casa·Fora"`. Sem isto, o nome dos times some.
    if "home_team" not in fields or "away_team" not in fields:
        split = _find_pair_field(sample, keys)
        if split:
            fields["home_team"] = {**split, "index": 0}
            fields["away_team"] = {**split, "index": 1}

    # Times em lista (`participants: [{name}, {name}]`) é o padrão mais comum
    # depois de campos nomeados, e não casa com os padrões acima.
    if "home_team" not in fields or "away_team" not in fields:
        for key in keys:
            leaf = key.rsplit(".", 1)[-1].lower()
            if leaf in {"participants", "competitors", "teams", "times"}:
                value = _dig(sample, key)
                if isinstance(value, list) and len(value) >= 2 and isinstance(value[0], dict):
                    name_key = next(
                        (k for k in value[0] if _matches(k, [r"name", r"nome", r"title"])), None
                    )
                    if name_key:
                        fields["home_team"] = f"{key}.0.{name_key}"
                        fields["away_team"] = f"{key}.1.{name_key}"
                        break

    return fields


NAME_LEAF = [r"^name$", r"^nome$", r"^title$", r"^label$", r"^shortname", r"^caption$"]


def _descend_to_leaf(sample: dict, path: str) -> str | None:
    """Garante que o caminho aponta para um valor escalar, não para um objeto.

    `{"tournament": {"name": "Série A"}}` casa o padrão de liga na chave
    `tournament`, mas ler ali devolve o dicionário inteiro. Aqui descemos até
    o campo de nome de dentro.
    """
    value = _dig(sample, path)
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return path
    if isinstance(value, dict):
        leaf = next((k for k in value if _matches(k, NAME_LEAF) and isinstance(value[k], str)), None)
        if leaf:
            return f"{path}.{leaf}"
        # Sem campo de nome óbvio, o primeiro escalar de texto serve.
        leaf = next((k for k, v in value.items() if isinstance(v, str)), None)
        return f"{path}.{leaf}" if leaf else None
    return None


def _looks_like_a_name(value: Any) -> bool:
    """Texto que pode ser nome de time — descarta IDs e placares numéricos."""
    return isinstance(value, str) and bool(value.strip()) and not value.strip().isdigit()


# Separadores usados por casas que juntam os dois times num campo só.
PAIR_SEPARATORS = ["·", " - ", " vs ", " v ", " x ", " @ ", "–", "—"]


def _find_pair_field(sample: dict, keys: list[str]) -> dict[str, str] | None:
    """Acha um campo do tipo "Casa·Fora" e devolve como separá-lo.

    Só aceita divisão em exatamente duas partes não vazias — "Sporting - CP -
    Braga" dividiria em três e seria pior que não dividir.
    """
    candidatos = [
        k
        for k in keys
        if re.search(r"match|event|fixture|game|name|nome", k.rsplit(".", 1)[-1].lower())
    ]
    for key in sorted(candidatos, key=lambda k: (k.count("."), len(k))):
        value = _dig(sample, key)
        if not isinstance(value, str) or len(value) < 3:
            continue
        for sep in PAIR_SEPARATORS:
            partes = [p.strip() for p in value.split(sep)]
            if len(partes) == 2 and all(partes):
                return {"path": key, "split": sep}
    return None


def _dig(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def infer_markets(sample: dict) -> list[dict[str, Any]]:
    """Encontra as seleções apostáveis e como ler cada mercado.

    Quando a casa achata vários mercados num array só — o padrão mais comum —
    cada mercado vira uma entrada com `filter_field`/`filter_value`, para o
    provedor recortar a fatia certa.
    """
    markets: list[dict[str, Any]] = []

    for grupo in market_labels(sample):
        markets.append(
            {
                "key": _guess_market_key(grupo.labels, len(grupo.labels)),
                "path": grupo.path,
                "outcome_field": grupo.label_field,
                "odds_field": grupo.odds_field,
                **(
                    {"filter_field": grupo.name_field, "filter_value": grupo.name}
                    if grupo.name_field
                    else {}
                ),
                **({"line_field": grupo.line_field} if grupo.line_field else {}),
                "_nome_na_casa": grupo.name,
                "_rotulos_encontrados": grupo.labels[:8],
                **({"_linhas_encontradas": grupo.line} if grupo.line else {}),
                "_precisa_revisao": True,
            }
        )

    return markets


def _guess_market_key(rotulos: list[str], n: int) -> str:
    """Chuta o tipo de mercado pelos rótulos das seleções."""
    texto = " ".join(rotulos).lower()
    if any(t in texto for t in ("over", "under", "mais de", "menos de", "peste", "sub")):
        return "over_under"
    if any(t in texto for t in ("sim", "yes", "não", "nao", "no", "gg", "ng")):
        return "btts"
    if any(t in texto for t in ("handicap", "spread")):
        return "asian_handicap"
    if n == 3 or any(t in texto for t in ("1", "x", "2", "draw", "empate")):
        return "1x2"
    return "1x2"


def extract_from_har(har: dict) -> list[tuple[str, Any]]:
    """Extrai (url, payload) de cada resposta JSON de um arquivo HAR."""
    out: list[tuple[str, Any]] = []
    for entry in har.get("log", {}).get("entries", []):
        response = entry.get("response", {})
        content = response.get("content", {})
        mime = content.get("mimeType", "")
        text = content.get("text")
        if not text or "json" not in mime.lower():
            continue
        try:
            out.append((entry.get("request", {}).get("url", ""), json.loads(text)))
        except (ValueError, TypeError):
            continue
    return out


def build_field_map(url: str, payload: Any) -> dict[str, Any] | None:
    """Monta um rascunho de `field_map.json` a partir de um payload."""
    candidates = [score_candidate(c) for c in walk_arrays(payload)]
    candidates = [c for c in candidates if c.score > 0]
    if not candidates:
        return None

    best = max(candidates, key=lambda c: c.score)
    sample = best.items[0]
    fields = infer_fields(sample)
    markets = infer_markets(sample)

    if not markets or "home_team" not in fields:
        return None

    return {
        "_gerado_por": "bet-ai discover — RASCUNHO, confira antes de usar",
        "_confianca": round(min(best.score / 8.0, 1.0), 2),
        "_como_foi_inferido": best.reasons,
        "_revisar": [
            "outcome_map: traduza os rótulos da casa para home/draw/away/over/under/yes/no",
            "key de cada mercado: a inferência chuta pelo texto dos rótulos",
            "line: over_under e handicap precisam da linha (2.5, -0.5...)",
        ],
        "url": url,
        "events_path": best.path,
        "fields": fields,
        "markets": markets,
    }


def summarize(value: Any, depth: int = 0, max_depth: int = 6, max_items: int = 3) -> Any:
    """Reduz um objeto a uma amostra legível da sua estrutura.

    Listas longas viram os primeiros itens mais uma contagem, textos longos são
    cortados. O objetivo é caber na tela e mostrar o formato — não os dados.
    """
    if depth > max_depth:
        return "..."
    if isinstance(value, dict):
        return {k: summarize(v, depth + 1, max_depth, max_items) for k, v in value.items()}
    if isinstance(value, list):
        amostra = [summarize(v, depth + 1, max_depth, max_items) for v in value[:max_items]]
        if len(value) > max_items:
            amostra.append(f"... (+{len(value) - max_items} itens)")
        return amostra
    if isinstance(value, str) and len(value) > 120:
        return value[:120] + "..."
    return value


def _melhor_rotulo(selecoes: list[dict], candidatos: list[str]) -> str | None:
    """Escolhe, entre os campos possíveis, o que melhor distingue as seleções.

    Vence quem tiver mais valores distintos e não-nulos: um campo preenchido
    em todas as seleções, com um valor diferente para cada, é exatamente o que
    serve de rótulo. Empate desfeito pela ordem em que os campos aparecem.
    """
    melhor, melhor_n = None, 0
    for chave in candidatos:
        valores = {
            str(_dig(s, chave)) for s in selecoes if _dig(s, chave) not in (None, "")
        }
        if len(valores) > melhor_n:
            melhor, melhor_n = chave, len(valores)
    return melhor if melhor_n >= 2 or (melhor and len(selecoes) == 1) else melhor


def _merge_keys(items: list[dict], limit: int = 40) -> dict:
    """Junta os itens numa amostra com todos os campos que aparecem.

    O primeiro valor não-nulo de cada campo vence, para que a checagem de tipo
    ("isto é texto?", "isto é uma odd plausível?") tenha o que examinar.
    """
    merged: dict = {}
    for item in items[:limit]:
        for key, value in item.items():
            if value is not None and (key not in merged or merged[key] is None):
                merged[key] = value
    return merged


@dataclass
class MarketSample:
    """Um mercado observado na captura, com os rótulos que a casa usa."""

    path: str
    name: str
    label_field: str
    labels: list[str]
    odds_field: str
    name_field: str | None = None
    line_field: str | None = None
    line: str | None = None


def market_labels(sample: dict) -> list[MarketSample]:
    """Agrupa as seleções por mercado e lista os rótulos de cada um.

    É o que falta para escrever o `outcome_map`: saber que a casa chama o
    empate de "X", de "Draw" ou de "0" — e que "Total goluri" é over/under.

    Casas que achatam todos os mercados num único array (o padrão mais comum)
    são desagrupadas aqui pelo campo de nome do mercado.
    """
    encontrados: list[MarketSample] = []

    for cand in walk_arrays(sample):
        if not cand.path:
            continue
        # Num array achatado as seleções não têm todas os mesmos campos: a do
        # 1X2 não traz linha, a do over/under traz. Olhar só o primeiro item
        # perderia o campo de linha inteiro.
        item = _merge_keys(cand.items)
        odds_key = next(
            (k for k in item if _matches(k, ODDS_KEYS) and _plausible_odds(item[k])), None
        )
        if not odds_key:
            continue

        # O nome do mercado tem que ser texto: `marketId: 8` identifica, mas
        # não diz nada; `marketName: "Total de Gols"` é o que orienta o mapa.
        # Preferência: `marketName` > qualquer campo de mercado com texto >
        # o que sobrar. Um `marketUuid` também é texto, mas agrupar por UUID
        # não diz nada a quem vai escrever o mapa.
        candidatos_nome = [k for k in item if re.search(r"market|mercado|bet_?type", k.lower())]
        nome_key = next(
            (k for k in candidatos_nome if re.search(r"name|nome|label", k.lower())),
            next(
                (
                    k
                    for k in candidatos_nome
                    if isinstance(item[k], str) and not re.search(r"uuid|guid|_?id$|tags", k.lower())
                ),
                next(iter(candidatos_nome), None),
            ),
        )
        rotulos_possiveis = [
            k for k in item if _matches(k, LABEL_KEYS) and isinstance(item[k], str)
        ]
        if not rotulos_possiveis:
            continue
        line_key = next((k for k in item if _matches(k, LINE_KEYS + [r"special"])), None)

        grupos: dict[str, list[dict]] = {}
        for sel in cand.items:
            nome = str(_dig(sel, nome_key)) if nome_key else "(mercado único)"
            grupos.setdefault(nome, []).append(sel)

        for nome, selecoes in grupos.items():
            # O campo de rótulo é escolhido por mercado, não uma vez para o
            # array inteiro: na Superbet o `code` identifica bem o 1X2 mas vem
            # nulo em "Total de Gols", onde quem distingue é o `name`.
            label_key = _melhor_rotulo(selecoes, rotulos_possiveis)
            if label_key is None:
                continue
            linhas = (
                sorted({str(_dig(s, line_key)) for s in selecoes if _dig(s, line_key) is not None})
                if line_key
                else []
            )
            encontrados.append(
                MarketSample(
                    path=cand.path,
                    name=nome,
                    label_field=label_key,
                    labels=sorted({str(_dig(s, label_key)) for s in selecoes})[:12],
                    odds_field=odds_key,
                    name_field=nome_key,
                    line_field=line_key,
                    line=", ".join(linhas[:6]) if linhas else None,
                )
            )

    return encontrados


def discover(payloads: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    """Roda a inferência sobre vários payloads, do mais promissor ao menos."""
    maps = [m for url, payload in payloads if (m := build_field_map(url, payload))]
    maps.sort(key=lambda m: m["_confianca"], reverse=True)
    return maps


# Termos que denunciam um painel de estatísticas ao vivo. Comparados contra os
# nomes dos campos, sem acento e em minúsculas. A lista mistura inglês e
# português porque as casas usam os dois, às vezes no mesmo payload.
TERMOS_DE_ESTATISTICA = [
    "shot",
    "chute",
    "finaliza",
    "ontarget",
    "nogol",
    "possession",
    "posse",
    "attack",
    "ataque",
    "dangerous",
    "perigoso",
    "corner",
    "escanteio",
    "xg",
    "expectedgoal",
    "statistic",
    "estatistica",
    "pressure",
    "pressao",
]


def _normalizar(texto: str) -> str:
    import unicodedata

    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.lower().replace("_", "").replace("-", "").replace(" ", "")


def _rotulo_de_estatistica(objeto: Any, termos: list[str]) -> str | None:
    """O objeto se identifica como uma estatística por um dos seus valores?

    É o formato dos provedores de dados ao vivo (Sportradar e semelhantes):
    o nome da métrica viaja como texto num campo genérico — `{"name": "Shots
    on target", "value": {"home": 6, "away": 2}}` — e procurar só por nome de
    campo passa direto por ele.
    """
    if not isinstance(objeto, dict):
        return None
    for chave, valor in objeto.items():
        if not isinstance(valor, str) or len(valor) > 60:
            continue
        if _normalizar(chave) in ("name", "nome", "label", "titulo", "type", "tipo"):
            if any(t in _normalizar(valor) for t in termos):
                return valor
    return None


def buscar_campos(
    data: Any, termos: list[str], caminho: str = "", profundidade: int = 0
) -> list[tuple[str, Any]]:
    """Todos os campos cujo nome contém algum dos termos, com o caminho até eles.

    Devolve pares (caminho, valor). O caminho é pontilhado e usa índices para
    listas, no mesmo formato que o `field_map` aceita.
    """
    if profundidade > 8:
        return []

    achados: list[tuple[str, Any]] = []
    if isinstance(data, dict):
        for chave, valor in data.items():
            aqui = f"{caminho}.{chave}" if caminho else str(chave)
            if isinstance(valor, (dict, list)):
                # O nome da estatística às vezes é um valor, não um campo:
                # {"name": "Shots on target", "value": {...}}. Nesse formato o
                # que interessa é o objeto inteiro, não a string solta.
                rotulo = _rotulo_de_estatistica(valor, termos)
                if rotulo:
                    achados.append((f"{aqui} [{rotulo}]", summarize(valor, max_depth=2)))
                else:
                    achados.extend(buscar_campos(valor, termos, aqui, profundidade + 1))
                continue
            if any(t in _normalizar(str(chave)) for t in termos):
                achados.append((aqui, valor))
    elif isinstance(data, list):
        # Só os primeiros itens: numa lista de 195 jogos, o formato do
        # primeiro já diz tudo e imprimir os 195 não ajudaria ninguém.
        for i, item in enumerate(data[:2]):
            achados.extend(
                buscar_campos(item, termos, f"{caminho}.{i}" if caminho else str(i), profundidade + 1)
            )
    return achados
