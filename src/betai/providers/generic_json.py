"""Provedor genérico para endpoints JSON de casas de apostas.

Quase todo site de apostas moderno é um front-end que consome o próprio
endpoint JSON interno. Este provedor lê esse endpoint a partir de um mapa de
campos declarado em arquivo, sem precisar de código novo por casa.

Antes de apontar isto para um site:

* leia os Termos de Uso — a maioria das casas proíbe coleta automatizada, e
  isso é uma decisão sua, não do código;
* o `robots.txt` é verificado e respeitado por padrão (`respect_robots`);
* há rate limit obrigatório entre requisições;
* prefira uma API licenciada (veja `the_odds_api.py`) sempre que existir uma.
"""

from __future__ import annotations

import re
import urllib.robotparser
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx

from ..models import (
    Event,
    Market,
    MarketKey,
    MatchState,
    Selection,
)
from .base import Provider, ProviderError, RateLimiter


_PLACEHOLDER = re.compile(
    r"\{(now|today)"           # base temporal
    r"(?:([+-])(\d+)([smhd]))?"  # deslocamento opcional: -7d, +2h
    r"(?::([^}]+))?\}"          # formato opcional: :%Y-%m-%d
)

_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def render_params(params: dict[str, str], now: datetime | None = None) -> dict[str, str]:
    """Resolve marcadores de data nos parâmetros da requisição.

    Vários endpoints de casas exigem uma data na query e recusam a requisição
    sem ela. Como a data muda a cada chamada, o mapa declara o marcador e o
    valor é calculado na hora:

        {"startDate": "{now-7d}"}       → "2026-07-25 18:34:00"
        {"date": "{today}"}             → "2026-08-01"
        {"from": "{now:%Y-%m-%dT%H:%M}"} → "2026-08-01T18:34"
    """
    agora = now or datetime.now()

    def resolve(match: re.Match[str]) -> str:
        base, sinal, quantidade, unidade, formato = match.groups()
        momento = agora
        if quantidade:
            delta = timedelta(**{_UNITS[unidade]: int(quantidade)})
            momento = momento - delta if sinal == "-" else momento + delta
        if formato:
            return momento.strftime(formato)
        return momento.strftime("%Y-%m-%d" if base == "today" else "%Y-%m-%d %H:%M:%S")

    return {chave: _PLACEHOLDER.sub(resolve, str(valor)) for chave, valor in params.items()}


def dig(data: Any, path: str, default: Any = None) -> Any:
    """Navega um JSON aninhado com um caminho pontilhado.

    Suporta índices de lista: `"events.0.markets.name"`.
    """
    current = data
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return default
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return default
    return current if current is not None else default


class FieldMap:
    """Declara onde cada campo vive no JSON do provedor.

    Exemplo de mapa (JSON no disco):

    ```json
    {
      "url": "https://exemplo.com/api/inplay",
      "events_path": "data.events",
      "fields": {
        "event_id": "id",
        "league": "competition.name",
        "home_team": "teams.0.name",
        "away_team": "teams.1.name",
        "minute": "clock.minute",
        "score_home": "score.home",
        "score_away": "score.away"
      },
      "markets": [
        {
          "key": "1x2",
          "path": "markets.match_odds.runners",
          "outcome_field": "name",
          "odds_field": "price",
          "outcome_map": {"1": "home", "X": "draw", "2": "away"}
        }
      ]
    }
    ```
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        self.url: str = spec["url"]
        self.events_path: str = spec.get("events_path", "")
        self.fields: dict[str, str] = spec.get("fields", {})
        self.markets: list[dict[str, Any]] = spec.get("markets", [])
        self.headers: dict[str, str] = spec.get("headers", {})
        self.params: dict[str, str] = spec.get("params", {})
        # Parâmetros para buscar jogos que ainda não começaram. É deles que
        # sai o baseline pré-jogo, sem o qual não há análise independente.
        self.params_upcoming: dict[str, str] | None = spec.get("params_upcoming")
        self.url_upcoming: str = spec.get("url_upcoming", self.url)

        # Detalhe por evento. Várias casas devolvem só o mercado principal na
        # listagem e guardam os outros vinte num endpoint por jogo. `url_event`
        # é um molde com `{id}`; `markets_event` são os mercados a extrair de
        # lá. Sem isso, a análise fica restrita ao 1X2.
        self.url_event: str | None = spec.get("url_event")
        self.params_event: dict[str, str] = spec.get("params_event", {})
        self.event_path: str = spec.get("event_path", self.events_path)
        self.markets_event: list[dict[str, Any]] = spec.get("markets_event", [])


class GenericJsonProvider(Provider):
    """Lê odds de um endpoint JSON descrito por um `FieldMap`."""

    name = "generic_json"

    def __init__(
        self,
        field_map: FieldMap,
        *,
        min_interval: float = 2.0,
        respect_robots: bool = True,
        timeout: float = 15.0,
        user_agent: str = "bet-ai/0.1 (analise pessoal)",
    ) -> None:
        self.map = field_map
        self.limiter = RateLimiter(min_interval)
        self.respect_robots = respect_robots
        self.user_agent = user_agent
        self.client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent, **field_map.headers},
        )
        self._robots_ok: bool | None = None

    def close(self) -> None:
        self.client.close()

    def _check_robots(self) -> None:
        if not self.respect_robots or self._robots_ok is not None:
            if self._robots_ok is False:
                raise ProviderError(f"robots.txt proíbe a coleta de {self.map.url}")
            return

        parsed = urlparse(self.map.url)
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        try:
            parser.read()
        except Exception:
            # Sem robots.txt acessível, seguimos — mas com rate limit.
            self._robots_ok = True
            return

        self._robots_ok = parser.can_fetch(self.user_agent, self.map.url)
        if not self._robots_ok:
            raise ProviderError(f"robots.txt proíbe a coleta de {self.map.url}")

    def _parse_markets(self, raw_event: Any, spec: dict[str, Any]) -> list[Market]:
        """Extrai de um evento cru todos os mercados que casam com um spec.

        Normalmente é um só. Mas `group_by` produz um mercado por valor
        distinto do campo — é o que separa "Total de Gols" em over/under 1.5,
        2.5, 3.5, que a casa entrega misturados num array único. Sem essa
        separação o livro não fecha (soma ~4 em vez de 1) e o mercado inteiro
        seria descartado como incoerente.
        """
        runners = dig(raw_event, spec["path"], [])
        if not isinstance(runners, list) or not runners:
            return []

        # Muitas casas achatam todos os mercados num array só, distinguindo-os
        # por um campo interno (`marketName`, `marketId`). O filtro recorta o
        # pedaço que pertence a este mercado.
        filter_field = spec.get("filter_field")
        if filter_field is not None:
            alvos = spec.get("filter_values") or [spec.get("filter_value")]
            alvos = {str(v) for v in alvos if v is not None}
            runners = [r for r in runners if str(dig(r, filter_field)) in alvos]
            if not runners:
                return []

        group_by = spec.get("group_by")
        if group_by is None:
            market = self._build_market(runners, spec, spec.get("line"))
            return [market] if market else []

        grupos: dict[str, list[Any]] = {}
        for runner in runners:
            valor = dig(runner, group_by)
            if valor is None:
                continue
            grupos.setdefault(str(valor), []).append(runner)

        markets = []
        for valor, do_grupo in grupos.items():
            market = self._build_market(do_grupo, spec, valor)
            if market:
                markets.append(market)
        return markets

    def _build_market(
        self, runners: list[Any], spec: dict[str, Any], line: Any
    ) -> Market | None:
        outcome_map: dict[str, str] = spec.get("outcome_map", {})
        # Rótulo por padrão de texto, para quando o nome carrega a linha
        # dentro dele ("Mais de 2.5", "Menos de 3.5") e um mapa fixo não dá
        # conta. Primeiro padrão que casar vence.
        patterns = [(re.compile(p, re.I), o) for p, o in spec.get("outcome_match", [])]

        line_value = _as_float(line)
        selections: list[Selection] = []
        for runner in runners:
            raw_outcome = str(dig(runner, spec.get("outcome_field", "name"), ""))
            odds = _as_float(dig(runner, spec.get("odds_field", "price")))
            if odds is None or odds <= 1.0:
                continue

            outcome = outcome_map.get(raw_outcome)
            if outcome is None:
                outcome = next(
                    (o for padrao, o in patterns if padrao.search(raw_outcome)),
                    raw_outcome.lower(),
                )

            line_field = spec.get("line_field")
            sel_line = _as_float(dig(runner, line_field)) if line_field else line_value
            selections.append(Selection(outcome=outcome, odds=odds, line=sel_line))

        if not selections:
            return None
        return Market(key=MarketKey(spec["key"]), selections=selections, line=line_value)

    def _field(self, raw: Any, spec: Any, default: Any = None) -> Any:
        """Lê um campo do evento.

        O spec normalmente é um caminho pontilhado. Também aceita a forma
        `{"path": ..., "split": "·", "index": 0}`, para casas que só publicam
        os dois times num texto único ("Casa·Fora", "Casa - Fora").
        """
        if spec is None:
            return default
        if isinstance(spec, str):
            return dig(raw, spec, default)

        value = dig(raw, spec.get("path", ""), None)
        if value is None:
            return default

        separator = spec.get("split")
        if separator is not None:
            partes = [p.strip() for p in str(value).split(separator)]
            index = int(spec.get("index", 0))
            value = partes[index] if 0 <= index < len(partes) else None
            if value is None:
                return default

        # Tradução de código para nome legível. Casas costumam publicar só o
        # id da liga; a tabela deixa o usuário dar nome aos que lhe importam.
        lookup = spec.get("lookup")
        if lookup:
            return lookup.get(str(value), spec.get("lookup_default", value))
        return value

    def _parse_event(self, raw: Any) -> Event | None:
        f = self.map.fields
        event_id = self._field(raw, f.get("event_id", "id"))
        home = self._field(raw, f.get("home_team", "home"))
        away = self._field(raw, f.get("away_team", "away"))
        if not event_id or not home or not away:
            return None

        minute = _as_int(self._field(raw, f.get("minute", "minute")))
        score_home = _as_int(self._field(raw, f.get("score_home", "score_home")))
        score_away = _as_int(self._field(raw, f.get("score_away", "score_away")))
        state = MatchState(
            minute=minute,
            period=_period_for(
                minute,
                status=self._field(raw, f.get("status")),
                placar=score_home + score_away,
            ),
            score_home=score_home,
            score_away=score_away,
            red_cards_home=_as_int(self._field(raw, f.get("red_cards_home"))),
            red_cards_away=_as_int(self._field(raw, f.get("red_cards_away"))),
        )

        markets = [m for spec in self.map.markets for m in self._parse_markets(raw, spec)]
        if not markets:
            return None

        return Event(
            event_id=str(event_id),
            league=str(self._field(raw, f.get("league", "league"), "desconhecida")),
            home_team=str(home),
            away_team=str(away),
            starts_at=datetime.now(timezone.utc),
            source=self.name,
            state=state,
            markets=markets,
        )

    def _get_json(self, url: str, params: dict[str, str]) -> Any:
        """Busca e decodifica a resposta, com plano B para compressão.

        Alguns CDNs devolvem Brotli mesmo quando o cliente não anunciou
        suporte, e a descompressão falha com "incorrect header check". Nesse
        caso repetimos pedindo a resposta sem compressão.
        """
        params = render_params(params)
        try:
            resp = self.client.get(url, params=params)
        except httpx.DecodingError:
            self.limiter.wait()
            resp = self.client.get(url, params=params, headers={"Accept-Encoding": "identity"})
        if resp.status_code >= 400:
            # O corpo de um 4xx quase sempre diz o que falta na requisição.
            # Sem ele, o usuário fica adivinhando qual parâmetro está errado.
            detalhe = resp.text[:300].strip().replace("\n", " ")
            raise ProviderError(
                f"{url} respondeu {resp.status_code}{f' — {detalhe}' if detalhe else ''}"
            )
        return resp.json()

    def _fetch(self, url: str, params: dict[str, str]) -> list[Event]:
        self._check_robots()
        self.limiter.wait()
        try:
            payload = self._get_json(url, params)
        except httpx.HTTPError as exc:
            raise ProviderError(f"falha ao ler {url}: {exc}") from exc
        except ValueError as exc:
            raise ProviderError(f"resposta de {url} não é JSON válido") from exc

        raw_events = dig(payload, self.map.events_path, payload) if self.map.events_path else payload
        if not isinstance(raw_events, list):
            raise ProviderError(
                f"events_path '{self.map.events_path}' não aponta para uma lista"
            )

        return [e for raw in raw_events if (e := self._parse_event(raw))]

    def fetch_live(self) -> Iterable[Event]:
        return self._fetch(self.map.url, self.map.params)

    def fetch_upcoming(self) -> Iterable[Event]:
        """Jogos que ainda não começaram — a fonte do baseline pré-jogo.

        Exige `params_upcoming` no mapa. Sem isso devolve vazio, e o baseline
        terá que vir da captura no início do jogo (veja `Pipeline`).
        """
        if self.map.params_upcoming is None:
            return []
        eventos = self._fetch(self.map.url_upcoming, self.map.params_upcoming)
        # O endpoint pode devolver jogos em andamento junto; só interessam os
        # que ainda não começaram.
        return [e for e in eventos if not e.state.is_live]

    @property
    def supports_details(self) -> bool:
        return bool(self.map.url_event and self.map.markets_event)

    def fetch_details(self, event_id: str) -> list[Market]:
        """Mercados extras de um jogo, do endpoint por evento.

        A listagem de várias casas traz só o mercado principal. Os outros
        exigem uma requisição por jogo — cara, por isso quem chama decide
        para quais jogos vale a pena (veja `collect_live_analyses`).
        """
        if not self.supports_details:
            return []

        self._check_robots()
        self.limiter.wait()
        url = self.map.url_event.format(id=event_id)
        try:
            payload = self._get_json(url, self.map.params_event)
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"falha ao ler detalhes de {event_id}: {exc}") from exc

        raw = dig(payload, self.map.event_path, payload) if self.map.event_path else payload
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        if raw is None:
            return []

        return [m for spec in self.map.markets_event for m in self._parse_markets(raw, spec)]


def _as_float(value: Any) -> float | None:
    """Converte para float tolerando texto e ausência.

    Linhas e odds chegam como string em boa parte das casas ("2.5", "1.85").
    """
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    """Converte para inteiro tolerando texto e ausência.

    Várias casas publicam placar e minuto como string ("59", "2"), e alguns
    campos vêm nulos em jogos que ainda não começaram.
    """
    if value is None:
        return 0
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


# Rótulos de status que significam "a bola está rolando", em variantes que
# as casas usam. Comparados em maiúsculas, sem espaços.
# O status declarado pela casa, quando informa a fase, é mais confiável que o
# minuto — que várias casas deixam vazio. Comparado em maiúsculas, sem
# espaços nem sublinhados.
STATUS_PRIMEIRO_TEMPO = {"1H", "FIRSTHALF", "PRIMEIROTEMPO", "1T"}
STATUS_SEGUNDO_TEMPO = {"2H", "SECONDHALF", "SEGUNDOTEMPO", "2T"}
STATUS_INTERVALO = {"HT", "HALFTIME", "INTERVALO", "PAUSED"}
STATUS_ENCERRADO = {"ENDED", "FINISHED", "CLOSED", "FT", "ENCERRADO", "FINALIZADO"}
# Ao vivo mas sem dizer a fase: cai para o minuto.
STATUS_AO_VIVO = {"STARTED", "LIVE", "INPLAY", "EMANDAMENTO"}


def _period_for(minute: int, status: object = None, placar: int = 0) -> str:
    """Em que fase o jogo está.

    Não dá para depender só do minuto: várias casas deixam o campo vazio em
    parte dos jogos, e um jogo tratado como pré-jogo enquanto as odds já
    refletem o placar produz divergência inteiramente artificial — o modelo
    calcula sem os gols que as odds já embutem.

    Por isso três sinais, nesta ordem: o status declarado pela casa, o minuto,
    e o placar. Gol antes do apito inicial não existe.
    """
    if status is not None:
        texto = str(status).upper().replace(" ", "").replace("_", "")
        if texto in STATUS_ENCERRADO:
            return "encerrado"
        if texto in STATUS_INTERVALO:
            return "intervalo"
        if texto in STATUS_PRIMEIRO_TEMPO:
            return "1h"
        if texto in STATUS_SEGUNDO_TEMPO:
            return "2h"
        if texto in STATUS_AO_VIVO:
            return "1h" if minute <= 45 else "2h"

    if minute <= 0:
        # Sem minuto mas com gol: o jogo começou e o relógio não veio.
        return "1h" if placar > 0 else "pre_match"
    if minute <= 45:
        return "1h"
    if minute < 90:
        return "2h"
    return "encerrado"
