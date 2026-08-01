"""Medição: o modelo acertou?

Até aqui o programa só opinava. Este módulo confronta cada opinião com o
placar final e responde as três perguntas que decidem se vale continuar:

1. **Ele é honesto?** Quando anuncia 70%, acontece 70% das vezes? Um modelo
   otimista manda apostar demais e quebra mesmo acertando bastante.
2. **Teria dado lucro?** Somando as apostas que ele apontou.
3. **Onde ele erra?** Por minuto e por mercado — é o que diz o que melhorar.

Duas decisões metodológicas que mudam o resultado e por isso ficam explícitas:

* **Uma aposta por seleção, não uma por ciclo.** O mesmo `over 2.5` reaparece
  a cada 30 segundos enquanto a vantagem persistir. Contar todas infla o
  volume e distorce o retorno; na prática você apostaria uma vez. Fica valendo
  a primeira aparição, com a odd daquele momento.
* **A calibração conta cada análise.** Aqui a repetição não atrapalha da mesma
  forma — mas também não são observações independentes, e o texto do relatório
  avisa disso.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .models import MarketKey
from .storage import Store


def resolve(market: str, outcome: str, line: float | None, score: tuple[int, int]) -> bool | None:
    """A seleção ganhou? `None` quando anula (push) ou não sabemos resolver."""
    casa, fora = score
    total = casa + fora
    diff = casa - fora

    if market == MarketKey.MATCH_ODDS.value:
        return {"home": diff > 0, "draw": diff == 0, "away": diff < 0}.get(outcome)

    if market == MarketKey.DOUBLE_CHANCE.value:
        return {
            "home_draw": diff >= 0,
            "home_away": diff != 0,
            "draw_away": diff <= 0,
        }.get(outcome)

    if market == MarketKey.BTTS.value:
        marcaram = casa > 0 and fora > 0
        return {"yes": marcaram, "no": not marcaram}.get(outcome)

    if market == MarketKey.OVER_UNDER.value:
        if line is None:
            return None
        if total == line:  # linha inteira batida na mosca: devolve o dinheiro
            return None
        return total > line if outcome == "over" else total < line

    if market == MarketKey.ASIAN_HANDICAP.value:
        if line is None:
            return None
        ajustado = diff + line
        if abs(ajustado) < 1e-9:
            return None
        return ajustado > 0 if outcome == "home" else ajustado < 0

    if market == MarketKey.CORRECT_SCORE.value:
        return outcome == f"{casa}-{fora}"

    return None


@dataclass
class Aposta:
    """Uma aposta apontada pelo modelo, já confrontada com o resultado."""

    event_id: str
    minute: int
    market: str
    outcome: str
    line: float | None
    odds: float
    model_probability: float
    ganhou: bool | None

    @property
    def lucro(self) -> float:
        """Retorno de uma unidade apostada. Zero quando a aposta anula."""
        if self.ganhou is None:
            return 0.0
        return self.odds - 1.0 if self.ganhou else -1.0


def apostas_resolvidas(analises: list[dict]) -> list[Aposta]:
    """Uma aposta por seleção apontada, na primeira vez que apareceu.

    Contar de novo a cada ciclo mediria a persistência do sinal, não o retorno
    de apostar nele.
    """
    vistas: set[tuple[str, str, str, float | None]] = set()
    out: list[Aposta] = []

    for a in analises:
        for bet in a["value_bets"]:
            chave = (a["event_id"], bet["market"], bet["outcome"], bet.get("line"))
            if chave in vistas:
                continue
            vistas.add(chave)
            out.append(
                Aposta(
                    event_id=a["event_id"],
                    minute=a["minute"],
                    market=bet["market"],
                    outcome=bet["outcome"],
                    line=bet.get("line"),
                    odds=bet["odds"],
                    model_probability=bet["model_probability"],
                    ganhou=resolve(bet["market"], bet["outcome"], bet.get("line"), a["score"]),
                )
            )
    return out


# Faixas de probabilidade da tabela de calibração.
FAIXAS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
          (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]


@dataclass
class Faixa:
    """Uma linha da tabela de calibração."""

    de: float
    ate: float
    n: int = 0
    acertos: int = 0
    soma_prevista: float = 0.0

    @property
    def previsto(self) -> float:
        return self.soma_prevista / self.n if self.n else 0.0

    @property
    def observado(self) -> float:
        return self.acertos / self.n if self.n else 0.0

    @property
    def desvio(self) -> float:
        """Positivo: o modelo prometeu mais do que entregou."""
        return self.previsto - self.observado


def calibracao(analises: list[dict], minimo: int = 20) -> list[Faixa]:
    """Compara a probabilidade anunciada com a frequência observada.

    Percorre todas as probabilidades de cada análise — não só as que viraram
    aposta —, porque a pergunta é sobre o modelo inteiro, não sobre o filtro
    de valor. Faixas com poucos casos são omitidas: com 3 observações,
    qualquer número é ruído.
    """
    faixas = [Faixa(de, ate) for de, ate in FAIXAS]

    for a in analises:
        for chave, p in a["probabilities"].items():
            # O último ponto separa a seleção: a linha tem ponto dentro
            # (`over_under@2.5.over`), então partir pelo primeiro erra.
            mercado, _, seletor = chave.rpartition(".")
            mercado, _, linha_txt = mercado.partition("@")
            linha = float(linha_txt) if linha_txt else None
            ganhou = resolve(mercado, seletor, linha, a["score"])
            if ganhou is None:
                continue
            i = min(int(p * 10), len(faixas) - 1)
            faixas[i].n += 1
            faixas[i].acertos += int(ganhou)
            faixas[i].soma_prevista += p

    return [f for f in faixas if f.n >= minimo]


@dataclass
class Retorno:
    """Resultado financeiro de um conjunto de apostas."""

    n: int = 0
    ganhas: int = 0
    anuladas: int = 0
    lucro: float = 0.0

    @property
    def roi(self) -> float:
        """Lucro por unidade apostada. 0.05 é 5% de retorno sobre o giro."""
        apostado = self.n - self.anuladas
        return self.lucro / apostado if apostado else 0.0

    @property
    def taxa_de_acerto(self) -> float:
        decididas = self.n - self.anuladas
        return self.ganhas / decididas if decididas else 0.0


def _somar(apostas: list[Aposta]) -> Retorno:
    r = Retorno(n=len(apostas))
    for a in apostas:
        if a.ganhou is None:
            r.anuladas += 1
            continue
        r.ganhas += int(a.ganhou)
        r.lucro += a.lucro
    return r


# Fatias de minuto do relatório: começo, meio e reta final têm dinâmicas
# diferentes e o modelo não tem por que acertar igual nas três.
FATIAS_DE_MINUTO = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 94)]


@dataclass
class Desempenho:
    """O relatório inteiro."""

    jogos: int = 0
    analises: int = 0
    geral: Retorno = field(default_factory=Retorno)
    por_mercado: dict[str, Retorno] = field(default_factory=dict)
    por_minuto: dict[str, Retorno] = field(default_factory=dict)
    faixas: list[Faixa] = field(default_factory=list)
    banca_final: float = 1.0


def _banca_por_kelly(apostas: list[Aposta], teto: float = 0.05) -> float:
    """Como a banca teria evoluído apostando a fração sugerida, em ordem.

    Começa em 1.0. Aposta composta: a fração incide sobre a banca corrente,
    que é o que o critério de Kelly de fato manda fazer.
    """
    banca = 1.0
    for a in sorted(apostas, key=lambda x: (x.event_id, x.minute)):
        if a.ganhou is None:
            continue
        fracao = min(max((a.model_probability * a.odds - 1) / (a.odds - 1) * 0.25, 0.0), teto)
        banca += banca * fracao * (a.odds - 1.0 if a.ganhou else -1.0)
        if banca <= 0:
            return 0.0
    return banca


def desempenho(store: Store) -> Desempenho:
    """Monta o relatório a partir do que está gravado."""
    analises = store.settled_analyses()
    apostas = apostas_resolvidas(analises)

    rel = Desempenho(
        jogos=len({a["event_id"] for a in analises}),
        analises=len(analises),
        geral=_somar(apostas),
        faixas=calibracao(analises),
        banca_final=_banca_por_kelly(apostas),
    )

    por_mercado: dict[str, list[Aposta]] = defaultdict(list)
    por_minuto: dict[str, list[Aposta]] = defaultdict(list)
    for a in apostas:
        por_mercado[a.market].append(a)
        for de, ate in FATIAS_DE_MINUTO:
            if de <= a.minute <= ate:
                por_minuto[f"{de}-{ate}'"].append(a)
                break

    rel.por_mercado = {k: _somar(v) for k, v in sorted(por_mercado.items())}
    rel.por_minuto = {
        f"{de}-{ate}'": _somar(por_minuto[f"{de}-{ate}'"])
        for de, ate in FATIAS_DE_MINUTO
        if por_minuto.get(f"{de}-{ate}'")
    }
    return rel
