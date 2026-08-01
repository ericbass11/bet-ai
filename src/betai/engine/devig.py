"""Remoção da margem da casa (vig/overround) das odds.

A odd oferecida embute a margem da casa, então `1/odd` sempre soma mais que 1.
Antes de comparar o modelo com o mercado é preciso extrair a probabilidade
"justa" implícita. Os quatro métodos abaixo distribuem a margem de formas
diferentes e discordam justamente nos azarões, que é onde a decisão importa.
"""

from __future__ import annotations

from typing import Literal, Sequence

Method = Literal["multiplicative", "additive", "power", "shin"]


def multiplicative(odds: Sequence[float]) -> list[float]:
    """Divide todas as implícitas pelo overround.

    Rápido e o padrão da indústria, mas assume que a casa aplica margem
    proporcional — o que superestima o azarão (favorite-longshot bias).
    """
    raw = [1.0 / o for o in odds]
    total = sum(raw)
    return [p / total for p in raw]


def additive(odds: Sequence[float]) -> list[float]:
    """Subtrai a margem em partes iguais de cada seleção.

    O oposto do multiplicativo: penaliza o azarão em termos relativos.
    """
    raw = [1.0 / o for o in odds]
    excess = (sum(raw) - 1.0) / len(raw)
    adjusted = [max(p - excess, 1e-9) for p in raw]
    total = sum(adjusted)
    return [p / total for p in adjusted]


def power(odds: Sequence[float], tol: float = 1e-10, max_iter: int = 200) -> list[float]:
    """Encontra k tal que sum(implícita_i ** k) == 1.

    Modela a margem como uma distorção exponencial. Fica entre o
    multiplicativo e o aditivo e costuma calibrar melhor mercados 1X2.
    """
    raw = [1.0 / o for o in odds]
    if abs(sum(raw) - 1.0) < tol:
        return list(raw)

    lo, hi = 0.001, 10.0
    for _ in range(max_iter):
        k = (lo + hi) / 2
        total = sum(p**k for p in raw)
        if abs(total - 1.0) < tol:
            break
        # sum(p^k) é decrescente em k porque todo p < 1
        if total > 1.0:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    adjusted = [p**k for p in raw]
    total = sum(adjusted)
    return [p / total for p in adjusted]


def shin(odds: Sequence[float], tol: float = 1e-10, max_iter: int = 200) -> list[float]:
    """Modelo de Shin (1993): a margem vem de apostadores informados.

    Estima a proporção `z` de dinheiro insider e remove esse componente.
    É o método com melhor embasamento teórico quando a margem é alta.
    """
    raw = [1.0 / o for o in odds]
    total = sum(raw)
    if abs(total - 1.0) < tol:
        return list(raw)

    def probs_for(z: float) -> list[float]:
        out = []
        for p in raw:
            disc = z**2 + 4 * (1 - z) * (p**2) / total
            out.append((max(disc, 0.0) ** 0.5 - z) / (2 * (1 - z)))
        return out

    lo, hi = 0.0, 0.9
    for _ in range(max_iter):
        z = (lo + hi) / 2
        s = sum(probs_for(z))
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = z
        else:
            hi = z
    result = probs_for((lo + hi) / 2)
    s = sum(result)
    return [p / s for p in result]


_METHODS = {
    "multiplicative": multiplicative,
    "additive": additive,
    "power": power,
    "shin": shin,
}


def remove_vig(odds: Sequence[float], method: Method = "power") -> list[float]:
    """Converte odds decimais em probabilidades justas normalizadas."""
    if not odds:
        return []
    if any(o <= 1.0 for o in odds):
        raise ValueError("odds decimais precisam ser maiores que 1.0")
    return _METHODS[method](odds)


def margin(odds: Sequence[float]) -> float:
    """Margem da casa em fração (0.06 == 6%)."""
    return sum(1.0 / o for o in odds) - 1.0


# Faixa plausível de overround. Uma casa opera entre 1% e 20% de margem;
# valores fora disso significam mercado suspenso, incompleto ou já resolvido.
MIN_OVERROUND = 0.99
MAX_OVERROUND = 1.35


def is_plausible(odds: Sequence[float]) -> bool:
    """O conjunto de odds forma um livro coerente?

    Esta checagem existe porque `remove_vig` normaliza qualquer entrada para
    somar 1 — inclusive um mercado com metade das seleções faltando. Um livro
    somando 0.40 vira três probabilidades inventadas, e comparar as odds
    originais contra elas produz "vantagens" de várias centenas por cento que
    são puro artefato.

    Overround abaixo de 1 seria a casa pagando para receber aposta; muito
    acima significa que o que chegou não é um mercado completo.
    """
    if len(odds) < 2 or any(o <= 1.0 for o in odds):
        return False
    total = sum(1.0 / o for o in odds)
    return MIN_OVERROUND <= total <= MAX_OVERROUND
