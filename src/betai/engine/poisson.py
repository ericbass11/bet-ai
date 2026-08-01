"""Modelo de gols: Poisson bivariado com correção de Dixon-Coles.

O núcleo quantitativo. A partir de duas taxas esperadas de gols (lambda do
mandante e do visitante) monta-se a matriz de placares, e todo mercado
(1X2, over/under, ambas marcam, handicap asiático, placar exato) sai de
somar as células certas dessa matriz.

Dixon-Coles corrige o defeito conhecido do Poisson independente: placares
baixos (0-0, 1-0, 0-1, 1-1) são mais correlacionados do que a independência
prevê.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

MAX_GOALS = 12
DEFAULT_RHO = -0.05


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam**k / math.factorial(k)


def _dixon_coles_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Fator de correção aplicado apenas aos quatro placares baixos."""
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1.0 + lam_h * rho
    if x == 1 and y == 0:
        return 1.0 + lam_a * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


@dataclass
class ScoreMatrix:
    """Distribuição conjunta de gols (mandante, visitante)."""

    lambda_home: float
    lambda_away: float
    rho: float = DEFAULT_RHO
    max_goals: int = MAX_GOALS
    grid: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.grid:
            self.grid = self._build()

    def _build(self) -> list[list[float]]:
        n = self.max_goals + 1
        home_pmf = [_poisson_pmf(i, self.lambda_home) for i in range(n)]
        away_pmf = [_poisson_pmf(j, self.lambda_away) for j in range(n)]
        grid = [
            [
                max(
                    home_pmf[i]
                    * away_pmf[j]
                    * _dixon_coles_tau(i, j, self.lambda_home, self.lambda_away, self.rho),
                    0.0,
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
        total = sum(sum(row) for row in grid)
        if total <= 0:
            raise ValueError("matriz de placares degenerada")
        return [[cell / total for cell in row] for row in grid]

    def prob(self, home_goals: int, away_goals: int) -> float:
        if home_goals > self.max_goals or away_goals > self.max_goals:
            return 0.0
        return self.grid[home_goals][away_goals]

    def shifted(self, base_home: int, base_away: int) -> "ScoreMatrix":
        """Desloca a matriz somando o placar já ocorrido.

        Usado ao vivo: a matriz modela apenas os gols restantes, e o placar
        atual entra como offset.
        """
        n = self.max_goals + 1 + max(base_home, base_away)
        grid = [[0.0] * n for _ in range(n)]
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                grid[i + base_home][j + base_away] += cell
        return ScoreMatrix(
            lambda_home=self.lambda_home,
            lambda_away=self.lambda_away,
            rho=self.rho,
            max_goals=n - 1,
            grid=grid,
        )

    # ---------- mercados ----------

    def match_odds(self) -> dict[str, float]:
        home = draw = away = 0.0
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if i > j:
                    home += cell
                elif i == j:
                    draw += cell
                else:
                    away += cell
        return {"home": home, "draw": draw, "away": away}

    def double_chance(self) -> dict[str, float]:
        m = self.match_odds()
        return {
            "home_draw": m["home"] + m["draw"],
            "home_away": m["home"] + m["away"],
            "draw_away": m["draw"] + m["away"],
        }

    def total_goals_distribution(self) -> list[float]:
        n = 2 * self.max_goals + 1
        dist = [0.0] * n
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                dist[i + j] += cell
        return dist

    def over_under(self, line: float) -> dict[str, float]:
        """Over/under incluindo linhas asiáticas (.0, .25, .5, .75).

        Linhas inteiras (2.0) devolvem `push`. Linhas de quarto (2.25) são a
        média de duas linhas adjacentes, porque metade do valor apostado vai
        para cada uma.
        """
        quarter = abs(line * 4 - round(line * 4)) < 1e-9 and abs(line * 2 - round(line * 2)) > 1e-9
        if quarter:
            low = line - 0.25
            high = line + 0.25
            a = self.over_under(low)
            b = self.over_under(high)
            return {k: (a[k] + b[k]) / 2 for k in ("over", "under", "push")}

        dist = self.total_goals_distribution()
        over = sum(p for total, p in enumerate(dist) if total > line)
        under = sum(p for total, p in enumerate(dist) if total < line)
        push = 1.0 - over - under
        return {"over": over, "under": under, "push": max(push, 0.0)}

    def btts(self) -> dict[str, float]:
        yes = sum(
            cell
            for i, row in enumerate(self.grid)
            for j, cell in enumerate(row)
            if i > 0 and j > 0
        )
        return {"yes": yes, "no": 1.0 - yes}

    def asian_handicap(self, line: float) -> dict[str, float]:
        """Handicap asiático do ponto de vista do mandante.

        `line = -0.5` significa que o mandante começa meio gol atrás.
        """
        quarter = abs(line * 4 - round(line * 4)) < 1e-9 and abs(line * 2 - round(line * 2)) > 1e-9
        if quarter:
            a = self.asian_handicap(line - 0.25)
            b = self.asian_handicap(line + 0.25)
            return {k: (a[k] + b[k]) / 2 for k in ("home", "away", "push")}

        home = away = push = 0.0
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                diff = (i + line) - j
                if diff > 1e-9:
                    home += cell
                elif diff < -1e-9:
                    away += cell
                else:
                    push += cell
        return {"home": home, "away": away, "push": push}

    def correct_score(self, top: int = 10) -> list[tuple[str, float]]:
        cells = [
            (f"{i}-{j}", cell)
            for i, row in enumerate(self.grid)
            for j, cell in enumerate(row)
        ]
        cells.sort(key=lambda kv: kv[1], reverse=True)
        return cells[:top]

    def expected_goals(self) -> tuple[float, float]:
        home = sum(i * sum(row) for i, row in enumerate(self.grid))
        away = sum(
            j * sum(self.grid[i][j] for i in range(len(self.grid)))
            for j in range(len(self.grid[0]))
        )
        return home, away


# ---------- calibração a partir das odds do mercado ----------


def _bisect(fn, lo: float, hi: float, target: float, tol: float = 1e-9, max_iter: int = 80) -> float:
    """Bisseção sobre uma função monotônica crescente."""
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        val = fn(mid)
        if abs(val - target) < tol:
            return mid
        if val < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def calibrate(
    p_home: float | None = None,
    p_away: float | None = None,
    p_over: float | None = None,
    over_line: float = 2.5,
    prior_total: float = 2.7,
    rho: float = DEFAULT_RHO,
    iterations: int = 12,
    base_home: int = 0,
    base_away: int = 0,
) -> tuple[float, float]:
    """Deriva (lambda_mandante, lambda_visitante) das probabilidades justas do mercado.

    A parametrização usada é `total = lh + la` e `supremacia = lh - la`:

    * o total é monotônico na probabilidade de over → bisseção no over/under;
    * a supremacia é monotônica em `P(casa) - P(fora)` → bisseção no 1X2.

    Os dois se acoplam levemente, então alterna-se entre eles até convergir.
    Sem mercado de over/under, usa-se `prior_total` da liga.

    Com `base_home`/`base_away` diferentes de zero, as odds são interpretadas
    como de um jogo em andamento e o resultado são as taxas dos gols
    **restantes**, não do jogo inteiro.
    """
    total = prior_total
    supremacy = 0.0
    diff_target = None if p_home is None or p_away is None else p_home - p_away

    def matrix(t: float, s: float) -> ScoreMatrix:
        lh = max((t + s) / 2, 1e-6)
        la = max((t - s) / 2, 1e-6)
        m = ScoreMatrix(lh, la, rho)
        return m.shifted(base_home, base_away) if base_home or base_away else m

    for _ in range(iterations):
        if p_over is not None:
            total = _bisect(
                lambda t: matrix(t, supremacy).over_under(over_line)["over"], 0.2, 9.0, p_over
            )

        if diff_target is not None:
            def diff_of(s: float) -> float:
                m = matrix(total, s).match_odds()
                return m["home"] - m["away"]

            supremacy = _bisect(diff_of, -total + 1e-6, total - 1e-6, diff_target)

    # Piso de 0.02: quando o mercado é muito desequilibrado a bisseção encosta
    # no limite e produziria lambda 0.00 — que significa "é impossível este
    # time marcar", coisa que nenhuma odd justifica.
    lam_home = max((total + supremacy) / 2, 0.02)
    lam_away = max((total - supremacy) / 2, 0.02)
    return lam_home, lam_away
