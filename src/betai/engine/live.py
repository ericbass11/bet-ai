"""Ajuste ao vivo: converte o modelo pré-jogo no modelo do tempo restante.

Três coisas mudam quando o jogo começa:

1. **Sobra menos jogo.** E gols não se distribuem uniformemente ao longo dos
   90 minutos — a taxa cresce no decorrer da partida.
2. **O placar mexe no comportamento.** Um time perdendo ataca mais; um time
   ganhando por muito administra. O efeito é real e assimétrico.
3. **Cartões vermelhos.** Mudam o jogo mais que qualquer outra variável
   observável em tempo real.

Todos os coeficientes aqui são heurísticas parametrizadas e documentadas —
existem para serem recalibradas contra dados históricos, não para serem
tratadas como verdade.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import MatchState
from .poisson import DEFAULT_RHO, ScoreMatrix

REGULATION_MINUTES = 90

# Acréscimos. O relógio das casas congela em 90' enquanto o jogo continua, e
# gol nos acréscimos é corriqueiro (~0.14 por partida). Sem isto, o modelo
# declara o placar decidido aos 90' — com probabilidade 100% e aposta enorme
# recomendada sobre um resultado que ainda pode mudar.
STOPPAGE_MINUTES = 4
MATCH_MINUTES = REGULATION_MINUTES + STOPPAGE_MINUTES

# Intensidade relativa de gols ao longo do jogo: sobe de ~0.75x no apito
# inicial para ~1.25x nos minutos finais, com média 1.0.
INTENSITY_START = 0.75
INTENSITY_END = 1.25


@dataclass
class LiveConfig:
    """Coeficientes do ajuste ao vivo. Recalibre contra o seu histórico."""

    # Multiplicador na taxa do time com um jogador a menos, por cartão.
    red_card_own: float = 0.72
    # Multiplicador na taxa do adversário do time com um jogador a menos.
    red_card_opponent: float = 1.30
    # Quanto o time que está perdendo acelera, por gol de desvantagem.
    chasing_boost: float = 0.12
    # Quanto o time que está ganhando desacelera, por gol de vantagem.
    leading_damp: float = 0.06
    # Teto do impacto do placar, para não explodir em goleadas.
    max_score_effect: float = 0.45
    # Peso do xG acumulado ao mesclar com a expectativa pré-jogo.
    xg_weight: float = 0.35
    rho: float = DEFAULT_RHO


def remaining_fraction(minute: int) -> float:
    """Fração da intensidade total de gols que ainda resta.

    Integra a intensidade crescente de `minute` até o fim do jogo, incluindo
    os acréscimos. Aos 90' ainda sobra ~5% — o suficiente para o modelo não
    tratar o placar como decidido enquanto a bola rola.
    """
    m = max(0, min(minute, MATCH_MINUTES))

    def cumulative(t: float) -> float:
        # Integral de (a + (b-a)*t/T) dt, de 0 a t.
        a, b, T = INTENSITY_START, INTENSITY_END, MATCH_MINUTES
        return a * t + (b - a) * t * t / (2 * T)

    full = cumulative(MATCH_MINUTES)
    return (full - cumulative(m)) / full


def _score_effect(goal_diff: int, cfg: LiveConfig) -> tuple[float, float]:
    """Multiplicadores (mandante, visitante) devidos ao placar corrente.

    `goal_diff` é gols do mandante menos gols do visitante.
    """
    if goal_diff == 0:
        return 1.0, 1.0

    magnitude = abs(goal_diff)
    boost = min(cfg.chasing_boost * magnitude, cfg.max_score_effect)
    damp = min(cfg.leading_damp * magnitude, cfg.max_score_effect)

    if goal_diff > 0:  # mandante na frente
        return 1.0 - damp, 1.0 + boost
    return 1.0 + boost, 1.0 - damp


def _red_card_effect(state: MatchState, cfg: LiveConfig) -> tuple[float, float]:
    """Multiplicadores por vantagem numérica líquida."""
    net = state.red_cards_home - state.red_cards_away
    if net == 0:
        return 1.0, 1.0
    if net > 0:  # mandante com menos jogadores
        return cfg.red_card_own**net, cfg.red_card_opponent**net
    n = -net
    return cfg.red_card_opponent**n, cfg.red_card_own**n


def _xg_blend(
    lam_home: float,
    lam_away: float,
    state: MatchState,
    elapsed_fraction: float,
    cfg: LiveConfig,
) -> tuple[float, float]:
    """Mescla a expectativa pré-jogo com o xG efetivamente produzido.

    Se um time gerou muito mais perigo do que se esperava até aqui, isso é
    informação sobre o resto do jogo. Mas extrapolar xG para 90 minutos exige
    dividir pela fração já jogada, e aos 20 minutos isso multiplica o ruído
    por cinco. Duas travas contra esse efeito:

    * o peso cresce com o tempo decorrido — 20 minutos de xG valem pouco,
      80 minutos valem quase tudo;
    * o resultado é limitado a metade/dobro da expectativa pré-jogo, porque
      nenhuma quantidade de xG acumulado justifica jogar fora o baseline.
    """
    stats = state.stats
    if stats.xg_home is None or stats.xg_away is None or elapsed_fraction < 0.15:
        return lam_home, lam_away

    w = cfg.xg_weight * elapsed_fraction

    def blend(lam: float, xg: float) -> float:
        observed = xg / elapsed_fraction  # taxa equivalente para 90 minutos
        return min(max((1 - w) * lam + w * observed, lam * 0.5), lam * 2.0)

    return blend(lam_home, stats.xg_home), blend(lam_away, stats.xg_away)


def live_matrix(
    lambda_home: float,
    lambda_away: float,
    state: MatchState,
    cfg: LiveConfig | None = None,
) -> ScoreMatrix:
    """Matriz de placares finais dado o estado atual da partida.

    `lambda_home`/`lambda_away` são as taxas para 90 minutos calibradas antes
    do jogo. O retorno já inclui o placar corrente somado.
    """
    cfg = cfg or LiveConfig()

    if not state.is_live:
        return ScoreMatrix(lambda_home, lambda_away, cfg.rho)

    remaining = remaining_fraction(state.minute)
    elapsed = 1.0 - remaining

    lam_h, lam_a = _xg_blend(lambda_home, lambda_away, state, elapsed, cfg)

    score_h, score_a = _score_effect(state.score_home - state.score_away, cfg)
    red_h, red_a = _red_card_effect(state, cfg)

    rem_home = lam_h * remaining * score_h * red_h
    rem_away = lam_a * remaining * score_a * red_a

    remaining_matrix = ScoreMatrix(max(rem_home, 1e-6), max(rem_away, 1e-6), cfg.rho)
    return remaining_matrix.shifted(state.score_home, state.score_away)
