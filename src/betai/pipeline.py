"""Pipeline de análise: das odds cruas às apostas de valor.

O fluxo, e o porquê de cada etapa:

1. **Devig** das odds da casa → probabilidades justas do mercado.
2. **Baseline**: taxas de gols para 90 minutos, calibradas pelas odds
   *pré-jogo*. É a única etapa em que copiamos o mercado de propósito — o
   mercado pré-jogo é eficiente, e discordar dele sem informação privada é
   arrogância.
3. **Modelo ao vivo**: aplica tempo restante, placar e cartões sobre o
   baseline. É *aqui* que nasce a divergência, porque o modelo de tempo/placar
   da casa pode ser pior que o nosso.
4. **Comparação** contra as odds ao vivo → EV, edge, Kelly.
5. **IA** (opcional) ajusta pontualmente o que o modelo não consegue enxergar.

Se não houver baseline pré-jogo, o passo 2 inverte as odds ao vivo — e nesse
caso o modelo concorda com o mercado por construção. O pipeline sinaliza isso
via `baseline_source` em vez de fingir que encontrou valor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine.devig import Method, is_plausible, remove_vig
from .engine.live import LiveConfig, live_matrix
from .engine.poisson import DEFAULT_RHO, ScoreMatrix, calibrate
from .engine.value import evaluate, filter_value
from .models import Analysis, Event, Market, MarketKey, ValueBet
from .storage import Store


def usable_markets(event: Event) -> list[Market]:
    """Mercados cujo livro fecha — os demais não dizem nada sobre o jogo.

    Um mercado com seleções faltando ou suspensas chega com as implícitas
    somando bem menos que 1. A remoção de margem normaliza qualquer coisa para
    somar 1, então esse mercado vira probabilidades inventadas e "vantagens"
    de centenas por cento. Descartar é a única leitura honesta.
    """
    return [m for m in event.markets if is_plausible([s.odds for s in m.selections])]


def market_probabilities(event: Event, method: Method = "power") -> dict[str, float]:
    """Probabilidades justas do mercado, uma chave por seleção.

    Chaves seguem o formato `mercado[@linha].seleção`, ex: `over_under@2.5.over`.
    """
    out: dict[str, float] = {}
    for mkt in usable_markets(event):
        odds = [s.odds for s in mkt.selections]
        fair = remove_vig(odds, method)
        suffix = f"@{mkt.line}" if mkt.line is not None else ""
        for sel, p in zip(mkt.selections, fair):
            out[f"{mkt.key.value}{suffix}.{sel.outcome}"] = p
    return out


def model_probabilities(matrix: ScoreMatrix, event: Event) -> dict[str, float]:
    """Probabilidades do modelo, nas mesmas chaves do mercado.

    Só gera linhas que a casa realmente oferece — não adianta ter opinião
    sobre um mercado inexistente.
    """
    out: dict[str, float] = {}

    for outcome, p in matrix.match_odds().items():
        out[f"{MarketKey.MATCH_ODDS.value}.{outcome}"] = p
    for outcome, p in matrix.double_chance().items():
        out[f"{MarketKey.DOUBLE_CHANCE.value}.{outcome}"] = p
    for outcome, p in matrix.btts().items():
        out[f"{MarketKey.BTTS.value}.{outcome}"] = p

    for mkt in usable_markets(event):
        if mkt.key != MarketKey.OVER_UNDER:
            continue
        line = mkt.line if mkt.line is not None else 2.5
        ou = matrix.over_under(line)
        out[f"{MarketKey.OVER_UNDER.value}@{line}.over"] = ou["over"]
        out[f"{MarketKey.OVER_UNDER.value}@{line}.under"] = ou["under"]

    for mkt in usable_markets(event):
        if mkt.key != MarketKey.ASIAN_HANDICAP:
            continue
        line = mkt.line if mkt.line is not None else 0.0
        ah = matrix.asian_handicap(line)
        out[f"{MarketKey.ASIAN_HANDICAP.value}@{line}.home"] = ah["home"]
        out[f"{MarketKey.ASIAN_HANDICAP.value}@{line}.away"] = ah["away"]

    return out


def _fair_inputs(event: Event, method: Method) -> tuple[float | None, float | None, float | None, float]:
    """Extrai P(casa), P(fora), P(over) e a linha do over das odds do evento."""
    p_home = p_away = p_over = None
    over_line = 2.5

    utilizaveis = usable_markets(event)
    mo = next((m for m in utilizaveis if m.key == MarketKey.MATCH_ODDS), None)
    if mo and len(mo.selections) >= 3:
        fair = remove_vig([s.odds for s in mo.selections], method)
        by_outcome = dict(zip([s.outcome for s in mo.selections], fair))
        p_home = by_outcome.get("home")
        p_away = by_outcome.get("away")

    ou_markets = [m for m in utilizaveis if m.key == MarketKey.OVER_UNDER]
    if ou_markets:
        # A linha mais próxima de 2.5 é a mais líquida e a mais informativa.
        mkt = min(ou_markets, key=lambda m: abs((m.line or 2.5) - 2.5))
        fair = remove_vig([s.odds for s in mkt.selections], method)
        by_outcome = dict(zip([s.outcome for s in mkt.selections], fair))
        p_over = by_outcome.get("over")
        over_line = mkt.line if mkt.line is not None else 2.5

    return p_home, p_away, p_over, over_line


def derive_baseline(
    event: Event,
    method: Method = "power",
    prior_total: float = 2.7,
    rho: float = DEFAULT_RHO,
) -> tuple[float, float]:
    """Taxas de gols para 90 minutos a partir das odds pré-jogo do evento."""
    p_home, p_away, p_over, over_line = _fair_inputs(event, method)
    return calibrate(
        p_home=p_home,
        p_away=p_away,
        p_over=p_over,
        over_line=over_line,
        prior_total=prior_total,
        rho=rho,
    )


@dataclass
class Pipeline:
    """Orquestra devig → baseline → modelo ao vivo → valor."""

    devig_method: Method = "power"
    prior_total: float = 2.7
    live_config: LiveConfig = field(default_factory=LiveConfig)
    min_edge: float = 0.03
    kelly_fraction: float = 0.25
    store: Store | None = None
    # Até que minuto uma observação ainda vale como baseline. Aos 5 minutos
    # as odds ao vivo praticamente ainda são as de pré-jogo, e a correção pelo
    # tempo decorrido é pequena o bastante para não distorcer. É o que permite
    # o `watch` se virar sozinho quando não existe endpoint de pré-jogo.
    early_baseline_minute: int = 5
    _baselines: dict[str, tuple[float, float]] = field(default_factory=dict)

    def _usable_as_baseline(self, event: Event) -> bool:
        return (
            not event.state.is_live
            or event.state.minute <= self.early_baseline_minute
        ) and bool(event.markets)

    def _derive(self, event: Event) -> tuple[float, float]:
        """Taxas para 90 minutos a partir de um evento pré-jogo ou recém-começado.

        Num jogo já iniciado, a calibração devolve as taxas dos gols restantes
        e é preciso reescalar. Só é seguro fazer isso cedo: aos 5 minutos o
        divisor é ~0.96, aos 87 seria 0.04 e o lambda explodiria.
        """
        if not event.state.is_live:
            return derive_baseline(
                event, self.devig_method, self.prior_total, self.live_config.rho
            )

        from .engine.live import remaining_fraction

        p_home, p_away, p_over, over_line = _fair_inputs(event, self.devig_method)
        rem_h, rem_a = calibrate(
            p_home=p_home,
            p_away=p_away,
            p_over=p_over,
            over_line=over_line,
            prior_total=self.prior_total,
            rho=self.live_config.rho,
            base_home=event.state.score_home,
            base_away=event.state.score_away,
        )
        restante = max(remaining_fraction(event.state.minute), 0.5)
        return rem_h / restante, rem_a / restante

    def register_baseline(self, event: Event) -> None:
        """Guarda o baseline de um evento, se ele ainda serve como referência.

        Vale para jogos que não começaram e para os que acabaram de começar —
        o primeiro que chegar fica, porque quanto mais cedo, melhor.
        """
        if event.event_id in self._baselines or not self._usable_as_baseline(event):
            return
        self._baselines[event.event_id] = self._derive(event)

    def _baseline_for(self, event: Event) -> tuple[tuple[float, float], str]:
        """Encontra o melhor baseline disponível e diz de onde ele veio."""
        if event.event_id in self._baselines:
            return self._baselines[event.event_id], "pre_match"

        # Procura no histórico o snapshot mais antigo que ainda sirva como
        # referência: antes do apito, ou nos primeiros minutos.
        if self.store is not None:
            for snap in self.store.snapshots_for(event.event_id):
                if self._usable_as_baseline(snap):
                    baseline = self._derive(snap)
                    self._baselines[event.event_id] = baseline
                    return baseline, "pre_match"

        if self._usable_as_baseline(event):
            baseline = self._derive(event)
            self._baselines[event.event_id] = baseline
            return baseline, "pre_match"

        # Última alternativa: inverter as odds ao vivo. O resultado são as
        # taxas dos gols **restantes** — não reescalamos para 90 minutos,
        # porque aos 87' isso significa dividir por 0.04 e produzir um lambda
        # de 50 gols.
        p_home, p_away, p_over, over_line = _fair_inputs(event, self.devig_method)
        rem = calibrate(
            p_home=p_home,
            p_away=p_away,
            p_over=p_over,
            over_line=over_line,
            prior_total=self.prior_total,
            rho=self.live_config.rho,
            base_home=event.state.score_home,
            base_away=event.state.score_away,
        )
        return rem, "live_inverted"

    def analyze(self, event: Event) -> Analysis:
        """Análise completa de um evento, sem a camada de IA."""
        (lam_home, lam_away), source = self._baseline_for(event)

        if source == "live_inverted":
            # Sem baseline pré-jogo não existe opinião independente: as taxas
            # vieram das próprias odds ao vivo. Aplicar por cima os efeitos de
            # placar e cartão — que essas odds já embutem — criaria divergência
            # artificial e encheria a tela de "valor" inexistente.
            matrix = ScoreMatrix(lam_home, lam_away, self.live_config.rho).shifted(
                event.state.score_home, event.state.score_away
            )
            return Analysis(
                event=event,
                lambda_home=lam_home,
                lambda_away=lam_away,
                baseline_source=source,
                probabilities=model_probabilities(matrix, event),
                value_bets=[],
            )

        matrix = live_matrix(lam_home, lam_away, event.state, self.live_config)
        model = model_probabilities(matrix, event)
        market = market_probabilities(event, self.devig_method)

        return Analysis(
            event=event,
            lambda_home=lam_home,
            lambda_away=lam_away,
            baseline_source=source,
            probabilities=model,
            value_bets=self._find_value(event, model, market),
        )

    def _find_value(
        self,
        event: Event,
        model: dict[str, float],
        market: dict[str, float],
    ) -> list[ValueBet]:
        bets: list[ValueBet] = []
        for mkt in usable_markets(event):
            suffix = f"@{mkt.line}" if mkt.line is not None else ""
            for sel in mkt.selections:
                key = f"{mkt.key.value}{suffix}.{sel.outcome}"
                if key not in model or key not in market:
                    continue
                bets.append(
                    evaluate(
                        market=mkt.key,
                        outcome=sel.outcome,
                        odds=sel.odds,
                        model_probability=model[key],
                        market_probability=market[key],
                        line=mkt.line if mkt.line is not None else sel.line,
                        kelly_fraction=self.kelly_fraction,
                    )
                )
        return filter_value(bets, min_edge=self.min_edge)

    def analyze_with_ai(
        self,
        event: Event,
        analyst,  # betai.ai.Analyst — importado tarde para não exigir a dependência
        context: str | None = None,
    ) -> Analysis:
        """Análise quantitativa seguida da revisão da IA.

        Se a chamada à IA falhar, devolve a análise quantitativa intacta — o
        motor não depende do modelo de linguagem para funcionar.
        """
        from .ai.analyst import apply_adjustments

        analysis = self.analyze(event)
        try:
            verdict = analyst.review(analysis, context)
        except Exception as exc:  # rede, cota, schema — nada disso invalida o modelo
            analysis.ai_summary = f"[IA indisponível: {exc}]"
            return analysis

        adjusted = apply_adjustments(
            analysis.probabilities, verdict, max_shift=analyst.max_shift
        )
        market = market_probabilities(event, self.devig_method)

        analysis.probabilities = adjusted
        analysis.ai_summary = verdict.summary
        analysis.ai_adjustments = {a.key: a.delta for a in verdict.adjustments}
        analysis.value_bets = self._find_value(event, adjusted, market)
        return analysis
