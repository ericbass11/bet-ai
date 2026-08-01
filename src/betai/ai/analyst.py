"""Camada de IA: Claude revisa e ajusta as probabilidades do modelo.

O papel da IA aqui é deliberadamente limitado. O modelo de Poisson é bom no
que é mensurável (gols, tempo, placar) e cego para o que é textual: notícia
de escalação, contexto de tabela, clima, um técnico que acabou de ser
demitido, um time que já está classificado. A IA lê esse contexto e propõe
um **ajuste delimitado** sobre a saída quantitativa.

Duas travas importantes:

* o ajuste é somado em pontos de probabilidade e limitado por `max_shift` —
  a IA corrige, nunca substitui o modelo;
* a saída é estruturada via schema, então não há parsing de texto livre.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..models import Analysis, Event

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """Você é um analista quantitativo de futebol. Recebe a saída de um \
modelo de Poisson/Dixon-Coles calibrado pelas odds do mercado, mais o contexto \
da partida.

Sua função é corrigir o modelo onde ele é estruturalmente cego, não recalculá-lo. \
O modelo já captura: gols esperados, tempo restante, placar corrente, cartões \
vermelhos e xG acumulado. Ele NÃO captura: notícias de escalação, desfalques, \
motivação de tabela, calendário, clima, arbitragem, contexto de rivalidade.

Regras:
- Proponha ajustes apenas onde houver um motivo concreto e citável no contexto \
fornecido. Sem motivo, o ajuste é 0.0.
- Ajustes são em pontos de probabilidade absolutos (0.03 = três pontos \
percentuais), positivos ou negativos, limitados ao teto informado.
- Não repita o que o modelo já mediu como se fosse informação nova. Placar e \
tempo já estão precificados.
- Se o contexto for pobre ou ausente, diga isso e devolva ajustes zerados. \
Ajuste sem base é ruído caro.
- Responda em português do Brasil, de forma direta e sem hedging decorativo."""


class SelectionAdjustment(BaseModel):
    """Ajuste proposto para uma seleção específica."""

    key: str = Field(
        description="Chave da seleção, exatamente como veio na entrada (ex: '1x2.home')"
    )
    delta: float = Field(
        description="Ajuste em pontos de probabilidade absolutos, positivo ou negativo"
    )
    rationale: str = Field(description="Motivo concreto do ajuste, em uma frase")


class AiVerdict(BaseModel):
    """Resposta estruturada do analista."""

    summary: str = Field(description="Leitura da partida em 2 a 4 frases")
    adjustments: list[SelectionAdjustment] = Field(
        default_factory=list, description="Ajustes propostos; vazio se não houver base"
    )
    key_factors: list[str] = Field(
        default_factory=list, description="Fatores decisivos identificados no contexto"
    )
    # Sem ge/le: structured outputs não suportam restrições numéricas, e o SDK
    # as removeria do schema para validar no cliente — o que derrubaria o
    # veredicto inteiro por causa de um número fora da faixa. Limitamos aqui.
    confidence: float = Field(
        default=0.0, description="Confiança nos ajustes propostos, de 0 a 1"
    )

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class Analyst:
    """Cliente Claude para revisão qualitativa das probabilidades."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL,
        max_shift: float = 0.05,
        effort: str = "medium",
    ) -> None:
        # O SDK resolve credenciais do ambiente (ANTHROPIC_API_KEY, perfil OAuth
        # do `ant auth login`, etc.) — passar api_key só é necessário quando se
        # quer injetar uma chave específica.
        import anthropic

        self.client = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        self.model = model
        self.max_shift = max_shift
        self.effort = effort

    def _payload(self, analysis: Analysis, context: str | None) -> str:
        event = analysis.event
        state = event.state
        book = {
            f"{m.key.value}{'@' + str(m.line) if m.line is not None else ''}.{s.outcome}": s.odds
            for m in event.markets
            for s in m.selections
        }
        return json.dumps(
            {
                "partida": event.label,
                "liga": event.league,
                "estado": {
                    "minuto": state.minute,
                    "periodo": state.period,
                    "placar": f"{state.score_home}-{state.score_away}",
                    "vermelhos": [state.red_cards_home, state.red_cards_away],
                    "xg": [state.stats.xg_home, state.stats.xg_away],
                    "chutes_no_gol": [
                        state.stats.shots_on_target_home,
                        state.stats.shots_on_target_away,
                    ],
                },
                "modelo": {
                    "lambda_mandante": round(analysis.lambda_home, 3),
                    "lambda_visitante": round(analysis.lambda_away, 3),
                    "probabilidades": {
                        k: round(v, 4) for k, v in analysis.probabilities.items()
                    },
                },
                "odds_da_casa": book,
                "teto_de_ajuste": self.max_shift,
                "contexto_externo": context or "nenhum contexto adicional fornecido",
            },
            ensure_ascii=False,
            indent=2,
        )

    def review(self, analysis: Analysis, context: str | None = None) -> AiVerdict:
        """Pede ao Claude uma revisão estruturada da análise quantitativa."""
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": self._payload(analysis, context)}],
            output_format=AiVerdict,
        )
        verdict = response.parsed_output
        if verdict is None:
            raise RuntimeError("Claude não devolveu uma saída válida no schema esperado")
        return verdict


def apply_adjustments(
    probabilities: dict[str, float],
    verdict: AiVerdict,
    max_shift: float = 0.05,
) -> dict[str, float]:
    """Aplica os ajustes da IA e renormaliza cada família de mercado.

    Cada delta é limitado a `max_shift` independentemente do que a IA pediu, e
    as probabilidades de um mesmo mercado são renormalizadas para somar 1 —
    o ajuste redistribui massa, não a cria.
    """
    adjusted = dict(probabilities)

    for adj in verdict.adjustments:
        if adj.key not in adjusted:
            continue
        delta = max(-max_shift, min(max_shift, adj.delta))
        adjusted[adj.key] = max(0.001, min(0.999, adjusted[adj.key] + delta))

    # Renormaliza por prefixo de mercado (tudo antes do último ponto).
    families: dict[str, list[str]] = {}
    for key in adjusted:
        family = key.rsplit(".", 1)[0]
        families.setdefault(family, []).append(key)

    for keys in families.values():
        total = sum(adjusted[k] for k in keys)
        # Só renormaliza famílias que já somavam ~1 antes (mercados fechados).
        original = sum(probabilities[k] for k in keys)
        if total > 0 and abs(original - 1.0) < 0.01:
            for k in keys:
                adjusted[k] /= total

    return adjusted


def analyst_from_env(**kwargs: Any) -> Analyst | None:
    """Constrói o analista se houver credencial disponível; senão, devolve None.

    Permite que o pipeline rode sem IA — o motor quantitativo é
    autossuficiente.
    """
    try:
        return Analyst(**kwargs)
    except Exception:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        raise
