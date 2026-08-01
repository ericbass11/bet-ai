"""Interface web local: uma página que se atualiza sozinha no navegador.

Roda inteiramente na máquina do usuário, com a biblioteca padrão do Python —
sem framework, sem servidor externo, sem nada para instalar além do que o
projeto já usa.

A arquitetura é a mais simples que resolve: uma thread coleta e analisa em
intervalo fixo, guarda o resultado em memória, e o servidor HTTP serve esse
resultado como JSON. A página busca o JSON e se redesenha.
"""

from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from .models import Analysis


@dataclass
class Estado:
    """O que a interface mostra. Compartilhado entre a thread e o servidor."""

    analises: list[dict] = field(default_factory=list)
    atualizado_em: str | None = None
    erro: str | None = None
    ciclos: int = 0
    intervalo: int = 60
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def atualizar(self, analises: list[Analysis]) -> None:
        with self._lock:
            self.analises = [_serializar(a) for a in analises]
            self.atualizado_em = datetime.now(timezone.utc).isoformat()
            self.erro = None
            self.ciclos += 1

    def falhar(self, erro: str) -> None:
        with self._lock:
            self.erro = erro
            self.ciclos += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            com_valor = [a for a in self.analises if a["apostas"]]
            return {
                "analises": self.analises,
                "atualizado_em": self.atualizado_em,
                "erro": self.erro,
                "ciclos": self.ciclos,
                "intervalo": self.intervalo,
                "resumo": {
                    "jogos": len(self.analises),
                    "com_valor": len(com_valor),
                    "sem_baseline": sum(
                        1 for a in self.analises if a["base"] == "live_inverted"
                    ),
                },
            }


def _serializar(a: Analysis) -> dict[str, Any]:
    from .pipeline import usable_markets

    e = a.event
    descartados = len(e.markets) - len(usable_markets(e))
    return {
        "mercados_descartados": descartados,
        "id": e.event_id,
        "casa": e.home_team,
        "fora": e.away_team,
        "liga": e.league,
        "minuto": e.state.minute,
        "periodo": e.state.period,
        "placar": f"{e.state.score_home}-{e.state.score_away}",
        "vermelhos": [e.state.red_cards_home, e.state.red_cards_away],
        "ao_vivo": e.state.is_live,
        "base": a.baseline_source,
        "lambda_casa": round(a.lambda_home, 2),
        "lambda_fora": round(a.lambda_away, 2),
        "resumo_ia": a.ai_summary,
        "apostas": [
            {
                "mercado": b.market.value,
                "linha": b.line,
                "selecao": b.outcome,
                "odd": b.odds,
                "justa": round(b.fair_odds, 2),
                "modelo": round(b.model_probability, 4),
                "edge": round(b.edge, 4),
                "kelly": round(b.kelly_fraction, 4),
            }
            for b in a.value_bets
        ],
    }


def loop_de_coleta(
    estado: Estado,
    coletar: Callable[[], list[Analysis]],
    intervalo: int,
    parar: threading.Event,
) -> None:
    """Coleta em intervalo fixo até receberem o sinal de parada.

    Uma falha de rede não derruba a interface: vira uma mensagem na tela e o
    ciclo seguinte tenta de novo.
    """
    while not parar.is_set():
        try:
            estado.atualizar(coletar())
        except Exception as exc:  # rede, provedor, mapeamento — tudo é recuperável
            estado.falhar(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        parar.wait(intervalo)


class Handler(BaseHTTPRequestHandler):
    estado: Estado

    def log_message(self, *args: Any) -> None:
        """Silencia o log de acesso — a página consulta a cada poucos segundos."""

    def _responder(self, corpo: bytes, tipo: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self) -> None:  # noqa: N802 — nome exigido pela biblioteca
        if self.path.startswith("/api/analises"):
            corpo = json.dumps(self.estado.snapshot(), ensure_ascii=False).encode()
            self._responder(corpo, "application/json; charset=utf-8")
        elif self.path in ("/", "/index.html"):
            self._responder(PAGINA.encode(), "text/html; charset=utf-8")
        else:
            self.send_error(404)


def servir(estado: Estado, host: str, porta: int) -> ThreadingHTTPServer:
    handler = type("HandlerComEstado", (Handler,), {"estado": estado})
    return ThreadingHTTPServer((host, porta), handler)


PAGINA = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bet-ai</title>
<style>
  :root {
    --bg: #f6f7f9; --card: #fff; --texto: #14181f; --fraco: #667085;
    --borda: #e3e6ea; --verde: #0f7b4f; --verde-bg: #e8f5ee;
    --amarelo: #8a6100; --amarelo-bg: #fdf3e0; --vermelho: #b42318;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1115; --card: #171a21; --texto: #e8eaed; --fraco: #9aa3af;
      --borda: #262b34; --verde: #4ade80; --verde-bg: #10291d;
      --amarelo: #fbbf24; --amarelo-bg: #2a2110; --vermelho: #f87171;
    }
  }
  :root[data-theme="light"] {
    --bg: #f6f7f9; --card: #fff; --texto: #14181f; --fraco: #667085;
    --borda: #e3e6ea; --verde: #0f7b4f; --verde-bg: #e8f5ee;
    --amarelo: #8a6100; --amarelo-bg: #fdf3e0; --vermelho: #b42318;
  }
  :root[data-theme="dark"] {
    --bg: #0f1115; --card: #171a21; --texto: #e8eaed; --fraco: #9aa3af;
    --borda: #262b34; --verde: #4ade80; --verde-bg: #10291d;
    --amarelo: #fbbf24; --amarelo-bg: #2a2110; --vermelho: #f87171;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--texto);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  }
  header {
    position: sticky; top: 0; z-index: 10; background: var(--card);
    border-bottom: 1px solid var(--borda); padding: 14px 20px;
    display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
  }
  h1 { font-size: 17px; margin: 0; letter-spacing: -0.01em; }
  .pilulas { display: flex; gap: 8px; flex-wrap: wrap; }
  .pilula {
    font-size: 13px; padding: 3px 10px; border-radius: 999px;
    background: var(--bg); border: 1px solid var(--borda); color: var(--fraco);
  }
  .pilula strong { color: var(--texto); }
  .controles { margin-left: auto; display: flex; gap: 14px; align-items: center;
               font-size: 13px; color: var(--fraco); }
  label { display: flex; gap: 6px; align-items: center; cursor: pointer; }
  main { padding: 20px; max-width: 1100px; margin: 0 auto; }
  .jogo {
    background: var(--card); border: 1px solid var(--borda); border-radius: 10px;
    padding: 14px 16px; margin-bottom: 12px;
  }
  .jogo.tem-valor { border-color: var(--verde); }
  .topo { display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; }
  .times { font-weight: 600; }
  .meta { color: var(--fraco); font-size: 13px; }
  .placar {
    font-variant-numeric: tabular-nums; font-weight: 600;
    background: var(--bg); border-radius: 6px; padding: 1px 8px;
  }
  .aviso {
    margin-top: 8px; font-size: 13px; padding: 7px 10px; border-radius: 7px;
    background: var(--amarelo-bg); color: var(--amarelo);
  }
  .erro {
    margin: 0 0 16px; padding: 12px 14px; border-radius: 8px;
    background: var(--card); border: 1px solid var(--vermelho);
    color: var(--vermelho); font-size: 14px;
  }
  .tabela-wrap { overflow-x: auto; margin-top: 10px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th {
    text-align: right; font-weight: 500; color: var(--fraco); font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.04em;
    padding: 4px 10px; border-bottom: 1px solid var(--borda);
  }
  th:first-child, td:first-child { text-align: left; }
  td { padding: 6px 10px; text-align: right; font-variant-numeric: tabular-nums; }
  tbody tr { background: var(--verde-bg); }
  tbody tr td:first-child { border-radius: 6px 0 0 6px; }
  tbody tr td:last-child { border-radius: 0 6px 6px 0; }
  .edge { color: var(--verde); font-weight: 600; }
  .vazio { color: var(--fraco); font-size: 13px; margin-top: 8px; }
  .ia { margin-top: 8px; font-size: 13px; }
  footer { color: var(--fraco); font-size: 12px; text-align: center; padding: 24px; }
  .carregando { color: var(--fraco); text-align: center; padding: 60px 20px; }
</style>
</head>
<body>
<header>
  <h1>bet-ai</h1>
  <div class="pilulas" id="pilulas"></div>
  <div class="controles">
    <label><input type="checkbox" id="so-valor"> só com valor</label>
    <span id="relogio"></span>
  </div>
</header>
<main id="conteudo"><p class="carregando">Carregando…</p></main>
<footer>
  Ferramenta de análise, não recomendação de aposta. Vantagem estimada não é lucro garantido.
</footer>

<script>
const pct = (v) => (v * 100).toFixed(1) + '%';
const soValor = document.getElementById('so-valor');
soValor.checked = localStorage.getItem('soValor') === '1';
soValor.addEventListener('change', () => {
  localStorage.setItem('soValor', soValor.checked ? '1' : '0');
  desenhar(ultimo);
});

let ultimo = null;

function cabecalho(j) {
  const quando = j.ao_vivo
    ? `${j.minuto}' <span class="placar">${j.placar}</span>`
    : 'ainda não começou';
  const vermelhos = (j.vermelhos[0] || j.vermelhos[1])
    ? ` · 🟥 ${j.vermelhos[0]}-${j.vermelhos[1]}` : '';
  return `<div class="topo">
      <span class="times">${j.casa} × ${j.fora}</span>
      <span class="meta">${j.liga} · ${quando}${vermelhos}</span>
      <span class="meta" style="margin-left:auto">
        gols esperados ${j.lambda_casa} / ${j.lambda_fora}</span>
    </div>`;
}

// Tradução do vocabulário interno para o que a pessoa lê na casa de apostas.
// "1x2 — home" não diz nada; "Palmeiras vence" diz.
function rotulo(a, j) {
  const n = a.linha != null ? a.linha : '';
  switch (a.mercado) {
    case '1x2':
      return { home: `${j.casa} vence`, draw: 'Empate', away: `${j.fora} vence` }[a.selecao];
    case 'double_chance':
      return {
        home_draw: `${j.casa} ou empate`,
        home_away: 'Sai do empate',
        draw_away: `${j.fora} ou empate`,
      }[a.selecao];
    case 'over_under':
      return a.selecao === 'over' ? `Mais de ${n} gols` : `Menos de ${n} gols`;
    case 'btts':
      return a.selecao === 'yes' ? 'Ambas marcam' : 'Nem todas marcam';
    case 'asian_handicap':
      return `${a.selecao === 'home' ? j.casa : j.fora} com handicap ${n}`;
    case 'correct_score':
      return `Placar exato ${a.selecao}`;
    default:
      return `${a.mercado} — ${a.selecao}`;
  }
}

function tabela(apostas, jogo) {
  const linhas = apostas.map(a => `<tr>
      <td><strong>${rotulo(a, jogo)}</strong></td>
      <td>${a.odd.toFixed(2)}</td>
      <td>${a.justa.toFixed(2)}</td>
      <td>${pct(a.modelo)}</td>
      <td class="edge">+${pct(a.edge)}</td>
      <td>${pct(a.kelly)}</td>
    </tr>`).join('');
  return `<div class="tabela-wrap"><table>
      <thead><tr>
        <th>aposta</th><th>paga</th><th>deveria pagar</th>
        <th>chance</th><th>vantagem</th><th>quanto apostar</th>
      </tr></thead>
      <tbody>${linhas}</tbody>
    </table></div>`;
}

function desenhar(dados) {
  if (!dados) return;
  ultimo = dados;

  const r = dados.resumo;
  document.getElementById('pilulas').innerHTML = `
    <span class="pilula"><strong>${r.jogos}</strong> jogos</span>
    <span class="pilula"><strong>${r.com_valor}</strong> com valor</span>
    ${r.sem_baseline ? `<span class="pilula">${r.sem_baseline} sem referência</span>` : ''}`;

  if (dados.atualizado_em) {
    const t = new Date(dados.atualizado_em);
    document.getElementById('relogio').textContent =
      'atualizado ' + t.toLocaleTimeString('pt-BR');
  }

  let html = '';
  if (dados.erro) {
    html += `<div class="erro"><strong>Falha na última coleta.</strong> ${dados.erro}
      <br>A interface segue tentando a cada ${dados.intervalo}s.</div>`;
  }

  let jogos = dados.analises;
  if (soValor.checked) jogos = jogos.filter(j => j.apostas.length);
  jogos = [...jogos].sort((a, b) => {
    const ea = a.apostas.length ? a.apostas[0].edge : -1;
    const eb = b.apostas.length ? b.apostas[0].edge : -1;
    return eb - ea;
  });

  if (!jogos.length) {
    html += `<p class="carregando">${
      dados.analises.length ? 'Nenhum jogo com valor no momento.' : 'Nenhum jogo ao vivo.'
    }</p>`;
  }

  for (const j of jogos) {
    html += `<div class="jogo ${j.apostas.length ? 'tem-valor' : ''}">${cabecalho(j)}`;
    if (j.mercados_descartados) {
      html += `<div class="aviso">${j.mercados_descartados} mercado(s) ignorado(s):
        as odds enviadas pela casa não formam um livro coerente — normalmente
        significa aposta suspensa. Analisar isso inventaria vantagem que não
        existe.</div>`;
    }
    if (j.base === 'live_inverted') {
      html += `<div class="aviso">Sem odds de antes do jogo — não dá para discordar
        do mercado. Deixe o programa rodando para capturar a referência no início
        das próximas partidas.</div>`;
    }
    if (j.resumo_ia) html += `<div class="ia"><strong>IA:</strong> ${j.resumo_ia}</div>`;
    html += j.apostas.length
      ? tabela(j.apostas, j)
      : '<p class="vazio">Sem diferença relevante contra o mercado.</p>';
    html += '</div>';
  }
  document.getElementById('conteudo').innerHTML = html;
}

async function buscar() {
  try {
    const r = await fetch('/api/analises');
    desenhar(await r.json());
  } catch (e) {
    document.getElementById('relogio').textContent = 'sem conexão com o programa';
  }
}
buscar();
setInterval(buscar, 5000);
</script>
</body>
</html>
"""
