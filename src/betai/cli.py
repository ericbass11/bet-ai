"""Interface de linha de comando do bet-ai."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from .config import Settings, load
from .engine.live import LiveConfig
from .models import Analysis, Event
from .pipeline import Pipeline, market_probabilities
from .providers import (
    FieldMap,
    GenericJsonProvider,
    MockProvider,
    Provider,
    ProviderError,
    TheOddsApiProvider,
)
from .storage import Store

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"


def build_provider(settings: Settings) -> Provider:
    if settings.provider == "mock":
        return MockProvider()
    if settings.provider == "the_odds_api":
        return TheOddsApiProvider(
            api_key=settings.odds_api_key,
            sport=settings.sport,
            regions=settings.regions,
            bookmaker=settings.bookmaker,
        )
    if settings.provider == "generic_json":
        if not settings.field_map_path:
            raise ProviderError("BETAI_FIELD_MAP precisa apontar para o arquivo de mapeamento")
        spec = json.loads(Path(settings.field_map_path).read_text(encoding="utf-8"))
        return GenericJsonProvider(FieldMap(spec))
    raise ProviderError(f"provedor desconhecido: {settings.provider}")


def build_pipeline(settings: Settings, store: Store | None) -> Pipeline:
    return Pipeline(
        devig_method=settings.devig_method,  # type: ignore[arg-type]
        prior_total=settings.prior_total_goals,
        live_config=LiveConfig(),
        min_edge=settings.min_edge,
        kelly_fraction=settings.kelly_fraction,
        store=store,
    )


# ---------- formatação ----------


def print_event_header(event: Event) -> None:
    state = event.state
    when = (
        f"{state.minute}' [{state.period}] {state.score_home}-{state.score_away}"
        if state.is_live
        else f"início {event.starts_at:%d/%m %H:%M}"
    )
    print(f"\n{BOLD}{event.label}{RESET}  {DIM}{event.league} · {when}{RESET}")


def print_analysis(analysis: Analysis, verbose: bool = False) -> None:
    print_event_header(analysis.event)

    if analysis.baseline_source == "live_inverted":
        print(
            f"  {YELLOW}sem odds pré-jogo: nenhuma análise independente possível.{RESET}"
            f" {DIM}Rode `bet-ai upcoming` antes do jogo começar.{RESET}"
        )

    print(
        f"  {DIM}λ mandante {analysis.lambda_home:.2f} · "
        f"λ visitante {analysis.lambda_away:.2f} · base: {analysis.baseline_source}{RESET}"
    )

    if verbose:
        market = market_probabilities(analysis.event)
        print(f"  {DIM}{'seleção':<28}{'modelo':>9}{'mercado':>10}{RESET}")
        for key in sorted(analysis.probabilities):
            if key not in market:
                continue
            print(
                f"  {key:<28}{analysis.probabilities[key]:>8.1%}"
                f"{market[key]:>10.1%}"
            )

    if analysis.ai_summary:
        print(f"  {BOLD}IA:{RESET} {analysis.ai_summary}")
        for key, delta in analysis.ai_adjustments.items():
            if abs(delta) > 1e-6:
                print(f"    {DIM}ajuste {key}: {delta:+.1%}{RESET}")

    if not analysis.value_bets:
        print(f"  {DIM}sem valor acima do limiar configurado{RESET}")
        return

    print(f"  {BOLD}Valor encontrado:{RESET}")
    for bet in analysis.value_bets:
        line = f"@{bet.line}" if bet.line is not None else ""
        print(
            f"    {GREEN}{bet.market.value}{line}.{bet.outcome:<12}{RESET}"
            f" odd {bet.odds:>6.2f}"
            f" | justa {bet.fair_odds:>6.2f}"
            f" | modelo {bet.model_probability:>6.1%}"
            f" | edge {bet.edge:>+6.1%}"
            f" | kelly {bet.kelly_fraction:>5.1%}"
        )


# ---------- comandos ----------


def build_analyst(args: argparse.Namespace, settings: Settings):
    if not getattr(args, "ai", False):
        return None
    from .ai import Analyst

    return Analyst(
        api_key=settings.anthropic_api_key,
        model=settings.model,
        max_shift=settings.max_ai_shift,
        effort=settings.effort,
    )


# Quantos jogos por ciclo recebem a busca de mercados extras. Cada um custa
# uma requisição a mais, com o intervalo mínimo do rate limit entre elas — com
# 190 jogos ao vivo, buscar todos levaria mais tempo que o ciclo inteiro. O
# corte é por ordem de chegada e os jogos de fora simplesmente ficam com o
# mercado principal neste ciclo.
MAX_DETALHES_POR_CICLO = 12


def _com_mercados_extras(provider: Provider, event: Event) -> Event:
    """Junta ao evento os mercados do endpoint por jogo, se houver.

    Uma falha aqui não pode derrubar o ciclo: o jogo segue analisável com o
    mercado principal, que é o que a listagem já trouxe.
    """
    try:
        return event.with_markets(provider.fetch_details(event.event_id))
    except ProviderError as exc:
        print(f"{DIM}Sem mercados extras para {event.label}: {exc}{RESET}", file=sys.stderr)
        return event


def collect_live_analyses(
    provider: Provider,
    store: Store | None,
    pipeline: Pipeline,
    analyst=None,
    context: str | None = None,
) -> list[Analysis]:
    """Um ciclo de coleta e análise. Não imprime nada.

    Separado da apresentação porque tanto o terminal quanto a interface web
    consomem o mesmo resultado.
    """
    # Odds pré-jogo primeiro: são elas que dão o baseline honesto. O snapshot
    # só é gravado uma vez por evento — em modo watch, regravar o mesmo
    # pré-jogo a cada minuto só engorda o banco.
    for event in provider.fetch_upcoming():
        novo = event.event_id not in pipeline._baselines
        pipeline.register_baseline(event)
        if store and novo:
            store.save_snapshot(event)

    analises: list[Analysis] = []
    restam_detalhes = MAX_DETALHES_POR_CICLO if provider.supports_details else 0
    for event in provider.fetch_live():
        if restam_detalhes and pipeline.wants_details(event):
            event = _com_mercados_extras(provider, event)
            restam_detalhes -= 1
        if store:
            store.save_snapshot(event)
        analysis = (
            pipeline.analyze_with_ai(event, analyst, context)
            if analyst
            else pipeline.analyze(event)
        )
        if store:
            store.save_analysis(analysis)
        analises.append(analysis)
    return analises


def run_live_cycle(
    provider: Provider,
    store: Store | None,
    pipeline: Pipeline,
    analyst,
    args: argparse.Namespace,
) -> None:
    """Um ciclo de análise ao vivo, imprimindo no terminal."""
    analises = collect_live_analyses(
        provider, store, pipeline, analyst, getattr(args, "context", None)
    )
    if not analises:
        print("Nenhum jogo ao vivo no momento.")
        return

    sem_baseline = sum(1 for a in analises if a.baseline_source == "live_inverted")
    mostrados = 0
    for analysis in analises:
        if getattr(args, "apenas_valor", False) and not analysis.value_bets:
            continue
        print_analysis(analysis, verbose=args.verbose)
        mostrados += 1

    print(f"\n{DIM}{len(analises)} jogos ao vivo, {mostrados} exibidos.{RESET}")
    if sem_baseline:
        print(
            f"{YELLOW}{sem_baseline} sem odds pré-jogo{RESET} — sem baseline não há como "
            f"discordar do mercado. {DIM}Deixe o `watch` rodando para capturá-lo no "
            f"início dos jogos.{RESET}"
        )


def cmd_live(args: argparse.Namespace, settings: Settings) -> int:
    provider = build_provider(settings)
    store = Store(settings.db_path) if not args.no_store else None
    pipeline = build_pipeline(settings, store)
    try:
        run_live_cycle(provider, store, pipeline, build_analyst(args, settings), args)
    finally:
        provider.close()
        if store:
            store.close()
    return 0


def cmd_upcoming(args: argparse.Namespace, settings: Settings) -> int:
    provider = build_provider(settings)
    store = Store(settings.db_path) if not args.no_store else None
    pipeline = build_pipeline(settings, store)
    try:
        events = list(provider.fetch_upcoming())
        if not events:
            print("Nenhum jogo futuro no feed.")
            return 0
        for event in events:
            pipeline.register_baseline(event)
            if store:
                store.save_snapshot(event)
            print_analysis(pipeline.analyze(event), verbose=args.verbose)
    finally:
        provider.close()
        if store:
            store.close()
    return 0


def cmd_watch(args: argparse.Namespace, settings: Settings) -> int:
    """Reanalisa em intervalo fixo — o modo de operação real ao vivo.

    Provedor, banco e baselines vivem por todo o loop: reabri-los a cada ciclo
    perderia os baselines em memória e recriaria a conexão sem necessidade.
    """
    provider = build_provider(settings)
    store = Store(settings.db_path) if not args.no_store else None
    pipeline = build_pipeline(settings, store)
    analyst = build_analyst(args, settings)

    print(f"Monitorando a cada {args.interval}s. Ctrl-C para parar.")
    try:
        while True:
            run_live_cycle(provider, store, pipeline, analyst, args)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nEncerrado.")
    finally:
        provider.close()
        if store:
            store.close()
    return 0


def cmd_web(args: argparse.Namespace, settings: Settings) -> int:
    """Sobe a interface no navegador e atualiza sozinha."""
    import threading
    import webbrowser

    from .web import Estado, loop_de_coleta, servir

    provider = build_provider(settings)
    store = Store(settings.db_path) if not args.no_store else None
    pipeline = build_pipeline(settings, store)
    analyst = build_analyst(args, settings)

    estado = Estado(intervalo=args.interval)
    parar = threading.Event()

    def coletar() -> list[Analysis]:
        return collect_live_analyses(provider, store, pipeline, analyst, args.context)

    coletor = threading.Thread(
        target=loop_de_coleta, args=(estado, coletar, args.interval, parar), daemon=True
    )
    coletor.start()

    try:
        servidor = servir(estado, args.host, args.port)
    except OSError as exc:
        # Porta ocupada é quase sempre uma interface anterior ainda no ar.
        # Sem esta mensagem o processo morre com um traceback de socket que
        # não diz o que fazer — e em segundo plano, morre calado.
        parar.set()
        print(
            f"Não consegui abrir a porta {args.port}: {exc}\n\n"
            f"Se já houver uma interface rodando, encerre-a:\n"
            f"    pkill -f 'bet-ai web'\n"
            f"Ou use outra porta:\n"
            f"    bet-ai web --port {args.port + 1}",
            file=sys.stderr,
        )
        return 1

    endereco = f"http://{'localhost' if args.host in ('', '0.0.0.0') else args.host}:{args.port}"
    print(f"Interface em {BOLD}{endereco}{RESET}  {DIM}(Ctrl-C para parar){RESET}")
    print(f"{DIM}Atualizando a cada {args.interval}s.{RESET}")
    if not args.sem_navegador:
        webbrowser.open(endereco)

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando.")
    finally:
        parar.set()
        servidor.shutdown()
        provider.close()
        if store:
            store.close()
    return 0


def cmd_history(args: argparse.Namespace, settings: Settings) -> int:
    with Store(settings.db_path) as store:
        rows = store.tracked_events()
        if not rows:
            print("Nenhum evento no banco ainda.")
            return 0
        print(f"{BOLD}{'evento':<24}{'partida':<40}{'snaps':>6}  último{RESET}")
        for row in rows:
            label = f"{row['home_team']} x {row['away_team']}"
            print(
                f"{row['event_id']:<24}{label:<40}{row['snapshots']:>6}"
                f"  {row['last_seen'][:19]}"
            )
    return 0


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    """Resumo das apostas de valor registradas, com resultado quando conhecido."""
    with Store(settings.db_path) as store:
        bets = list(store.value_bet_history(args.event_id))
        if not bets:
            print("Nenhuma aposta de valor registrada.")
            return 0

        settled = [b for b in bets if b["final_score"] is not None]
        print(f"{len(bets)} apostas de valor registradas, {len(settled)} com resultado.")
        avg_edge = sum(b["edge"] for b in bets) / len(bets)
        print(f"Edge médio: {avg_edge:+.2%}")
        for bet in bets[-args.limit :]:
            line = f"@{bet['line']}" if bet["line"] is not None else ""
            print(
                f"  {bet['created_at'][:19]} {bet['minute']:>3}' "
                f"{bet['market']}{line}.{bet['outcome']:<12} "
                f"odd {bet['odds']:>6.2f} edge {bet['edge']:>+6.1%}"
            )
    return 0


def cmd_settle(args: argparse.Namespace, settings: Settings) -> int:
    with Store(settings.db_path) as store:
        store.save_result(args.event_id, args.home, args.away)
    print(f"Placar final de {args.event_id}: {args.home}-{args.away}")
    return 0


def cmd_discover(args: argparse.Namespace, settings: Settings) -> int:
    """Infere um field_map a partir de uma captura do DevTools."""
    from .discover import build_field_map, discover, extract_from_har

    path = Path(args.captura)
    if not path.exists():
        print(f"Arquivo não encontrado: {path}", file=sys.stderr)
        return 1

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "log" in raw and "entries" in raw.get("log", {}):
        payloads = extract_from_har(raw)
        print(f"HAR com {len(payloads)} respostas JSON.")
    else:
        payloads = [(args.url or "COLE_A_URL_AQUI", raw)]

    maps = discover(payloads)
    if not maps:
        print(
            "Não encontrei nenhuma lista de eventos com odds nesta captura.\n"
            "Verifique se capturou a requisição certa (a que traz os jogos, não\n"
            "a de imagens ou telemetria) e se salvou o HAR *com* o conteúdo.",
            file=sys.stderr,
        )
        return 1

    best = maps[0]
    print(f"\n{BOLD}Melhor candidato{RESET} (confiança {best['_confianca']:.0%})")
    print(f"  url: {best['url']}")
    print(f"  lista de eventos: {best['events_path'] or '(raiz)'}")
    print(f"  {DIM}sinais: {', '.join(best['_como_foi_inferido'])}{RESET}")
    print(f"  campos: {', '.join(best['fields'])}")
    print(f"  mercados: {len(best['markets'])}")
    for mkt in best["markets"]:
        rotulos = ", ".join(mkt.get("_rotulos_encontrados", []))
        print(f"    {mkt['key']:<16} {mkt['path']}  {DIM}[{rotulos}]{RESET}")

    if len(maps) > 1:
        print(f"  {DIM}(+{len(maps) - 1} outras respostas geraram mapa){RESET}")

    out = Path(args.output)
    out.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRascunho salvo em {BOLD}{out}{RESET}")
    print(
        f"{YELLOW}Revise antes de usar:{RESET} a inferência acerta a estrutura, mas o "
        f"`outcome_map` (traduzir os rótulos da casa para home/draw/away) e a `line` "
        f"de cada mercado precisam de conferência humana."
    )
    return 0


def cmd_inspect(args: argparse.Namespace, settings: Settings) -> int:
    """Mostra a estrutura de um evento da captura, para montar o mapeamento.

    Imprime apenas o corpo da resposta pública de odds — nunca cabeçalhos,
    cookies ou dados de sessão, que é o que torna um HAR sensível.
    """
    from .discover import (
        extract_from_har,
        market_labels,
        score_candidate,
        summarize,
        walk_arrays,
    )
    from .providers.generic_json import dig

    path = Path(args.captura)
    if not path.exists():
        print(f"Arquivo não encontrado: {path}", file=sys.stderr)
        return 1

    raw = json.loads(path.read_text(encoding="utf-8"))
    payloads = (
        extract_from_har(raw)
        if isinstance(raw, dict) and "entries" in raw.get("log", {})
        else [("", raw)]
    )
    if args.contendo:
        payloads = [(u, p) for u, p in payloads if args.contendo in u]
    if not payloads:
        print(f"Nenhuma resposta com '{args.contendo}' na URL.", file=sys.stderr)
        return 1

    # Uma resposta por URL, com o melhor candidato de cada. Mostrar só o
    # vencedor global esconde o endpoint por evento: a lista geral tem 195
    # jogos e ganha na pontuação, mesmo trazendo menos mercados.
    achados: list[tuple[str, Any]] = []
    for url, payload in payloads:
        candidatos = [c for c in (score_candidate(c) for c in walk_arrays(payload)) if c.items]
        if candidatos:
            achados.append((url, max(candidatos, key=lambda c: c.score)))

    if not achados:
        print("Nenhuma lista de eventos encontrada.", file=sys.stderr)
        return 1

    achados.sort(key=lambda par: len(market_labels(par[1].items[0])), reverse=True)

    print(f"{BOLD}Respostas com jogos nesta captura{RESET} "
          f"(ordenadas por quantidade de mercados):")
    for i, (url, cand) in enumerate(achados):
        mercados = market_labels(cand.items[0])
        curta = url.split("?")[0].rsplit("/", 2)
        curta = "/".join(curta[-2:]) if len(curta) > 1 else url
        print(
            f"  [{i}] {len(cand.items):>4} jogos · {len(mercados):>2} mercados"
            f"  {DIM}…/{curta}{RESET}"
        )
    print(f"\n{DIM}Use --indice N para detalhar outra. Detalhando a [{args.indice}]:{RESET}\n")

    if not 0 <= args.indice < len(achados):
        print(f"Índice fora da faixa (0 a {len(achados) - 1}).", file=sys.stderr)
        return 1

    url, cand = achados[args.indice]
    evento = cand.items[0]

    print(f"{BOLD}Origem{RESET}: {url}")
    print(f"{BOLD}Eventos em{RESET}: {cand.path or '(raiz)'} ({len(cand.items)} no total)\n")

    mercados = market_labels(evento)

    if args.mercado:
        # Detalhe cru de um mercado: é o que permite escrever o outcome_map
        # sem adivinhar qual campo carrega o rótulo.
        alvo = [m for m in mercados if args.mercado.lower() in m.name.lower()]
        if not alvo:
            print(f"Nenhum mercado com '{args.mercado}' no nome.", file=sys.stderr)
            return 1
        for m in alvo[:3]:
            print(f"\n{BOLD}{m.name}{RESET}  {DIM}({m.path}){RESET}")
            selecoes = [
                s
                for s in dig(evento, m.path, [])
                if m.name_field is None or str(dig(s, m.name_field)) == m.name
            ]
            for sel in selecoes[:8]:
                enxuto = {
                    k: v
                    for k, v in sel.items()
                    if k not in {"uuid", "marketUuid", "extra", "tags", "marketTags"}
                }
                print(f"  {json.dumps(enxuto, ensure_ascii=False)}")
            if len(selecoes) > 8:
                print(f"  {DIM}... (+{len(selecoes) - 8} seleções){RESET}")
        return 0

    print(f"{BOLD}Mercados encontrados{RESET} (para montar o outcome_map):")
    uteis = [m for m in mercados if len(m.labels) >= 2]
    for m in uteis[: args.mercados]:
        linha = f"  linha: {m.line}" if m.line else ""
        print(f"  {BOLD}{m.name}{RESET}")
        print(f"    {m.path}.{m.label_field} → {', '.join(m.labels)}{linha}")
    if len(uteis) > args.mercados:
        print(f"  {DIM}... (+{len(uteis) - args.mercados}; use --mercados N){RESET}")
    if not uteis:
        print(f"  {DIM}nenhum rótulo de texto encontrado{RESET}")
    print(f"\n{DIM}Use --mercado \"nome\" para ver as seleções cruas de um deles.{RESET}")

    print(f"\n{BOLD}Estrutura de um evento{RESET}:")
    print(json.dumps(summarize(evento), ensure_ascii=False, indent=2)[: args.limite])
    return 0


def _listar_captura(raw: Any, payloads: list[tuple[str, Any]]) -> int:
    """Mostra tudo que a captura contém, agrupado por domínio.

    Serve para responder uma pergunta anterior à busca: o tráfego que eu queria
    chegou a ser gravado? Se o domínio do painel não aparece aqui, procurar
    campo dentro das respostas nunca ia encontrar nada.

    Imprime endereço sem a query string — é lá que viajam tokens de sessão.
    """
    from urllib.parse import urlparse

    entradas = raw.get("log", {}).get("entries", []) if isinstance(raw, dict) else []
    com_ws = sum(1 for e in entradas if e.get("_webSocketMessages"))

    print(f"{BOLD}A captura tem{RESET}: {len(entradas)} requisição(ões) no total, "
          f"{len(payloads)} resposta(s) em JSON, {com_ws} conexão(ões) WebSocket.\n")

    por_dominio: dict[str, set[str]] = {}
    for url, _ in payloads:
        partes = urlparse(url.split(" [")[0])
        por_dominio.setdefault(partes.netloc or "(sem domínio)", set()).add(partes.path)

    for dominio in sorted(por_dominio, key=lambda d: -len(por_dominio[d])):
        caminhos = sorted(por_dominio[dominio])
        print(f"{BOLD}{dominio}{RESET}  {DIM}({len(caminhos)} endereço(s)){RESET}")
        for caminho in caminhos[:15]:
            print(f"    {caminho}")
        if len(caminhos) > 15:
            print(f"    {DIM}... e mais {len(caminhos) - 15}{RESET}")
        print()

    if not com_ws:
        print(
            f"{DIM}Nenhum WebSocket foi gravado. Se o painel usa um, a conexão já\n"
            f"estava aberta antes da gravação começar — recarregue a página (F5)\n"
            f"com a aba Network já aberta e capture de novo.{RESET}"
        )
    return 0


def cmd_estatisticas(args: argparse.Namespace, settings: Settings) -> int:
    """Procura, numa captura, a resposta que traz as estatísticas ao vivo.

    O `inspect` procura mercados e odds; um painel de estatísticas não tem nem
    uma coisa nem outra, e por isso passava batido. Aqui a busca é pelo nome
    dos campos: chutes, posse, ataques perigosos, xG.

    Imprime apenas nomes de campos e valores numéricos das respostas — nunca
    cabeçalhos nem cookies, que é o que torna um HAR sensível.
    """
    from .discover import (
        TERMOS_DE_ESTATISTICA,
        _normalizar,
        buscar_campos,
        extract_from_har,
        extract_websocket_from_har,
    )

    path = Path(args.captura)
    if not path.exists():
        print(f"Arquivo não encontrado: {path}", file=sys.stderr)
        return 1

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "entries" in raw.get("log", {}):
        payloads = extract_from_har(raw)
        # Painel ao vivo costuma transmitir por WebSocket em vez de fazer
        # requisições; sem olhar os quadros, o dado some da captura.
        quadros = extract_websocket_from_har(raw)
        if quadros:
            print(f"{DIM}{len(quadros)} mensagem(ns) de WebSocket na captura.{RESET}\n")
        payloads = payloads + quadros
    else:
        payloads = [("", raw)]

    if args.listar:
        return _listar_captura(raw, payloads)

    if args.contendo:
        # Domínio já identificado: o que falta é ver o formato da resposta
        # para escrever o mapeamento, e aí a busca por palavra atrapalha —
        # os campos podem se chamar qualquer coisa.
        alvo = [(u, p) for u, p in payloads if args.contendo in u]
        if not alvo:
            print(f"Nenhuma resposta com '{args.contendo}' no endereço.", file=sys.stderr)
            return 1
        from .discover import summarize

        for url, payload in alvo[: args.limite]:
            print(f"{BOLD}{url.split('?')[0]}{RESET}")
            print(json.dumps(summarize(payload), ensure_ascii=False, indent=2))
            print()
        return 0

    termos = [_normalizar(t) for t in (args.termo or [])] or TERMOS_DE_ESTATISTICA
    if args.termo:
        print(f"{DIM}Procurando por: {', '.join(args.termo)}{RESET}\n")

    achados = [(url, campos) for url, p in payloads if (campos := buscar_campos(p, termos))]
    if not achados:
        print(
            "Nenhum campo de estatística nesta captura.\n\n"
            "O painel talvez carregue de outro domínio. No DevTools, deixe o filtro\n"
            "em Fetch/XHR, abra o painel de estatísticas com a aba Network já\n"
            "gravando, e salve de novo. Se ainda assim não vier, o painel pode ser\n"
            "um iframe de terceiro — nesse caso o dado não está ao nosso alcance.",
            file=sys.stderr,
        )
        return 1

    achados.sort(key=lambda par: len(par[1]), reverse=True)
    print(f"{BOLD}Respostas com campos de estatística{RESET} (mais promissoras primeiro):")
    for i, (url, campos) in enumerate(achados[: args.limite]):
        print(f"\n{BOLD}[{i}]{RESET} {len(campos)} campo(s)  {DIM}{url.split('?')[0]}{RESET}")
        vistos: set[str] = set()
        for caminho, valor in campos:
            # Um caminho por formato: numa lista de 195 jogos os campos se
            # repetem, e o que interessa é o nome, não os 195 valores.
            generico = re.sub(r"\.\d+\.", ".N.", caminho)
            if generico in vistos:
                continue
            vistos.add(generico)
            print(f"    {generico:<50} = {valor!r}")
    return 0


def cmd_probe(args: argparse.Namespace, settings: Settings) -> int:
    """Consulta uma URL e descreve o que voltou. Rode da sua própria máquina."""
    import httpx

    from .discover import build_field_map

    ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    try:
        resp = httpx.get(args.url, timeout=20.0, headers={"User-Agent": ua}, follow_redirects=True)
    except httpx.HTTPError as exc:
        print(f"Falha de rede: {exc}", file=sys.stderr)
        return 1

    print(f"HTTP {resp.status_code} · {len(resp.content)} bytes · {resp.headers.get('content-type', '?')}")
    if resp.status_code == 403:
        print(
            f"{YELLOW}403 — pode ser geobloqueio, WAF ou exigência de sessão. "
            f"Se o site só atende o Brasil, isto precisa rodar de um IP brasileiro.{RESET}"
        )
        print(resp.text[:300])
        return 1
    if resp.status_code != 200:
        print(resp.text[:300])
        return 1

    try:
        payload = resp.json()
    except ValueError:
        print(f"{YELLOW}A resposta não é JSON — provavelmente é o HTML da página.{RESET}")
        print("Use o DevTools para achar a chamada XHR que traz os jogos.")
        return 1

    fmap = build_field_map(args.url, payload)
    if not fmap:
        print("JSON válido, mas sem estrutura de eventos+odds reconhecível.")
        return 1

    print(f"Lista de eventos em '{fmap['events_path'] or '(raiz)'}', "
          f"{len(fmap['markets'])} mercados, confiança {fmap['_confianca']:.0%}")
    Path(args.output).write_text(json.dumps(fmap, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rascunho salvo em {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bet-ai",
        description="Análise de jogos de futebol ao vivo com modelo de gols + IA",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-v", "--verbose", action="store_true", help="mostra todas as probabilidades")
        p.add_argument("--no-store", action="store_true", help="não grava no banco")

    live = sub.add_parser("live", help="analisa os jogos ao vivo agora")
    add_common(live)
    live.add_argument(
        "--apenas-valor",
        action="store_true",
        dest="apenas_valor",
        help="mostra só os jogos com aposta de valor",
    )
    live.add_argument("--ai", action="store_true", help="ativa a revisão do Claude")
    live.add_argument("--context", help="contexto externo passado à IA (notícias, desfalques)")
    live.set_defaults(func=cmd_live)

    upcoming = sub.add_parser("upcoming", help="analisa jogos que ainda vão começar")
    add_common(upcoming)
    upcoming.set_defaults(func=cmd_upcoming)

    watch = sub.add_parser("watch", help="reanalisa em intervalo fixo")
    add_common(watch)
    watch.add_argument("--interval", type=int, default=60, help="segundos entre ciclos")
    watch.add_argument(
        "--apenas-valor", action="store_true", dest="apenas_valor",
        help="mostra só os jogos com aposta de valor",
    )
    watch.add_argument("--ai", action="store_true")
    watch.add_argument("--context")
    watch.set_defaults(func=cmd_watch)

    web = sub.add_parser("web", help="abre a interface no navegador")
    add_common(web)
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--interval", type=int, default=60, help="segundos entre coletas")
    web.add_argument(
        "--sem-navegador", action="store_true", dest="sem_navegador",
        help="não abre o navegador automaticamente",
    )
    web.add_argument("--ai", action="store_true", help="ativa a revisão do Claude")
    web.add_argument("--context", help="contexto externo passado à IA")
    web.set_defaults(func=cmd_web)

    history = sub.add_parser("history", help="lista eventos já capturados")
    history.set_defaults(func=cmd_history)

    report = sub.add_parser("report", help="resumo das apostas de valor registradas")
    report.add_argument("--event-id")
    report.add_argument("--limit", type=int, default=20)
    report.set_defaults(func=cmd_report)

    discover = sub.add_parser(
        "discover",
        help="infere um field_map a partir de um HAR ou JSON capturado no DevTools",
    )
    discover.add_argument("captura", help="arquivo .har ou .json salvo do navegador")
    discover.add_argument("--url", help="URL do endpoint (se a captura for um .json solto)")
    discover.add_argument("-o", "--output", default="field_map.json")
    discover.set_defaults(func=cmd_discover)

    inspect = sub.add_parser(
        "inspect",
        help="mostra a estrutura de um evento da captura (só o corpo público das odds)",
    )
    inspect.add_argument("captura", help="arquivo .har ou .json")
    inspect.add_argument("--contendo", help="filtra pelas URLs que contenham este texto")
    inspect.add_argument(
        "--indice", type=int, default=0, help="qual das respostas detalhar (padrão 0)"
    )
    inspect.add_argument("--mercado", help="detalha as seleções cruas do mercado com este nome")
    inspect.add_argument("--mercados", type=int, default=25, help="quantos mercados listar")
    inspect.add_argument("--limite", type=int, default=6000, help="máximo de caracteres")
    inspect.set_defaults(func=cmd_inspect)

    estat = sub.add_parser(
        "estatisticas", help="acha as estatísticas ao vivo (chutes, posse) numa captura"
    )
    estat.add_argument("captura", help="arquivo .har ou .json do DevTools")
    estat.add_argument(
        "--termo", action="append", help="procura por este nome de campo (pode repetir)"
    )
    estat.add_argument("--limite", type=int, default=8, help="quantas respostas detalhar")
    estat.add_argument(
        "--listar",
        action="store_true",
        help="lista tudo que a captura contém, por domínio, sem procurar nada",
    )
    estat.add_argument(
        "--contendo", help="mostra o formato das respostas com este trecho no endereço"
    )
    estat.set_defaults(func=cmd_estatisticas)

    probe = sub.add_parser(
        "probe", help="consulta uma URL e descreve o JSON que voltou (rode da sua máquina)"
    )
    probe.add_argument("url")
    probe.add_argument("-o", "--output", default="field_map.json")
    probe.set_defaults(func=cmd_probe)

    settle = sub.add_parser("settle", help="registra o placar final de um evento")
    settle.add_argument("event_id")
    settle.add_argument("home", type=int)
    settle.add_argument("away", type=int)
    settle.set_defaults(func=cmd_settle)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load()
    try:
        return args.func(args, settings)
    except ProviderError as exc:
        print(f"Erro no provedor: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
