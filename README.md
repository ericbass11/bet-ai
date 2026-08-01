# bet-ai

Análise quantitativa de jogos de futebol ao vivo. Captura os metadados e as
odds de um provedor de apostas, modela a probabilidade de cada evento com um
modelo de gols, compara com o que a casa está pagando e sinaliza onde há
divergência.

A IA entra no fim da cadeia, não no começo: o número sai do modelo estatístico;
o Claude ajusta pontualmente o que o modelo não consegue enxergar (desfalques,
contexto de tabela, notícia de última hora), dentro de um teto de ajuste.

## De onde vem a vantagem

O ponto que determina se o projeto tem alguma chance:

**O mercado pré-jogo é eficiente.** Ele agrega mais informação do que qualquer
modelo caseiro. Discordar dele sem informação privada não é edge, é ruído. Por
isso o sistema **calibra o modelo pelas odds pré-jogo de propósito** — copia o
mercado onde o mercado é bom.

**O mercado ao vivo é mais fraco.** A casa precisa reprecificar centenas de
jogos em tempo real, e o modelo ao vivo dela costuma ser mais simples do que o
pré-jogo: decaimento aproximado do tempo, tratamento grosseiro de placar e
cartão. É aí que se busca divergência.

Traduzindo para o código: `derive_baseline()` copia o mercado, `live_matrix()`
diverge dele.

## Pipeline

```
odds da casa
    │
    ├─ devig ─────────────► probabilidade justa do mercado
    │                        (multiplicativo | aditivo | power | shin)
    │
    ├─ baseline pré-jogo ─► λ mandante, λ visitante para 90 minutos
    │                        (calibrado por bisseção em total × supremacia)
    │
    ├─ modelo ao vivo ────► matriz de placares finais
    │                        tempo restante × placar × cartões × xG
    │
    ├─ mercados ──────────► 1X2, dupla chance, over/under, ambas marcam,
    │                        handicap asiático, placar exato
    │
    ├─ comparação ────────► edge, EV, Kelly fracionado
    │
    └─ IA (opcional) ─────► ajuste delimitado + justificativa
```

## Instalação

Requer Python 3.10+.

```bash
git clone https://github.com/ericbass11/bet-ai.git
cd bet-ai
git checkout claude/football-analysis-ai-5kc8mn

uv venv && uv pip install -e ".[dev]"
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Sem `uv`, o equivalente com as ferramentas padrão:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip          # necessário: instalação editável exige pip ≥ 21.3
pip install -e ".[dev]"
```

O upgrade do pip não é opcional. Versões anteriores à 21.3 não implementam a
PEP 660 e recusam projetos sem `setup.py`, com a mensagem enganosa
*"Directory cannot be installed in editable mode"* — o diretório está certo,
o pip é que é velho. O venv do macOS costuma vir com uma dessas.

Confira que ficou de pé:

```bash
pytest        # 106 testes, sem rede
bet-ai live   # jogos sintéticos do provedor mock
```

## Uso

O provedor padrão é o `mock`, que gera jogos sintéticos coerentes — dá para
rodar tudo sem chave de API e sem rede:

```bash
bet-ai upcoming            # jogos que ainda vão começar
bet-ai live                # jogos ao vivo, com as apostas de valor
bet-ai live -v             # inclui a tabela modelo × mercado
bet-ai watch --interval 60 # reanalisa a cada 60s, gravando a série temporal
bet-ai history             # eventos já capturados no banco
bet-ai settle mock-0 2 1   # registra o placar final
bet-ai report              # resumo das apostas de valor registradas
bet-ai discover captura.har # infere o field_map de uma casa a partir do DevTools
```

Com a camada de IA:

```bash
export ANTHROPIC_API_KEY=...
bet-ai live --ai --context "Mandante sem o artilheiro (suspenso). Visitante já classificado."
```

O `--context` é o que a IA tem de novo em relação ao modelo. Sem contexto, ela
deve devolver ajustes zerados — e o prompt instrui exatamente isso.

## Provedores

| Provedor | Quando usar |
|---|---|
| `mock` | Desenvolvimento e testes. Sem rede, determinístico. |
| `the_odds_api` | Dados reais. API licenciada, com plano gratuito. É o caminho recomendado. |
| `generic_json` | Endpoint JSON de uma casa específica, mapeado por arquivo de configuração. |

```bash
export BETAI_PROVIDER=the_odds_api
export THE_ODDS_API_KEY=...
export BETAI_SPORT=soccer_brazil_campeonato
bet-ai live
```

Para o `generic_json`, aponte `BETAI_FIELD_MAP` para um arquivo como
`examples/field_map.json`, que descreve onde cada campo vive no JSON do
provedor. O provedor verifica o `robots.txt` e aplica rate limit por padrão.

### Descobrindo o mapeamento de uma casa

Escrever o `field_map.json` na mão exige rastrear cada campo dentro do JSON da
casa. O comando `discover` faz isso por inferência:

```bash
# 1. No navegador, abra a página de jogos ao vivo da casa.
# 2. DevTools → Network → filtre por Fetch/XHR
# 3. Botão direito → "Save all as HAR with content"
bet-ai discover captura.har -o examples/minha_casa.json
```

Ele encontra qual das respostas capturadas traz os jogos (descartando
telemetria, imagens e afins), deduz onde estão times, placar, minuto e odds, e
salva um rascunho com nível de confiança. Também aceita um `.json` solto
copiado do painel Response.

O que a inferência **não** resolve sozinho, e você precisa conferir:

- `outcome_map` — traduzir os rótulos da casa (`"1"`, `"Mais de 2.5"`) para o
  vocabulário interno (`home`, `over`);
- `key` de cada mercado — é chutada pelo texto dos rótulos;
- `line` — over/under e handicap precisam da linha explícita.

Se você estiver num IP com acesso ao site, `bet-ai probe <url>` consulta o
endpoint direto e faz a mesma inferência sem passar pelo HAR.

### Geobloqueio

Casas brasileiras reguladas (`.bet.br`) restringem acesso por região — é
exigência legal, não antibot. De fora do Brasil a resposta é `403` no edge do
CDN, antes de qualquer verificação de `robots.txt`:

```
HTTP/2 403 · server: CloudFront
"Our location services have detected you are in a country
 that this site does not offer its services to."
```

Não há configuração no bet-ai que contorne isso, e contornar seria burlar um
controle de acesso deliberado. A captura precisa ser feita de dentro do país.
É exatamente para esse caso que existe o fluxo via HAR: você captura no seu
navegador, o `discover` monta o mapa.

**Sobre coletar de sites de apostas:** a maioria das casas proíbe coleta
automatizada nos Termos de Uso. O código respeita `robots.txt` e espaça as
requisições, mas a decisão de apontá-lo para um site específico — e a
responsabilidade por ela — é sua. Onde existir API licenciada, use a API.

## Configuração

Tudo por variável de ambiente (veja `.env.example`):

| Variável | Padrão | O que faz |
|---|---|---|
| `BETAI_PROVIDER` | `mock` | `mock`, `the_odds_api` ou `generic_json` |
| `BETAI_DEVIG` | `power` | Método de remoção de margem |
| `BETAI_PRIOR_TOTAL` | `2.7` | Gols esperados quando não há mercado de over/under |
| `BETAI_MIN_EDGE` | `0.03` | Edge mínimo para reportar uma aposta |
| `BETAI_KELLY_FRACTION` | `0.25` | Fração de Kelly (¼ por padrão) |
| `BETAI_MAX_AI_SHIFT` | `0.05` | Teto do ajuste da IA, em pontos de probabilidade |
| `BETAI_MODEL` | `claude-opus-5` | Modelo usado na revisão |
| `BETAI_DB` | `./bet-ai.db` | Caminho do SQLite |

## Decisões de modelagem

**Dixon-Coles em vez de Poisson puro.** Poisson independente subestima 0-0,
1-0, 0-1 e 1-1 — os quatro placares mais comuns do futebol. A correção `tau`
conserta exatamente essas quatro células.

**Calibração por total × supremacia.** Em vez de otimizar (λh, λa)
diretamente, reparametriza-se para `total = λh + λa` e `supremacia = λh − λa`.
O total é monotônico na probabilidade de over, e a supremacia é monotônica em
`P(casa) − P(fora)` — duas bisseções alternadas resolvem sem dependência de
otimizador numérico.

**Intensidade de gols crescente.** Gols não se distribuem uniformemente pelos
90 minutos; a taxa sobe ao longo do jogo. No intervalo ainda resta ~56% dos
gols esperados, não 50%.

**Kelly fracionado por padrão.** Kelly cheio maximiza o crescimento
logarítmico assumindo que a probabilidade é *conhecida*. Aqui ela é
*estimada*, e qualquer erro de estimativa vira aposta grande demais. ¼ de
Kelly é o padrão de uso.

**Todos os coeficientes ao vivo são heurísticas.** Os valores em `LiveConfig`
(impacto de cartão vermelho, efeito de placar, peso do xG) são pontos de
partida plausíveis, não verdades calibradas. O ponto de ter o SQLite gravando
snapshots é poder recalibrá-los contra o seu próprio histórico.

## Limitações conhecidas

- **Sem baseline pré-jogo, não há vantagem.** Se o sistema só vê o jogo já em
  andamento, ele inverte as odds ao vivo para obter o baseline — e aí concorda
  com o mercado por construção. A análise reporta `baseline_source:
  live_inverted` em vez de fingir que encontrou valor. Rode `bet-ai upcoming`
  antes dos jogos começarem.
- **A `the_odds_api` não fornece placar ao vivo.** Sem feed de placar, o motor
  trata o jogo como pré-jogo. Para análise ao vivo real, é preciso combinar
  com um provedor de placar.
- **Sem backtest automatizado ainda.** A infraestrutura está no lugar (o
  schema grava snapshots, análises e resultados), mas o cálculo de ROI
  histórico não foi implementado. `bet-ai report` mostra as apostas
  registradas e o edge médio, não o retorno realizado.
- **Os coeficientes ao vivo não foram calibrados** contra dados reais.

## Testes

```bash
pytest
```

106 testes cobrindo os quatro métodos de devig, a matriz de placares e todos os
mercados derivados dela, a calibração ida-e-volta, o ajuste ao vivo, o
dimensionamento por Kelly, os provedores, a inferência de mapeamento e o
pipeline ponta a ponta.

## Aviso

Isto é uma ferramenta de análise, não uma recomendação de aposta. Modelo com
edge positivo não é lucro garantido: a estimativa pode estar errada, a casa
pode limitar a conta, e variância de curto prazo destrói bankroll mal
dimensionado. Aposte apenas o que puder perder.
