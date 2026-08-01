#!/bin/bash
# Clique duas vezes neste arquivo no Finder para abrir o bet-ai.
#
# Existe porque o programa é usado por quem não programa: sem isto seriam
# quatro comandos decorados na ordem certa, e um deles (`source`) falha
# calado quando rodado da pasta errada.

cd "$(dirname "$0")" || exit 1

if [ ! -f .venv/bin/activate ]; then
  echo "A instalação não está completa nesta pasta."
  echo "Falta o ambiente do Python (.venv)."
  echo
  echo "Pressione Enter para fechar."
  read -r
  exit 1
fi

source .venv/bin/activate

echo "Buscando atualizações..."
git pull --quiet 2>/dev/null || echo "(sem internet para atualizar — seguindo com a versão local)"

# Uma interface anterior esquecida aberta ocupa a porta e derruba esta.
if pgrep -f "bet-ai web" > /dev/null; then
  echo "Encerrando a interface anterior..."
  pkill -f "bet-ai web"
  sleep 1
fi

echo
echo "Abrindo o bet-ai. Deixe esta janela aberta."
echo "Para parar: feche a janela ou pressione Control+C."
echo

bet-ai web

echo
echo "Encerrado. Pressione Enter para fechar."
read -r
