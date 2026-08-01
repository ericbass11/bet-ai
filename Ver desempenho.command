#!/bin/bash
# Clique duas vezes para ver se o modelo está acertando.

cd "$(dirname "$0")" || exit 1

if [ ! -f .venv/bin/activate ]; then
  echo "A instalação não está completa nesta pasta."
  echo "Pressione Enter para fechar."
  read -r
  exit 1
fi

source .venv/bin/activate
git pull --quiet 2>/dev/null

echo
bet-ai desempenho
echo
echo "Pressione Enter para fechar."
read -r
