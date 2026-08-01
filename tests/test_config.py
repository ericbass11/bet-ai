"""Testes da configuração."""

import pytest

from betai.config import Settings, load, load_dotenv


@pytest.fixture(autouse=True)
def ambiente_limpo(monkeypatch):
    for chave in list(__import__("os").environ):
        if chave.startswith("BETAI_"):
            monkeypatch.delenv(chave, raising=False)


def test_padroes_sem_configuracao(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = load()
    assert s.provider == "mock"
    assert s.devig_method == "power"
    assert s.min_edge == 0.03


def test_le_o_arquivo_env(tmp_path, monkeypatch):
    """Eu documentava `cp .env.example .env` e nada lia o arquivo."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "BETAI_PROVIDER=generic_json\n"
        "BETAI_FIELD_MAP=examples/superbet.json\n"
        "BETAI_MIN_EDGE=0.05\n",
        encoding="utf-8",
    )
    s = load()
    assert s.provider == "generic_json"
    assert s.field_map_path == "examples/superbet.json"
    assert s.min_edge == 0.05


def test_variavel_exportada_vence_o_arquivo(tmp_path, monkeypatch):
    """Quem exporta na mão quer testar algo pontual sem editar o .env."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("BETAI_PROVIDER=generic_json\n", encoding="utf-8")
    monkeypatch.setenv("BETAI_PROVIDER", "mock")
    assert load().provider == "mock"


def test_ignora_comentarios_e_linhas_vazias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "# comentário\n\n  \nBETAI_SPORT=soccer_brazil_campeonato\nlinha sem igual\n",
        encoding="utf-8",
    )
    assert load().sport == "soccer_brazil_campeonato"


def test_remove_aspas_do_valor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('BETAI_MODEL="claude-opus-5"\n', encoding="utf-8")
    assert load().model == "claude-opus-5"


def test_sem_arquivo_nao_quebra(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_dotenv() == {}


def test_settings_le_o_ambiente_na_instanciacao(monkeypatch):
    """Se os padrões fossem avaliados na importação, o .env nunca teria efeito
    e mudanças no ambiente seriam ignoradas depois do primeiro import."""
    monkeypatch.setenv("BETAI_PROVIDER", "the_odds_api")
    assert Settings().provider == "the_odds_api"
    monkeypatch.setenv("BETAI_PROVIDER", "generic_json")
    assert Settings().provider == "generic_json"
