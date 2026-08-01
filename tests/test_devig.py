import math

import pytest

from betai.engine.devig import margin, remove_vig

# Mercado 1X2 típico com ~6% de margem.
ODDS_1X2 = [2.10, 3.40, 3.60]


@pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
def test_probabilidades_somam_um(method):
    probs = remove_vig(ODDS_1X2, method)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)
    assert all(0 < p < 1 for p in probs)


@pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
def test_preserva_a_ordem_das_selecoes(method):
    probs = remove_vig(ODDS_1X2, method)
    # Odd menor tem que virar probabilidade maior, em qualquer método.
    assert probs[0] > probs[1]
    assert probs[1] < probs[2] or math.isclose(probs[1], probs[2], abs_tol=0.02)


@pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
def test_mercado_sem_margem_fica_inalterado(method):
    fair = [4.0, 4.0, 4.0, 4.0]
    probs = remove_vig(fair, method)
    assert all(math.isclose(p, 0.25, abs_tol=1e-6) for p in probs)


def test_metodos_discordam_no_azarao():
    """O motivo de existirem quatro métodos: eles precificam a cauda diferente."""
    longshot = [1.20, 8.00, 15.00]
    mult = remove_vig(longshot, "multiplicative")
    add = remove_vig(longshot, "additive")
    # O aditivo tira margem absoluta igual de todos, o que corta mais do azarão
    # em termos relativos.
    assert add[2] < mult[2]


def test_margem_calculada():
    assert margin([2.0, 2.0]) == pytest.approx(0.0, abs=1e-9)
    assert margin([1.90, 1.90]) == pytest.approx(0.0526, abs=1e-3)


def test_odds_invalidas_rejeitadas():
    with pytest.raises(ValueError):
        remove_vig([1.0, 2.0])
    with pytest.raises(ValueError):
        remove_vig([0.5, 2.0])
