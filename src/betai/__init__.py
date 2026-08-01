"""bet-ai — análise quantitativa de jogos de futebol ao vivo."""

from .models import Analysis, Event, Market, MarketKey, MatchState, Selection, ValueBet
from .pipeline import Pipeline, derive_baseline, market_probabilities, model_probabilities
from .storage import Store

__version__ = "0.1.0"

__all__ = [
    "Event",
    "Market",
    "MarketKey",
    "MatchState",
    "Selection",
    "ValueBet",
    "Analysis",
    "Pipeline",
    "derive_baseline",
    "market_probabilities",
    "model_probabilities",
    "Store",
]
