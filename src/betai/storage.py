"""Persistência em SQLite: snapshots de odds e análises ao longo do tempo.

A série temporal é o ativo mais valioso do projeto. Uma odd isolada não diz
muito; a trajetória dela ao longo do jogo mostra como o mercado reagiu, e é
o que permite depois medir se o modelo estava certo (backtest) e recalibrar
os coeficientes ao vivo.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from .models import Analysis, Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id      TEXT    NOT NULL,
    source        TEXT    NOT NULL,
    league        TEXT    NOT NULL,
    home_team     TEXT    NOT NULL,
    away_team     TEXT    NOT NULL,
    captured_at   TEXT    NOT NULL,
    minute        INTEGER NOT NULL,
    score_home    INTEGER NOT NULL,
    score_away    INTEGER NOT NULL,
    payload       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_event ON snapshots(event_id, captured_at);

CREATE TABLE IF NOT EXISTS analyses (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       TEXT    NOT NULL,
    created_at     TEXT    NOT NULL,
    minute         INTEGER NOT NULL,
    lambda_home    REAL    NOT NULL,
    lambda_away    REAL    NOT NULL,
    probabilities  TEXT    NOT NULL,
    value_bets     TEXT    NOT NULL,
    ai_summary     TEXT
);

CREATE INDEX IF NOT EXISTS idx_analyses_event ON analyses(event_id, created_at);

CREATE TABLE IF NOT EXISTS results (
    event_id    TEXT PRIMARY KEY,
    score_home  INTEGER NOT NULL,
    score_away  INTEGER NOT NULL,
    settled_at  TEXT    NOT NULL
);
"""


class Store:
    """Camada de persistência. Use como context manager."""

    def __init__(self, path: str | Path = "bet-ai.db") -> None:
        self.path = str(path)
        # `check_same_thread=False` porque a interface web coleta numa thread
        # e o processo principal serve noutra. O sqlite3 proíbe isso por
        # padrão para evitar corrida — aqui a corrida é evitada pelo lock
        # abaixo, que serializa todo acesso à conexão.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, closing(self.conn.cursor()) as cur:
            cur.executescript(SCHEMA)
            self.conn.commit()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ---------- escrita ----------

    def save_snapshot(self, event: Event) -> int:
        """Grava o estado + odds de um evento num instante."""
        with self._lock, closing(self.conn.cursor()) as cur:
            cur.execute(
                """INSERT INTO snapshots
                   (event_id, source, league, home_team, away_team, captured_at,
                    minute, score_home, score_away, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.source,
                    event.league,
                    event.home_team,
                    event.away_team,
                    event.captured_at.isoformat(),
                    event.state.minute,
                    event.state.score_home,
                    event.state.score_away,
                    event.model_dump_json(),
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid or 0)

    def save_analysis(self, analysis: Analysis) -> int:
        with self._lock, closing(self.conn.cursor()) as cur:
            cur.execute(
                """INSERT INTO analyses
                   (event_id, created_at, minute, lambda_home, lambda_away,
                    probabilities, value_bets, ai_summary)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis.event.event_id,
                    datetime.now(timezone.utc).isoformat(),
                    analysis.event.state.minute,
                    analysis.lambda_home,
                    analysis.lambda_away,
                    json.dumps(analysis.probabilities),
                    json.dumps([b.model_dump() for b in analysis.value_bets]),
                    analysis.ai_summary,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid or 0)

    def save_result(self, event_id: str, score_home: int, score_away: int) -> None:
        """Registra o placar final. É o que fecha o ciclo para backtest."""
        with self._lock, closing(self.conn.cursor()) as cur:
            cur.execute(
                """INSERT OR REPLACE INTO results (event_id, score_home, score_away, settled_at)
                   VALUES (?, ?, ?, ?)""",
                (event_id, score_home, score_away, datetime.now(timezone.utc).isoformat()),
            )
            self.conn.commit()

    # ---------- leitura ----------

    def snapshots_for(self, event_id: str) -> list[Event]:
        """Linha do tempo completa de um evento, em ordem de captura."""
        with self._lock, closing(self.conn.cursor()) as cur:
            rows = cur.execute(
                "SELECT payload FROM snapshots WHERE event_id = ? ORDER BY captured_at",
                (event_id,),
            ).fetchall()
        return [Event.model_validate_json(row["payload"]) for row in rows]

    def tracked_events(self) -> list[sqlite3.Row]:
        """Um registro por evento visto, com contagem de snapshots."""
        with self._lock, closing(self.conn.cursor()) as cur:
            return cur.execute(
                """SELECT event_id, league, home_team, away_team,
                          COUNT(*) AS snapshots,
                          MIN(captured_at) AS first_seen,
                          MAX(captured_at) AS last_seen
                   FROM snapshots
                   GROUP BY event_id
                   ORDER BY last_seen DESC"""
            ).fetchall()

    def result_for(self, event_id: str) -> tuple[int, int] | None:
        """Placar final de um evento, ou `None` se ainda não foi registrado."""
        with self._lock, closing(self.conn.cursor()) as cur:
            row = cur.execute(
                "SELECT score_home, score_away FROM results WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return (row["score_home"], row["score_away"]) if row else None

    # Depois de quanto tempo sem notícia um jogo é dado por encerrado. Uma
    # partida dura ~2h; sumir do feed por 3 horas não deixa outra leitura.
    ABANDONO_HORAS = 3

    def settle_finished(self, agora: datetime | None = None) -> list[tuple[str, int, int]]:
        """Anota o placar final dos jogos que terminaram. Devolve os anotados.

        O placar final não precisa ser buscado: o programa já viu o jogo em
        todos os minutos, e o último estado observado *é* o resultado. Hoje
        isso era descartado, e sem resultado não há como medir se o modelo
        acerta — que é a única pergunta que importa no fim.

        Um jogo é dado por encerrado quando a casa diz que encerrou, ou quando
        some do feed por horas. O segundo caso existe porque nem toda casa
        publica o status final: o jogo simplesmente desaparece da lista.
        """
        agora = agora or datetime.now(timezone.utc)
        limite = (agora - timedelta(hours=self.ABANDONO_HORAS)).isoformat()

        with self._lock, closing(self.conn.cursor()) as cur:
            rows = cur.execute(
                """SELECT s.event_id, s.payload, s.captured_at
                     FROM snapshots s
                     JOIN (SELECT event_id, MAX(captured_at) AS ultimo
                             FROM snapshots GROUP BY event_id) u
                       ON u.event_id = s.event_id AND u.ultimo = s.captured_at
                    WHERE s.event_id NOT IN (SELECT event_id FROM results)"""
            ).fetchall()

        anotados: list[tuple[str, int, int]] = []
        for row in rows:
            evento = Event.model_validate_json(row["payload"])
            encerrado = evento.state.period == "encerrado"
            sumiu = row["captured_at"] < limite and evento.state.minute > 0
            if not (encerrado or sumiu):
                continue
            anotados.append(
                (row["event_id"], evento.state.score_home, evento.state.score_away)
            )

        for event_id, casa, fora in anotados:
            self.save_result(event_id, casa, fora)
        return anotados

    def settled_analyses(self) -> list[dict]:
        """Análises de jogos cujo placar final já é conhecido.

        É a matéria-prima da medição: de um lado o que o modelo disse, do
        outro o que aconteceu. Análise de jogo sem resultado fica de fora —
        não dá para pontuar o que ainda não terminou.
        """
        with self._lock, closing(self.conn.cursor()) as cur:
            rows = cur.execute(
                """SELECT a.event_id, a.created_at, a.minute, a.probabilities,
                          a.value_bets, r.score_home, r.score_away
                     FROM analyses a
                     JOIN results r ON r.event_id = a.event_id
                    ORDER BY a.event_id, a.created_at"""
            ).fetchall()

        return [
            {
                "event_id": row["event_id"],
                "created_at": row["created_at"],
                "minute": row["minute"],
                "probabilities": json.loads(row["probabilities"]),
                "value_bets": json.loads(row["value_bets"]),
                "score": (row["score_home"], row["score_away"]),
            }
            for row in rows
        ]

    def value_bet_history(self, event_id: str | None = None) -> list[dict]:
        """Todas as apostas de valor registradas, com o placar final se houver.

        Devolve uma lista, não um gerador: um gerador manteria o lock aberto
        durante a iteração, e qualquer chamada ao banco feita pelo consumidor
        no meio do laço travaria o processo.
        """
        query = """SELECT a.event_id, a.created_at, a.minute, a.value_bets,
                          r.score_home, r.score_away
                   FROM analyses a
                   LEFT JOIN results r ON r.event_id = a.event_id"""
        params: tuple = ()
        if event_id:
            query += " WHERE a.event_id = ?"
            params = (event_id,)
        query += " ORDER BY a.created_at"

        with self._lock, closing(self.conn.cursor()) as cur:
            rows = cur.execute(query, params).fetchall()

        return [
            {
                "event_id": row["event_id"],
                "created_at": row["created_at"],
                "minute": row["minute"],
                "final_score": (
                    None
                    if row["score_home"] is None
                    else (row["score_home"], row["score_away"])
                ),
                **bet,
            }
            for row in rows
            for bet in json.loads(row["value_bets"])
        ]
