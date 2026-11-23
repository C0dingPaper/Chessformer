"""Validated records and deterministic game-level dataset splitting."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chess

Split = Literal["train", "validation", "test"]
SPLITS: tuple[Split, ...] = ("train", "validation", "test")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetConfig:
    max_plies: int = 40
    seed: str = "chessformer-v1"
    validation_fraction: float = 0.1
    test_fraction: float = 0.1

    def __post_init__(self) -> None:
        if type(self.max_plies) is not int or self.max_plies <= 0:
            raise ValueError("max_plies must be a positive integer")
        if not isinstance(self.seed, str) or not self.seed:
            raise ValueError("seed must be a nonempty string")
        for fraction in (self.validation_fraction, self.test_fraction):
            if (
                isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction)
                or not 0 <= fraction < 1
            ):
                raise ValueError("split fractions must be finite and in [0, 1)")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("split fractions must leave a nonempty training fraction")


@dataclass(frozen=True)
class MoveRecord:
    """A position BEFORE its target move; ply is relative to the game's start.
    Result is a label/metadata only. It must never become a prediction feature.
    """

    game_id: str
    split: Split
    ply: int
    fen: str
    move_uci: str
    player_elo: int
    opponent_elo: int
    result: str
    time_control: str

    def __post_init__(self) -> None:
        for name in ("game_id", "fen", "move_uci", "result", "time_control"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split: {self.split!r}")
        if type(self.ply) is not int or self.ply < 0:
            raise ValueError("ply must be a nonnegative integer")
        for elo in (self.player_elo, self.opponent_elo):
            if type(elo) is not int or elo <= 0:
                raise ValueError("ratings must be positive integers")
        if self.result not in ("1-0", "0-1", "1/2-1/2"):
            raise ValueError("result must describe a finished game")
        board = chess.Board(self.fen)
        if not board.is_valid():
            raise ValueError("record contains an invalid standard chess position")
        move = chess.Move.from_uci(self.move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal target move {self.move_uci!r}")


def position_key(board: chess.Board) -> str:
    """Merge opening transpositions, preserving turn, castling and legal EP.

    Move counters/history are intentionally excluded: this baseline predicts
    opening moves and does not adjudicate repetition or fifty-move draws.
    """
    return " ".join(board.fen(en_passant="legal").split()[:4])


def split_for_game(game_id: str, config: DatasetConfig) -> Split:
    """Stable across processes, input order and Python versions."""
    encoded = json.dumps([config.seed, game_id], ensure_ascii=True).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    fraction = value / 2**64
    if fraction < config.test_fraction:
        return "test"
    if fraction < config.test_fraction + config.validation_fraction:
        return "validation"
    return "train"


def read_records(path: Path) -> Iterator[MoveRecord]:
    """Stream validated JSONL, identifying the source line on bad input."""
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object")
                yield MoveRecord(**payload)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
