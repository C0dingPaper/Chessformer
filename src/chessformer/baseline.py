"""A reproducible, rating-conditioned move-frequency baseline."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chess

from chessformer.schema import MoveRecord, position_key

_FORMAT = "chessformer.frequency-policy"
_VERSION = 1


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _record_board(record: MoveRecord) -> chess.Board:
    _positive_int(record.player_elo, "player_elo")
    try:
        board = chess.Board(record.fen)
        move = chess.Move.from_uci(record.move_uci)
    except ValueError as exc:
        raise ValueError(f"Invalid position or move in game {record.game_id}") from exc
    if not board.is_valid() or move not in board.legal_moves:
        raise ValueError(f"Illegal position or move in game {record.game_id}")
    return board


class FrequencyPolicy:
    """Predict legal moves using rating-band counts and explicit backoff.

    ``fit`` replaces the model only after the entire training iterable validates.
    Positions ignore move clocks; turn, castling and legal en passant are retained.
    Smoothing must be finite and at least 1e-12 to avoid numerical underflow.
    """

    def __init__(self, bucket_width: int = 200, smoothing: float = 1.0) -> None:
        self.bucket_width = _positive_int(bucket_width, "bucket_width")
        if (
            isinstance(smoothing, bool)
            or not isinstance(smoothing, (int, float))
            or not math.isfinite(smoothing)
            or smoothing < 1e-12
        ):
            raise ValueError("smoothing must be finite and at least 1e-12")
        self.smoothing = float(smoothing)
        self._counts: dict[str, dict[int, Counter[str]]] = {}

    def _band(self, elo: int) -> int:
        return _positive_int(elo, "elo") // self.bucket_width * self.bucket_width

    def fit(self, records: Iterable[MoveRecord]) -> None:
        counts: dict[str, dict[int, Counter[str]]] = {}
        for record in records:
            if record.split != "train":
                raise ValueError("FrequencyPolicy.fit accepts only training records")
            board = _record_board(record)
            bands = counts.setdefault(position_key(board), {})
            moves = bands.setdefault(self._band(record.player_elo), Counter())
            moves[record.move_uci] += 1
        if not counts:
            raise ValueError("Cannot fit a policy without training records")
        self._counts = counts

    def _lookup(self, board: chess.Board, elo: int) -> tuple[Counter[str], str]:
        band = self._band(elo)
        if not board.is_valid():
            raise ValueError("Cannot predict from an invalid chess position")
        bands = self._counts.get(position_key(board), {})
        if band in bands:
            return bands[band], "rating_band"
        if bands:
            counts: Counter[str] = Counter()
            for values in bands.values():
                counts.update(values)
            return counts, "position"
        return Counter(), "uniform"

    def predict(self, board: chess.Board, elo: int) -> dict[str, float]:
        """Return every legal move, ordered by probability then UCI string."""
        counts, _ = self._lookup(board, elo)
        moves = [move.uci() for move in board.legal_moves]
        if not moves:
            return {}
        # Scaling avoids an infinite denominator for large, finite smoothing.
        scale = max(self.smoothing, max((counts[move] for move in moves), default=0))
        weights = {
            move: counts[move] / scale + self.smoothing / scale for move in moves
        }
        total = sum(weights.values())
        return dict(
            sorted(
                ((move, weight / total) for move, weight in weights.items()),
                key=lambda item: (-item[1], item[0]),
            )
        )

    def support(self, board: chess.Board, elo: int) -> dict[str, object]:
        """Describe data coverage; counts are observations, not confidence bounds."""
        counts, source = self._lookup(board, elo)
        band = self._band(elo)
        return {
            "source": source,
            "observations": sum(counts.values()),
            "rating_band_lower": band,
            "rating_band_upper_exclusive": band + self.bucket_width,
        }

    def save(self, path: Path) -> None:
        payload = {
            "format": _FORMAT,
            "schema_version": _VERSION,
            "bucket_width": self.bucket_width,
            "smoothing": self.smoothing,
            "counts": {
                key: {str(band): dict(moves) for band, moves in bands.items()}
                for key, bands in self._counts.items()
            },
        }
        serialized = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        with path.open("x", encoding="utf-8") as target:
            target.write(serialized)

    @classmethod
    def load(cls, path: Path) -> FrequencyPolicy:
        """Load and validate a versioned JSON artifact without executing code."""
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or raw.get("format") != _FORMAT
            or type(raw.get("schema_version")) is not int
            or raw["schema_version"] != _VERSION
        ):
            raise ValueError("Unsupported frequency-policy artifact format or version")
        if "bucket_width" not in raw or "smoothing" not in raw:
            raise ValueError("Model artifact is missing its configuration")
        policy = cls(raw["bucket_width"], raw["smoothing"])
        if not isinstance(raw.get("counts"), dict):
            raise ValueError("Model artifact must contain position counts")
        for key, bands in raw["counts"].items():
            try:
                board = chess.Board(key + " 0 1")
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid position key in model artifact") from exc
            if not board.is_valid() or position_key(board) != key:
                raise ValueError(
                    "Invalid or noncanonical position key in model artifact"
                )
            if not isinstance(bands, dict) or not bands:
                raise ValueError("Position counts must contain nonempty rating bands")
            parsed_bands: dict[int, Counter[str]] = {}
            for label, moves in bands.items():
                try:
                    band = int(label)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Invalid rating band in model artifact") from exc
                if (
                    str(band) != label
                    or band < 0
                    or band % policy.bucket_width
                    or not isinstance(moves, dict)
                    or not moves
                ):
                    raise ValueError("Invalid rating band or move counts in artifact")
                parsed_moves: Counter[str] = Counter()
                for uci, count in moves.items():
                    try:
                        move = chess.Move.from_uci(uci)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("Invalid move in model artifact") from exc
                    if move not in board.legal_moves:
                        raise ValueError("Illegal move in model artifact")
                    parsed_moves[uci] = _positive_int(count, "move count")
                parsed_bands[band] = parsed_moves
            policy._counts[key] = parsed_bands
        return policy


class _Metrics:
    def __init__(self) -> None:
        self.count = 0
        self.top1 = 0
        self.top3 = 0
        self.nll = 0.0

    def add(self, predictions: dict[str, float], target: str) -> None:
        ranking = list(predictions)
        self.count += 1
        self.top1 += int(ranking[0] == target)
        self.top3 += int(target in ranking[:3])
        probability = predictions[target]
        self.nll += -math.log(probability) if probability else math.inf

    def report(self) -> dict[str, int | float | None]:
        return {
            "positions": self.count,
            "top1_accuracy": self.top1 / self.count if self.count else None,
            "top3_accuracy": self.top3 / self.count if self.count else None,
            "negative_log_likelihood": self.nll / self.count if self.count else None,
        }


def evaluate(
    policy: FrequencyPolicy, records: Iterable[MoveRecord]
) -> dict[str, object]:
    """Score held-out labels without updating counts or using outcome labels."""
    overall = _Metrics()
    by_band: dict[int, _Metrics] = {}
    coverage: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    for record in records:
        if record.split not in ("validation", "test"):
            raise ValueError("Evaluation accepts only validation or test records")
        board = _record_board(record)
        predictions = policy.predict(board, record.player_elo)
        overall.add(predictions, record.move_uci)
        band = policy._band(record.player_elo)
        by_band.setdefault(band, _Metrics()).add(predictions, record.move_uci)
        source = str(policy.support(board, record.player_elo)["source"])
        coverage[source] += 1
        splits[record.split] += 1
    return {
        **overall.report(),
        "by_rating_band": {
            f"{band}-{band + policy.bucket_width - 1}": metrics.report()
            for band, metrics in sorted(by_band.items())
        },
        "support_sources": {
            source: coverage[source]
            for source in ("rating_band", "position", "uniform")
        },
        "splits": dict(sorted(splits.items())),
    }
