"""Behavioral checks for the statistical baseline and held-out evaluation."""

import json
import math
from dataclasses import replace
from pathlib import Path

import chess
import pytest

from chessformer.baseline import FrequencyPolicy, evaluate
from chessformer.schema import MoveRecord


def record(
    move: str = "e2e4", elo: int = 1200, fen: str = chess.STARTING_FEN
) -> MoveRecord:
    return MoveRecord(
        game_id="sample",
        split="train",
        ply=0,
        fen=fen,
        move_uci=move,
        player_elo=elo,
        opponent_elo=1400,
        result="1-0",
        time_control="600+0",
    )


def test_ratings_change_ordering_and_counts_back_off() -> None:
    model = FrequencyPolicy()
    model.fit([record("e2e4", 1200)] * 3 + [record("d2d4", 1800)] * 2)
    board = chess.Board()
    assert next(iter(model.predict(board, 1250))) == "e2e4"
    assert next(iter(model.predict(board, 1850))) == "d2d4"
    assert next(iter(model.predict(board, 2400))) == "e2e4"
    assert model.support(board, 1250)["observations"] == 3
    assert model.support(board, 2400)["source"] == "position"
    assert model.support(board, 2400)["observations"] == 5


def test_uniform_fallback_is_normalized_legal_and_deterministic() -> None:
    model = FrequencyPolicy()
    board = chess.Board()
    predictions = model.predict(board, 1000)
    assert list(predictions) == sorted(predictions)
    assert set(predictions) == {move.uci() for move in board.legal_moves}
    assert sum(predictions.values()) == pytest.approx(1)
    assert all(value == pytest.approx(1 / 20) for value in predictions.values())
    assert model.support(board, 1000)["source"] == "uniform"


def test_fit_rejects_held_out_records_without_mutating_model() -> None:
    model = FrequencyPolicy()
    model.fit([record("d2d4")])
    before = model.predict(chess.Board(), 1200)
    with pytest.raises(ValueError, match="training"):
        model.fit([record(), replace(record(), split="validation")])
    assert model.predict(chess.Board(), 1200) == before


@pytest.mark.parametrize(
    ("fen", "move"),
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
    ],
)
def test_special_moves_are_legal_prediction_targets(fen: str, move: str) -> None:
    model = FrequencyPolicy()
    model.fit([record(move=move, fen=fen)])
    probabilities = model.predict(chess.Board(fen), 1200)
    assert next(iter(probabilities)) == move
    assert sum(probabilities.values()) == pytest.approx(1)


def test_clocks_do_not_fragment_positions_but_castling_rights_do() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    model = FrequencyPolicy()
    model.fit([record("e1g1", fen=board.fen())])
    board.fullmove_number = 12
    board.halfmove_clock = 9
    assert model.support(board, 1200)["source"] == "rating_band"
    board.castling_rights = chess.BB_EMPTY
    assert model.support(board, 1200)["source"] == "uniform"
    assert "e1g1" not in model.predict(board, 1200)


def test_round_trip_and_corrupt_model_rejection(tmp_path: Path) -> None:
    model = FrequencyPolicy(bucket_width=100, smoothing=0.5)
    model.fit([record()] * 2 + [record("d2d4", 1500)])
    path = tmp_path / "model.json"
    model.save(path)
    restored = FrequencyPolicy.load(path)
    assert restored.bucket_width == 100
    assert restored.smoothing == 0.5
    assert restored.predict(chess.Board(), 1200) == model.predict(chess.Board(), 1200)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        model.save(path)
    assert path.read_bytes() == original
    raw = json.loads(path.read_text())
    next(iter(next(iter(raw["counts"].values())).values()))["e2e5"] = 4
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Illegal move"):
        FrequencyPolicy.load(path)


def test_held_out_metrics_use_labels_without_updating_counts() -> None:
    model = FrequencyPolicy()
    model.fit([record()] * 2)
    validation = replace(record(), split="validation")
    before = model.predict(chess.Board(), 1200)
    metrics = evaluate(model, [validation])
    assert metrics["positions"] == 1
    assert metrics["top1_accuracy"] == 1
    assert metrics["top3_accuracy"] == 1
    assert metrics["negative_log_likelihood"] == pytest.approx(-math.log(3 / 22))
    assert metrics["support_sources"] == {
        "rating_band": 1,
        "position": 0,
        "uniform": 0,
    }
    assert metrics["splits"] == {"validation": 1}
    assert model.predict(chess.Board(), 1200) == before
    with pytest.raises(ValueError, match="validation or test"):
        evaluate(model, [record()])


def test_empty_evaluation_reports_absent_metrics() -> None:
    result = evaluate(FrequencyPolicy(), [])
    assert result["positions"] == 0
    assert result["top1_accuracy"] is None
    assert result["negative_log_likelihood"] is None


def test_empty_training_is_rejected() -> None:
    with pytest.raises(ValueError, match="training records"):
        FrequencyPolicy().fit([])


def test_large_finite_smoothing_still_normalizes() -> None:
    model = FrequencyPolicy(smoothing=1e308)
    model.fit([record()])
    probabilities = model.predict(chess.Board(), 1200)
    assert sum(probabilities.values()) == pytest.approx(1)
    assert all(value > 0 for value in probabilities.values())


def test_minimum_smoothing_keeps_unseen_moves_scoreable() -> None:
    model = FrequencyPolicy(smoothing=1e-12)
    model.fit([record()] * 2)
    probabilities = model.predict(chess.Board(), 1200)
    assert sum(probabilities.values()) == pytest.approx(1)
    assert all(value > 0 for value in probabilities.values())
    metrics = evaluate(model, [replace(record("d2d4"), split="test")])
    nll = metrics["negative_log_likelihood"]
    assert isinstance(nll, float)
    assert math.isfinite(nll)


@pytest.mark.parametrize("width", [0, -1, True])
def test_invalid_bucket_width_rejected(width: int) -> None:
    with pytest.raises(ValueError, match="bucket_width"):
        FrequencyPolicy(bucket_width=width)


@pytest.mark.parametrize("smoothing", [0, -1, 5e-324, 1e-13, math.nan, math.inf, True])
def test_invalid_smoothing_rejected(smoothing: float) -> None:
    with pytest.raises(ValueError, match="smoothing"):
        FrequencyPolicy(smoothing=smoothing)
