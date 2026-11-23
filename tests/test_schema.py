from dataclasses import replace
from pathlib import Path

import chess
import pytest

from chessformer.schema import (
    DatasetConfig,
    MoveRecord,
    position_key,
    read_records,
    split_for_game,
)


def record() -> MoveRecord:
    return MoveRecord(
        game_id="example",
        split="train",
        ply=0,
        fen=chess.STARTING_FEN,
        move_uci="e2e4",
        player_elo=1500,
        opponent_elo=1600,
        result="1-0",
        time_control="600+0",
    )


def test_record_rejects_illegal_target_and_bad_ratings() -> None:
    with pytest.raises(ValueError, match="illegal target"):
        replace(record(), move_uci="e2e5")
    with pytest.raises(ValueError, match="ratings"):
        replace(record(), player_elo=0)
    with pytest.raises(ValueError, match="ratings"):
        replace(record(), player_elo=True)
    with pytest.raises(ValueError, match="finished"):
        replace(record(), result="*")


@pytest.mark.parametrize(
    "config",
    [
        {"max_plies": 0},
        {"max_plies": True},
        {"seed": ""},
        {"test_fraction": float("nan")},
        {"validation_fraction": -0.1},
        {"test_fraction": 0.5, "validation_fraction": 0.5},
    ],
)
def test_invalid_config(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DatasetConfig(**config)  # type: ignore[arg-type]


def test_game_split_is_order_independent_and_seeded() -> None:
    ids = [f"game-{index}" for index in range(200)]
    config = DatasetConfig()
    forward = {game: split_for_game(game, config) for game in ids}
    backward = {game: split_for_game(game, config) for game in reversed(ids)}
    assert forward == backward
    assert set(forward.values()) == {"train", "validation", "test"}
    assert any(
        forward[game] != split_for_game(game, DatasetConfig(seed="other"))
        for game in ids
    )


def test_position_key_handles_transpositions_and_preserves_rights() -> None:
    first, second = chess.Board(), chess.Board()
    for san in ("Nf3", "Nf6", "g3", "g6"):
        first.push_san(san)
    for san in ("g3", "g6", "Nf3", "Nf6"):
        second.push_san(san)
    assert first.fen() != second.fen()  # different halfmove counters
    assert position_key(first) == position_key(second)
    second.castling_rights = 0
    assert position_key(first) != position_key(second)


def test_position_key_retains_only_legal_en_passant() -> None:
    board = chess.Board()
    board.push_san("e4")
    assert position_key(board).endswith(" -")
    for san in ("a6", "e5", "d5"):
        board.push_san(san)
    without_ep = board.copy()
    without_ep.ep_square = None
    assert position_key(board) != position_key(without_ep)


def test_bad_jsonl_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("\n{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:2:"):
        list(read_records(path))
