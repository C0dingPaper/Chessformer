"""The ingestion contract: reject broken games and never split a game."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import chess
import pytest

from chessformer.data import ingest_pgn
from chessformer.schema import DatasetConfig, MoveRecord, read_records


def _pgn(
    *,
    site: str = "https://lichess.org/abcd1234",
    time_control: str = "600+0",
    white_elo: str = "1200",
    black_elo: str = "1600",
    result: str = "1-0",
    moves: str = "1. e4 e5 2. Nf3 Nc6",
    extra: str = "",
) -> str:
    return (
        '[Event "Rated Rapid game"]\n'
        f'[Site "{site}"]\n'
        '[Date "2026.01.01"]\n'
        '[White "Alice"]\n'
        '[Black "Bob"]\n'
        f'[Result "{result}"]\n'
        f'[WhiteElo "{white_elo}"]\n'
        f'[BlackElo "{black_elo}"]\n'
        f'[TimeControl "{time_control}"]\n'
        f"{extra}\n"
        f"{moves} {result}\n\n"
    )


def _ingest(
    tmp_path: Path, pgn: str, config: DatasetConfig | None = None
) -> tuple[dict[str, object], list[MoveRecord], Path]:
    source = tmp_path / "games.pgn"
    source.write_text(pgn, encoding="utf-8")
    output = tmp_path / "dataset"
    manifest = ingest_pgn(source, output, config or DatasetConfig())
    records = [
        record
        for split in ("train", "validation", "test")
        for record in read_records(output / f"{split}.jsonl")
    ]
    return manifest, records, output


def test_extracts_pre_move_positions_and_moving_players_rating(tmp_path: Path) -> None:
    manifest, records, output = _ingest(tmp_path, _pgn())
    assert manifest["games_read"] == manifest["games_accepted"] == 1
    assert manifest["records_written"] == 4
    assert manifest["games_skipped"] == 0
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert not (output / "seen.sqlite3").exists()
    assert records[0].fen == chess.STARTING_FEN
    assert records[0].player_elo == records[1].opponent_elo == 1200
    assert records[1].player_elo == records[0].opponent_elo == 1600
    assert records[1].move_uci == "e7e5"
    assert [record.ply for record in records] == list(range(4))
    assert len({record.split for record in records}) == 1
    board = chess.Board()
    for record in records:
        assert record.fen == board.fen()
        board.push_uci(record.move_uci)


@pytest.mark.parametrize(
    ("headers", "reason"),
    [
        ({"white_elo": "?"}, "invalid_ratings"),
        ({"black_elo": "0"}, "invalid_ratings"),
        ({"result": "*"}, "unfinished_game"),
        ({"time_control": "300+0"}, "non_rapid_time_control"),
        ({"time_control": "1500+0"}, "non_rapid_time_control"),
        ({"time_control": "?"}, "non_rapid_time_control"),
        ({"moves": ""}, "no_moves"),
        ({"moves": "1. e4 e5 2. Bh6"}, "malformed_game"),
        ({"moves": "1. e4 e5 2. --"}, "illegal_move"),
        ({"extra": '[FEN "invalid"]'}, "malformed_game"),
        ({"extra": '[Variant "Atomic"]'}, "unsupported_variant"),
    ],
)
def test_filters_invalid_games(
    tmp_path: Path, headers: dict[str, str], reason: str
) -> None:
    manifest, records, _ = _ingest(tmp_path, _pgn(**headers))
    assert records == []
    assert manifest["games_read"] == manifest["games_skipped"] == 1
    assert manifest["skip_reasons"] == {reason: 1}


@pytest.mark.parametrize("time_control", ["480+0", "300+5", "600+0", "1499+0"])
def test_rapid_boundaries(tmp_path: Path, time_control: str) -> None:
    manifest, _, _ = _ingest(tmp_path, _pgn(time_control=time_control))
    assert manifest["games_accepted"] == 1


def test_deduplicates_lichess_game_urls_and_fallbacks(tmp_path: Path) -> None:
    source = (
        _pgn()
        + _pgn(site="https://lichess.org/abcd1234/black")
        + _pgn(site="https://lichess.org/abcd1234ABCD/white")
        + _pgn(site="https://elsewhere.example/one")
        + _pgn(site="https://elsewhere.example/two")
    )
    manifest, records, _ = _ingest(tmp_path, source)
    assert manifest["games_read"] == 5
    assert manifest["games_accepted"] == 2
    assert manifest["skip_reasons"] == {"duplicate_game": 3}
    assert len({record.game_id for record in records}) == 2


def test_rejects_spoofed_lichess_hostname(tmp_path: Path) -> None:
    manifest, records, _ = _ingest(
        tmp_path, _pgn(site="https://lichess.org.example/abcd1234")
    )
    assert manifest["games_accepted"] == 1
    assert records[0].game_id.startswith("sha256:")


def test_limits_plies_but_rejects_broken_tails(tmp_path: Path) -> None:
    manifest, records, _ = _ingest(
        tmp_path,
        _pgn() + _pgn(site="https://lichess.org/bad01234", moves="1. d4 d5 2. Bh5"),
        DatasetConfig(max_plies=1),
    )
    assert manifest["games_accepted"] == 1
    assert manifest["skip_reasons"] == {"malformed_game": 1}
    assert len(records) == 1


def test_valid_setup_position_and_black_to_move(tmp_path: Path) -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    manifest, records, _ = _ingest(
        tmp_path,
        _pgn(extra=f'[SetUp "1"]\n[FEN "{board.fen()}"]', moves="1... e5"),
    )
    assert manifest["games_accepted"] == 1
    assert records[0].fen == board.fen()
    assert records[0].player_elo == 1600


def test_deterministic_records_independent_of_input_order(tmp_path: Path) -> None:
    source = tmp_path / "games.pgn"
    games = [_pgn(site=f"https://lichess.org/game{i:04}") for i in range(10)]
    source.write_text("".join(games))
    first = tmp_path / "first"
    second = tmp_path / "second"
    ingest_pgn(source, first, DatasetConfig())
    source.write_text("".join(reversed(games)))
    ingest_pgn(source, second, DatasetConfig())
    for split in ("train", "validation", "test"):
        assert sorted((first / f"{split}.jsonl").read_text().splitlines()) == sorted(
            (second / f"{split}.jsonl").read_text().splitlines()
        )


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    manifest, _, output = _ingest(tmp_path, _pgn())
    with pytest.raises(FileExistsError, match="already exists"):
        ingest_pgn(tmp_path / "games.pgn", output, DatasetConfig())
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_decode_failure_cleans_staging_and_does_not_publish(tmp_path: Path) -> None:
    source = tmp_path / "broken.pgn"
    source.write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError):
        ingest_pgn(source, tmp_path / "dataset", DatasetConfig())
    assert sorted(path.name for path in tmp_path.iterdir()) == ["broken.pgn"]


def test_missing_zstandard_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "games.pgn.zst"
    source.write_bytes(b"unused")

    def missing(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ValueError, match="chessformer\\[data\\]"):
        ingest_pgn(source, tmp_path / "dataset", DatasetConfig())
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize(
    "moves",
    ["1. e4 e5 2. Nf3", "1. e4 (1. d4 d5 1-0) e5 2. Nf3"],
)
@pytest.mark.parametrize("ending", ["", "\n\n"])
def test_rejects_missing_mainline_termination_marker(
    tmp_path: Path, moves: str, ending: str
) -> None:
    truncated = _pgn(moves=moves).rsplit("1-0", 1)[0].rstrip() + ending
    manifest, records, _ = _ingest(tmp_path, truncated)
    assert records == []
    assert manifest["games_accepted"] == 0
    assert manifest["skip_reasons"] == {"malformed_game": 1}


def test_mainline_marker_accepts_annotated_game(tmp_path: Path) -> None:
    manifest, records, _ = _ingest(
        tmp_path, _pgn(moves="1. e4 (1. d4 d5 1/2-1/2) e5 2. Nf3")
    )
    assert manifest["games_accepted"] == 1
    assert [record.move_uci for record in records] == ["e2e4", "e7e5", "g1f3"]


@pytest.mark.parametrize("concatenated_frames", [False, True])
def test_ingests_real_zstandard_stream(
    tmp_path: Path, concatenated_frames: bool
) -> None:
    zstandard = pytest.importorskip("zstandard")
    pgn = (_pgn() + _pgn(site="https://lichess.org/game0002")).encode("utf-8")
    # Split inside a PGN token too: text decoding must span frame boundaries.
    boundary = len(pgn) // 3
    chunks = [pgn[:boundary], pgn[boundary:]] if concatenated_frames else [pgn]
    compressor = zstandard.ZstdCompressor(write_checksum=True)
    compressed = b"".join(compressor.compress(chunk) for chunk in chunks)
    source = tmp_path / "games.pgn.zst"
    source.write_bytes(compressed)
    output = tmp_path / "dataset"
    manifest = ingest_pgn(source, output, DatasetConfig())
    assert manifest["games_accepted"] == 2
    assert manifest["records_written"] == 8
    records = [
        record
        for split in ("train", "validation", "test")
        for record in read_records(output / f"{split}.jsonl")
    ]
    assert len(records) == 8
    assert {record.game_id for record in records} == {
        "lichess:abcd1234",
        "lichess:game0002",
    }


@pytest.mark.parametrize("corruption", ["bad_frame", "checksum"])
def test_corrupted_zstandard_stream_cleans_staging(
    tmp_path: Path, corruption: str
) -> None:
    zstandard = pytest.importorskip("zstandard")
    if corruption == "bad_frame":
        compressed = b"This is not a Zstandard frame"
    else:
        compressed_bytes = bytearray(
            zstandard.ZstdCompressor(write_checksum=True).compress(_pgn().encode())
        )
        compressed_bytes[-1] ^= 1
        compressed = bytes(compressed_bytes)
    source = tmp_path / "games.pgn.zst"
    source.write_bytes(compressed)
    with pytest.raises(ValueError, match="Cannot decompress PGN archive"):
        ingest_pgn(source, tmp_path / "dataset", DatasetConfig())
    assert sorted(path.name for path in tmp_path.iterdir()) == ["games.pgn.zst"]
