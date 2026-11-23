"""Exercise the boundaries between PGN parsing, storage, fitting and inference."""

import json
import subprocess
import sys
from pathlib import Path

import chess
import pytest

from chessformer.cli import main
from chessformer.schema import read_records


@pytest.fixture
def sample_pgn(tmp_path: Path) -> Path:
    games = []
    for index in range(80):
        line = "1. e4 e5 2. Nf3 Nc6" if index % 2 else "1. d4 d5 2. c4 e6"
        games.append(
            f'[Event "Synthetic rapid test"]\n'
            f'[Site "https://lichess.org/{index:08d}"]\n'
            f'[WhiteElo "{1200 if index % 2 else 2000}"]\n'
            '[BlackElo "1600"]\n[TimeControl "600+0"]\n'
            f'[Result "1-0"]\n\n{line} 1-0\n'
        )
    path = tmp_path / "sample.pgn"
    path.write_text("\n".join(games), encoding="utf-8")
    return path


def test_end_to_end_workflow(
    sample_pgn: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "dataset"
    model = tmp_path / "model.json"
    metrics = tmp_path / "metrics.json"
    assert main(["ingest", str(sample_pgn), "--output", str(dataset)]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["games_accepted"] == 80
    assert manifest["records_written"] == 320
    game_sets = [
        {record.game_id for record in read_records(dataset / f"{split}.jsonl")}
        for split in ("train", "validation", "test")
    ]
    assert all(game_sets)
    assert sum(len(group) for group in game_sets) == len(set.union(*game_sets))
    assert (
        main(["train", "--data", str(dataset / "train.jsonl"), "--output", str(model)])
        == 0
    )
    train_report = json.loads(capsys.readouterr().out)
    assert train_report["positions"] == len(game_sets[0]) * 4
    assert (
        main(
            [
                "evaluate",
                "--model",
                str(model),
                "--data",
                str(dataset / "test.jsonl"),
                "--output",
                str(metrics),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(metrics.read_text())
    assert report["positions"] == len(game_sets[2]) * 4
    assert 0 <= report["top1_accuracy"] <= report["top3_accuracy"] <= 1
    assert report["negative_log_likelihood"] > 0
    assert (
        main(["predict", "--model", str(model), "--elo", "1200", "--top-k", "20"]) == 0
    )
    predictions = json.loads(capsys.readouterr().out)
    assert predictions["moves"][0]["uci"] == "e2e4"
    assert predictions["displayed_probability_mass"] == pytest.approx(1)
    assert {move["uci"] for move in predictions["moves"]} == {
        move.uci() for move in chess.Board().legal_moves
    }
    original = model.read_bytes()
    assert (
        main(["train", "--data", str(dataset / "train.jsonl"), "--output", str(model)])
        == 1
    )
    assert "already exists" in capsys.readouterr().err
    assert model.read_bytes() == original


def test_reject_held_out_training_without_creating_model(
    sample_pgn: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "dataset"
    assert main(["ingest", str(sample_pgn), "--output", str(dataset)]) == 0
    capsys.readouterr()
    model = tmp_path / "model.json"
    assert (
        main(
            [
                "train",
                "--data",
                str(dataset / "validation.jsonl"),
                "--output",
                str(model),
            ]
        )
        == 1
    )
    assert "only training" in capsys.readouterr().err
    assert not model.exists()


def test_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "chessformer", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ingest" in result.stdout
    assert "predict" in result.stdout
