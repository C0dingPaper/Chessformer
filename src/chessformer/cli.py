"""Small, scriptable interface for the data and statistical baseline workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import chess

from chessformer import __version__
from chessformer.baseline import FrequencyPolicy, evaluate
from chessformer.data import ingest_pgn
from chessformer.schema import DatasetConfig, MoveRecord, read_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chessformer", description="Elo-aware chess opening experiments"
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="extract rapid PGNs into split JSONL")
    ingest.add_argument("input", type=Path, help="local .pgn or .pgn.zst archive")
    ingest.add_argument(
        "--output", type=Path, required=True, help="new dataset directory"
    )
    ingest.add_argument("--max-plies", type=int, default=40)
    ingest.add_argument("--seed", default="chessformer-v1")
    ingest.add_argument("--validation-fraction", type=float, default=0.1)
    ingest.add_argument("--test-fraction", type=float, default=0.1)

    train = commands.add_parser("train", help="fit a rating-aware frequency baseline")
    train.add_argument("--data", type=Path, required=True, help="training JSONL")
    train.add_argument("--output", type=Path, required=True, help="new model JSON file")
    train.add_argument("--bucket-width", type=int, default=200)
    train.add_argument("--smoothing", type=float, default=1.0)

    test = commands.add_parser("evaluate", help="score validation or test JSONL")
    test.add_argument("--model", type=Path, required=True)
    test.add_argument("--data", type=Path, required=True)
    test.add_argument("--output", type=Path, help="optional new metrics JSON file")

    predict = commands.add_parser("predict", help="predict legal human moves")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--elo", type=int, required=True)
    predict.add_argument("--fen", default=chess.STARTING_FEN)
    predict.add_argument("--top-k", type=int, default=5)
    return parser


def _counted_records(path: Path, counts: dict[str, int]) -> Iterator[MoveRecord]:
    for record in read_records(path):
        counts["positions"] += 1
        yield record


def _write_report(path: Path, report: dict[str, object]) -> None:
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        target.write(serialized)


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "ingest":
        return ingest_pgn(
            args.input,
            args.output,
            DatasetConfig(
                max_plies=args.max_plies,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
            ),
        )
    if args.command == "train":
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        policy = FrequencyPolicy(args.bucket_width, args.smoothing)
        counts = {"positions": 0}
        policy.fit(_counted_records(args.data, counts))
        if counts["positions"] == 0:
            raise ValueError("training data contains no positions")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        policy.save(args.output)
        return {"model": str(args.output), **counts}
    policy = FrequencyPolicy.load(args.model)
    if args.command == "evaluate":
        report = evaluate(policy, read_records(args.data))
        if args.output is not None:
            _write_report(args.output, report)
        return report
    if args.elo <= 0 or args.top_k <= 0:
        raise ValueError("elo and top-k must be positive")
    board = chess.Board(args.fen)
    if not board.is_valid():
        raise ValueError("fen must describe a valid standard chess position")
    probabilities = policy.predict(board, args.elo)
    selected = list(probabilities.items())[: args.top_k]
    moves = [
        {
            "uci": uci,
            "san": board.san(chess.Move.from_uci(uci)),
            "probability": probability,
        }
        for uci, probability in selected
    ]
    return {
        "fen": board.fen(),
        "elo": args.elo,
        "support": policy.support(board, args.elo),
        "moves": moves,
        "displayed_probability_mass": sum(probability for _, probability in selected),
        "legal_moves": len(probabilities),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = _run(args)
        serialized = json.dumps(report, indent=2, allow_nan=False)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        print(f"chessformer: {exc}", file=sys.stderr)
        return 1
    print(serialized)
    return 0
