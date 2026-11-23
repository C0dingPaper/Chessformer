"""Streaming, game-disjoint training-data extraction from rapid-game PGNs."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import re
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TextIO
from urllib.parse import urlparse

import chess
import chess.pgn

from chessformer import __version__
from chessformer.schema import DatasetConfig, MoveRecord, split_for_game

_SPLITS = ("train", "validation", "test")
_RESULTS = {"1-0", "0-1", "1/2-1/2"}


class _MainlineBuilder(chess.pgn.BaseVisitor[chess.pgn.Game]):
    """Avoid retaining annotations and variations from large archival PGNs."""

    def begin_game(self) -> None:
        self.game = chess.pgn.Game()
        self.node: chess.pgn.GameNode = self.game
        self.saw_result = False

    def begin_headers(self) -> chess.pgn.Headers:
        return self.game.headers

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        self.game.headers[tagname] = tagvalue

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        self.node = self.node.add_main_variation(move)

    def result(self) -> chess.pgn.Game:
        return self.game

    def begin_variation(self) -> chess.pgn.SkipType:
        return chess.pgn.SKIP

    def visit_comment(self, comment: str) -> None:
        pass

    def handle_error(self, error: Exception) -> None:
        # Default GameBuilder logs every bad game. Keep aggregate counts instead.
        self.game.errors.append(error)

    def end_game(self) -> None:
        if not self.saw_result:
            self.handle_error(ValueError("Missing mainline termination marker"))

    def visit_result(self, result: str) -> None:
        self.saw_result = True
        header_result = self.game.headers.get("Result", "*")
        if header_result != "*" and header_result != result:
            self.handle_error(ValueError("Header and movetext results disagree"))
        if header_result == "*":
            self.game.headers["Result"] = result


@contextmanager
def _open_pgn(path: Path) -> Iterator[TextIO]:
    if path.name.endswith(".zst"):
        try:
            zstandard = importlib.import_module("zstandard")
        except ImportError as error:
            raise ValueError(
                "Reading .pgn.zst requires zstandard; "
                "install it with: pip install 'chessformer[data]'"
            ) from error
        try:
            with path.open("rb") as raw:
                with zstandard.ZstdDecompressor().stream_reader(
                    raw, read_across_frames=True
                ) as decompressed:
                    with io.TextIOWrapper(decompressed, encoding="utf-8-sig") as text:
                        yield text
        except zstandard.ZstdError as error:
            raise ValueError(
                f"Cannot decompress PGN archive {path}: {error}"
            ) from error
    else:
        with path.open("r", encoding="utf-8-sig") as text:
            yield text


def _is_rapid(time_control: str) -> bool:
    """Use Lichess's estimated duration: base seconds plus 40 increments."""
    match = re.fullmatch(r"([0-9]+)\+([0-9]+)", time_control)
    if match is None:
        return False
    base, increment = map(int, match.groups())
    return 480 <= base + 40 * increment < 1500


def _ratings(game: chess.pgn.Game) -> tuple[int, int] | None:
    values = [game.headers.get(name, "") for name in ("WhiteElo", "BlackElo")]
    if not all(re.fullmatch(r"[0-9]+", value) for value in values):
        return None
    white, black = map(int, values)
    return (white, black) if white > 0 and black > 0 else None


def _game_id(game: chess.pgn.Game, initial_fen: str, moves: list[chess.Move]) -> str:
    site = urlparse(game.headers.get("Site", ""))
    if site.scheme in {"http", "https"} and site.hostname == "lichess.org":
        match = re.fullmatch(
            r"/([A-Za-z0-9]{8})(?:[A-Za-z0-9]{4})?(?:/(?:white|black))?/?",
            site.path,
        )
        if match:
            return f"lichess:{match.group(1)}"
    # Site URLs can contain unstable query strings. The fallback uses game facts.
    payload = {
        "headers": {
            name: game.headers.get(name, "")
            for name in (
                "Event",
                "Date",
                "UTCDate",
                "UTCTime",
                "Round",
                "White",
                "Black",
                "WhiteElo",
                "BlackElo",
                "Result",
                "TimeControl",
            )
        },
        "fen": initial_fen,
        "moves": [move.uci() for move in moves],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _validate_game(
    game: chess.pgn.Game,
) -> tuple[chess.Board, list[chess.Move], tuple[int, int]] | str:
    if game.errors:
        return "malformed_game"
    try:
        board = game.board()
    except ValueError:
        return "invalid_position"
    if type(board) is not chess.Board or board.chess960:
        return "unsupported_variant"
    if not board.is_valid():
        return "invalid_position"
    if game.headers.get("Result") not in _RESULTS:
        return "unfinished_game"
    ratings = _ratings(game)
    if ratings is None:
        return "invalid_ratings"
    if not _is_rapid(game.headers.get("TimeControl", "")):
        return "non_rapid_time_control"
    initial_board = board.copy(stack=False)
    moves = list(game.mainline_moves())
    if not moves:
        return "no_moves"
    # Validate beyond the extraction horizon too: a broken tail rejects the game.
    for move in moves:
        if move not in board.legal_moves:
            return "illegal_move"
        board.push(move)
    return initial_board, moves, ratings


def ingest_pgn(
    input_path: Path, output_dir: Path, config: DatasetConfig
) -> dict[str, object]:
    """Extract validated opening records and a manifest into a new directory.

    The corpus streams one game at a time. SQLite tracks duplicate identities on
    disk, and a staging directory prevents failed runs from appearing complete.
    No existing dataset directory is overwritten.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Dataset output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    games_read = 0
    games_accepted = 0
    records_written = 0
    skipped: Counter[str] = Counter()
    splits = {split: {"games": 0, "records": 0} for split in _SPLITS}
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as staging_name:
        staging = Path(staging_name)
        database = staging / "seen.sqlite3"
        with closing(sqlite3.connect(database)) as seen:
            seen.execute("CREATE TABLE games (id TEXT PRIMARY KEY) WITHOUT ROWID")
            with _open_pgn(input_path) as source:
                with (
                    (staging / "train.jsonl").open("w", encoding="utf-8") as train,
                    (staging / "validation.jsonl").open("w", encoding="utf-8") as val,
                    (staging / "test.jsonl").open("w", encoding="utf-8") as test,
                ):
                    outputs = {"train": train, "validation": val, "test": test}
                    while (
                        game := chess.pgn.read_game(source, Visitor=_MainlineBuilder)
                    ) is not None:
                        games_read += 1
                        validated = _validate_game(game)
                        if isinstance(validated, str):
                            skipped[validated] += 1
                            continue
                        board, moves, ratings = validated
                        game_id = _game_id(game, board.fen(), moves)
                        inserted = seen.execute(
                            "INSERT OR IGNORE INTO games VALUES (?)", (game_id,)
                        ).rowcount
                        if not inserted:
                            skipped["duplicate_game"] += 1
                            continue
                        split = split_for_game(game_id, config)
                        games_accepted += 1
                        splits[split]["games"] += 1
                        for ply, move in enumerate(moves[: config.max_plies]):
                            white_elo, black_elo = ratings
                            record = MoveRecord(
                                game_id=game_id,
                                split=split,
                                ply=ply,
                                fen=board.fen(),
                                move_uci=move.uci(),
                                player_elo=white_elo if board.turn else black_elo,
                                opponent_elo=black_elo if board.turn else white_elo,
                                result=game.headers["Result"],
                                time_control=game.headers["TimeControl"],
                            )
                            outputs[split].write(
                                json.dumps(asdict(record), separators=(",", ":")) + "\n"
                            )
                            board.push(move)
                            records_written += 1
                            splits[split]["records"] += 1
        database.unlink()
        manifest: dict[str, object] = {
            "schema_version": 1,
            "input_path": str(input_path),
            "input_bytes": input_path.stat().st_size,
            "package_version": __version__,
            "config": asdict(config),
            "games_read": games_read,
            "games_accepted": games_accepted,
            "games_skipped": sum(skipped.values()),
            "skip_reasons": dict(sorted(skipped.items())),
            "records_written": records_written,
            "splits": splits,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Check again so a directory created during ingestion is never replaced.
        if output_dir.exists():
            raise FileExistsError(f"Dataset output already exists: {output_dir}")
        staging.rename(output_dir)
    return manifest
