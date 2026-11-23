# First experiment: rating-aware move frequencies

This is a reproducible statistical baseline, not a trained neural network or a
soundness-verified repertoire generator. Its purpose is to establish the data and
evaluation workflow before introducing PyTorch and search.

## Dataset contract (version 1)

`ingest` streams a local PGN or, with the `data` extra, a `.pgn.zst` archive.
It writes `manifest.json` and `train.jsonl`, `validation.jsonl`, `test.jsonl` to a
new directory. Failed ingestion removes its temporary output. Existing datasets
are never overwritten. Duplicate identities are tracked in a temporary SQLite
database rather than an in-memory set. Each game is held in memory individually;
side variations and comments are discarded.

Accepted games must have a valid standard-chess starting position, legal mainline,
positive integer ratings for both players, a finished movetext result that agrees
with any declared finished header result, and a simple
`base+increment` time control. Rapid uses **480 <= base + 40 * increment < 1500**
seconds, following the [Lichess classification](https://lichess.org/faq#time-controls).
Both rated and casual games with these metadata are currently accepted. Missing
metadata, unsupported variants and malformed games are counted as skips. Move
legality is checked through the entire mainline, even when only the first 40 plies
are extracted. A PGN's declared final result is metadata, not independent proof
that the recorded game was played to completion.

Each JSONL object has these fields:

| Field | Meaning |
| --- | --- |
| `game_id` | Normalized Lichess game ID, or SHA-256 of identifying headers, starting FEN and mainline |
| `split` | `train`, `validation`, or `test` |
| `ply` | Zero-based index relative to the supplied game's starting position |
| `fen` | Position immediately **before** the target move |
| `move_uci` | Legal target move, including promotion suffix if present |
| `player_elo` | Rating of the player whose turn it is |
| `opponent_elo` | Other player's rating |
| `result` | White-oriented PGN result; metadata only |
| `time_control` | Original `base+increment` string |

The manifest records schema/package versions, input path and size, extraction
config, skip counts, and game/position totals by split. Retain the original
archive: path and size do not identify its contents cryptographically.
Zstandard decoding errors fail the run and remove staged output. The streaming
decoder does not establish that the final compressed frame is complete: a
truncated archive that still ends after a complete PGN game can be accepted.
Verify downloaded archive sizes/checksums independently before real experiments.

A seeded SHA-256 hash assigns whole games to approximately 80/10/10 splits;
fractions and seed are configurable. Assignment is independent of input order.
Deduplication applies within a single ingestion run. Reingesting the same game
with the same identity and seed gives the same split. Non-Lichess exports use
fallback identity: altered identifying metadata can defeat duplicate detection.

Common opening positions intentionally occur across game splits, as they will in
real games. This measures prediction on new games with familiar positions, not
generalization to unseen openings. Do not describe this as position-disjoint or
time-based evaluation. Chronological holdouts, cross-source duplicate audits,
and distribution reports by time control/opening remain required before scaling.

## Baseline and evaluation

The only features used for prediction are the current position and the moving
player's Elo. Result, opponent Elo and future moves do not enter the policy.
Position keys retain pieces, side to move, castling rights and **legal** en
passant; move counters are excluded so opening transpositions share counts. This
is unsuitable for adjudicating repetition and fifty-move draws.

The default rating buckets have width 200 (e.g. 1400–1599). For each position:

1. Use move counts from the requested Elo bucket when available.
2. Otherwise pool that position's counts over all Elo buckets.
3. For an unseen position, use a uniform distribution over legal moves.

Additive smoothing (default 1.0 per legal move, minimum 1e-12) keeps unseen legal
moves possible. The lower bound avoids numerical underflow with tiny pseudocounts.
All legal moves are normalized together. `predict --top-k` only limits display;
it does not renormalize the displayed moves. `support` exposes the backoff source
and observation count, which are coverage diagnostics, not confidence intervals.
JSON model files include a format/version and are validated on load.

`train` accepts only training records; `evaluate` accepts only validation/test
records. Evaluation streams positions without updating the model, and reports:

- Top-1 and top-3 move accuracy (deterministic UCI ordering breaks ties).
- Mean negative log-likelihood in natural-log units; lower is better.
- The same metrics by moving-player rating bucket.
- Position counts by backoff source and held-out split.

Metrics are weighted by positions, so games with longer extracted openings
contribute more. Empty holdouts report null metrics, not perfect scores.
Inspect validation results for configuration selection. Reserve the test split
for the final comparison, without tuning against its outcomes. Artifact checks
cannot prevent someone relabeling or manually mixing records; preserve the
original split files and manifest.

## Milestone acceptance and next experiments

The foundation is accepted when automated tests establish legal extraction
(including black-to-move ratings, special moves and malformed tails), stable
game splits and duplicate handling, normalized legal inference, artifact round
trips, train/holdout separation, and an end-to-end CLI run. Synthetic fixture
metrics demonstrate the software only; they do not establish chess strength.

Next, process 100,000 real rapid games and record archive provenance, wall time,
peak memory, disk size, skip rates, and coverage by Elo. Compare the rating-aware
baseline with uniform and pooled-position baselines on the same holdout. Check
whether rating conditioning improves NLL overall and within adequately sampled
Elo bands; report sample sizes and game-level uncertainty rather than claiming
an improvement from aggregate accuracy alone. Set concrete neural-model
acceptance thresholds after measuring this baseline, before training/tuning the
neural candidates. Calibration, temporal evaluation, Stockfish checks and
branching repertoire search are later milestones.

This JSONL/count-table prototype is intended for the first data experiment.
Training keeps distinct position/bucket counts in memory; large-scale storage,
Parquet sharding, throughput tuning and neural feature encodings are future work.
