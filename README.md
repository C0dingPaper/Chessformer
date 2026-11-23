## Chessformer
An Elo-aware chess opening specialist trained on human rapid games.
Status: Early development. A rapid-PGN data pipeline, Elo-aware statistical move predictor, evaluation workflow, and CLI are implemented. Neural models, repertoire search, and engine verification below remain goals.

### Progress so far

The first working development milestone is complete:

- Python package and CLI (`ingest`, `train`, `evaluate`, and `predict`).
- Streaming PGN ingestion, validated position/move/rating JSONL records,
  disk-backed game deduplication, deterministic whole-game splits, and dataset manifests.
- A smoothed statistical move predictor using 200-point rating buckets, with
  pooled-position and uniform-legal-move fallbacks when data is sparse.
- Held-out move accuracy and negative log-likelihood reports by Elo band.
- A synthetic demo dataset and saved baseline workflow, plus automated tests
  (133 tests recorded at the first milestone).

The files in `data/demo/` are generated from synthetic games. They exercise the
workflow; they do not measure real player behavior. `checkpoints/demo.json` is
the saved statistical baseline, not a neural network. No real-game benchmark,
trained neural network, repertoire search, or Stockfish verification is complete yet.

### Next milestone: four rating groups

The first real-data experiment will target **50,000 games per group**, or
approximately **200,000 games overall**, from a downloaded Lichess monthly
archive. The model should support all four groups, including strong play and
lower-rated opponent behavior.

The proposed boundaries below turn the target ratings into usable ranges.
They are planning defaults to confirm before implementing the sampler; these
four groups are not yet supported by the CLI.

| Group label | Lichess rapid rating | Training games | Validation games | Test games | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| 500 | Below 750 | 40,000 | 5,000 | 5,000 | 50,000 |
| 1000 | 750–1249 | 40,000 | 5,000 | 5,000 | 50,000 |
| 1500 | 1250–1849 | 40,000 | 5,000 | 5,000 | 50,000 |
| 2200 | 1850 and above | 40,000 | 5,000 | 5,000 | 50,000 |
| **Total target** | | **160,000** | **20,000** | **20,000** | **200,000** |

These are approximate 80/10/10 split targets, subject to available qualifying
games. Group labels are representative ratings, not exact-rating filters or
minimum ratings; for example, the proposed 2200 group includes players below 2200.
Each move-prediction example uses the rating of the player making that move.
Mixed-group games can contribute examples to multiple groups, so the sampler
must define quota attribution and report unique-game and per-group counts
separately. The 200,000 total is a planning target, not a guarantee of that many
distinct games under overlapping group membership. Every game must remain in
one split across all groups to prevent leakage.

### Immediate next steps

1. Store the planned roughly 31 GB `.pgn.zst` download in `data/raw/`.
   Raw archives and generated datasets are already excluded from Git. Keep the
   archive compressed and stream it; full extraction is unnecessary.
2. Add configurable rating groups and per-group accepted-game quotas to
   ingestion. Define mixed-group quota attribution, preserve whole-game split
   isolation, and record sampling rules and coverage in the manifest.
   The current CLI processes the supplied archive without a game-count limit.
3. Run a small real-data pilot to verify rapid filtering, ratings, deduplication,
   disk usage, processing speed, and group coverage before the full run.
4. Build the four-group dataset in a new directory such as `data/rapid_v1/`,
   targeting 50,000 games per group and reporting any shortfalls.
5. Train and evaluate the statistical baseline on real data, compare pooled and
   rating-aware predictions, and report performance and coverage for every group.
   Choose settings on validation data and reserve test data for final comparisons.
6. Implement one rating-conditioned neural model trained across all four groups,
   compare the planned compact CNN and transformer against the baseline, and
   evaluate each group separately.
7. After move prediction is validated, add branching repertoire search and
   Stockfish soundness checks.

### Run the first working prototype

Requires Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

# A small synthetic fixture exercises the workflow; it is not a real-game benchmark.
python -m chessformer ingest examples/rapid_sample.pgn --output data/demo
python -m chessformer train --data data/demo/train.jsonl --output checkpoints/demo.json
python -m chessformer evaluate --model checkpoints/demo.json --data data/demo/validation.jsonl
python -m chessformer predict --model checkpoints/demo.json --elo 1200 --top-k 5
```

The installed `chessformer` command is equivalent to `python -m chessformer`.
Ingestion, training, and metrics output paths must be new; use a new destination
for another experiment. For real local Lichess `.pgn.zst` archives, install
`python -m pip install -e '.[data]'` first, then pass the archive to `ingest`.
No game download or GPU is required for the demo.

The baseline counts moves by position and 200-point Elo bucket, backs off to
pooled position counts, and uses a uniform legal distribution for unseen
positions. Predictions include observation counts and the fallback source.
These are human-move estimates, not engine-approved opening recommendations.
Use validation data to choose settings, and reserve `test.jsonl` for the final
comparison. See [the dataset contract and experiment plan](docs/experiments.md)
for filtering rules, split guarantees, evaluation metrics, and limitations.

Development checks:

```bash
python -m pytest
ruff check src tests
ruff format --check src tests
mypy src
```

Overview
Chess Opening AI is an independent machine-learning project that aims to generate sound, practical, rating-specific opening repertoires. Inspired by the practical opening preparation offered by Lotus Chess, the system will learn how humans play the opening at different rapid ratings, identify likely inaccuracies, and construct branching lines designed to perform well against real opponents.

The long-term data target is 200 million rapid chess games. The project is not intended to become a general-purpose chess engine or to outperform Stockfish at full-game search. Instead, it will use learned human behavior, opening statistics, targeted search, and engine verification to act as an opening-preparation companion.

It will be developed and evaluated independently before any integration with Chess Master Coach (CMC).
Motivation
The objectively best move is not always the most useful practical repertoire recommendation for every human player. Opponents at different ratings choose different moves, make different mistakes, and respond differently to complexity.

Traditional opening books describe established theory; engine analysis evaluates positions; database explorers show historical results. This project explores how to combine these strengths with an AI trained to predict rating-dependent human decisions.

The goal is not to promote unsound traps with impressive-looking win rates. It is to discover robust, defensible opening choices that offer practical opportunities against the mistakes opponents are actually likely to make.
Three core optimization goals
1. Practical effectiveness: Favor positions and continuations associated with favorable practical outcomes for the target rapid-rating population.
2. Opponent predictability: Estimate which legal replies opponents at that rating are likely to choose.
3. Repertoire robustness: Prepare reliable answers to common, uncommon, and strong defensive replies—not only tempting traps.
Soundness is a constraint throughout: Stockfish analysis will help reject objectively poor or tactically refutable recommendations.
Intended experience
A standalone request will specify:
Input	Purpose
Rapid Elo	Select the target opponent-behavior distribution (initially assume comparable opponent rating).
Color	Generate a White or Black repertoire.
Opening or starting position	Choose the position or opening family to prepare.
Optional: style	Adjust preferences for balanced, tactical, or positional play without overriding soundness.
Optional: depth and breadth	Control how many moves and branches to cover.


The output should be a legal, branching opening repertoire containing recommended moves, estimated opponent-response probabilities, critical alternatives, common mistakes and their punishments, engine-based soundness checks, and supporting explanations. The system should also expose confidence or data-coverage limitations where appropriate.

A repertoire is not a single forced sequence: opponents can deviate at any move, and the tree needs useful replies when they do.
Proposed architecture
Human rapid-game PGNs (target: 200M games)
                 |
       Data validation & extraction
                 |
     Position / move / rating records
                 |
         +-------+--------+
         |                |
  Opening statistics   Rating-conditioned
  & baseline models    neural network
                          |
                  Human move-policy model
                  (later: outcome / mistake models)
         |                |
         +-------+--------+
                 |
       Probability-weighted
        repertoire search
                 |
        Stockfish verification
                 |
       Elo-specific opening tree
                 |
      Standalone evaluation/testing
                 |
      Future: Chess Master Coach
1. Human opponent model
The first neural network will predict the probability of each legal move from a chess position, conditioned on the moving player's rapid rating. We will begin with a compact baseline and compare a CNN/ResNet against a rating-conditioned transformer before selecting a model for larger-scale training.

Future experiments may include additional heads for win/draw/loss estimates and engine-labeled mistake probabilities. Game outcomes and future moves will be targets or evaluation data, never leaked into next-move prediction inputs.
2. Repertoire search
A separate generator will traverse opening positions, use the learned model to estimate likely opponent replies, and choose our continuations based on practical value. It must cover strong defensive alternatives rather than optimize only for a single predicted line. Historical frequencies and statistical baselines will be retained for comparison.
3. Engine verification
Stockfish will assess critical branches and identify tactical or positional weaknesses. Engine evaluation is a soundness check, not a substitute for modeling human behavior. We do not plan to run expensive engine analysis on every position in all 200 million games.
4. Optional reinforcement learning research
After supervised models and the repertoire generator are independently validated, we may train a repertoire-selection agent against Elo-conditioned simulated human opponents. The goal would be to improve practical repertoire choices, not to build a standalone self-play chess engine.

This remains experimental: a simulator can be inaccurate, and an RL agent can exploit the simulator's errors. Improvements must be confirmed on held-out real games and through soundness checks before adoption.
Data strategy
Planned source: Lichess Open Database and other appropriately licensed sources if needed.
- Use rapid games only for the primary training corpus.
- Retain opening moves, positions, player ratings, game results, and available time-control metadata.
- Parse, validate, filter, and deduplicate records where appropriate.
- Start with approximately 200,000 games across four rating groups (50,000 per group), then expand through 1M → 10M → 50M → 200M based on measured value.
- Hold out entire games for validation and testing; use time-based holdouts to check performance on more recent play.
- Avoid leakage from repeated positions, overlapping game records, and future-game information.
- Track coverage by Elo band, opening family, and time control; account for sparse positions and sampling bias.
The 200-million-game figure is a target, not a statement that the data has already been downloaded or processed. Storage, processing throughput, and compute requirements will be measured before scaling.
Model evaluation and reliability
Model training accuracy alone is insufficient. We will evaluate both human move prediction and repertoire quality.
Area	Proposed checks
Move prediction	Held-out top-1/top-k accuracy, negative log-likelihood, and calibration by Elo band.
Elo sensitivity	Whether predicted moves and recommended lines change appropriately across ratings.
Repertoire legality	Every move, promotion, and branch must be legal.
Soundness	Stockfish evaluations of recommended moves and critical opponent replies.
Robustness	Coverage of likely replies, sound responses to rare/strong defenses, and transposition handling.
Practical value	Historical out-of-sample performance estimates and, later, controlled user testing.
Baseline comparisons	Opening-frequency and rating-filtered statistical recommenders.
Reproducibility	Versioned datasets, configuration, seeds, metrics, and checkpoints.


Historical win rate is not proof of causation: player strength, selection bias, and the rest of the game affect results. Reliability criteria and numerical acceptance thresholds will be set before major experiments.
Development roadmap
- [x] Phase 1 — Foundations: Define data schemas, reproducible experiments, evaluation criteria, and the repository structure.
- [ ] Phase 2 — Data prototype: Process and verify the first four-group rapid dataset (target: 50,000 games per group); create position/move/rating training records.
- [ ] Phase 3 — Baselines: Implement rating-aware opening statistics and a simple move predictor.
- [ ] Phase 4 — Neural prototype: Train and compare a compact CNN and rating-conditioned transformer on held-out games.
- [ ] Phase 5 — Repertoire generator: Build opening-tree search with transpositions, branching coverage, and Stockfish verification.
- [ ] Phase 6 — Scaling: Move through progressively larger datasets, measuring whether additional data improves results.
- [ ] Phase 7 — Advanced experiments: Evaluate outcome/mistake heads and, optionally, reinforcement learning against simulated opponents.
- [ ] Phase 8 — Independent validation: Test performance, calibration, robustness, and reliability against established baselines.
- [ ] Phase 9 — Future CMC integration: Expose a stable inference interface only after standalone validation.

Implemented groundwork for phases 2–3: streaming PGN ingestion, validated
JSONL records, deterministic game-level splits, disk-backed deduplication,
manifest generation, a smoothed rating-aware frequency policy, and held-out
accuracy/NLL reports by Elo band. The four-group real-data milestone and
baseline comparisons have not been run.
Planned tech stack
Technology choices are provisional and will be finalized during implementation.
Area	Tools / libraries	Role
Language	Python	Data processing, training, search, evaluation.
Deep learning	PyTorch	Model definition, automatic differentiation, GPU training.
Numerical/data work	NumPy, Polars	Feature construction and efficient tabular processing.
Chess data and rules	python-chess	PGN parsing, FEN positions, legal moves, UCI engine access.
Chess analysis	Stockfish	Soundness verification and targeted analysis.
Game-file decompression	Zstandard	Streaming compressed chess archives.
Analytical storage	Parquet, DuckDB	Compact records and opening-statistics queries.
Experiments	TensorBoard or MLflow	Track configurations, losses, metrics, and model versions.
Testing	pytest	Data, chess legality, and model-pipeline tests.
Code quality	Ruff, mypy	Linting, formatting, and static checks.
Reproducible environments	uv or pip + virtual environments	Dependency management.
Future service interface	FastAPI, Pydantic	Typed, versioned inference requests and responses.
Version control	Git, GitHub	Source code, issues, milestones, and documentation.


No React frontend or CMC-specific dependencies are required during the initial independent model-development stages.
Learning goals
This is also a from-scratch educational research project. The intention is to understand and implement the major components, not simply assemble an opaque pretrained solution.

By developing this project, I aim to gain practical knowledge of:
- Deep learning fundamentals: tensors, gradients, backpropagation, optimization, and regularization.
- Supervised learning: training loops, validation, loss functions, model calibration, and generalization.
- Neural architectures: CNNs, embeddings, attention, and transformers.
- Domain modeling: chess positions, legal-move masking, ratings, and game-outcome labels.
- Large-scale data engineering: streaming PGNs, data quality, columnar storage, and efficient sampling.
- Experimental ML: reproducibility, statistical baselines, ablation studies, and error analysis.
- Search and decision-making: opening trees, transpositions, opponent distributions, and constrained optimization.
- Reinforcement learning: reward design, simulation, policy improvement, and model-exploitation risks.
- Software engineering: modular design, automated tests, profiling, interfaces, and maintainable documentation.
Learning approach: Study the underlying theory, reproduce core techniques in small experiments, build the pieces independently, and scale only after understanding and validating them.
Study resources
- Dive into Deep Learning — applied neural-network foundations and PyTorch.
- Build a Large Language Model (From Scratch) — transformer implementation concepts.
- Designing Machine Learning Systems — reliable, scalable ML projects.
- Reinforcement Learning: An Introduction — foundational RL.
- Understanding Deep Learning — deeper theoretical reference.
Future integration: Chess Master Coach
Chess Master Coach (CMC) is a separate chess-training application. This repository focuses on training and validating the opening specialist on its own first.

After successful standalone evaluation, CMC could supply a user's rapid Elo, playing color, and requested opening/position to a versioned inference API. The specialist would return a structured, Elo-specific opening tree that CMC could present and use for study or practice.

Potential benefits to CMC:
- Personalized opening preparation instead of one generic repertoire for everyone.
- Training lines based on realistic opponent behavior at the user's level.
- Sound continuations against both common mistakes and good defense.
- Repertoires that can be revisited as the user's rapid Elo changes.
CMC integration is explicitly out of scope until the independent model and repertoire generator pass evaluation. The model should remain usable without CMC.
Scope and limitations
In scope: rapid-game learning, Elo-conditioned move prediction, opening-tree generation, practical evaluation, and soundness verification.

Out of scope for the initial release: a general-purpose playing engine, endgame mastery, replacing Stockfish, a complete CMC integration, and any claim of guaranteed wins or universal opening superiority.

Because game distributions shift and rare positions contain little data, the system will need calibration, uncertainty handling, and periodic re-evaluation. Its recommendations should be treated as preparation guidance, not proof that an opening guarantees victory.
Repository status
This repository contains the initial data/statistical-baseline implementation,
automated tests, a synthetic demo fixture, and the longer-term roadmap. It has
no real-game benchmark results or trained neural weights yet. Large raw game
archives, generated datasets, and model checkpoints should not be committed
directly to Git.
Acknowledgments and inspiration
Inspired by the practical, human-focused opening-preparation philosophy of Lotus Chess, and informed by research into Maia, Maia3, neural chess models, and classical chess-engine analysis. This is an independent project and is not affiliated with those teams.
