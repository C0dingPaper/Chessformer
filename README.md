## Chessformer
An Elo-aware chess opening specialist trained on human rapid games.
Status: Planning / early development. The architecture, training approach, and features below are goals, not claims of a completed model.

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
- Start with 100,000 games, then expand through 1M → 10M → 50M → 200M based on measured value.
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
- [ ] Phase 1 — Foundations: Define data schemas, reproducible experiments, evaluation criteria, and the repository structure.
- [ ] Phase 2 — Data prototype: Process and verify the first 100,000 rapid games; create position/move/rating training records.
- [ ] Phase 3 — Baselines: Implement rating-aware opening statistics and a simple move predictor.
- [ ] Phase 4 — Neural prototype: Train and compare a compact CNN and rating-conditioned transformer on held-out games.
- [ ] Phase 5 — Repertoire generator: Build opening-tree search with transpositions, branching coverage, and Stockfish verification.
- [ ] Phase 6 — Scaling: Move through progressively larger datasets, measuring whether additional data improves results.
- [ ] Phase 7 — Advanced experiments: Evaluate outcome/mistake heads and, optionally, reinforcement learning against simulated opponents.
- [ ] Phase 8 — Independent validation: Test performance, calibration, robustness, and reliability against established baselines.
- [ ] Phase 9 — Future CMC integration: Expose a stable inference interface only after standalone validation.
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
This repository is currently a project specification and implementation roadmap. Code, datasets, benchmarks, trained weights, and usage instructions will be added as development proceeds. Large raw game archives and model checkpoints should not be committed directly to Git.
Acknowledgments and inspiration
Inspired by the practical, human-focused opening-preparation philosophy of Lotus Chess, and informed by research into Maia, Maia3, neural chess models, and classical chess-engine analysis. This is an independent project and is not affiliated with those teams.
