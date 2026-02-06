# AutoProver (MuZero + Knowledge Graph)

Theorem proving system that uses **MuZero** (Schrittwieser et al., 2020) with a
Knowledge‑Graph backbone to learn to prove Lean 4 / Mathlib theorems.

## Quick Start (data prep)

```bash
# 1. Build the Lean project + fetch Mathlib cache
lake build
lake exe cache get

# 2. Smoke-test the Lean oracle
python3 autoprover/scripts/smoke_oracle.py

# 3. Extract theorem declarations
python3 autoprover/scripts/extract_deps.py \
    --out autoprover/data/decls.jsonl --limit 100000

# 4. Build knowledge graph
python3 autoprover/scripts/build_kg.py \
    --in autoprover/data/decls.jsonl \
    --out autoprover/data/kg.graph.json

# 5. Pretrain KG embeddings
python3 autoprover/scripts/pretrain_embeddings.py \
    --graph autoprover/data/kg.graph.json \
    --out autoprover/data/kg.emb.pt
```

## Training (full MuZero)

The full training pipeline implements the MuZero paper end-to-end:

| Feature | Implementation |
|---|---|
| Representation / Dynamics / Prediction | `muzero/model.py` – ResBlock stacks with LayerNorm |
| K‑step unroll loss (policy + value + reward) | `muzero/trainer.py` → `compute_loss()` |
| MCTS with PUCT + Dirichlet noise | `muzero/mcts.py` – MinMaxStats normalisation |
| Self‑play trajectory generation | `muzero/self_play.py` → `run_episode()` |
| Prioritised replay + n‑step bootstrap | `muzero/replay.py` |
| MuZero Reanalyze | `muzero/reanalyze.py` |
| Target network for value stability | trainer copies weights every N steps |
| Curriculum learning | depth-based gating in config |
| Checkpointing (periodic + best val) | `trainer.py` → `save_checkpoint()` |
| Metrics / Logging | success rate, avg length, lean calls, reward |

### Launch training

```bash
python3 autoprover/scripts/train_full_muzero.py \
    --decls autoprover/data/decls.jsonl \
    --graph autoprover/data/kg.graph.json \
    --emb   autoprover/data/kg.emb.pt \
    --steps 100000 --batch 64 --simulations 50 \
    --checkpoint-dir autoprover/data/checkpoints
```

Key flags:
- `--unroll 5` – K‑step unroll depth (default 5)
- `--td-steps 10` – n‑step bootstrap horizon
- `--reanalyze-interval 50` – re‑run MCTS on stored trajectories
- `--resume /path/to/ckpt.pt` – continue from a checkpoint

### Evaluate

```bash
python3 autoprover/scripts/eval_full.py \
    --checkpoint autoprover/data/checkpoints/best.pt \
    --split val --episodes 100
```

### Offline Reanalyze

```bash
python3 autoprover/scripts/reanalyze.py \
    --checkpoint autoprover/data/checkpoints/step_5000.pt \
    --replay     autoprover/data/replay.pkl \
    --fraction 0.5
```

## Legacy MVP Training

The original MVP scripts are still available:

```bash
python3 autoprover/scripts/train_muzero.py \
    --decls autoprover/data/decls.jsonl \
    --graph autoprover/data/kg.graph.json \
    --emb   autoprover/data/kg.emb.pt

python3 autoprover/scripts/eval_muzero.py \
    --decls autoprover/data/decls.jsonl --limit 50
```

## Architecture

```
observation ──► ObsEncoder ──► RepresentationNet ──► s₀
                                                      │
            ┌──────────────────── MCTS loop ──────────┤
            │                                         ▼
            │  action_vec ──► DynamicsNet(sₜ,a) ──► sₜ₊₁, rₜ
            │                                         │
            │                 PredictionNet(sₜ₊₁) ──► πₜ₊₁, vₜ₊₁
            └─────────────────────────────────────────┘
```

## Action Space

| Action | Oracle payload |
|---|---|
| `apply lemma` | `{"kind":"apply","lemma":"..."}` |
| `exact lemma` | `{"kind":"exact","lemma":"..."}` |
| `rw lemma` | `{"kind":"rw","lemma":"...","direction":"forward"}` |
| `rw ← lemma` | `{"kind":"rw","lemma":"...","direction":"backward"}` |
| `intro` | `{"kind":"intro"}` |

Direction filtering: `rw` candidates are only generated when the LHS/RHS
tokens overlap with the current goal.

## Notes

- The Lean oracle uses `lake env lean --run` with `import Mathlib` at runtime.
- Run `lake exe cache get` first to ensure all `*.olean` files exist.
- For speed the oracle runs as a long‑lived subprocess (Python `LeanOracle` class).

## Files

- `lean/ExtractDeps.lean` – Lean extractor for Mathlib declarations
- `lean/Oracle.lean` – JSON stdin/stdout Lean tactic server
- `kg/build_graph.py` – knowledge graph builder
- `kg/pretrain_embeddings.py` – contrastive pretrain for node embeddings
- `muzero/config.py` – all hyper‑parameters
- `muzero/model.py` – Representation / Dynamics / Prediction networks
- `muzero/mcts.py` – MCTS with PUCT + Dirichlet noise
- `muzero/env.py` – Lean proof environment
- `muzero/replay.py` – prioritised trajectory replay buffer
- `muzero/self_play.py` – self-play episode generation
- `muzero/reanalyze.py` – MuZero Reanalyze
- `muzero/trainer.py` – full training loop
- `scripts/train_full_muzero.py` – CLI for training
- `scripts/eval_full.py` – CLI for evaluation
- `scripts/reanalyze.py` – CLI for offline reanalyze
