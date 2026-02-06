#!/usr/bin/env python3
"""Train full MuZero pipeline for theorem proving.

Usage:
    python3 autoprover/scripts/train_full_muzero.py [OPTIONS]

Example:
    python3 autoprover/scripts/train_full_muzero.py \\
        --decls autoprover/data/decls.jsonl \\
        --graph autoprover/data/kg.graph.json \\
        --emb   autoprover/data/kg.emb.pt \\
        --steps 100000 --batch 64

See autoprover/muzero/config.py for all tuneable hyper-parameters.
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autoprover.muzero.config import MuZeroConfig
from autoprover.muzero.trainer import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Full MuZero training")
    # data
    ap.add_argument("--decls", default="autoprover/data/decls.jsonl")
    ap.add_argument("--graph", default="autoprover/data/kg.graph.json")
    ap.add_argument("--emb", default="autoprover/data/kg.emb.pt")
    # training
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--unroll", type=int, default=5, help="K unroll steps")
    ap.add_argument("--td-steps", type=int, default=10)
    ap.add_argument("--simulations", type=int, default=50)
    # replay
    ap.add_argument("--replay-capacity", type=int, default=100_000)
    # reanalyze
    ap.add_argument("--reanalyze-interval", type=int, default=50)
    ap.add_argument("--reanalyze-fraction", type=float, default=0.5)
    # eval
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=50)
    # checkpoint
    ap.add_argument("--checkpoint-dir", default="autoprover/data/checkpoints")
    ap.add_argument("--checkpoint-interval", type=int, default=500)
    ap.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    # hidden/obs dims
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--obs-dim", type=int, default=128)
    ap.add_argument("--max-proof-steps", type=int, default=30)
    args = ap.parse_args()

    cfg = MuZeroConfig(
        decls_path=args.decls,
        graph_path=args.graph,
        emb_path=args.emb,
        total_training_steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        num_unroll_steps=args.unroll,
        td_steps=args.td_steps,
        num_simulations=args.simulations,
        replay_capacity=args.replay_capacity,
        reanalyze_interval=args.reanalyze_interval,
        reanalyze_fraction=args.reanalyze_fraction,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        hidden_dim=args.hidden_dim,
        obs_dim=args.obs_dim,
        max_proof_steps=args.max_proof_steps,
    )

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
