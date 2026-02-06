#!/usr/bin/env python3
"""Evaluate a trained MuZero model on val / test theorems.

Usage:
    python3 autoprover/scripts/eval_full.py \\
        --checkpoint autoprover/data/checkpoints/best.pt \\
        --split val --episodes 100
"""

import argparse
import json
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from autoprover.muzero.config import MuZeroConfig
from autoprover.muzero.env import ProofEnv
from autoprover.muzero.mcts import MCTS
from autoprover.muzero.model import ActionEncoder, MuZeroNet, ObsEncoder
from autoprover.muzero.self_play import (
    ACTION_TYPES,
    TheoremTask,
    build_tasks,
    run_episode,
)
from autoprover.muzero.trainer import (
    build_vocab_from_decls,
    load_checkpoint,
    split_tasks,
)
from autoprover.utils.decls import build_decl_map, load_decls
from autoprover.utils.lean_oracle import LeanOracle


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate MuZero prover")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--decls", default="autoprover/data/decls.jsonl")
    ap.add_argument("--graph", default="autoprover/data/kg.graph.json")
    ap.add_argument("--emb", default="autoprover/data/kg.emb.pt")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--simulations", type=int, default=50)
    ap.add_argument("--max-proof-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = MuZeroConfig(
        decls_path=args.decls,
        graph_path=args.graph,
        emb_path=args.emb,
        num_simulations=args.simulations,
        max_proof_steps=args.max_proof_steps,
        eval_episodes=args.episodes,
    )

    decls = load_decls(cfg.decls_path)
    with open(cfg.graph_path, "r") as f:
        graph = json.load(f)
    decl_map = build_decl_map(decls)
    all_tasks = build_tasks(decls, graph)
    train_tasks, val_tasks, test_tasks = split_tasks(all_tasks, cfg)
    eval_tasks = val_tasks if args.split == "val" else test_tasks

    vocab = build_vocab_from_decls(decls, cfg.vocab_max_size)

    try:
        emb_data = torch.load(cfg.emb_path, map_location="cpu")
        lemma_emb = emb_data["embeddings"]
        node_to_idx = emb_data["node_to_idx"]
    except Exception:
        data = np.load(cfg.emb_path)
        lemma_emb = torch.tensor(data["embeddings"]).float()
        node_to_idx = data["node_to_idx"].item()

    obs_encoder = ObsEncoder(len(vocab), token_emb_dim=cfg.token_emb_dim,
                             kg_emb_dim=cfg.kg_emb_dim, out_dim=cfg.obs_dim)
    action_encoder = ActionEncoder(lemma_emb, num_action_types=len(ACTION_TYPES), out_dim=cfg.action_emb_dim)
    model = MuZeroNet(obs_dim=cfg.obs_dim, hidden_dim=cfg.hidden_dim,
                      action_dim=cfg.action_emb_dim, n_res_blocks=cfg.num_res_blocks)

    load_checkpoint(args.checkpoint, model, obs_encoder, action_encoder)
    model.eval()
    obs_encoder.eval()
    action_encoder.eval()

    oracle = LeanOracle()
    env = ProofEnv(oracle, max_steps=cfg.max_proof_steps)
    mcts = MCTS(c_puct=cfg.c_puct, num_simulations=cfg.num_simulations,
                dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_frac=0.0, discount=cfg.discount)

    samples = eval_tasks[:cfg.eval_episodes] if len(eval_tasks) >= cfg.eval_episodes else eval_tasks
    successes = 0
    total_len = 0
    total_lean = 0
    total_reward = 0.0

    print(f"Evaluating on {len(samples)} {args.split} tasks …")
    for i, task in enumerate(samples):
        traj = run_episode(
            task, env, model, obs_encoder, action_encoder, vocab,
            graph, decl_map, node_to_idx, lemma_emb, mcts, cfg,
            training_step=cfg.total_training_steps,
        )
        ok = "✓" if traj.success else "✗"
        total_len += len(traj.steps)
        total_lean += env.lean_calls
        total_reward += traj.total_reward
        if traj.success:
            successes += 1
        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            print(f"  [{i+1}/{len(samples)}] running success={successes}/{i+1} "
                  f"({successes/(i+1):.1%})")

    oracle.close()
    n = max(len(samples), 1)
    print("\n── Results ──")
    print(f"  split:            {args.split}")
    print(f"  episodes:         {n}")
    print(f"  success rate:     {successes}/{n} = {successes/n:.2%}")
    print(f"  avg proof length: {total_len/n:.1f}")
    print(f"  avg lean calls:   {total_lean/n:.1f}")
    print(f"  avg reward:       {total_reward/n:.3f}")


if __name__ == "__main__":
    main()
