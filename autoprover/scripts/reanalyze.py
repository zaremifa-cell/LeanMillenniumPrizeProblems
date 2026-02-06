#!/usr/bin/env python3
"""Standalone reanalyze pass over a saved replay buffer.

Usage:
    python3 autoprover/scripts/reanalyze.py \\
        --checkpoint autoprover/data/checkpoints/step_5000.pt \\
        --replay     autoprover/data/replay.pkl

If no --replay is given, this script generates fresh self-play
trajectories, saves them, then reanalyzes.
"""

import argparse
import json
import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from autoprover.muzero.config import MuZeroConfig
from autoprover.muzero.model import ActionEncoder, MuZeroNet, ObsEncoder
from autoprover.muzero.reanalyze import reanalyze_trajectories
from autoprover.muzero.replay import ReplayBuffer
from autoprover.muzero.self_play import ACTION_TYPES, TheoremTask, build_tasks
from autoprover.muzero.trainer import build_vocab_from_decls, load_checkpoint
from autoprover.utils.decls import build_decl_map, load_decls


def main() -> None:
    ap = argparse.ArgumentParser(description="MuZero Reanalyze (offline)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--replay", default=None, help="pickled ReplayBuffer")
    ap.add_argument("--decls", default="autoprover/data/decls.jsonl")
    ap.add_argument("--graph", default="autoprover/data/kg.graph.json")
    ap.add_argument("--emb", default="autoprover/data/kg.emb.pt")
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--output", default=None, help="save updated replay buffer here")
    args = ap.parse_args()

    cfg = MuZeroConfig(
        decls_path=args.decls,
        graph_path=args.graph,
        emb_path=args.emb,
        reanalyze_fraction=args.fraction,
    )

    decls = load_decls(cfg.decls_path)
    with open(cfg.graph_path, "r") as f:
        graph = json.load(f)
    decl_map = build_decl_map(decls)
    all_tasks = build_tasks(decls, graph)
    tasks_by_name = {t.name: t for t in all_tasks}
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

    if args.replay and os.path.exists(args.replay):
        with open(args.replay, "rb") as f:
            replay: ReplayBuffer = pickle.load(f)
        print(f"Loaded replay buffer: {replay.num_trajectories()} trajectories, {len(replay)} steps")
    else:
        print("No replay buffer provided. Nothing to reanalyze.")
        return

    n = reanalyze_trajectories(
        replay, model, obs_encoder, action_encoder, vocab,
        graph, decl_map, node_to_idx, lemma_emb, cfg,
        tasks_by_name=tasks_by_name,
    )
    print(f"Reanalyzed {n} trajectories")

    out_path = args.output or args.replay
    with open(out_path, "wb") as f:
        pickle.dump(replay, f)
    print(f"Saved updated replay to {out_path}")


if __name__ == "__main__":
    main()
