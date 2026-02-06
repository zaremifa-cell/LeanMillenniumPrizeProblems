"""Full MuZero training loop.

Implements:
 - K-step model unrolling with policy + value + reward loss
 - Self-play episode generation using MCTS
 - Prioritised replay with n-step bootstrap targets
 - MuZero Reanalyze (periodic MCTS re-evaluation)
 - Target network for value stability
 - Curriculum learning by theorem depth
 - Checkpointing (periodic + best validation)
 - Metrics logging
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from autoprover.muzero.config import MuZeroConfig
from autoprover.muzero.env import ProofEnv
from autoprover.muzero.mcts import MCTS
from autoprover.muzero.model import Action, ActionEncoder, MuZeroNet, ObsEncoder
from autoprover.muzero.reanalyze import reanalyze_trajectories
from autoprover.muzero.replay import ReplayBuffer, SampleBatch, Trajectory
from autoprover.muzero.self_play import (
    ACTION_TYPES,
    TheoremTask,
    action_candidates,
    build_tasks,
    encode_actions,
    encode_obs,
    run_episode,
    select_premises,
)
from autoprover.utils.decls import build_decl_map, load_decls
from autoprover.utils.lean_oracle import LeanOracle
from autoprover.utils.tokenize import tokenize
from autoprover.utils.vocab import Vocab


# ──────────────────────── Vocab builder ──────────────────────────────

def build_vocab_from_decls(decls: List[Dict], max_size: int = 12_000) -> Vocab:
    tokens: List[str] = []
    for d in decls:
        tokens.extend(d.get("type_expr_tokens", []))
        tokens.extend(tokenize(d.get("type_expr", "")))
    return Vocab(tokens, max_size=max_size)


# ──────────────────────── Data splitting ─────────────────────────────

def split_tasks(
    tasks: List[TheoremTask], cfg: MuZeroConfig,
) -> Tuple[List[TheoremTask], List[TheoremTask], List[TheoremTask]]:
    random.shuffle(tasks)
    n = len(tasks)
    tr = min(cfg.train_size, n)
    va = min(cfg.val_size, n - tr)
    te = min(cfg.test_size, n - tr - va)
    return tasks[:tr], tasks[tr:tr + va], tasks[tr + va:tr + va + te]


# ──────────────────────── Metrics tracker ────────────────────────────

@dataclass
class Metrics:
    total_episodes: int = 0
    successes: int = 0
    total_steps_all: int = 0
    total_lean_calls: int = 0
    total_reward: float = 0.0
    episode_lengths: List[int] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    eval_success_rates: List[Tuple[int, float]] = field(default_factory=list)

    def log_episode(self, traj: Trajectory, lean_calls: int) -> None:
        self.total_episodes += 1
        self.total_reward += traj.total_reward
        self.episode_lengths.append(len(traj.steps))
        if traj.success:
            self.successes += 1
        self.total_steps_all += len(traj.steps)
        self.total_lean_calls += lean_calls

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.total_episodes, 1)

    @property
    def avg_length(self) -> float:
        return sum(self.episode_lengths[-100:]) / max(len(self.episode_lengths[-100:]), 1)

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.total_episodes, 1)

    def summary(self, step: int) -> str:
        return (
            f"[step {step}] episodes={self.total_episodes} "
            f"success={self.success_rate:.2%} avg_len={self.avg_length:.1f} "
            f"avg_reward={self.avg_reward:.3f} lean_calls={self.total_lean_calls} "
            f"loss={self.loss_history[-1]:.4f}" if self.loss_history else
            f"[step {step}] episodes={self.total_episodes} "
            f"success={self.success_rate:.2%} avg_len={self.avg_length:.1f}"
        )


# ──────────────────────── K-step loss ────────────────────────────────

def compute_loss(
    batch: SampleBatch,
    model: MuZeroNet,
    target_model: MuZeroNet,
    obs_encoder: ObsEncoder,
    action_encoder: ActionEncoder,
    vocab: Vocab,
    node_to_idx: Dict[str, int],
    kg_emb_weight: Optional[Tensor],
    graph: Dict,
    decl_map: Dict[str, Dict],
    cfg: MuZeroConfig,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute the full MuZero loss over a batch with K-step unrolling."""
    total_policy_loss = torch.tensor(0.0)
    total_value_loss = torch.tensor(0.0)
    total_reward_loss = torch.tensor(0.0)
    count = 0

    for i in range(len(batch.observations)):
        obs = batch.observations[i]
        acts = batch.action_indices[i]
        rews = batch.rewards[i]
        pols = batch.policy_targets[i]
        vals = batch.value_targets[i]
        w = float(batch.weights[i])

        goal = obs["goal"]
        context = obs.get("context", [])
        ast_size = obs.get("ast_size", 1)
        binder_depth = obs.get("binder_depth", 0)
        head_symbol = obs.get("head_symbol", "")
        type_consts = obs.get("type_consts", [])

        tok, ast_f, kg = encode_obs(
            vocab, goal, context, ast_size, binder_depth, head_symbol,
            kg_emb_weight, node_to_idx, type_consts,
        )
        obs_vec = obs_encoder(tok, ast_f, kg)
        state = model.encode(obs_vec)       # (1, H)

        # action candidates for logits (rebuild from goal)
        goal_tokens = tokenize(goal)
        premises = []  # simplified: we use a small surrogate set
        all_actions = action_candidates(goal_tokens, premises, decl_map)
        if not all_actions:
            all_actions = [Action(kind="intro")]
        a_vecs = encode_actions(all_actions, node_to_idx, action_encoder)

        # initial prediction
        logits, value = model.predict(state, a_vecs)

        # policy loss at t=0
        target_pol = _make_policy_tensor(pols[0], logits.shape[-1])
        total_policy_loss = total_policy_loss + w * _cross_entropy(logits.squeeze(0), target_pol)
        total_value_loss = total_value_loss + w * F.mse_loss(value.squeeze(), torch.tensor(vals[0]))

        # K-step unroll
        for k in range(min(cfg.num_unroll_steps, len(rews))):
            act_idx = acts[k]
            if act_idx < a_vecs.shape[0]:
                a_vec = a_vecs[act_idx].unsqueeze(0)
            else:
                a_vec = a_vecs[0].unsqueeze(0)  # fallback

            state, reward_pred = model.dynamics_step(state, a_vec)
            logits_k, value_k = model.predict(state, a_vecs)

            # reward loss
            total_reward_loss = total_reward_loss + w * F.mse_loss(reward_pred.squeeze(), torch.tensor(rews[k]))

            # policy + value at step k+1
            if k + 1 < len(pols):
                tp = _make_policy_tensor(pols[k + 1], logits_k.shape[-1])
                total_policy_loss = total_policy_loss + w * _cross_entropy(logits_k.squeeze(0), tp)
            if k + 1 < len(vals):
                total_value_loss = total_value_loss + w * F.mse_loss(value_k.squeeze(), torch.tensor(vals[k + 1]))

        count += 1

    n = max(count, 1)
    loss = (
        cfg.policy_loss_weight * total_policy_loss / n
        + cfg.value_loss_weight * total_value_loss / n
        + cfg.reward_loss_weight * total_reward_loss / n
    )
    info = {
        "policy_loss": float(total_policy_loss / n),
        "value_loss": float(total_value_loss / n),
        "reward_loss": float(total_reward_loss / n),
        "total_loss": float(loss),
    }
    return loss, info


def _make_policy_tensor(pol: List[float], size: int) -> Tensor:
    if len(pol) >= size:
        return torch.tensor(pol[:size], dtype=torch.float32)
    return torch.tensor(pol + [0.0] * (size - len(pol)), dtype=torch.float32)


def _cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    """Cross-entropy between logits and a soft target distribution."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum()


# ──────────────────────── Eval helpers ───────────────────────────────

@torch.no_grad()
def evaluate(
    tasks: List[TheoremTask],
    env: ProofEnv,
    model: MuZeroNet,
    obs_encoder: ObsEncoder,
    action_encoder: ActionEncoder,
    vocab: Vocab,
    graph: Dict,
    decl_map: Dict[str, Dict],
    node_to_idx: Dict[str, int],
    kg_emb_weight: Optional[Tensor],
    cfg: MuZeroConfig,
) -> Dict[str, float]:
    mcts = MCTS(c_puct=cfg.c_puct, num_simulations=cfg.num_simulations,
                dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_frac=0.0, discount=cfg.discount)
    successes, total_len, total_lean = 0, 0, 0
    samples = tasks[:cfg.eval_episodes] if len(tasks) >= cfg.eval_episodes else tasks
    for task in samples:
        traj = run_episode(
            task, env, model, obs_encoder, action_encoder, vocab,
            graph, decl_map, node_to_idx, kg_emb_weight, mcts, cfg, training_step=cfg.total_training_steps,
        )
        if traj.success:
            successes += 1
        total_len += len(traj.steps)
        total_lean += env.lean_calls
    n = max(len(samples), 1)
    return {
        "success_rate": successes / n,
        "avg_proof_length": total_len / n,
        "avg_lean_calls": total_lean / n,
    }


# ──────────────────────── Checkpointing ──────────────────────────────

def save_checkpoint(
    path: str,
    step: int,
    model: MuZeroNet,
    obs_encoder: ObsEncoder,
    action_encoder: ActionEncoder,
    optimizer: torch.optim.Optimizer,
    metrics: Metrics,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "obs_encoder": obs_encoder.state_dict(),
        "action_encoder": action_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": {
            "total_episodes": metrics.total_episodes,
            "successes": metrics.successes,
            "success_rate": metrics.success_rate,
        },
    }, path)


def load_checkpoint(
    path: str,
    model: MuZeroNet,
    obs_encoder: ObsEncoder,
    action_encoder: ActionEncoder,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    obs_encoder.load_state_dict(ckpt["obs_encoder"])
    action_encoder.load_state_dict(ckpt["action_encoder"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)


# ──────────────────────── Main train loop ────────────────────────────

def train(
    cfg: MuZeroConfig,
    resume_path: Optional[str] = None,
) -> None:
    print(f"Loading data from {cfg.decls_path} …")
    decls = load_decls(cfg.decls_path)
    with open(cfg.graph_path, "r") as f:
        graph = json.load(f)
    decl_map = build_decl_map(decls)
    all_tasks = build_tasks(decls, graph)
    if not all_tasks:
        raise RuntimeError("No tasks found – check decls.jsonl")

    train_tasks, val_tasks, test_tasks = split_tasks(all_tasks, cfg)
    print(f"Tasks: {len(train_tasks)} train / {len(val_tasks)} val / {len(test_tasks)} test")

    vocab = build_vocab_from_decls(decls, cfg.vocab_max_size)

    # Load KG embeddings
    try:
        emb_data = torch.load(cfg.emb_path, map_location="cpu")
        lemma_emb: Tensor = emb_data["embeddings"]
        node_to_idx: Dict[str, int] = emb_data["node_to_idx"]
    except Exception:
        data = np.load(cfg.emb_path)
        lemma_emb = torch.tensor(data["embeddings"]).float()
        node_to_idx = data["node_to_idx"].item()

    kg_emb_weight = lemma_emb

    # ── Build networks ──
    obs_encoder = ObsEncoder(len(vocab), token_emb_dim=cfg.token_emb_dim,
                             kg_emb_dim=cfg.kg_emb_dim, out_dim=cfg.obs_dim)
    action_encoder = ActionEncoder(lemma_emb, num_action_types=len(ACTION_TYPES), out_dim=cfg.action_emb_dim)
    model = MuZeroNet(obs_dim=cfg.obs_dim, hidden_dim=cfg.hidden_dim,
                      action_dim=cfg.action_emb_dim, n_res_blocks=cfg.num_res_blocks)
    target_model = copy.deepcopy(model)
    target_model.eval()

    params = list(obs_encoder.parameters()) + list(action_encoder.parameters()) + list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # LR warmup + cosine decay
    def lr_lambda(step: int) -> float:
        if step < cfg.lr_warmup_steps:
            return step / max(cfg.lr_warmup_steps, 1)
        progress = (step - cfg.lr_warmup_steps) / max(cfg.total_training_steps - cfg.lr_warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    replay = ReplayBuffer(
        capacity=cfg.replay_capacity, td_steps=cfg.td_steps,
        discount=cfg.discount, unroll_steps=cfg.num_unroll_steps,
        priority_alpha=cfg.priority_alpha,
    )

    start_step = 0
    if resume_path and os.path.exists(resume_path):
        start_step = load_checkpoint(resume_path, model, obs_encoder, action_encoder, optimizer)
        target_model.load_state_dict(model.state_dict())
        print(f"Resumed from step {start_step}")

    metrics = Metrics()
    best_val_rate = -1.0
    tasks_by_name = {t.name: t for t in all_tasks}

    oracle = LeanOracle()
    env = ProofEnv(oracle, max_steps=cfg.max_proof_steps)
    mcts_play = MCTS(c_puct=cfg.c_puct, num_simulations=cfg.num_simulations,
                     dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_frac=cfg.dirichlet_frac,
                     discount=cfg.discount)

    print("Starting training …")
    t0 = time.time()

    for step in range(start_step, cfg.total_training_steps):
        # ── self-play ──
        for _ in range(cfg.selfplay_episodes_per_step):
            max_depth = cfg.curriculum_max_depth(step)
            eligible = [t for t in train_tasks if t.depth <= max_depth]
            if not eligible:
                eligible = train_tasks
            task = random.choice(eligible)

            model.eval()
            obs_encoder.eval()
            action_encoder.eval()

            traj = run_episode(
                task, env, model, obs_encoder, action_encoder, vocab,
                graph, decl_map, node_to_idx, kg_emb_weight,
                mcts_play, cfg, training_step=step, fallback_tasks=train_tasks,
            )
            replay.save_trajectory(traj)
            metrics.log_episode(traj, env.lean_calls)

        # ── training step ──
        if len(replay) >= cfg.min_replay_size:
            model.train()
            obs_encoder.train()
            action_encoder.train()

            batch = replay.sample(cfg.batch_size, beta=cfg.priority_beta(step))
            optimizer.zero_grad()
            loss, loss_info = compute_loss(
                batch, model, target_model, obs_encoder, action_encoder,
                vocab, node_to_idx, kg_emb_weight, graph, decl_map, cfg,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            metrics.loss_history.append(loss_info["total_loss"])

        # ── target network update ──
        if step > 0 and step % cfg.target_net_update_interval == 0:
            target_model.load_state_dict(model.state_dict())

        # ── reanalyze ──
        if step > 0 and step % cfg.reanalyze_interval == 0 and replay.num_trajectories() > 0:
            model.eval()
            obs_encoder.eval()
            action_encoder.eval()
            n_re = reanalyze_trajectories(
                replay, model, obs_encoder, action_encoder, vocab,
                graph, decl_map, node_to_idx, kg_emb_weight, cfg,
                training_step=step, tasks_by_name=tasks_by_name,
            )

        # ── logging ──
        if step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            print(f"{metrics.summary(step)}  replay={len(replay)}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={elapsed:.0f}s")

        # ── evaluation ──
        if step > 0 and step % cfg.eval_interval == 0 and val_tasks:
            model.eval()
            obs_encoder.eval()
            action_encoder.eval()
            eval_res = evaluate(
                val_tasks, env, model, obs_encoder, action_encoder, vocab,
                graph, decl_map, node_to_idx, kg_emb_weight, cfg,
            )
            sr = eval_res["success_rate"]
            metrics.eval_success_rates.append((step, sr))
            print(f"  [eval] success={sr:.2%} avg_len={eval_res['avg_proof_length']:.1f} "
                  f"lean_calls={eval_res['avg_lean_calls']:.1f}")
            if sr > best_val_rate:
                best_val_rate = sr
                save_checkpoint(
                    os.path.join(cfg.checkpoint_dir, "best.pt"),
                    step, model, obs_encoder, action_encoder, optimizer, metrics,
                )
                print(f"  ✓ new best val {sr:.2%}")

        # ── periodic checkpoint ──
        if step > 0 and step % cfg.checkpoint_interval == 0:
            save_checkpoint(
                os.path.join(cfg.checkpoint_dir, f"step_{step}.pt"),
                step, model, obs_encoder, action_encoder, optimizer, metrics,
            )

    # ── final checkpoint ──
    save_checkpoint(
        os.path.join(cfg.checkpoint_dir, "final.pt"),
        cfg.total_training_steps, model, obs_encoder, action_encoder, optimizer, metrics,
    )
    oracle.close()
    print(f"Training done. Final success rate: {metrics.success_rate:.2%}")
