"""Centralised configuration for the full MuZero training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MuZeroConfig:
    # ── network dims ──
    vocab_max_size: int = 12_000
    token_emb_dim: int = 64
    obs_dim: int = 128
    hidden_dim: int = 256
    action_emb_dim: int = 128
    kg_emb_dim: int = 64
    num_res_blocks: int = 4

    # ── MCTS ──
    num_simulations: int = 50
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.25
    dirichlet_frac: float = 0.25
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_decay_steps: int = 5_000

    # ── environment ──
    max_proof_steps: int = 30
    max_premises: int = 64

    # ── training ──
    num_unroll_steps: int = 5          # K in the paper
    td_steps: int = 10                 # n-step bootstrap
    discount: float = 0.997
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lr_warmup_steps: int = 500
    total_training_steps: int = 100_000
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.25
    reward_loss_weight: float = 1.0

    # ── replay ──
    replay_capacity: int = 100_000
    min_replay_size: int = 256
    priority_alpha: float = 1.0
    priority_beta_init: float = 0.4
    priority_beta_final: float = 1.0

    # ── target network ──
    target_net_update_interval: int = 200

    # ── self-play ──
    selfplay_episodes_per_step: int = 2

    # ── reanalyze ──
    reanalyze_fraction: float = 0.5
    reanalyze_interval: int = 50

    # ── curriculum ──
    curriculum_stages: List[int] = field(default_factory=lambda: [0, 2_000, 5_000])
    curriculum_max_depths: List[int] = field(default_factory=lambda: [3, 6, 999])

    # ── eval ──
    eval_interval: int = 500
    eval_episodes: int = 50

    # ── checkpointing / logging ──
    checkpoint_dir: str = "autoprover/data/checkpoints"
    checkpoint_interval: int = 500
    log_interval: int = 50

    # ── data paths ──
    decls_path: str = "autoprover/data/decls.jsonl"
    graph_path: str = "autoprover/data/kg.graph.json"
    emb_path: str = "autoprover/data/kg.emb.pt"

    # ── splits ──
    train_size: int = 80_000
    val_size: int = 10_000
    test_size: int = 10_000

    # ── action types (indices) ──
    action_types: List[str] = field(
        default_factory=lambda: ["apply", "exact", "rw_fwd", "rw_bwd", "intro"]
    )

    def curriculum_max_depth(self, training_step: int) -> int:
        depth = self.curriculum_max_depths[0]
        for stage, d in zip(self.curriculum_stages, self.curriculum_max_depths):
            if training_step >= stage:
                depth = d
        return depth

    def temperature(self, training_step: int) -> float:
        frac = min(training_step / max(self.temperature_decay_steps, 1), 1.0)
        return self.temperature_init + frac * (self.temperature_final - self.temperature_init)

    def priority_beta(self, training_step: int) -> float:
        frac = min(training_step / max(self.total_training_steps, 1), 1.0)
        return self.priority_beta_init + frac * (self.priority_beta_final - self.priority_beta_init)
