"""Prioritised replay buffer with trajectory storage for MuZero.

Stores full game trajectories and samples positions for K-step unrolling.
Supports n-step bootstrapped value targets and priority sampling.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────── Data structures ──────────────────────────────

@dataclass
class StepRecord:
    """One step inside a trajectory."""
    obs: Dict                        # {goal, context, ast_size, binder_depth, head_symbol, kg_emb_key}
    action_idx: int
    reward: float
    policy_target: List[float]       # MCTS visit-count distribution
    value_target: float              # will be updated by n-step bootstrap / reanalyze
    root_value: float                # raw MCTS root value
    action_vecs_key: Optional[str] = None   # identifier for caching action-vecs


@dataclass
class Trajectory:
    """A completed proof-search episode."""
    task_name: str
    steps: List[StepRecord] = field(default_factory=list)
    success: bool = False
    total_reward: float = 0.0


@dataclass
class SampleBatch:
    """Flat batch returned by the replay buffer."""
    observations: List[Dict]
    action_indices: List[List[int]]          # K+1 actions per position
    rewards: List[List[float]]               # K rewards
    policy_targets: List[List[List[float]]]  # K+1 policies
    value_targets: List[List[float]]         # K+1 values
    weights: np.ndarray                      # importance-sampling weights


# ───────────────────── Transition (backward compat) ──────────────────

@dataclass
class Transition:
    obs: dict
    action_idx: int
    reward: float
    policy_target: List[float]
    value_target: float


# ───────────────────── Replay Buffer ─────────────────────────────────

class ReplayBuffer:
    """Stores full trajectories and samples positions for K-step unroll."""

    def __init__(
        self,
        capacity: int = 100_000,
        td_steps: int = 10,
        discount: float = 0.997,
        unroll_steps: int = 5,
        priority_alpha: float = 1.0,
    ) -> None:
        self.capacity = capacity          # measured in *trajectories*
        self.td_steps = td_steps
        self.discount = discount
        self.unroll_steps = unroll_steps
        self.alpha = priority_alpha

        self.trajectories: List[Trajectory] = []
        self.priorities: List[float] = []      # per trajectory
        self._total_steps = 0

    # ── storage ──

    def save_trajectory(self, traj: Trajectory) -> None:
        """Compute n-step value targets then store."""
        self._compute_value_targets(traj)
        if len(self.trajectories) >= self.capacity:
            removed = self.trajectories.pop(0)
            self.priorities.pop(0)
            self._total_steps -= len(removed.steps)
        self.trajectories.append(traj)
        self.priorities.append(self._trajectory_priority(traj))
        self._total_steps += len(traj.steps)

    def update_trajectory(self, idx: int, traj: Trajectory) -> None:
        """Replace trajectory at *idx* (used by reanalyze)."""
        self._compute_value_targets(traj)
        old_len = len(self.trajectories[idx].steps)
        self.trajectories[idx] = traj
        self.priorities[idx] = self._trajectory_priority(traj)
        self._total_steps += len(traj.steps) - old_len

    # ── backward compat push ──

    def push(self, t: Transition) -> None:
        """MVP compat: wrap a single transition into a 1-step trajectory."""
        sr = StepRecord(
            obs=t.obs, action_idx=t.action_idx, reward=t.reward,
            policy_target=t.policy_target, value_target=t.value_target,
            root_value=t.value_target,
        )
        traj = Trajectory(task_name="", steps=[sr], success=(t.reward >= 1.0))
        self.save_trajectory(traj)

    # ── sampling ──

    def sample(self, batch_size: int, beta: float = 0.4) -> SampleBatch:
        """Priority-weighted sampling of positions, each with K-step unroll."""
        probs = self._priority_probs()
        indices = np.random.choice(len(self.trajectories), size=batch_size, p=probs, replace=True)

        obs_list, act_list, rew_list, pol_list, val_list = [], [], [], [], []
        weights = np.zeros(batch_size, dtype=np.float32)

        for i, traj_idx in enumerate(indices):
            traj = self.trajectories[traj_idx]
            if len(traj.steps) == 0:
                continue
            pos = random.randint(0, len(traj.steps) - 1)

            acts, rews, pols, vals = [], [], [], []
            for k in range(self.unroll_steps + 1):
                t = pos + k
                if t < len(traj.steps):
                    step = traj.steps[t]
                    acts.append(step.action_idx)
                    rews.append(step.reward)
                    pols.append(step.policy_target)
                    vals.append(step.value_target)
                else:
                    # absorbing state – zero padding
                    acts.append(0)
                    rews.append(0.0)
                    pols.append(pols[-1] if pols else [0.0])
                    vals.append(0.0)

            obs_list.append(traj.steps[pos].obs)
            act_list.append(acts)
            rew_list.append(rews[1:])      # rewards for transitions 0..K-1
            pol_list.append(pols)
            val_list.append(vals)

            # IS weight
            w = (1.0 / (len(self.trajectories) * probs[traj_idx])) ** beta
            weights[i] = w

        weights /= weights.max() + 1e-8
        return SampleBatch(obs_list, act_list, rew_list, pol_list, val_list, weights)

    # ── helpers ──

    def _compute_value_targets(self, traj: Trajectory) -> None:
        """n-step bootstrapped value targets (paper eq. in appendix)."""
        T = len(traj.steps)
        for t in range(T):
            bootstrap = 0.0
            discount_power = 1.0
            value = 0.0
            for n in range(self.td_steps):
                idx = t + n
                if idx >= T:
                    break
                value += discount_power * traj.steps[idx].reward
                discount_power *= self.discount
            # bootstrap from root_value of step t+n if available
            boot_idx = t + self.td_steps
            if boot_idx < T:
                value += discount_power * traj.steps[boot_idx].root_value
            traj.steps[t].value_target = value

    @staticmethod
    def _trajectory_priority(traj: Trajectory) -> float:
        if not traj.steps:
            return 1e-6
        return max(abs(s.root_value - s.value_target) for s in traj.steps) + 1e-6

    def _priority_probs(self) -> np.ndarray:
        if not self.priorities:
            return np.array([1.0])
        p = np.array(self.priorities, dtype=np.float64) ** self.alpha
        return p / p.sum()

    def __len__(self) -> int:
        return self._total_steps

    def num_trajectories(self) -> int:
        return len(self.trajectories)
