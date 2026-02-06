"""MCTS with PUCT for MuZero (variable action-space, Dirichlet noise)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class Node:
    state: Optional[Tensor] = None
    prior: float = 0.0
    reward: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0


class MinMaxStats:
    """Normalise values into [0,1] for stable PUCT."""

    def __init__(self) -> None:
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, v: float) -> None:
        self.min_val = min(self.min_val, v)
        self.max_val = max(self.max_val, v)

    def normalize(self, v: float) -> float:
        span = self.max_val - self.min_val
        if span < 1e-8:
            return v
        return (v - self.min_val) / span


class MCTS:
    """Full MuZero MCTS with PUCT selection and Dirichlet root noise."""

    def __init__(
        self,
        c_puct: float = 2.5,
        num_simulations: int = 50,
        dirichlet_alpha: float = 0.25,
        dirichlet_frac: float = 0.25,
        discount: float = 0.997,
    ) -> None:
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.discount = discount

    # ──────────────────────── public ────────────────────────

    @torch.no_grad()
    def run(
        self,
        model,
        root_state: Tensor,
        action_vecs: Tensor,
        add_noise: bool = True,
    ) -> Tuple[List[float], float, Node]:
        """Run simulations and return (visit counts, root value, root node)."""
        num_actions = action_vecs.shape[0]
        root = Node(state=root_state)
        mm = MinMaxStats()

        # expand root
        logits, value = model.predict(root_state, action_vecs)
        priors = torch.softmax(logits.squeeze(0), 0).cpu().numpy()
        self._expand(root, priors, float(value.squeeze()))

        # Dirichlet noise at root
        if add_noise and num_actions > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * num_actions)
            eps = self.dirichlet_frac
            for a, child in root.children.items():
                child.prior = (1 - eps) * child.prior + eps * noise[a]

        mm.update(root.value)

        # simulations
        for _ in range(self.num_simulations):
            node = root
            path: List[Node] = [node]
            actions_taken: List[int] = []

            # ── select ──
            while node.expanded:
                action, child = self._select_child(node, mm)
                actions_taken.append(action)
                if child.state is None:
                    # dynamics expansion
                    a_vec = action_vecs[action].unsqueeze(0)
                    ns, reward = model.dynamics_step(
                        node.state.unsqueeze(0) if node.state.dim() == 1 else node.state,
                        a_vec,
                    )
                    child.state = ns.squeeze(0)
                    child.reward = float(reward.item())
                path.append(child)
                node = child

            # ── evaluate leaf ──
            if node.state is not None:
                lg, val = model.predict(node.state, action_vecs)
                pr = torch.softmax(lg.squeeze(0), 0).cpu().numpy()
                leaf_value = float(val.squeeze())
                self._expand(node, pr, leaf_value)
            else:
                leaf_value = 0.0

            # ── backprop ──
            self._backpropagate(path, leaf_value, mm)

        # collect visit counts
        visits = [0.0] * num_actions
        for a, ch in root.children.items():
            visits[a] = float(ch.visit_count)
        return visits, root.value, root

    # ──────────────────────── internals ─────────────────────

    def _expand(self, node: Node, priors: np.ndarray, value: float) -> None:
        for a in range(len(priors)):
            if a not in node.children:
                node.children[a] = Node(prior=float(priors[a]))
        node.value_sum += value
        node.visit_count += 1

    def _select_child(self, node: Node, mm: MinMaxStats) -> Tuple[int, Node]:
        total = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(total + 1)
        best_score = float("-inf")
        best_a, best_c = -1, list(node.children.values())[0]
        for a, child in node.children.items():
            q = mm.normalize(child.value) if child.visit_count > 0 else 0.0
            exploration = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + exploration
            if score > best_score:
                best_score = score
                best_a = a
                best_c = child
        return best_a, best_c

    def _backpropagate(self, path: List[Node], leaf_value: float, mm: MinMaxStats) -> None:
        value = leaf_value
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            mm.update(node.value)
            value = node.reward + self.discount * value
