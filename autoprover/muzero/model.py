"""Full MuZero networks: Representation, Dynamics, Prediction.

Follows the MuZero paper (Schrittwieser et al., 2020) adapted for
theorem-proving with variable-size action spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ───────────────────────── Action dataclass ──────────────────────────

@dataclass
class Action:
    kind: str                       # apply | exact | rw | intro
    lemma: Optional[str] = None
    direction: Optional[str] = None  # forward | backward


# ───────────────────────── Building blocks ───────────────────────────

class ResBlock(nn.Module):
    """Pre-activation residual block with LayerNorm."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = F.relu(self.ln(x))
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out + residual


def _res_stack(dim: int, n: int) -> nn.Sequential:
    return nn.Sequential(*[ResBlock(dim) for _ in range(n)])


# ───────────────────── Observation encoder ───────────────────────────

class ObsEncoder(nn.Module):
    """Encodes (goal tokens, AST features, hyp tokens, KG embedding) → obs vec."""

    def __init__(
        self,
        vocab_size: int,
        token_emb_dim: int = 64,
        kg_emb_dim: int = 64,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, token_emb_dim, padding_idx=0)
        # AST features: ast_size (1) + binder_depth (1) + head_sym_id (1) = 3
        feat_input = token_emb_dim + 3 + kg_emb_dim
        self.goal_proj = nn.Sequential(
            nn.Linear(feat_input, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim),
        )
        self.hyp_proj = nn.Sequential(
            nn.Linear(token_emb_dim, out_dim), nn.ReLU(),
        )
        self.combine = nn.Linear(out_dim * 2, out_dim)

    def forward(
        self,
        token_ids: Tensor,          # (B, T)
        ast_features: Tensor,       # (B, 3)
        kg_emb: Tensor,             # (B, kg_dim)
        hyp_token_ids: Optional[Tensor] = None,  # (B, H)
    ) -> Tensor:
        tok = self.token_emb(token_ids).mean(dim=1)          # (B, D)
        goal = self.goal_proj(torch.cat([tok, ast_features.float(), kg_emb.float()], -1))
        if hyp_token_ids is not None and hyp_token_ids.numel() > 0:
            hyp = self.hyp_proj(self.token_emb(hyp_token_ids).mean(dim=1))
        else:
            hyp = torch.zeros_like(goal)
        return self.combine(torch.cat([goal, hyp], -1))       # (B, out_dim)


# ───────────────────── Action encoder ────────────────────────────────

class ActionEncoder(nn.Module):
    def __init__(self, lemma_emb: Tensor, num_action_types: int = 5, out_dim: int = 128) -> None:
        super().__init__()
        self.lemma_emb = nn.Embedding.from_pretrained(lemma_emb, freeze=False)
        self.type_emb = nn.Embedding(num_action_types, out_dim)
        self.proj = nn.Linear(lemma_emb.shape[1] + out_dim, out_dim)

    def forward(self, lemma_idx: Tensor, type_idx: Tensor) -> Tensor:
        return self.proj(torch.cat([self.lemma_emb(lemma_idx), self.type_emb(type_idx)], -1))


# ───────────────────── Representation h_θ ────────────────────────────

class RepresentationNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_blocks: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            _res_stack(hidden_dim, n_blocks),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)


# ───────────────────── Dynamics g_θ ──────────────────────────────────

class DynamicsNet(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, n_blocks: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.ReLU(),
            _res_stack(hidden_dim, n_blocks),
            nn.LayerNorm(hidden_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: Tensor, action_vec: Tensor) -> Tuple[Tensor, Tensor]:
        ns = self.net(torch.cat([state, action_vec], -1))
        return ns, self.reward_head(ns).squeeze(-1)


# ───────────────────── Prediction f_θ ────────────────────────────────

class PredictionNet(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.policy_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: Tensor, action_vecs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        state:       (B, H) or (H,)
        action_vecs: (A, H)
        Returns policy_logits (B, A), value (B,)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        proj = self.policy_proj(state)          # (B, H)
        logits = proj @ action_vecs.T           # (B, A)
        value = self.value_head(state).squeeze(-1)
        return logits, value


# ───────────────────── Combined MuZeroNet ────────────────────────────

class MuZeroNet(nn.Module):
    def __init__(
        self,
        obs_dim: int = 128,
        hidden_dim: int = 256,
        action_dim: int = 128,
        n_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.representation = RepresentationNet(obs_dim, hidden_dim, n_res_blocks)
        self.dynamics = DynamicsNet(hidden_dim, action_dim, n_res_blocks)
        self.prediction = PredictionNet(hidden_dim, action_dim)

        # backward compat aliases
        self.value_head = self.prediction.value_head
        self.reward_head = self.dynamics.reward_head

    # ── public API ──

    def encode(self, obs_vec: Tensor) -> Tensor:
        return self.representation(obs_vec)

    def dynamics_step(self, state: Tensor, action_vec: Tensor) -> Tuple[Tensor, Tensor]:
        return self.dynamics(state, action_vec)

    def predict(self, state: Tensor, action_vecs: Tensor) -> Tuple[Tensor, Tensor]:
        return self.prediction(state, action_vecs)

    def initial_inference(self, obs_vec: Tensor, action_vecs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """h + f → (state, policy_logits, value)"""
        s = self.encode(obs_vec)
        p, v = self.predict(s, action_vecs)
        return s, p, v

    def recurrent_inference(
        self, state: Tensor, action_vec: Tensor, action_vecs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """g + f → (next_state, reward, policy_logits, value)"""
        ns, r = self.dynamics_step(state, action_vec)
        p, v = self.predict(ns, action_vecs)
        return ns, r, p, v
