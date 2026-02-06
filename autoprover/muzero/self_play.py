"""Self-play: generate proof trajectories using MCTS + the Lean oracle."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from autoprover.muzero.config import MuZeroConfig
from autoprover.muzero.env import GoalState, ProofEnv
from autoprover.muzero.mcts import MCTS
from autoprover.muzero.model import Action, ActionEncoder, MuZeroNet, ObsEncoder
from autoprover.muzero.replay import StepRecord, Trajectory
from autoprover.utils.lean_oracle import LeanOracle
from autoprover.utils.tokenize import tokenize
from autoprover.utils.vocab import Vocab

ACTION_TYPES = {"apply": 0, "exact": 1, "rw_fwd": 2, "rw_bwd": 3, "intro": 4}


# ─────────────────────── Task definition ──────────────────────────────

class TheoremTask:
    __slots__ = ("name", "type_expr", "type_consts", "depth")

    def __init__(self, name: str, type_expr: str, type_consts: List[str], depth: int) -> None:
        self.name = name
        self.type_expr = type_expr
        self.type_consts = type_consts
        self.depth = depth


# ─────────────────────── Helpers ──────────────────────────────────────

def build_tasks(decls: List[Dict], graph: Dict) -> List[TheoremTask]:
    tasks: List[TheoremTask] = []
    nodes = graph.get("nodes", {})
    for d in decls:
        name = d["decl_name"]
        te = d.get("type_expr", "")
        if not te:
            continue
        tc = d.get("type_consts", [])
        depth = nodes.get(name, {}).get("depth", 0)
        tasks.append(TheoremTask(name, te, tc, depth))
    return tasks


def select_premises(task: TheoremTask, graph: Dict, topk: int = 64) -> List[str]:
    candidates: set = set()
    edges = graph.get("edges", {})
    nodes = graph.get("nodes", {})
    for c in task.type_consts:
        candidates.add(c)
        for dep in edges.get(c, []):
            candidates.add(dep)
    candidates = [c for c in candidates if c in nodes and nodes[c].get("kind") in ("theorem", "definition")]
    candidates.sort(key=lambda n: nodes[n].get("degree", 0), reverse=True)
    return candidates[:topk] if candidates else []


def action_candidates(goal_tokens: List[str], lemma_pool: List[str], decl_map: Dict[str, Dict]) -> List[Action]:
    actions: List[Action] = [Action(kind="intro")]
    goal_set = set(goal_tokens)
    for lemma in lemma_pool:
        actions.append(Action(kind="apply", lemma=lemma))
        actions.append(Action(kind="exact", lemma=lemma))
        te = decl_map.get(lemma, {}).get("type_expr", "")
        if "=" in te or "↔" in te:
            sep = "=" if "=" in te else "↔"
            lhs, rhs = te.split(sep, 1)
            if set(tokenize(lhs)) & goal_set:
                actions.append(Action(kind="rw", lemma=lemma, direction="forward"))
            if set(tokenize(rhs)) & goal_set:
                actions.append(Action(kind="rw", lemma=lemma, direction="backward"))
    return actions


def action_to_payload(a: Action) -> Dict:
    d: Dict = {"kind": a.kind}
    if a.lemma:
        d["lemma"] = a.lemma
    if a.direction:
        d["direction"] = a.direction
    return d


def encode_obs(
    vocab: Vocab,
    goal: str,
    context: List[Dict[str, str]],
    ast_size: int,
    binder_depth: int,
    head_symbol: str,
    kg_emb_weight: Optional[Tensor],
    node_to_idx: Optional[Dict[str, int]],
    goal_consts: Optional[List[str]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return (token_ids, ast_features[3], kg_emb)."""
    tokens = tokenize(goal)
    for entry in context:
        tokens.extend(tokenize(entry.get("type", "")))
    token_ids = torch.tensor(vocab.encode(tokens) or [0], dtype=torch.long).unsqueeze(0)

    head_id = vocab.token_to_id.get(head_symbol, 0)
    ast_features = torch.tensor([[float(ast_size), float(binder_depth), float(head_id)]])

    # KG embedding: average of occurring constants
    if kg_emb_weight is not None and node_to_idx is not None and goal_consts:
        idxs = [node_to_idx[c] for c in goal_consts if c in node_to_idx]
        if idxs:
            kg = kg_emb_weight[idxs].mean(dim=0, keepdim=True)
        else:
            kg = torch.zeros(1, kg_emb_weight.shape[1])
    else:
        kg_dim = kg_emb_weight.shape[1] if kg_emb_weight is not None else 64
        kg = torch.zeros(1, kg_dim)

    return token_ids, ast_features, kg


def encode_actions(
    actions: List[Action],
    node_to_idx: Dict[str, int],
    action_encoder: ActionEncoder,
) -> Tensor:
    lemma_idxs, type_idxs = [], []
    for a in actions:
        if a.kind == "intro":
            lemma_idxs.append(0)
            type_idxs.append(ACTION_TYPES["intro"])
        else:
            lemma_idxs.append(node_to_idx.get(a.lemma or "", 0))
            if a.kind == "rw":
                type_idxs.append(ACTION_TYPES["rw_fwd"] if a.direction == "forward" else ACTION_TYPES["rw_bwd"])
            else:
                type_idxs.append(ACTION_TYPES.get(a.kind, 0))
    return action_encoder(torch.tensor(lemma_idxs, dtype=torch.long), torch.tensor(type_idxs, dtype=torch.long))


# ─────────────────────── Self-play episode ────────────────────────────

@torch.no_grad()
def run_episode(
    task: TheoremTask,
    env: ProofEnv,
    model: MuZeroNet,
    obs_encoder: ObsEncoder,
    action_encoder: ActionEncoder,
    vocab: Vocab,
    graph: Dict,
    decl_map: Dict[str, Dict],
    node_to_idx: Dict[str, int],
    kg_emb_weight: Optional[Tensor],
    mcts: MCTS,
    cfg: MuZeroConfig,
    training_step: int = 0,
    fallback_tasks: Optional[List[TheoremTask]] = None,
) -> Trajectory:
    """Play one full episode, return a Trajectory."""
    state = env.reset(task.type_expr)
    traj = Trajectory(task_name=task.name)
    temperature = cfg.temperature(training_step)

    done = False
    while not done and state.goals:
        goal = state.goals[0]
        goal_tokens = tokenize(goal)

        # candidate actions
        premises = select_premises(task, graph, cfg.max_premises)
        if not premises and fallback_tasks:
            premises = [t.name for t in random.sample(fallback_tasks, min(cfg.max_premises, len(fallback_tasks)))]
        actions = action_candidates(goal_tokens, premises, decl_map)
        if not actions:
            actions = [Action(kind="intro")]

        # encode
        tok, ast_f, kg = encode_obs(
            vocab, goal, state.context,
            state.goal_ast_sizes[0] if state.goal_ast_sizes else 1,
            state.binder_depth, state.head_symbol,
            kg_emb_weight, node_to_idx, task.type_consts,
        )
        obs_vec = obs_encoder(tok, ast_f, kg)
        s = model.encode(obs_vec)
        a_vecs = encode_actions(actions, node_to_idx, action_encoder)

        # MCTS
        visits, root_value, _ = mcts.run(model, s.squeeze(0), a_vecs, add_noise=True)

        # temperature-based policy
        total_v = sum(visits)
        if total_v == 0:
            policy = [1.0 / len(visits)] * len(visits)
        else:
            if temperature < 0.05:
                # greedy
                policy = [0.0] * len(visits)
                policy[int(max(range(len(visits)), key=lambda i: visits[i]))] = 1.0
            else:
                adjusted = [v ** (1.0 / temperature) for v in visits]
                s_adj = sum(adjusted)
                policy = [a / s_adj for a in adjusted] if s_adj > 0 else [1.0 / len(visits)] * len(visits)

        # sample action
        action_idx = random.choices(range(len(policy)), weights=policy, k=1)[0]
        action = actions[action_idx]

        next_state, reward, done, info = env.step(action_to_payload(action))

        traj.steps.append(StepRecord(
            obs={"goal": goal, "context": state.context,
                 "ast_size": state.goal_ast_sizes[0] if state.goal_ast_sizes else 1,
                 "binder_depth": state.binder_depth,
                 "head_symbol": state.head_symbol,
                 "type_consts": task.type_consts},
            action_idx=action_idx,
            reward=reward,
            policy_target=policy,
            value_target=0.0,      # will be set by replay buffer
            root_value=root_value,
        ))
        traj.total_reward += reward
        state = next_state

    traj.success = (traj.total_reward >= 1.0 - 1e-6 or (state.goals is not None and len(state.goals) == 0))
    return traj
