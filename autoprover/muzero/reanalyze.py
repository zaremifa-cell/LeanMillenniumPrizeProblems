"""MuZero Reanalyze: re-run MCTS on stored trajectories to improve targets."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch
from torch import Tensor

from autoprover.muzero.config import MuZeroConfig
from autoprover.muzero.mcts import MCTS
from autoprover.muzero.model import ActionEncoder, MuZeroNet, ObsEncoder
from autoprover.muzero.replay import ReplayBuffer, StepRecord, Trajectory
from autoprover.muzero.self_play import (
    ACTION_TYPES,
    TheoremTask,
    action_candidates,
    encode_actions,
    encode_obs,
    select_premises,
)
from autoprover.utils.tokenize import tokenize
from autoprover.utils.vocab import Vocab


@torch.no_grad()
def reanalyze_trajectories(
    replay: ReplayBuffer,
    model: MuZeroNet,
    obs_encoder: ObsEncoder,
    action_encoder: ActionEncoder,
    vocab: Vocab,
    graph: Dict,
    decl_map: Dict[str, Dict],
    node_to_idx: Dict[str, int],
    kg_emb_weight: Optional[Tensor],
    cfg: MuZeroConfig,
    training_step: int = 0,
    tasks_by_name: Optional[Dict[str, TheoremTask]] = None,
) -> int:
    """Re-run MCTS on a fraction of stored trajectories to update policy/value targets.

    Returns the number of trajectories reanalyzed.
    """
    n_traj = replay.num_trajectories()
    if n_traj == 0:
        return 0

    num_to_reanalyze = max(1, int(n_traj * cfg.reanalyze_fraction))
    indices = random.sample(range(n_traj), min(num_to_reanalyze, n_traj))

    mcts = MCTS(
        c_puct=cfg.c_puct,
        num_simulations=cfg.num_simulations,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_frac=0.0,   # no noise during reanalyze
        discount=cfg.discount,
    )

    count = 0
    for traj_idx in indices:
        traj = replay.trajectories[traj_idx]
        if len(traj.steps) == 0:
            continue

        new_steps: List[StepRecord] = []
        for step in traj.steps:
            obs = step.obs
            goal = obs["goal"]
            context = obs.get("context", [])
            ast_size = obs.get("ast_size", 1)
            binder_depth = obs.get("binder_depth", 0)
            head_symbol = obs.get("head_symbol", "")
            type_consts = obs.get("type_consts", [])

            # rebuild action candidates
            goal_tokens = tokenize(goal)
            task_obj = tasks_by_name.get(traj.task_name) if tasks_by_name else None
            if task_obj:
                premises = select_premises(task_obj, graph, cfg.max_premises)
            else:
                premises = []
            actions = action_candidates(goal_tokens, premises, decl_map)
            if not actions:
                from autoprover.muzero.model import Action
                actions = [Action(kind="intro")]

            tok, ast_f, kg = encode_obs(
                vocab, goal, context, ast_size, binder_depth, head_symbol,
                kg_emb_weight, node_to_idx, type_consts,
            )
            obs_vec = obs_encoder(tok, ast_f, kg)
            s = model.encode(obs_vec)
            a_vecs = encode_actions(actions, node_to_idx, action_encoder)

            visits, root_value, _ = mcts.run(model, s.squeeze(0), a_vecs, add_noise=False)

            total_v = sum(visits)
            if total_v > 0:
                new_policy = [v / total_v for v in visits]
            else:
                new_policy = [1.0 / len(visits)] * len(visits)

            new_steps.append(StepRecord(
                obs=obs,
                action_idx=step.action_idx,
                reward=step.reward,
                policy_target=new_policy,
                value_target=step.value_target,
                root_value=root_value,
            ))

        new_traj = Trajectory(
            task_name=traj.task_name,
            steps=new_steps,
            success=traj.success,
            total_reward=traj.total_reward,
        )
        replay.update_trajectory(traj_idx, new_traj)
        count += 1

    return count
