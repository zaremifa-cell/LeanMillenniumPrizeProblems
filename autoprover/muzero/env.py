"""Lean proof environment wrapping the Oracle for RL episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from autoprover.utils.lean_oracle import LeanOracle
from autoprover.utils.tokenize import tokenize


@dataclass
class GoalState:
    goals: List[str]
    goal_ast_sizes: List[int]
    context: List[Dict[str, str]]
    # extra features for observation encoding
    binder_depth: int = 0
    head_symbol: str = ""


def _binder_depth(expr: str) -> int:
    """Cheap heuristic: count leading ∀/fun/Π binders."""
    depth = 0
    for tok in tokenize(expr):
        if tok in ("forall", "∀", "fun", "Π", "Pi", "→"):
            depth += 1
        else:
            break
    return depth


def _head_symbol(expr: str) -> str:
    toks = tokenize(expr)
    for t in toks:
        if t not in ("forall", "∀", "fun", "Π", "Pi", "(", ")", ":", ",", "→", "↔"):
            return t
    return ""


class ProofEnv:
    """Single-episode environment.  Call reset() then step() repeatedly."""

    def __init__(self, oracle: LeanOracle, max_steps: int = 30) -> None:
        self.oracle = oracle
        self.max_steps = max_steps
        self.steps = 0
        self.state: Optional[GoalState] = None
        self.lean_calls = 0

    def reset(self, goal_expr: str) -> GoalState:
        self.steps = 0
        self.lean_calls = 0
        self.state = GoalState(
            goals=[goal_expr],
            goal_ast_sizes=[len(tokenize(goal_expr))],
            context=[],
            binder_depth=_binder_depth(goal_expr),
            head_symbol=_head_symbol(goal_expr),
        )
        return self.state

    def step(self, action: Dict[str, Any]) -> Tuple[GoalState, float, bool, Dict[str, Any]]:
        assert self.state is not None and self.state.goals, "call reset() first"
        self.steps += 1
        self.lean_calls += 1

        goal = self.state.goals[0]
        ctx = self.state.context
        try:
            resp = self.oracle.request(goal, ctx, action)
        except Exception as e:
            resp = {"ok": False, "error": str(e)}

        info: Dict[str, Any] = {"oracle_ok": resp.get("ok", False)}

        if not resp.get("ok", False):
            reward = -0.2
            done = self.steps >= self.max_steps
            return self.state, reward, done, info

        new_goals: List[str] = resp.get("goals", [])
        goal_ast_sizes: List[int] = resp.get("goal_ast_sizes", [])
        context: List[Dict[str, str]] = resp.get("context", [])

        old_count = len(self.state.goals)
        new_count = len(new_goals)

        if new_count == 0:
            reward = 1.0
            done = True
        else:
            done = self.steps >= self.max_steps
            reward = 0.0
            if new_count < old_count:
                reward += 0.5
            elif new_count > old_count:
                reward -= 0.1
            elif self.state.goal_ast_sizes and goal_ast_sizes:
                old_s = self.state.goal_ast_sizes[0]
                new_s = goal_ast_sizes[0]
                reward += 0.1 * (old_s - new_s) / max(old_s, 1)

        if not goal_ast_sizes:
            goal_ast_sizes = [len(tokenize(g)) for g in new_goals]

        first_goal = new_goals[0] if new_goals else ""
        self.state = GoalState(
            goals=new_goals,
            goal_ast_sizes=goal_ast_sizes,
            context=context,
            binder_depth=_binder_depth(first_goal),
            head_symbol=_head_symbol(first_goal),
        )
        info["lean_calls"] = self.lean_calls
        return self.state, reward, done, info
