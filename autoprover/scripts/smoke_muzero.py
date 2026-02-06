#!/usr/bin/env python3
"""Quick smoke test for the full MuZero pipeline components."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from autoprover.muzero.model import MuZeroNet, ObsEncoder, ActionEncoder
from autoprover.muzero.mcts import MCTS
from autoprover.muzero.replay import ReplayBuffer, Trajectory, StepRecord
from autoprover.muzero.config import MuZeroConfig

def main():
    cfg = MuZeroConfig()

    # ── encoders & model ──
    lemma_emb = torch.randn(100, 64)
    obs_enc = ObsEncoder(500, token_emb_dim=64, kg_emb_dim=64, out_dim=128)
    act_enc = ActionEncoder(lemma_emb, num_action_types=5, out_dim=128)
    model = MuZeroNet(obs_dim=128, hidden_dim=256, action_dim=128, n_res_blocks=4)

    tok = torch.randint(0, 500, (1, 10))
    ast = torch.tensor([[5.0, 2.0, 3.0]])
    kg  = torch.randn(1, 64)
    obs_vec = obs_enc(tok, ast, kg)
    print("obs_vec:", obs_vec.shape)

    all_a = act_enc(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    s, p, v = model.initial_inference(obs_vec, all_a)
    print("state:", s.shape, "logits:", p.shape, "value:", v.shape)

    a_vec = act_enc(torch.tensor([1]), torch.tensor([0]))
    ns, r, p2, v2 = model.recurrent_inference(s, a_vec, all_a)
    print("next_state:", ns.shape, "reward:", r.shape,
          "logits:", p2.shape, "value:", v2.shape)

    # ── MCTS ──
    mcts = MCTS(c_puct=2.5, num_simulations=10)
    a_vecs = act_enc(torch.randint(0, 100, (20,)), torch.randint(0, 5, (20,)))
    visits, root_v, root = mcts.run(model, s.squeeze(0), a_vecs, add_noise=True)
    print("MCTS visits sum:", sum(visits), "root_value:", round(root_v, 4))

    # ── Replay buffer ──
    replay = ReplayBuffer(capacity=100, td_steps=5, discount=0.99, unroll_steps=3)
    traj = Trajectory(task_name="test")
    for i in range(8):
        traj.steps.append(StepRecord(
            obs={"goal": "True", "context": [], "ast_size": 1,
                 "binder_depth": 0, "head_symbol": "True", "type_consts": []},
            action_idx=i % 3,
            reward=0.1 * i,
            policy_target=[0.5, 0.3, 0.2],
            value_target=0.0,
            root_value=0.5,
        ))
    replay.save_trajectory(traj)
    batch = replay.sample(4)
    print("Batch obs count:", len(batch.observations),
          "acts shape:", len(batch.action_indices[0]))

    # ── Loss / gradient flow check ──
    B, K, A = 2, 3, 5
    all_acts = act_enc(torch.arange(A), torch.zeros(A, dtype=torch.long))

    obs_t = obs_enc(torch.randint(0, 500, (B, 10)),
                     torch.tensor([[1.0, 0.0, 0.0]] * B),
                     torch.randn(B, 64))
    s, logits, val = model.initial_inference(obs_t, all_acts)

    # K‑step unroll
    total_loss = torch.tensor(0.0)
    target_pol = torch.ones(B, A) / A
    total_loss += torch.nn.functional.cross_entropy(logits, target_pol)
    total_loss += torch.nn.functional.mse_loss(val, torch.zeros(B))

    state = s
    for k in range(K):
        a_vec = act_enc(torch.zeros(B, dtype=torch.long),
                        torch.zeros(B, dtype=torch.long))
        ns, rew, lgt, v = model.recurrent_inference(state, a_vec, all_acts)
        total_loss += torch.nn.functional.cross_entropy(lgt, target_pol)
        total_loss += torch.nn.functional.mse_loss(v, torch.zeros(B))
        total_loss += torch.nn.functional.mse_loss(rew, torch.zeros(B))
        state = ns

    total_loss.backward()
    grad_ok = all(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.parameters() if p.requires_grad)
    print("Loss:", round(total_loss.item(), 4), "gradients_ok:", grad_ok)
    print("Loss:", round(total_loss.item(), 4), "gradients_ok:", grad_ok)
    print("\n\u2705 ALL SMOKE TESTS PASSED")

if __name__ == "__main__":
    main()
