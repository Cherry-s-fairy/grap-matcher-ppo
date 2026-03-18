# main.py
"""
Training entry-point: Resource-Aware Task Graph Shaping in Heterogeneous UAV Swarms
------------------------------------------------------------------------------------
This script demonstrates the full training loop using a minimal hand-rolled
PPO-style update.  To use Stable-Baselines3 instead, see the commented block
at the bottom of this file.

Run:
    python main.py
"""

from __future__ import annotations

import random
import time
from collections import deque
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.params import (
    GAMMA,
    LR,
    CLIP_EPS,
    ENTROPY_COEF,
    VALUE_LOSS_COEF,
    PPO_EPOCHS,
    BATCH_SIZE,
    ROLLOUT_STEPS,
    TOTAL_TIMESTEPS,
    MAX_EPISODE_STEPS,
)
from env.uav_env import UAVTaskEnv
from models.gnn_policy import NodeLevelActorCritic


# ---------------------------------------------------------------------------
# Helper: convert numpy obs dict → torch tensor dict
# ---------------------------------------------------------------------------

def obs_to_torch(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: torch.as_tensor(v, dtype=torch.float32 if v.dtype == np.float32 else torch.long).to(device)
        for k, v in obs.items()
    }


# ---------------------------------------------------------------------------
# Rollout buffer (simplified, single env)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.obs: List[Dict[str, torch.Tensor]] = []
        self.actions:    List[int]   = []
        self.log_probs:  List[float] = []
        self.rewards:    List[float] = []
        self.values:     List[float] = []
        self.dones:      List[bool]  = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.actions)

    def compute_returns(self, last_value: float, gamma: float) -> torch.Tensor:
        returns = []
        R = last_value
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + gamma * R * (1.0 - float(done))
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)


# ---------------------------------------------------------------------------
# PPO update step
# ---------------------------------------------------------------------------

def ppo_update(
    policy: NodeLevelActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    last_value: float,
    device: torch.device,
):
    returns    = buffer.compute_returns(last_value, GAMMA).to(device)
    old_log_ps = torch.tensor(buffer.log_probs, dtype=torch.float32).to(device)
    actions_t  = torch.tensor(buffer.actions,   dtype=torch.long).to(device)
    values_t   = torch.tensor(buffer.values,    dtype=torch.float32).to(device)

    advantages = (returns - values_t).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        # mini-batch (random indices)
        idx = torch.randperm(len(buffer.actions))
        for start in range(0, len(buffer.actions), BATCH_SIZE):
            mb_idx = idx[start: start + BATCH_SIZE]

            mb_obs_list = [buffer.obs[i] for i in mb_idx.tolist()]
            mb_obs: Dict[str, torch.Tensor] = {}
            for key in mb_obs_list[0]:
                mb_obs[key] = torch.stack([o[key] for o in mb_obs_list], dim=0).to(device)

            logits, new_values = policy(mb_obs)

            # Re-apply action mask with the stored observations so log_probs
            # are consistent between rollout collection and update.
            mask          = policy.compute_action_mask(mb_obs)
            masked_logits = logits.masked_fill(~mask, float("-inf"))
            dist          = torch.distributions.Categorical(logits=masked_logits)
            new_log_ps    = dist.log_prob(actions_t[mb_idx])
            entropy       = dist.entropy().mean()

            ratio      = (new_log_ps - old_log_ps[mb_idx]).exp()
            adv_mb     = advantages[mb_idx]

            surr1 = ratio * adv_mb
            surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_mb
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(new_values.squeeze(-1), returns[mb_idx])

            loss = actor_loss + VALUE_LOSS_COEF * critic_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Using device: {device}")

    env    = UAVTaskEnv(seed=42)
    policy = NodeLevelActorCritic().to(device)
    optim_ = optim.Adam(policy.parameters(), lr=LR)
    buffer = RolloutBuffer(capacity=ROLLOUT_STEPS)

    ep_rewards: deque = deque(maxlen=20)
    ep_reward = 0.0
    timestep  = 0
    episode   = 0

    obs_np, _ = env.reset()
    obs = obs_to_torch(obs_np, device)

    t0 = time.time()

    while timestep < TOTAL_TIMESTEPS:
        # ---- collect rollout ----
        buffer.clear()
        for _ in range(ROLLOUT_STEPS):
            with torch.no_grad():
                action, log_prob, _, value = policy.get_action_and_value(
                    {k: v.unsqueeze(0) for k, v in obs.items()}
                )

            next_obs_np, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            buffer.add(obs, action.item(), log_prob.item(), reward, value.item(), done)
            ep_reward += reward
            timestep  += 1

            if done:
                ep_rewards.append(ep_reward)
                episode += 1
                if episode % 20 == 0:
                    elapsed = time.time() - t0
                    mean_r  = np.mean(ep_rewards)
                    fps     = timestep / elapsed
                    print(
                        f"[ep {episode:5d} | step {timestep:7d}] "
                        f"mean_reward={mean_r:7.2f}  fps={fps:.0f}  "
                        f"latency={info['latency_ms']:.1f}ms  "
                        f"nodes={info['num_task_nodes']}  "
                        f"action={info['action_taken']}"
                    )
                ep_reward = 0.0
                obs_np, _ = env.reset()
                obs = obs_to_torch(obs_np, device)
            else:
                obs = obs_to_torch(next_obs_np, device)

        # ---- PPO update ----
        with torch.no_grad():
            _, _, _, last_value = policy.get_action_and_value(
                {k: v.unsqueeze(0) for k, v in obs.items()}
            )
        ppo_update(policy, optim_, buffer, last_value.item(), device)

    print("[main] Training complete.")
    torch.save(policy.state_dict(), "policy.pt")
    print("[main] Model saved to policy.pt")


# ---------------------------------------------------------------------------
# Stable-Baselines3 alternative (uncomment to use)
# ---------------------------------------------------------------------------
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from models.gnn_policy import DualGraphExtractor
#
# def train_sb3():
#     env = make_vec_env(UAVTaskEnv, n_envs=4)
#     policy_kwargs = dict(
#         features_extractor_class=DualGraphExtractor,
#         features_extractor_kwargs=dict(features_dim=128),
#     )
#     model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs,
#                 learning_rate=LR, n_steps=ROLLOUT_STEPS, batch_size=BATCH_SIZE,
#                 verbose=1)
#     model.learn(total_timesteps=TOTAL_TIMESTEPS)
#     model.save("sb3_policy")


if __name__ == "__main__":
    train()
