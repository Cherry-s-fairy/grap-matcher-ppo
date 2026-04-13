# train_with_metrics.py
"""
Single-seed training run for RTGS.
Records per-episode metrics:
  (a) Scheduling Latency (ms)
  (b) Task Success Rate
  (c) Deadline Miss Rate
  (d) Resource Utilization
Plots all four learning curves after training.
"""

from __future__ import annotations

import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.params import (
    GAMMA, LR, CLIP_EPS, ENTROPY_COEF,
    VALUE_LOSS_COEF, PPO_EPOCHS, BATCH_SIZE,
    ROLLOUT_STEPS, TOTAL_TIMESTEPS, DEADLINE_MS,
)
from env.uav_env import UAVTaskEnv
from models.gnn_policy import NodeLevelActorCritic


# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


# ── helpers ────────────────────────────────────────────────────────────────────
def obs_to_torch(obs: dict, device: torch.device) -> dict:
    return {
        k: torch.as_tensor(
            v, dtype=torch.float32 if v.dtype == np.float32 else torch.long
        ).to(device)
        for k, v in obs.items()
    }


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs:       list = []
        self.actions:   list = []
        self.log_probs: list = []
        self.rewards:   list = []
        self.values:    list = []
        self.dones:     list = []

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
        returns, R = [], last_value
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + gamma * R * (1.0 - float(done))
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)


def ppo_update(policy, optimizer, buffer: RolloutBuffer,
               last_value: float, device: torch.device):
    returns   = buffer.compute_returns(last_value, GAMMA).to(device)
    old_lp    = torch.tensor(buffer.log_probs, dtype=torch.float32).to(device)
    actions_t = torch.tensor(buffer.actions,   dtype=torch.long).to(device)
    values_t  = torch.tensor(buffer.values,    dtype=torch.float32).to(device)

    adv = (returns - values_t).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        idx = torch.randperm(len(buffer.actions))
        for start in range(0, len(buffer.actions), BATCH_SIZE):
            mb_idx      = idx[start: start + BATCH_SIZE]
            mb_obs_list = [buffer.obs[i] for i in mb_idx.tolist()]
            mb_obs      = {
                k: torch.stack([o[k] for o in mb_obs_list]).to(device)
                for k in mb_obs_list[0]
            }

            logits, new_vals = policy(mb_obs)
            mask          = policy.compute_action_mask(mb_obs)
            masked_logits = logits.masked_fill(~mask, float("-inf"))
            dist          = torch.distributions.Categorical(logits=masked_logits)
            new_lp        = dist.log_prob(actions_t[mb_idx])
            entropy       = dist.entropy().mean()

            ratio  = (new_lp - old_lp[mb_idx]).exp()
            adv_mb = adv[mb_idx]
            surr   = torch.min(
                ratio * adv_mb,
                ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_mb,
            )
            loss = (
                -surr.mean()
                + VALUE_LOSS_COEF * nn.functional.mse_loss(
                    new_vals.squeeze(-1), returns[mb_idx]
                )
                - ENTROPY_COEF * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


# ── training loop ──────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RTGS] device={device}  seed={SEED}\n")

    env    = UAVTaskEnv(seed=SEED)
    policy = NodeLevelActorCritic().to(device)
    optim_ = optim.Adam(policy.parameters(), lr=LR)
    buffer = RolloutBuffer()

    # ── per-episode metric logs ──────────────────────────────────────────────
    log_latency  : list[float] = []   # (a) mean scheduling latency per episode (ms)
    log_success  : list[float] = []   # (b) mean task success rate per episode
    log_dmiss    : list[float] = []   # (c) deadline miss rate per episode
    log_util     : list[float] = []   # (d) mean UAV resource utilisation per episode

    # step-level accumulators for the current episode
    ep_lat, ep_suc, ep_dm, ep_util = [], [], [], []

    timestep = 0
    episode  = 0
    t0       = time.time()

    obs_np, _ = env.reset(seed=SEED)
    obs = obs_to_torch(obs_np, device)

    while timestep < TOTAL_TIMESTEPS:
        buffer.clear()

        # ── collect one rollout ─────────────────────────────────────────────
        for _ in range(ROLLOUT_STEPS):
            with torch.no_grad():
                action, log_prob, _, value = policy.get_action_and_value(
                    {k: v.unsqueeze(0) for k, v in obs.items()}
                )

            next_obs_np, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            # accumulate step metrics
            ep_lat.append(info["latency_ms"])
            ep_suc.append(info["success_rate"])
            ep_dm.append(float(info["latency_ratio"] > 1.0))
            # feedback[3] = avg_uav_utilization (see SchedulingFeedback.as_vector)
            ep_util.append(float(next_obs_np["feedback"][3]))

            buffer.add(obs, action.item(), log_prob.item(), reward, value.item(), done)
            timestep += 1

            if done:
                episode += 1
                log_latency.append(float(np.mean(ep_lat)))
                log_success.append(float(np.mean(ep_suc)))
                log_dmiss.append(float(np.mean(ep_dm)))
                log_util.append(float(np.mean(ep_util)))
                ep_lat, ep_suc, ep_dm, ep_util = [], [], [], []

                if episode % 200 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  ep {episode:5d} | step {timestep:7d} | {elapsed:6.0f}s"
                        f"  lat={log_latency[-1]:7.1f}ms"
                        f"  suc={log_success[-1]:.3f}"
                        f"  miss={log_dmiss[-1]:.3f}"
                        f"  util={log_util[-1]:.3f}"
                    )

                obs_np, _ = env.reset()
                obs = obs_to_torch(obs_np, device)
            else:
                obs = obs_to_torch(next_obs_np, device)

        # ── PPO update ──────────────────────────────────────────────────────
        with torch.no_grad():
            _, _, _, last_val = policy.get_action_and_value(
                {k: v.unsqueeze(0) for k, v in obs.items()}
            )
        ppo_update(policy, optim_, buffer, last_val.item(), device)

    elapsed = time.time() - t0
    print(f"\n[RTGS] training complete — {episode} episodes | "
          f"{timestep} steps | {elapsed:.0f}s")

    torch.save(policy.state_dict(), "policy_seed42.pt")
    print("[RTGS] model  → policy_seed42.pt")

    np.savez(
        "training_metrics_seed42.npz",
        latency=log_latency,
        success=log_success,
        deadline_miss=log_dmiss,
        utilization=log_util,
    )
    print("[RTGS] metrics → training_metrics_seed42.npz")

    plot_metrics(log_latency, log_success, log_dmiss, log_util)


# ── plotting ───────────────────────────────────────────────────────────────────
def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Simple moving-average smoothing."""
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def plot_metrics(lat, suc, dmiss, util):
    W = 100   # smoothing window (episodes)

    data = np.array(lat), np.array(suc), np.array(dmiss), np.array(util)
    n    = len(lat)
    eps_raw    = np.arange(1, n + 1)
    eps_smooth = np.arange(W, n + 1)   # valid output of convolve(..., 'valid')

    titles  = [
        "(a) Scheduling Latency",
        "(b) Task Success Rate",
        "(c) Deadline Miss Rate",
        "(d) Resource Utilization",
    ]
    ylabels = ["Latency (ms)", "Success Rate", "Miss Rate", "Utilization"]
    colors  = ["#2C7BB6", "#1A9641", "#D7191C", "#E87722"]
    hrefs   = [DEADLINE_MS, 1.0, 0.0, None]   # optional horizontal reference lines
    href_labels = [f"Deadline = {DEADLINE_MS:.0f} ms", "Rate = 1.0", "Rate = 0", None]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(
        f"RTGS Training Curves  (seed = {SEED})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, arr, title, ylabel, color, href, hlabel in zip(
        axes.flat, data, titles, ylabels, colors, hrefs, href_labels
    ):
        # raw curve (very faint)
        ax.plot(eps_raw, arr, color=color, alpha=0.15, linewidth=0.7, zorder=1)

        # smoothed curve
        if n >= W:
            ax.plot(
                eps_smooth, smooth(arr, W),
                color=color, linewidth=2.2,
                label=f"MA-{W}", zorder=3,
            )

        # optional horizontal reference line
        if href is not None:
            ax.axhline(href, color="gray", linewidth=1.0,
                       linestyle="--", alpha=0.7, label=hlabel, zorder=2)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("training_curves_seed42.png", dpi=200, bbox_inches="tight")
    plt.savefig("training_curves_seed42.pdf",            bbox_inches="tight")
    print("[plot] training_curves_seed42.png / .pdf")
    plt.close()


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
