# evaluate.py
"""
Unified evaluation script for all baselines and our method.

Usage
-----
# 1. Evaluate a single method (no training needed for NS/HS/RND):
    python evaluate.py --method ns
    python evaluate.py --method hs
    python evaluate.py --method rnd

# 2. Train + evaluate RL methods:
    python evaluate.py --method rl_global --train
    python evaluate.py --method rl_node   --train

# 3. Run all methods in sequence:
    python evaluate.py --all --train

# 4. Multiple seeds (for paper statistics):
    python evaluate.py --all --train --seeds 0 1 2 3 4

Results saved to: results/<method>_seed<N>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.params import (
    GAMMA, LR, CLIP_EPS, ENTROPY_COEF, VALUE_LOSS_COEF,
    PPO_EPOCHS, BATCH_SIZE, ROLLOUT_STEPS, TOTAL_TIMESTEPS,
    MAX_EPISODE_STEPS, ACTION_DIM,
)
from env.uav_env import UAVTaskEnv
from models.gnn_policy import NodeLevelActorCritic, ActorCriticPolicy
from core.task_shaping import AdaptiveTaskShaper

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def obs_to_torch(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
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
        self.obs: list = []
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
        returns = []
        R = last_value
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + gamma * R * (1.0 - float(done))
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def compute_gae(self, last_value: float, gamma: float, lam: float = 0.95):
        advantages = []
        gae = 0.0
        vals = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * vals[t + 1] * (1.0 - float(self.dones[t])) - vals[t]
            gae   = delta + gamma * lam * (1.0 - float(self.dones[t])) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return (torch.tensor(advantages, dtype=torch.float32),
                torch.tensor(returns,    dtype=torch.float32))


def ppo_update(policy, optimizer, buffer, last_value, device, use_mask=True):
    advantages_t, returns_t = buffer.compute_gae(last_value, GAMMA)
    advantages_t = advantages_t.to(device)
    returns_t    = returns_t.to(device)
    old_log_ps   = torch.tensor(buffer.log_probs, dtype=torch.float32).to(device)
    actions_t    = torch.tensor(buffer.actions,   dtype=torch.long).to(device)
    old_values_t = torch.tensor(buffer.values,    dtype=torch.float32).to(device)

    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        idx = torch.randperm(len(buffer.actions))
        for start in range(0, len(buffer.actions), BATCH_SIZE):
            mb_idx = idx[start: start + BATCH_SIZE]
            mb_obs_list = [buffer.obs[i] for i in mb_idx.tolist()]
            mb_obs: Dict[str, torch.Tensor] = {}
            for key in mb_obs_list[0]:
                mb_obs[key] = torch.stack([o[key] for o in mb_obs_list], dim=0).to(device)

            logits, new_values = policy(mb_obs)
            new_values = new_values.squeeze(-1)

            if use_mask and hasattr(policy, "compute_action_mask"):
                mask          = policy.compute_action_mask(mb_obs)
                masked_logits = logits.masked_fill(~mask, float("-inf"))
            else:
                masked_logits = logits

            dist       = torch.distributions.Categorical(logits=masked_logits)
            new_log_ps = dist.log_prob(actions_t[mb_idx])
            entropy    = dist.entropy().mean()

            ratio  = (new_log_ps - old_log_ps[mb_idx]).exp()
            adv_mb = advantages_t[mb_idx]

            surr1 = ratio * adv_mb
            surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_mb
            actor_loss  = -torch.min(surr1, surr2).mean()

            # Value function clipping (PPO standard): prevents large critic updates
            # that destabilise the policy via bad advantage estimates.
            ret_mb = returns_t[mb_idx]
            v_unclipped = (new_values - ret_mb).pow(2)
            v_clipped   = old_values_t[mb_idx] + (new_values - old_values_t[mb_idx]).clamp(
                -CLIP_EPS, CLIP_EPS
            )
            v_loss_clipped = (v_clipped - ret_mb).pow(2)
            critic_loss = 0.5 * torch.max(v_unclipped, v_loss_clipped).mean()

            loss = actor_loss + VALUE_LOSS_COEF * critic_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


# ---------------------------------------------------------------------------
# RL training loop (generic — works for both rl_global and rl_node)
# ---------------------------------------------------------------------------

def train_rl(policy, env_seed: int, device: torch.device, method_tag: str,
             log_path: Path | None = None,
             total_timesteps: int | None = None):
    """
    Train policy for total_timesteps steps (default: TOTAL_TIMESTEPS from params).
    Saves a training log (timestep, mean_reward, mean_latency, mean_success)
    every LOG_INTERVAL episodes to log_path (JSON), enabling reward-curve plots.
    """
    LOG_INTERVAL = 20   # record one data point every N completed episodes
    n_steps = total_timesteps if total_timesteps is not None else TOTAL_TIMESTEPS
    n_updates = n_steps // ROLLOUT_STEPS   # total number of PPO update calls

    optimizer = optim.Adam(policy.parameters(), lr=LR)
    # Linear LR decay: ramp from LR down to LR/10 over all updates.
    # This prevents the policy from over-optimising in the late stage, which
    # caused the ~13-point reward collapse seen in prior runs.
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_updates
    )
    buffer    = RolloutBuffer()
    env       = UAVTaskEnv(seed=env_seed)

    ep_rewards:    deque = deque(maxlen=50)
    ep_latencies:  deque = deque(maxlen=50)
    ep_successes:  deque = deque(maxlen=50)
    ep_reward  = 0.0
    last_info  = {}
    timestep   = 0
    episode    = 0

    # Best-model tracking: save the state_dict that achieved the highest
    # rolling mean reward (window=50 eps).  Prevents returning a degraded
    # checkpoint when training collapses near the end.
    best_mean_reward  = -float("inf")
    best_state_dict   = copy.deepcopy(policy.state_dict())
    BEST_MODEL_MIN_EP = 50   # require at least this many episodes before tracking

    # Training log: list of dicts {step, reward, latency, success}
    train_log: List[Dict] = []

    obs_np, _ = env.reset()
    obs = obs_to_torch(obs_np, device)
    t0 = time.time()

    print(f"[{method_tag}] Training on seed={env_seed} for {n_steps} steps ...")

    while timestep < n_steps:
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
            last_info  = info
            timestep  += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_latencies.append(last_info.get("latency_ms", 0.0))
                ep_successes.append(last_info.get("success_rate", 0.0))
                episode += 1

                # Save log point every LOG_INTERVAL episodes
                if episode % LOG_INTERVAL == 0:
                    mean_r = float(np.mean(ep_rewards))
                    train_log.append({
                        "step":    timestep,
                        "reward":  mean_r,
                        "latency": float(np.mean(ep_latencies)),
                        "success": float(np.mean(ep_successes)),
                    })
                    # Track best model (only after warm-up)
                    if episode >= BEST_MODEL_MIN_EP and mean_r > best_mean_reward:
                        best_mean_reward = mean_r
                        best_state_dict  = copy.deepcopy(policy.state_dict())

                if episode % 50 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  [ep {episode:5d} | step {timestep:7d}]"
                        f"  reward={np.mean(ep_rewards):6.2f}"
                        f"  latency={np.mean(ep_latencies):6.1f}ms"
                        f"  success={np.mean(ep_successes):.3f}"
                        f"  fps={timestep/elapsed:.0f}"
                        f"  lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                ep_reward = 0.0
                obs_np, _ = env.reset()
                obs = obs_to_torch(obs_np, device)
            else:
                obs = obs_to_torch(next_obs_np, device)

        with torch.no_grad():
            _, _, _, last_val = policy.get_action_and_value(
                {k: v.unsqueeze(0) for k, v in obs.items()}
            )
        ppo_update(policy, optimizer, buffer, last_val.item(), device)
        scheduler.step()

    print(f"[{method_tag}] Training complete.")
    print(f"[{method_tag}] Best rolling mean reward: {best_mean_reward:.3f} "
          f"(final: {float(np.mean(ep_rewards)):.3f})")

    # Restore best checkpoint (prevents returning a collapsed final policy)
    policy.load_state_dict(best_state_dict)
    print(f"[{method_tag}] Restored best checkpoint (reward={best_mean_reward:.3f})")

    # Persist training log
    if log_path is not None:
        with open(log_path, "w") as f:
            json.dump(train_log, f)
        print(f"[{method_tag}] Training log -> {log_path}")

    return policy


# ---------------------------------------------------------------------------
# Evaluation loop (100 episodes, records per-episode metrics)
# ---------------------------------------------------------------------------

def evaluate(policy_fn, env_seed: int, n_episodes: int = 100) -> Dict[str, list]:
    """
    policy_fn(obs_np) -> action_int

    Returns a dict of per-episode lists for all tracked metrics.
    """
    env = UAVTaskEnv(seed=env_seed + 9999)   # different seed from training
    metrics = {
        "reward": [],
        "success_rate": [],
        "latency_ms": [],
        "deadline_miss": [],
        "reschedule_count": [],
        "num_task_nodes": [],
        "episode_steps": [],
    }

    for _ in range(n_episodes):
        obs_np, _ = env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        done      = False
        last_info = {}

        while not done:
            action = policy_fn(obs_np)
            obs_np, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps  += 1
            last_info  = info

        metrics["reward"].append(ep_reward)
        metrics["success_rate"].append(last_info.get("success_rate", 0.0))
        metrics["latency_ms"].append(last_info.get("latency_ms", 0.0))
        metrics["deadline_miss"].append(
            1 if last_info.get("latency_ms", 0.0) > 1000.0 else 0
        )
        metrics["reschedule_count"].append(last_info.get("reschedule_count", 0))
        metrics["num_task_nodes"].append(last_info.get("num_task_nodes", 0))
        metrics["episode_steps"].append(ep_steps)

    return metrics


def summarise(metrics: Dict[str, list]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Policy functions for non-RL baselines
# ---------------------------------------------------------------------------

def make_ns_policy():
    """No Shaping: always noop (action 0)."""
    def policy(obs_np):
        return 0
    return policy


def make_hs_policy():
    """Heuristic Shaping: rule-based suggest_action -> translate to node-level action."""
    shaper = AdaptiveTaskShaper()

    def policy(obs_np):
        from core.task_shaping import SchedulingFeedback
        fb_vec = obs_np["feedback"]
        fb = SchedulingFeedback(
            latency_ratio       = float(fb_vec[0]),
            success_rate        = float(fb_vec[1]),
            reschedule_count    = int(fb_vec[2]),
            avg_uav_utilization = float(fb_vec[3]),
            min_link_bw_ratio   = float(fb_vec[4]),
            min_battery_ratio   = float(fb_vec[5]),
        )
        op = shaper.suggest_action(fb)
        # op=0 -> noop(0); op=1 -> split node 0 (action 1); op=2 -> merge pair (0,1) (action N+1)
        from configs.params import MAX_TASK_NODES
        if op == 0:
            return 0
        elif op == 1:
            return 1              # split padded position 0 (heaviest by observation order)
        else:
            return MAX_TASK_NODES + 1   # merge first valid pair
    return policy


def make_rnd_policy(seed: int = 0):
    """Random Shaping: uniform over valid actions (uses action masking)."""
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")

    # Borrow the mask computation from NodeLevelActorCritic
    dummy = NodeLevelActorCritic()

    def policy(obs_np):
        obs_t = {
            k: torch.as_tensor(
                v, dtype=torch.float32 if v.dtype == np.float32 else torch.long
            ).unsqueeze(0)
            for k, v in obs_np.items()
        }
        with torch.no_grad():
            mask = dummy.compute_action_mask(obs_t).squeeze(0).numpy()
        valid = np.where(mask)[0]
        return int(rng.choice(valid))

    return policy


def make_rl_policy(policy_module: nn.Module, device: torch.device):
    """Wrap a trained RL policy as a callable policy_fn (deterministic argmax at eval)."""
    policy_module.eval()

    def policy(obs_np):
        obs_t = {
            k: torch.as_tensor(
                v, dtype=torch.float32 if v.dtype == np.float32 else torch.long
            ).unsqueeze(0).to(device)
            for k, v in obs_np.items()
        }
        with torch.no_grad():
            logits, _ = policy_module(obs_t)
            if hasattr(policy_module, "compute_action_mask"):
                mask = policy_module.compute_action_mask(obs_t)
                logits = logits.masked_fill(~mask, float("-inf"))
            action = logits.argmax(dim=-1)
        return action.item()

    return policy


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

METHOD_CHOICES = ["ns", "hs", "rnd", "rl_global", "rl_node"]


def run_one(method: str, seed: int, train: bool, device: torch.device,
            n_eval: int, total_timesteps: int | None = None):
    out_path = RESULTS_DIR / f"{method}_seed{seed}.json"
    if out_path.exists():
        print(f"[skip] {out_path} already exists — delete to re-run.")
        return

    print(f"\n{'='*60}")
    print(f"  Method: {method.upper()}   seed={seed}")
    print(f"{'='*60}")

    # Build policy_fn -------------------------------------------------------
    if method == "ns":
        policy_fn = make_ns_policy()

    elif method == "hs":
        policy_fn = make_hs_policy()

    elif method == "rnd":
        policy_fn = make_rnd_policy(seed=seed)

    elif method == "rl_global":
        model_path = RESULTS_DIR / f"rl_global_seed{seed}.pt"
        log_path   = RESULTS_DIR / f"rl_global_seed{seed}_trainlog.json"
        policy = ActorCriticPolicy(action_dim=ACTION_DIM, features_dim=128).to(device)
        if train:
            policy = train_rl(policy, seed, device, "rl_global",
                              log_path=log_path, total_timesteps=total_timesteps)
            torch.save(policy.state_dict(), model_path)
        elif model_path.exists():
            policy.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise FileNotFoundError(
                f"{model_path} not found — run with --train first."
            )
        policy_fn = make_rl_policy(policy, device)

    elif method == "rl_node":
        model_path = RESULTS_DIR / f"rl_node_seed{seed}.pt"
        log_path   = RESULTS_DIR / f"rl_node_seed{seed}_trainlog.json"
        policy = NodeLevelActorCritic().to(device)
        if train:
            policy = train_rl(policy, seed, device, "rl_node",
                              log_path=log_path, total_timesteps=total_timesteps)
            torch.save(policy.state_dict(), model_path)
        elif model_path.exists():
            policy.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise FileNotFoundError(
                f"{model_path} not found — run with --train first."
            )
        policy_fn = make_rl_policy(policy, device)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Evaluate ---------------------------------------------------------------
    print(f"[{method}] Evaluating {n_eval} episodes …")
    metrics = evaluate(policy_fn, env_seed=seed, n_episodes=n_eval)
    summary = summarise(metrics)

    result = {"method": method, "seed": seed, "summary": summary, "episodes": metrics}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{method}] Results -> {out_path}")
    print(f"  reward={summary['reward']:.3f}  success={summary['success_rate']:.3f}"
          f"  latency={summary['latency_ms']:.1f}ms"
          f"  miss_rate={summary['deadline_miss']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHOD_CHOICES, default=None,
                        help="Single method to run")
    parser.add_argument("--all",   action="store_true",
                        help="Run all 5 methods")
    parser.add_argument("--train", action="store_true",
                        help="Train RL methods before evaluating")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="Random seeds (e.g. --seeds 0 1 2 3 4)")
    parser.add_argument("--n_eval", type=int, default=100,
                        help="Number of evaluation episodes per method/seed")
    parser.add_argument("--device", default="auto",
                        help="cuda / cpu / auto")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override TOTAL_TIMESTEPS for RL training (e.g. 50000 for quick test)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[main] Device: {device}")

    methods = METHOD_CHOICES if args.all else [args.method]
    if not methods or methods == [None]:
        parser.print_help()
        return

    for seed in args.seeds:
        for method in methods:
            run_one(method, seed, args.train, device, args.n_eval,
                    total_timesteps=args.timesteps)

    print("\n[main] All done. Run `python plot_results.py` to generate figures.")


if __name__ == "__main__":
    main()
