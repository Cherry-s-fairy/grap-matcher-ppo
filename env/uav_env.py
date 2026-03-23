# env/uav_env.py
"""
UAVTaskEnv — Gymnasium environment for Resource-Aware Task Graph Shaping
in Heterogeneous UAV Swarms.

Episode flow
------------
1. reset()   : sample a random Task DAG + fixed heterogeneous Resource Graph.
               Feedback is initialised to zeros.
2. step(a)   : apply shaping action → greedy-match tasks to UAVs
               → compute reward → update feedback state.
3. Repeat until deadline exceeded or max steps reached.

Observation (dict)
------------------
  task_x    : (N_t, 2)   node features [cpu_demand, data_size]  — padded to MAX_TASK_NODES
  task_edge : (2, E_t)   COO edge index of the task DAG
  task_batch: (N_t,)     all zeros (single graph per env step)
  res_x     : (N_r, 3)   [cpu_cap, bandwidth, battery]
  res_edge  : (2, E_r)   fully-connected resource graph edges
  res_batch : (N_r,)     all zeros
  feedback  : (3,)       [latency_ratio, success_rate, reschedule_count]

Action space
------------
Discrete(3):  0=noop, 1=split heaviest node, 2=merge two cheapest leaves

Reward
------
  +1   for each task matched within deadline
  -0.5 for each reschedule
  -1   if total latency exceeds deadline
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces

import torch

from configs.params import (
    NUM_UAVS,
    UAV_PROFILES,
    MAX_TASK_NODES,
    DEADLINE_MS,
    TRANSFER_LATENCY_BASE,
    MAX_EPISODE_STEPS,
    BW_MAX_MBPS,
    BW_MIN_MBPS,
    ACTION_DIM,
    decode_node_action,
)
from core.task_shaping import (
    AdaptiveTaskShaper,
    SchedulingFeedback,
    generate_random_task_dag,
)
from core.mobility import UAVMobilityModel


# ---------------------------------------------------------------------------
# Greedy matching / scheduling  (placeholder for learned matcher)
# ---------------------------------------------------------------------------

def greedy_match(
    task_graph: nx.DiGraph,
    eff_cpu: np.ndarray,                # (N,) effective CPU per UAV (after derating)
    bw_matrix: Optional[np.ndarray] = None,  # (N, N) link BW in Mbps; None → use base
) -> Tuple[float, float, int, float, np.ndarray, np.ndarray]:
    """
    Topological-order greedy assignment with mobility-aware transfer latency.

    CPU is consumed per assignment within one scheduling round.  Transfer
    latency is computed from the actual bandwidth between the UAV running a
    predecessor task and the UAV running its successor — making it sensitive
    to the current topology.

    Parameters
    ----------
    task_graph : DAG of tasks
    eff_cpu    : (N,) effective CPU capacity per UAV (accounts for low-battery derating)
    bw_matrix  : (N, N) current link bandwidths in Mbps.  If None, falls back to
                 TRANSFER_LATENCY_BASE (backward-compatible static mode).

    Returns
    -------
    latency_ms      : makespan of the schedule (ms)
    success_rate    : fraction of tasks that executed successfully
    reschedule_count: tasks that had to be force-assigned to an overloaded UAV
    avg_utilization : fraction of total effective fleet CPU consumed
    cpu_per_uav     : (N,) CPU units consumed per UAV — fed to energy model
    tx_per_uav      : (N,) MB transmitted per UAV    — fed to energy model
    """
    N = len(eff_cpu)
    capacity  = {i: float(eff_cpu[i]) for i in range(N)}
    remaining = dict(capacity)
    assignments: Dict[int, int] = {}    # task_node → uav_id
    task_finish: Dict[int, float] = {}
    reschedule_count = 0
    failed = 0

    # Per-UAV energy accounting
    cpu_per_uav = np.zeros(N, dtype=np.float32)
    tx_per_uav  = np.zeros(N, dtype=np.float32)

    try:
        order = list(nx.topological_sort(task_graph))
    except nx.NetworkXUnfeasible:
        return DEADLINE_MS * 2, 0.0, len(task_graph), 1.0, cpu_per_uav, tx_per_uav

    for node in order:
        cpu_demand = task_graph.nodes[node].get("cpu", 1.0)

        # Earliest start: wait for all predecessors to finish
        pred_finish = max(
            (task_finish.get(p, 0.0) for p in task_graph.predecessors(node)),
            default=0.0,
        )

        # candidates: active UAVs (eff_cpu > 0) with enough remaining budget
        candidates = sorted(
            [(uid, cap) for uid, cap in remaining.items()
             if cap >= cpu_demand and capacity[uid] > 0.0],
            key=lambda x: -x[1],
        )

        if not candidates:
            reschedule_count += 1
            # Fall back to least-degraded active UAV
            active = {uid: cap for uid, cap in remaining.items() if capacity[uid] > 0.0}
            if not active:
                failed += 1
                task_finish[node] = pred_finish
                continue
            best_uid = max(active, key=lambda u: active[u])
            avail = remaining[best_uid]
            if avail < cpu_demand * 0.5:
                failed += 1
                task_finish[node] = pred_finish
                continue
            exec_ms = cpu_demand / max(avail, 0.01) * 100
        else:
            best_uid = candidates[0][0]
            exec_ms  = cpu_demand / remaining[best_uid] * 100

        # Transfer latency: use actual link BW between predecessor UAV and this UAV
        transfer_ms = 0.0
        for pred in task_graph.predecessors(node):
            transfer_mb = task_graph[pred][node].get("transfer_mb", 1.0)
            pred_uav = assignments.get(pred)
            if bw_matrix is not None and pred_uav is not None:
                bw = float(bw_matrix[pred_uav, best_uid])
                # latency (ms) = data (MB) / bandwidth (Mbps) * 1000
                transfer_ms += (transfer_mb / max(bw, BW_MIN_MBPS)) * 1000.0
            else:
                transfer_ms += transfer_mb * TRANSFER_LATENCY_BASE
            tx_per_uav[best_uid] += transfer_mb

        remaining[best_uid]  = max(remaining[best_uid] - cpu_demand, 0.0)
        cpu_per_uav[best_uid] += cpu_demand
        assignments[node]    = best_uid
        task_finish[node]    = pred_finish + transfer_ms + exec_ms

    total_latency = max(task_finish.values(), default=0.0)
    success_rate  = (len(task_graph) - failed) / max(len(task_graph), 1)

    total_cap   = sum(c for c in capacity.values() if c > 0)
    total_used  = sum(capacity[uid] - remaining[uid] for uid in capacity)
    avg_util    = min(total_used / max(total_cap, 1e-6), 1.0)

    return total_latency, success_rate, reschedule_count, avg_util, cpu_per_uav, tx_per_uav


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class UAVTaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_task_nodes: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._rng    = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self._num_task_nodes = num_task_nodes  # None → random each episode
        self._shaper  = AdaptiveTaskShaper()
        self._mobility = UAVMobilityModel(rng=self._np_rng)

        N  = MAX_TASK_NODES
        Nr = NUM_UAVS

        self.observation_space = spaces.Dict({
            "task_x":    spaces.Box(0.0, 10.0, shape=(N, 2),      dtype=np.float32),
            "task_edge": spaces.Box(0,   N,    shape=(2, N * N),   dtype=np.int64),
            "task_batch":spaces.Box(0,   0,    shape=(N,),         dtype=np.int64),
            # res_x is now DYNAMIC: [eff_cpu_ratio, avg_bw_ratio, battery_ratio]
            "res_x":     spaces.Box(0.0, 1.0,  shape=(Nr, 3),     dtype=np.float32),
            "res_edge":  spaces.Box(0,   Nr,   shape=(2, Nr * Nr), dtype=np.int64),
            "res_batch": spaces.Box(0,   0,    shape=(Nr,),        dtype=np.int64),
            # feedback: 6 signals including min_link_bw_ratio and min_battery_ratio
            "feedback":  spaces.Box(-np.inf, np.inf, shape=(6,),  dtype=np.float32),
        })
        # Node-level action space: 1 noop + N split + C(N,2) merge = 56 actions
        self.action_space = spaces.Discrete(ACTION_DIM)

        # episode state
        self._task_graph: nx.DiGraph = nx.DiGraph()
        self._feedback   = SchedulingFeedback()
        self._step_count = 0
        self._prev_latency_ratio = 1.0   # neutral: "at deadline", any improvement → positive delta
        self._prev_success_rate  = 0.0   # neutral: "no success", any success → positive delta

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng    = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)
            self._mobility = UAVMobilityModel(rng=self._np_rng)

        n = self._num_task_nodes or self._rng.randint(3, MAX_TASK_NODES)
        self._task_graph = generate_random_task_dag(n, rng=self._rng)
        self._mobility.reset()
        self._feedback   = SchedulingFeedback()
        self._step_count = 0
        self._prev_latency_ratio = 1.0   # neutral: "at deadline", any improvement → positive delta
        self._prev_success_rate  = 0.0   # neutral: "no success", any success → positive delta

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        self._step_count += 1

        # 1. Decode flat action → (op_type, padded_node_i, padded_node_j)
        #    and dispatch to targeted shaping (resource-aware node selection)
        op, pidx_i, pidx_j = decode_node_action(int(action))
        node_ids = sorted(self._task_graph.nodes)    # padded pos → actual node ID

        actual_i = node_ids[pidx_i] if 0 <= pidx_i < len(node_ids) else -1
        actual_j = node_ids[pidx_j] if 0 <= pidx_j < len(node_ids) else -1

        self._task_graph, shape_info = self._shaper.shape_targeted(
            self._task_graph, op, actual_i, actual_j
        )

        # 2. Read current mobility state
        bw_matrix = self._mobility.bandwidth_matrix()   # (N, N) Mbps
        eff_cpu   = self._mobility.effective_cpu()      # (N,) derated CPU

        # 3. Greedy matching with mobility-aware transfer latency
        latency, success_rate, reschedule_count, avg_util, cpu_used, tx_used = greedy_match(
            self._task_graph, eff_cpu, bw_matrix
        )

        # 4. Deduct energy from the fleet based on this scheduling round
        self._mobility.consume_energy(cpu_used, tx_used)

        # 5. Advance UAV positions for next step
        self._mobility.step()

        # 6. Update feedback — includes topology (link BW) and energy (battery) signals
        latency_ratio    = latency / DEADLINE_MS
        min_bw_ratio     = self._mobility.min_active_link_bw() / BW_MAX_MBPS
        min_battery      = self._mobility.min_battery_ratio()
        self._feedback = SchedulingFeedback(
            latency_ratio=latency_ratio,
            success_rate=success_rate,
            reschedule_count=reschedule_count,
            avg_uav_utilization=avg_util,
            min_link_bw_ratio=min_bw_ratio,
            min_battery_ratio=min_battery,
        )

        # 7. Reward
        reward = self._compute_reward(latency, success_rate, reschedule_count)
        self._prev_latency_ratio = latency_ratio
        self._prev_success_rate  = success_rate

        # 8. Termination
        terminated = latency > DEADLINE_MS * 2          # catastrophic failure
        truncated  = self._step_count >= MAX_EPISODE_STEPS

        info = {
            "latency_ms":      latency,
            "latency_ratio":   latency_ratio,
            "success_rate":    success_rate,
            "reschedule_count":reschedule_count,
            "num_task_nodes":  len(self._task_graph),
            **shape_info,
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _compute_reward(self, latency, success_rate, reschedule_count):
        latency_ratio = latency / DEADLINE_MS
        n_nodes = len(self._task_graph)

        # --- 绝对质量信号（维持好性能也有奖励）---
        r  = 2.0 * success_rate
        r += 1.5 * max(0.0, 1.0 - latency_ratio)

        # --- Delta 奖励：鼓励比上一步更好（核心改进）---
        r += 1.0 * (self._prev_latency_ratio - latency_ratio)  # 延迟降低 → 正奖励
        r += 0.5 * (success_rate - self._prev_success_rate)    # 成功率提升 → 正奖励

        # --- 惩罚：reschedule 惩罚从 0.1 提升到 0.3 ---
        r -= 0.3 * reschedule_count

        # --- 平滑 deadline 惩罚（去掉 -2.0 硬截断）---
        if latency_ratio > 1.0:
            r -= 2.0 * (latency_ratio - 1.0)

        # --- 拓扑膨胀惩罚 ---
        r -= 0.05 * max(0, n_nodes - 12)

        return float(r)

    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build the padded numpy observation dict."""
        N  = MAX_TASK_NODES
        Nr = NUM_UAVS
        g  = self._task_graph

        # --- task graph node features (padded) ---
        task_x = np.zeros((N, 2), dtype=np.float32)
        node_ids = sorted(g.nodes)
        id_map   = {nid: i for i, nid in enumerate(node_ids)}
        for nid in node_ids:
            i = id_map[nid]
            task_x[i, 0] = g.nodes[nid].get("cpu", 0.0)
            task_x[i, 1] = g.nodes[nid].get("data_size", 0.0)

        # --- task graph edges (COO, padded) ---
        edges = [(id_map[u], id_map[v]) for u, v in g.edges]
        max_edges = N * N
        task_edge = np.zeros((2, max_edges), dtype=np.int64)
        for k, (u, v) in enumerate(edges[:max_edges]):
            task_edge[0, k] = u
            task_edge[1, k] = v

        task_batch = np.zeros(N, dtype=np.int64)

        # --- resource graph node features (dynamic from mobility model) ---
        # [eff_cpu_ratio, avg_link_bw_ratio, battery_ratio] — all ∈ [0, 1]
        res_x = self._mobility.node_features()           # (Nr, 3)

        # fully connected resource graph
        res_edges = [(i, j) for i in range(Nr) for j in range(Nr) if i != j]
        res_edge = np.zeros((2, Nr * Nr), dtype=np.int64)
        for k, (u, v) in enumerate(res_edges):
            res_edge[0, k] = u
            res_edge[1, k] = v

        res_batch = np.zeros(Nr, dtype=np.int64)

        feedback = np.array(self._feedback.as_vector(), dtype=np.float32)

        return {
            "task_x":    task_x,
            "task_edge": task_edge,
            "task_batch":task_batch,
            "res_x":     res_x,
            "res_edge":  res_edge,
            "res_batch": res_batch,
            "feedback":  feedback,
        }

    # ------------------------------------------------------------------
    def render(self):
        pass
