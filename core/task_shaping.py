# core/task_shaping.py
"""
Adaptive Task Shaping
---------------------
Observes feedback from the previous scheduling round (latency ratio,
success rate, reschedule count) and mutates the Task Graph DAG topology
to make future matching easier / more efficient.

Two primitive operations are supported:
  - split_node  : decompose a heavy task into two lighter sub-tasks
  - merge_nodes : collapse two lightweight parallel tasks into one

These primitives are intentionally simple so the RL agent can learn
*when* to apply them via the action space rather than hard-coding policy.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

import networkx as nx

from configs.params import (
    MAX_TASK_NODES,
    SHAPING_MERGE_THRESHOLD,
    SHAPING_SPLIT_THRESHOLD,
    MAX_SHAPING_STEPS,
)


# ---------------------------------------------------------------------------
# Feedback dataclass
# ---------------------------------------------------------------------------

class SchedulingFeedback:
    """
    Carries performance signals from the previous matching step.

    Attributes
    ----------
    latency_ratio       : achieved_latency / deadline  (>1.0 means deadline missed)
    success_rate        : fraction of tasks successfully offloaded in [0, 1]
    reschedule_count    : number of tasks that had to be rescheduled
    avg_uav_utilization : fraction of total UAV CPU capacity consumed in [0, 1]
                          High value → fleet is heavily loaded → agent should consider merge
    """

    def __init__(
        self,
        latency_ratio: float = 0.0,
        success_rate: float = 1.0,
        reschedule_count: int = 0,
        avg_uav_utilization: float = 0.0,
        min_link_bw_ratio: float = 1.0,
    ):
        self.latency_ratio       = float(latency_ratio)
        self.success_rate        = float(success_rate)
        self.reschedule_count    = int(reschedule_count)
        self.avg_uav_utilization = float(avg_uav_utilization)
        # min_link_bw_ratio: weakest active link / BW_MAX_MBPS ∈ [0, 1]
        # Low value → topology is fragmented → merge tasks to cut cross-link traffic
        self.min_link_bw_ratio   = float(min_link_bw_ratio)

    def as_vector(self) -> List[float]:
        """Return a fixed-length (5) feature vector for use as RL state input."""
        return [
            self.latency_ratio,
            self.success_rate,
            float(self.reschedule_count),
            self.avg_uav_utilization,
            self.min_link_bw_ratio,
        ]


# ---------------------------------------------------------------------------
# Core shaping logic
# ---------------------------------------------------------------------------

class AdaptiveTaskShaper:
    """
    Stateless transformer that applies topology mutations to a Task DAG.

    The RL environment calls `shape(graph, feedback, action)` each step.
    The `action` integer selects which primitive (or no-op) to apply.

    Action space (discrete):
        0 — no-op
        1 — split the most CPU-heavy node
        2 — merge two CPU-lightest parallel leaf nodes
    """

    ACTION_NOOP  = 0
    ACTION_SPLIT = 1
    ACTION_MERGE = 2
    NUM_ACTIONS  = 3

    # ------------------------------------------------------------------
    def shape(
        self,
        graph: nx.DiGraph,
        feedback: SchedulingFeedback,
        action: int,
    ) -> Tuple[nx.DiGraph, Dict]:
        """
        Apply the chosen shaping action and return the mutated graph
        together with a dict of metadata (for logging / reward shaping).

        Returns
        -------
        new_graph : nx.DiGraph  — mutated copy (original is never modified)
        info      : dict        — {'action_taken': str, 'delta_nodes': int}
        """
        g = copy.deepcopy(graph)
        info: Dict = {"action_taken": "noop", "delta_nodes": 0}

        if action == self.ACTION_SPLIT:
            g, info = self._try_split(g, feedback)
        elif action == self.ACTION_MERGE:
            g, info = self._try_merge(g, feedback)

        return g, info

    # ------------------------------------------------------------------
    # Heuristic: rule-based suggestions (used by wrapper / curriculum)
    # ------------------------------------------------------------------

    def suggest_action(self, feedback: SchedulingFeedback) -> int:
        """
        Heuristic to bootstrap training.  Priority order:

          1. Link quality collapse (mobility) → merge to cut cross-UAV traffic
             When min_link_bw_ratio < 0.15, inter-UAV data transfers dominate
             latency; merging tasks collocates computation, eliminating those
             transfers entirely.

          2. Fleet saturated (high util + low success) → merge to ease scheduling

          3. Deadline pressure but fleet has headroom → split to expose parallelism

          4. Success rate collapsed (energy depletion / over-subscription) → merge

          5. Otherwise no-op

        The RL agent is expected to discover superior policies beyond this heuristic.
        """
        # Topology-aware: weak links dominate transfer latency → merge
        if feedback.min_link_bw_ratio < 0.15:
            return self.ACTION_MERGE

        # Resource-aware: fleet is saturated → reduce task granularity
        if (feedback.avg_uav_utilization > 0.85
                and feedback.success_rate < 0.8):
            return self.ACTION_MERGE

        # Latency-aware: deadline pressure → split heavy node to expose parallelism
        if feedback.latency_ratio > SHAPING_SPLIT_THRESHOLD:
            return self.ACTION_SPLIT

        # Success-aware: many tasks failing → fleet overwhelmed → merge
        if feedback.success_rate < SHAPING_MERGE_THRESHOLD:
            return self.ACTION_MERGE

        return self.ACTION_NOOP

    # ------------------------------------------------------------------
    # Primitive: split
    # ------------------------------------------------------------------

    def _try_split(
        self, g: nx.DiGraph, feedback: SchedulingFeedback
    ) -> Tuple[nx.DiGraph, Dict]:
        """
        Find the node with the highest CPU demand and split it into two
        child nodes with half the CPU demand each, preserving all edges.
        """
        if len(g.nodes) >= MAX_TASK_NODES:
            return g, {"action_taken": "split_skipped_max", "delta_nodes": 0}

        # pick heaviest node
        target = max(g.nodes, key=lambda n: g.nodes[n].get("cpu", 0.0))
        cpu = g.nodes[target]["cpu"]
        data_size = g.nodes[target].get("data_size", 1.0)

        # new node ids
        new_id = max(g.nodes) + 1
        child_a, child_b = target, new_id

        # update target in place (becomes child_a)
        g.nodes[child_a]["cpu"]       = cpu / 2.0
        g.nodes[child_a]["data_size"] = data_size / 2.0

        # add child_b
        g.add_node(
            child_b,
            cpu=cpu / 2.0,
            data_size=data_size / 2.0,
        )

        # child_a → child_b (sequential dependency)
        g.add_edge(child_a, child_b, transfer_mb=data_size / 2.0)

        # re-attach child_b to all original successors of child_a
        for succ in list(g.successors(child_a)):
            if succ != child_b:
                g.add_edge(child_b, succ, transfer_mb=g[child_a][succ].get("transfer_mb", 1.0))
                g.remove_edge(child_a, succ)

        return g, {"action_taken": "split", "delta_nodes": 1}

    # ------------------------------------------------------------------
    # Primitive: merge
    # ------------------------------------------------------------------

    def _try_merge(
        self, g: nx.DiGraph, feedback: SchedulingFeedback
    ) -> Tuple[nx.DiGraph, Dict]:
        """
        Find two parallel leaf nodes (no successors) with the lowest
        combined CPU demand and merge them into one.
        """
        leaves = [n for n in g.nodes if g.out_degree(n) == 0]
        if len(leaves) < 2:
            return g, {"action_taken": "merge_skipped_no_leaves", "delta_nodes": 0}

        # sort by cpu ascending, pick two cheapest
        leaves.sort(key=lambda n: g.nodes[n].get("cpu", 0.0))
        a, b = leaves[0], leaves[1]

        merged_cpu       = g.nodes[a].get("cpu", 0.0) + g.nodes[b].get("cpu", 0.0)
        merged_data_size = g.nodes[a].get("data_size", 1.0) + g.nodes[b].get("data_size", 1.0)

        # keep node a, update attributes
        g.nodes[a]["cpu"]       = merged_cpu
        g.nodes[a]["data_size"] = merged_data_size

        # redirect all predecessors of b to a
        for pred in list(g.predecessors(b)):
            if pred != a:
                g.add_edge(pred, a, transfer_mb=g[pred][b].get("transfer_mb", 1.0))

        g.remove_node(b)

        return g, {"action_taken": "merge", "delta_nodes": -1}

    # ------------------------------------------------------------------
    # Targeted primitives: agent selects WHICH node to operate on
    # ------------------------------------------------------------------

    def split_node(self, graph: nx.DiGraph, node_id: int) -> Tuple[nx.DiGraph, Dict]:
        """
        Split a SPECIFIC node (identified by node_id in the NetworkX graph).

        Unlike _try_split which always picks the heaviest node, here the
        caller (RL agent via node-level policy) chose this node because
        it attends to the resource graph and sees that its assigned UAV
        has sufficient spare CPU to handle two lighter sub-tasks.
        """
        g = copy.deepcopy(graph)
        if node_id not in g.nodes:
            return g, {"action_taken": "split_skipped_invalid", "delta_nodes": 0}
        if len(g.nodes) >= MAX_TASK_NODES:
            return g, {"action_taken": "split_skipped_max", "delta_nodes": 0}

        cpu       = g.nodes[node_id]["cpu"]
        data_size = g.nodes[node_id].get("data_size", 1.0)
        new_id    = max(g.nodes) + 1

        g.nodes[node_id]["cpu"]       = cpu / 2.0
        g.nodes[node_id]["data_size"] = data_size / 2.0
        g.add_node(new_id, cpu=cpu / 2.0, data_size=data_size / 2.0)
        g.add_edge(node_id, new_id, transfer_mb=data_size / 2.0)

        for succ in list(g.successors(node_id)):
            if succ != new_id:
                g.add_edge(new_id, succ,
                           transfer_mb=g[node_id][succ].get("transfer_mb", 1.0))
                g.remove_edge(node_id, succ)

        return g, {"action_taken": "split", "delta_nodes": 1, "target": node_id}

    def merge_node_pair(
        self, graph: nx.DiGraph, node_i: int, node_j: int
    ) -> Tuple[nx.DiGraph, Dict]:
        """
        Merge two SPECIFIC nodes (identified by node IDs in NetworkX graph).

        The agent selects this pair because cross-graph attention reveals
        that both nodes would be assigned to UAVs connected by a weak link,
        so collapsing them eliminates that cross-UAV transfer entirely.

        Validity: merging is skipped if there exists any directed path
        between the two nodes (would create a cycle in the DAG).
        """
        if node_i not in graph.nodes or node_j not in graph.nodes or node_i == node_j:
            return graph, {"action_taken": "merge_skipped_invalid", "delta_nodes": 0}

        # Cycle-safety check: reject if nodes are topologically related
        if nx.has_path(graph, node_i, node_j) or nx.has_path(graph, node_j, node_i):
            return graph, {"action_taken": "merge_skipped_cycle", "delta_nodes": 0}

        g = copy.deepcopy(graph)

        # Absorb node_j into node_i
        g.nodes[node_i]["cpu"] = (
            g.nodes[node_i].get("cpu", 0.0) + g.nodes[node_j].get("cpu", 0.0)
        )
        g.nodes[node_i]["data_size"] = (
            g.nodes[node_i].get("data_size", 0.0) + g.nodes[node_j].get("data_size", 0.0)
        )

        for pred in list(g.predecessors(node_j)):
            if pred != node_i and not g.has_edge(pred, node_i):
                g.add_edge(pred, node_i,
                           transfer_mb=g[pred][node_j].get("transfer_mb", 1.0))
        for succ in list(g.successors(node_j)):
            if succ != node_i and not g.has_edge(node_i, succ):
                g.add_edge(node_i, succ,
                           transfer_mb=g[node_j][succ].get("transfer_mb", 1.0))

        g.remove_node(node_j)
        return g, {"action_taken": "merge", "delta_nodes": -1,
                   "targets": (node_i, node_j)}

    def shape_targeted(
        self,
        graph: nx.DiGraph,
        op_type: int,
        node_i: int,
        node_j: int = -1,
    ) -> Tuple[nx.DiGraph, Dict]:
        """
        Dispatch to the appropriate targeted primitive.

        op_type : 0 = noop, 1 = split(node_i), 2 = merge(node_i, node_j)
        node_i/j: actual NetworkX node IDs (not padded positions).
        """
        if op_type == self.ACTION_SPLIT:
            return self.split_node(graph, node_i)
        elif op_type == self.ACTION_MERGE:
            return self.merge_node_pair(graph, node_i, node_j)
        return graph, {"action_taken": "noop", "delta_nodes": 0}


# ---------------------------------------------------------------------------
# Utility: generate a random DAG for episode initialisation
# ---------------------------------------------------------------------------

def generate_random_task_dag(
    num_nodes: int,
    rng: random.Random | None = None,
) -> nx.DiGraph:
    """
    Build a random DAG with `num_nodes` tasks.
    Edges go from lower to higher node id to guarantee acyclicity.
    """
    from configs.params import TASK_CPU_DEMAND_RANGE, TASK_DATA_SIZE_RANGE

    rng = rng or random.Random()
    g = nx.DiGraph()

    for i in range(num_nodes):
        g.add_node(
            i,
            cpu=rng.uniform(*TASK_CPU_DEMAND_RANGE),
            data_size=rng.uniform(*TASK_DATA_SIZE_RANGE),
        )

    # random edges (guaranteed DAG: only i→j with i<j)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < 0.3:
                g.add_edge(
                    i, j,
                    transfer_mb=rng.uniform(*TASK_DATA_SIZE_RANGE),
                )

    # ensure connectivity: chain 0→1→2→...
    for i in range(num_nodes - 1):
        if not g.has_edge(i, i + 1):
            g.add_edge(i, i + 1, transfer_mb=rng.uniform(*TASK_DATA_SIZE_RANGE))

    return g
