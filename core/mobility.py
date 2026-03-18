# core/mobility.py
"""
UAV Mobility and Heterogeneous Energy Model
-------------------------------------------
Implements a Random Waypoint mobility model within a bounded 2-D arena.
Each UAV moves toward a randomly chosen waypoint; upon arrival a new
waypoint is selected.  Battery is consumed by both computation and
wireless transmission.

Key outputs consumed by the environment:
  - bandwidth_matrix()  : (N, N) float32 — current inter-UAV link BW (Mbps)
  - node_features()     : (N, 3) float32 — dynamic [cpu_ratio, avg_bw_ratio, battery_ratio]
  - effective_cpu()     : (N,)  float32  — CPU capacity after low-battery derating
  - min_active_link_bw(): float           — weakest link among active-UAV pairs

Academic motivation
-------------------
Static resource-graph assumptions are violated in real UAV swarms because:
  1. Mobility causes time-varying link bandwidths → inter-task transfer
     latency is non-stationary.
  2. Heterogeneous battery depletion shrinks the effective fleet over time.
  3. The Task Shaping agent must learn to MERGE tasks whose communication
     partners have weak links, and SPLIT tasks on high-CPU nodes before
     those nodes deplete their batteries — behaviours that cannot emerge
     in static environments.
"""

from __future__ import annotations

import numpy as np

from configs.params import (
    NUM_UAVS,
    UAV_PROFILES,
    AREA_SIZE_M,
    UAV_SPEED_MPS,
    SIM_STEP_S,
    COMM_RANGE_M,
    BW_HALF_DIST_M,
    BW_MAX_MBPS,
    BW_MIN_MBPS,
    ENERGY_CPU_COEF,
    ENERGY_TX_COEF,
    BATTERY_LOW_RATIO,
)


class UAVMobilityModel:
    """
    Random Waypoint mobility + heterogeneous energy model for a UAV swarm.

    State
    -----
    positions  : (N, 2) float32  — current XY position in [0, AREA_SIZE_M]²
    waypoints  : (N, 2) float32  — current movement target
    battery    : (N,)  float32  — remaining energy fraction in [0, 1]

    Usage
    -----
    mob = UAVMobilityModel(rng=np.random.default_rng(42))
    mob.reset()
    for step in episode:
        bw = mob.bandwidth_matrix()        # read before step to get current state
        feats = mob.node_features()
        mob.consume_energy(cpu_alloc, tx_mb)   # deduct energy from scheduling
        mob.step()                             # advance positions
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng or np.random.default_rng()
        self._n   = NUM_UAVS

        # Static per-profile CPU and BW ceilings (used for normalisation)
        self._cpu_cap = np.array([p["cpu"] for p in UAV_PROFILES], dtype=np.float32)
        self._init_battery = np.array([p["battery"] for p in UAV_PROFILES], dtype=np.float32)

        # Mutable state (initialised by reset)
        self.positions = np.zeros((self._n, 2), dtype=np.float32)
        self.waypoints = np.zeros((self._n, 2), dtype=np.float32)
        self.battery   = np.zeros(self._n, dtype=np.float32)

        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> "UAVMobilityModel":
        """Randomise positions/waypoints; restore battery to profile values."""
        self.positions = self._rng.uniform(0, AREA_SIZE_M, size=(self._n, 2)).astype(np.float32)
        self.waypoints = self._rng.uniform(0, AREA_SIZE_M, size=(self._n, 2)).astype(np.float32)
        self.battery   = self._init_battery.copy()
        return self

    # ------------------------------------------------------------------
    def step(self) -> None:
        """
        Advance each active UAV one simulation tick (SIM_STEP_S seconds)
        toward its current waypoint.  A new waypoint is drawn on arrival.
        Depleted UAVs stay in place.
        """
        step_dist = UAV_SPEED_MPS * SIM_STEP_S
        for i in range(self._n):
            if self.battery[i] <= 0.0:
                continue                         # offline: no movement
            direction = self.waypoints[i] - self.positions[i]
            dist = float(np.linalg.norm(direction))
            if dist <= step_dist:
                self.positions[i] = self.waypoints[i].copy()
                self.waypoints[i] = self._rng.uniform(0, AREA_SIZE_M, size=2).astype(np.float32)
            else:
                self.positions[i] += (direction / dist) * step_dist

    # ------------------------------------------------------------------
    def consume_energy(
        self,
        cpu_alloc: np.ndarray,
        tx_mb: np.ndarray,
    ) -> None:
        """
        Deduct battery based on computation and transmission load.

        Parameters
        ----------
        cpu_alloc : (N,) — CPU units consumed per UAV this scheduling round
        tx_mb     : (N,) — MB transmitted per UAV this scheduling round
        """
        delta = ENERGY_CPU_COEF * cpu_alloc + ENERGY_TX_COEF * tx_mb
        self.battery = np.clip(self.battery - delta, 0.0, 1.0)

    # ------------------------------------------------------------------
    def bandwidth_matrix(self) -> np.ndarray:
        """
        Compute the (N, N) pairwise link-bandwidth matrix.

        Model: exponential decay with distance
            BW(d) = BW_MIN + (BW_MAX - BW_MIN) * exp(-d / BW_HALF_DIST)
        Links to/from depleted UAVs are set to 0.

        Returns
        -------
        bw : (N, N) float32  — bandwidth in Mbps; diagonal = 0
        """
        bw = np.zeros((self._n, self._n), dtype=np.float32)
        bw_range = BW_MAX_MBPS - BW_MIN_MBPS

        for i in range(self._n):
            if self.battery[i] <= 0.0:
                continue
            for j in range(self._n):
                if i == j or self.battery[j] <= 0.0:
                    continue
                d = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if d > COMM_RANGE_M:
                    bw[i, j] = BW_MIN_MBPS          # out-of-range floor
                else:
                    bw[i, j] = BW_MIN_MBPS + bw_range * float(
                        np.exp(-d / BW_HALF_DIST_M)
                    )
        return bw

    # ------------------------------------------------------------------
    def effective_cpu(self) -> np.ndarray:
        """
        Return effective CPU capacity per UAV after low-battery derating.

        UAVs below BATTERY_LOW_RATIO enter low-power mode (50 % CPU).
        Depleted UAVs (battery = 0) contribute 0 CPU.

        Returns
        -------
        (N,) float32
        """
        eff = self._cpu_cap.copy()
        low_power = (self.battery > 0.0) & (self.battery < BATTERY_LOW_RATIO)
        depleted  = self.battery <= 0.0
        eff[low_power] *= 0.5
        eff[depleted]   = 0.0
        return eff

    # ------------------------------------------------------------------
    def node_features(self) -> np.ndarray:
        """
        Build dynamic per-UAV feature matrix for the resource graph GNN.

        Features (all normalised to [0, 1]):
          col 0 — effective_cpu / max_cpu_cap  : compute availability
          col 1 — mean link BW to active peers / BW_MAX_MBPS : connectivity
          col 2 — battery remaining fraction   : energy state

        Returns
        -------
        (N, 3) float32
        """
        bw_matrix = self.bandwidth_matrix()                      # (N, N)
        eff_cpu   = self.effective_cpu()                         # (N,)

        # Mean BW to active peers (exclude self and depleted UAVs)
        active_mask = (self.battery > 0.0).astype(np.float32)   # (N,)
        peer_count  = max(active_mask.sum() - 1.0, 1.0)
        avg_bw      = bw_matrix.sum(axis=1) / peer_count        # (N,)

        features = np.stack([
            eff_cpu / max(self._cpu_cap.max(), 1e-6),
            avg_bw  / BW_MAX_MBPS,
            self.battery,
        ], axis=1).astype(np.float32)                            # (N, 3)

        return features

    # ------------------------------------------------------------------
    def min_active_link_bw(self) -> float:
        """
        Return the minimum bandwidth among all active-UAV pairs.
        Used as a feedback signal to trigger merge when links are weak.

        Returns 0.0 if fewer than 2 UAVs are active.
        """
        bw = self.bandwidth_matrix()
        active = self.battery > 0.0
        if active.sum() < 2:
            return 0.0
        # Only consider pairs where both ends are active
        mask = np.outer(active, active)
        np.fill_diagonal(mask, False)
        active_bw = bw[mask]
        return float(active_bw.min()) if len(active_bw) > 0 else 0.0
