# models/gnn_policy.py
"""
GNN-based feature extractor for the dual-graph state.

Architecture
------------
Two separate GNN encoders (shared weights optional):
  - TaskGraphEncoder  : embeds each task node  → pooled task-graph vector
  - ResourceGraphEncoder: embeds each UAV node → pooled resource-graph vector

The concatenated embedding is passed to an Actor-Critic head.

Usage with Stable-Baselines3
-----------------------------
Subclass `BaseFeaturesExtractor` and register via policy_kwargs:

    policy_kwargs = dict(
        features_extractor_class=DualGraphExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

For the standalone / custom PPO path the same `DualGraphExtractor` can
be used as a plain `nn.Module`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

from configs.params import (
    GNN_HIDDEN_DIM, GNN_NUM_LAYERS, EMBED_DIM,
    MAX_TASK_NODES, ACTION_DIM, N_MERGE_ACTIONS,
    decode_node_action,
)


# ---------------------------------------------------------------------------
# Low-level GNN encoder (works with or without PyG)
# ---------------------------------------------------------------------------

class _GNNEncoder(nn.Module):
    """
    Stack of GraphSAGE layers → global mean pool → linear projection.

    Falls back to a simple MLP mean-pool when PyG is not installed,
    so the project can be imported for unit-testing without heavy deps.
    """

    def __init__(self, in_channels: int, hidden: int, out_dim: int, num_layers: int):
        super().__init__()
        self._use_pyg = _HAS_PYG

        if self._use_pyg:
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(in_channels, hidden))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden, hidden))
            self.norms = nn.ModuleList(
                [nn.LayerNorm(hidden) for _ in range(num_layers)]
            )
        else:
            # MLP fallback: treat node features as a set (no edges)
            layers = [nn.Linear(in_channels, hidden), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            self.mlp = nn.Sequential(*layers)

        self.proj = nn.Linear(hidden, out_dim)
        self.act  = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (B, N, F) or (N, F)   node features — padded to fixed N
        edge_index : (B, 2, E) or (2, E)   COO edge index — padded to fixed E
        batch      : (B, N)   or (N,)       unused in padded mode (kept for API compat)

        Returns
        -------
        (B, out_dim) or (out_dim,)  graph-level embedding per batch element
        """
        # Normalise to batched 3-D: (B, N, F) / (B, 2, E)
        squeeze = (x.dim() == 2)
        if squeeze:
            x          = x.unsqueeze(0)           # (1, N, F)
            edge_index = edge_index.unsqueeze(0)   # (1, 2, E)

        B, N, _ = x.shape

        if self._use_pyg:
            # Flatten all B graphs into one big graph for a single PyG forward pass.
            # Shift edge indices of graph b by b*N so they reference the correct nodes.
            x_flat     = x.reshape(B * N, -1)                              # (B*N, F)
            offsets    = torch.arange(B, device=x.device).view(B, 1, 1) * N  # (B,1,1)
            ei_flat    = (edge_index + offsets).permute(1, 0, 2).reshape(2, -1)  # (2, B*E)
            batch_flat = torch.arange(B, device=x.device).repeat_interleave(N)   # (B*N,)

            for conv, norm in zip(self.convs, self.norms):
                x_flat = self.act(norm(conv(x_flat, ei_flat)))
            x_out = global_mean_pool(x_flat, batch_flat)                   # (B, hidden)
        else:
            # MLP path: apply to last dim, then mean-pool over node axis.
            # Zero-padded nodes have all-zero features → contribute ~0 to the mean,
            # which is an acceptable approximation for a research scaffold.
            x_out = self.mlp(x).mean(dim=1)                                # (B, hidden)

        out = self.proj(x_out)                                             # (B, out_dim)
        return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------
# Dual-graph feature extractor
# ---------------------------------------------------------------------------

class DualGraphExtractor(nn.Module):
    """
    Encodes (task_graph, resource_graph, feedback_vector) into a single
    flat feature vector suitable for an Actor-Critic head.

    Input dict keys expected from the environment observation:
        "task_x"        : (N_t, task_node_feat_dim)
        "task_edge"     : (2, E_t)
        "task_batch"    : (N_t,)
        "res_x"         : (N_r, res_node_feat_dim)
        "res_edge"      : (2, E_r)
        "res_batch"     : (N_r,)
        "feedback"      : (4,)   [latency_ratio, success_rate, reschedule_count, avg_uav_utilization]
    """

    TASK_NODE_FEAT_DIM = 2   # [cpu_demand, data_size]
    RES_NODE_FEAT_DIM  = 3   # [cpu_cap, bandwidth, battery]
    FEEDBACK_DIM       = 6   # [latency_ratio, success_rate, reschedule_count, avg_uav_utilization, min_link_bw_ratio, min_battery_ratio]

    def __init__(self, features_dim: int = 128):
        super().__init__()
        self.task_enc = _GNNEncoder(
            in_channels=self.TASK_NODE_FEAT_DIM,
            hidden=GNN_HIDDEN_DIM,
            out_dim=EMBED_DIM,
            num_layers=GNN_NUM_LAYERS,
        )
        self.res_enc = _GNNEncoder(
            in_channels=self.RES_NODE_FEAT_DIM,
            hidden=GNN_HIDDEN_DIM,
            out_dim=EMBED_DIM,
            num_layers=GNN_NUM_LAYERS,
        )
        fused_dim = EMBED_DIM + EMBED_DIM + self.FEEDBACK_DIM
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )
        self.features_dim = features_dim

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        task_emb = self.task_enc(
            obs["task_x"], obs["task_edge"], obs["task_batch"]
        )
        res_emb = self.res_enc(
            obs["res_x"], obs["res_edge"], obs["res_batch"]
        )
        feedback = obs["feedback"]                  # (B, 3)
        if feedback.dim() == 1:
            feedback = feedback.unsqueeze(0)

        fused = torch.cat([task_emb, res_emb, feedback], dim=-1)
        return self.fusion(fused)


# ---------------------------------------------------------------------------
# Minimal Actor-Critic head (placeholder — swap in SB3 PPO for full training)
# ---------------------------------------------------------------------------

class ActorCriticPolicy(nn.Module):
    """
    Thin Actor-Critic wrapper around DualGraphExtractor.

    action_dim should equal AdaptiveTaskShaper.NUM_ACTIONS (3).
    """

    def __init__(self, action_dim: int = 3, features_dim: int = 128):
        super().__init__()
        self.extractor = DualGraphExtractor(features_dim=features_dim)

        self.actor  = nn.Linear(features_dim, action_dim)
        self.critic = nn.Linear(features_dim, 1)

    def forward(self, obs: Dict[str, torch.Tensor]):
        feat   = self.extractor(obs)
        logits = self.actor(feat)
        value  = self.critic(feat)
        return logits, value

    def get_action_and_value(self, obs: Dict[str, torch.Tensor]):
        logits, value = self.forward(obs)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ===========================================================================
# NODE-LEVEL POLICY  (resource-aware node selection)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Node-level GNN encoder — returns (B, N, d), no pooling
# ---------------------------------------------------------------------------

class _GNNNodeEncoder(nn.Module):
    """
    Same SAGEConv stack as _GNNEncoder but skips the global mean-pool,
    returning per-node embeddings (B, N, out_dim).

    These node-level vectors feed the cross-graph attention and the
    per-node scoring heads, making node selection resource-aware.
    """

    def __init__(self, in_channels: int, hidden: int, out_dim: int, num_layers: int):
        super().__init__()
        self._use_pyg = _HAS_PYG

        if self._use_pyg:
            self.convs = nn.ModuleList(
                [SAGEConv(in_channels, hidden)]
                + [SAGEConv(hidden, hidden) for _ in range(num_layers - 1)]
            )
            self.norms = nn.ModuleList(
                [nn.LayerNorm(hidden) for _ in range(num_layers)]
            )
        else:
            layers: list = [nn.Linear(in_channels, hidden), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            self.mlp = nn.Sequential(*layers)

        self.proj = nn.Linear(hidden, out_dim)
        self.act  = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (B, N, F)  — padded node features
        edge_index : (B, 2, E)  — padded COO edge index

        Returns
        -------
        (B, N, out_dim)  — per-node embeddings, padding rows included
        """
        B, N, _ = x.shape

        if self._use_pyg:
            x_flat  = x.reshape(B * N, -1)
            offsets = torch.arange(B, device=x.device).view(B, 1, 1) * N
            ei_flat = (edge_index + offsets).permute(1, 0, 2).reshape(2, -1)
            for conv, norm in zip(self.convs, self.norms):
                x_flat = self.act(norm(conv(x_flat, ei_flat)))
            x_out = x_flat.reshape(B, N, -1)               # (B, N, hidden)
        else:
            x_out = self.mlp(x)                             # (B, N, hidden)

        return self.proj(x_out)                             # (B, N, out_dim)


# ---------------------------------------------------------------------------
# 2. Cross-graph attention — task nodes attend to resource nodes
# ---------------------------------------------------------------------------

class CrossGraphAttention(nn.Module):
    """
    Task-to-Resource cross-attention.

    For each task node i, compute a weighted sum over resource nodes:

        enriched(i) = sum_r  softmax_r[ q(task_i) · k(res_r) / √d ]  · v(res_r)

    This is the architectural innovation that makes node selection truly
    resource-aware: the per-node split/merge scores are conditioned on
    which UAV each task node "attends to" and that UAV's current
    CPU / bandwidth / battery state.
    """

    def __init__(self, task_dim: int, res_dim: int, attn_dim: int):
        super().__init__()
        self.q_proj   = nn.Linear(task_dim, attn_dim, bias=False)
        self.k_proj   = nn.Linear(res_dim,  attn_dim, bias=False)
        self.v_proj   = nn.Linear(res_dim,  attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, task_dim)
        self.scale    = attn_dim ** 0.5

    def forward(
        self,
        task_emb: torch.Tensor,   # (B, N_t, task_dim)
        res_emb:  torch.Tensor,   # (B, N_r, res_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        out         : (B, N_t, task_dim) — resource-enriched task embeddings
        attn_weights: (B, N_t, N_r)     — soft task-to-UAV assignment weights
                      Used downstream to compute per-pair link-quality signals
                      for the merge scoring head.
        """
        Q    = self.q_proj(task_emb)                              # (B, N_t, d)
        K    = self.k_proj(res_emb)                               # (B, N_r, d)
        V    = self.v_proj(res_emb)                               # (B, N_r, d)
        attn = torch.softmax(Q @ K.transpose(-1, -2) / self.scale, dim=-1)  # (B, N_t, N_r)
        ctx  = attn @ V                                           # (B, N_t, d)
        return self.out_proj(ctx), attn                           # also expose attn weights


# ---------------------------------------------------------------------------
# 3. NodeLevelActorCritic — full node-level policy
# ---------------------------------------------------------------------------

class NodeLevelActorCritic(nn.Module):
    """
    Resource-aware node-level Actor-Critic for task graph shaping.

    Action space: Discrete(ACTION_DIM = 56)
        index 0         → noop
        index 1..N      → split(node at padded position i-1)
        index N+1..N+C  → merge(node i, node j)  [C = C(N,2) = 45 pairs]

    Key properties vs. the old ActorCriticPolicy
    ─────────────────────────────────────────────
    • Per-node split scores:  every task node gets a separate logit derived
      from its resource-enriched embedding.  The agent learns to prefer
      splitting nodes whose cross-attention weight falls on high-CPU UAVs.

    • Per-pair merge scores:  the merge logit for (i,j) is MLP(emb_i ‖ emb_j).
      The agent learns to prefer merging nodes whose embeddings both point
      to a weak-link UAV pair, eliminating the cross-UAV transfer.

    • Action masking:  invalid actions (padding, graph-at-max, non-leaf pairs)
      are set to -inf before sampling, ensuring the env never receives
      a structurally invalid operation.
    """

    # Pre-compute flat pair indices once at class definition time
    _PAIR_I: list = [i for i in range(MAX_TASK_NODES)
                     for j in range(i + 1, MAX_TASK_NODES)]
    _PAIR_J: list = [j for i in range(MAX_TASK_NODES)
                     for j in range(i + 1, MAX_TASK_NODES)]

    def __init__(self):
        super().__init__()
        N  = MAX_TASK_NODES
        d  = EMBED_DIM                         # 32
        fb = 6                                  # feedback vector dim (incl. min_battery)

        # Node-level GNN encoders
        self.task_gnn = _GNNNodeEncoder(2, GNN_HIDDEN_DIM, d, GNN_NUM_LAYERS)
        self.res_gnn  = _GNNNodeEncoder(3, GNN_HIDDEN_DIM, d, GNN_NUM_LAYERS)

        # Cross-graph attention: task <- resource (also returns attn weights)
        self.cross_attn = CrossGraphAttention(d, d, d)

        # FIX 1: Global state uses GNN mean + raw resource bottleneck (min+mean of res_x)
        # This preserves "which UAV is the weakest link" information lost by plain mean.
        # res_raw_dim = min(3) + mean(3) across Nr UAVs = 6
        res_raw_dim = 6
        global_dim  = d + d + res_raw_dim + fb  # 32 + 32 + 6 + 6 = 76
        self.noop_head  = nn.Linear(global_dim, 1)
        self.value_head = nn.Linear(global_dim, 1)

        # Per-node split head
        self.split_head = nn.Linear(d, 1)

        # FIX 2: Per-pair merge head adds link-quality signal:
        #   cat(emb_i, emb_j, link_quality) where link_quality = dot(attn_i, attn_j)
        #   High dot product -> both tasks attend to same UAV -> no cross-link needed
        #   Low dot product  -> tasks on different UAVs -> merge eliminates cross-link
        self.merge_head = nn.Sequential(
            nn.Linear(2 * d + 1, d),           # +1 for link_quality scalar
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        # Register pair index buffers so they move to the right device automatically
        pair_i = torch.tensor(self._PAIR_I, dtype=torch.long)
        pair_j = torch.tensor(self._PAIR_J, dtype=torch.long)
        self.register_buffer("pair_i", pair_i)
        self.register_buffer("pair_j", pair_j)

    # ------------------------------------------------------------------
    def _encode(self, obs: Dict[str, torch.Tensor]):
        task_node = self.task_gnn(obs["task_x"],  obs["task_edge"])   # (B, N, d)
        res_node  = self.res_gnn(obs["res_x"],    obs["res_edge"])    # (B, Nr, d)

        # 掩码零填充节点，阻止其污染嵌入
        task_real = (obs["task_x"].abs().sum(-1, keepdim=True) > 0)  # (B, N, 1)
        task_node = task_node * task_real                             # 清零 padding 行

        # Cross-graph attention: enrich each task node with UAV context
        # Also capture attn_weights (B, N, Nr) for merge link-quality scoring
        cross_ctx, attn_weights = self.cross_attn(task_node, res_node)
        task_enriched = task_node + cross_ctx                         # residual (B, N, d)

        # FIX 1: Replace plain res_node mean with explicit resource bottleneck.
        # min captures the weakest/most-depleted UAV; mean captures general fleet state.
        # This prevents the noop/value heads from being blind to individual UAV failures.
        res_x_raw = obs["res_x"]                                      # (B, Nr, 3)
        if res_x_raw.dim() == 2:
            res_x_raw = res_x_raw.unsqueeze(0)
        res_bottleneck = torch.cat([
            res_x_raw.min(dim=1).values,   # (B, 3) — weakest UAV per feature
            res_x_raw.mean(dim=1),         # (B, 3) — fleet average per feature
        ], dim=-1)                                                     # (B, 6)

        # masked mean（只对真实节点求均值）
        n_real = task_real.float().sum(dim=1).clamp(min=1.0)         # (B, 1)
        task_mean = task_enriched.sum(dim=1) / n_real                # (B, d)

        feedback = obs["feedback"]
        if feedback.dim() == 1:
            feedback = feedback.unsqueeze(0)
        global_state = torch.cat(
            [task_mean, res_node.mean(dim=1), res_bottleneck, feedback],
            dim=-1,
        )                                                              # (B, 76)

        return task_enriched, global_state, attn_weights

    # ------------------------------------------------------------------
    def forward(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        all_logits : (B, ACTION_DIM)  — raw (unmasked) action logits
        value      : (B, 1)
        """
        task_enriched, global_state, attn_weights = self._encode(obs)

        # NOOP logit — single scalar per batch item
        noop_logit   = self.noop_head(global_state)                   # (B, 1)

        # SPLIT logits — one per task node
        split_logits = self.split_head(task_enriched).squeeze(-1)    # (B, N)

        # MERGE logits — one per (i,j) pair
        emb_i = task_enriched[:, self.pair_i, :]                     # (B, C, d)
        emb_j = task_enriched[:, self.pair_j, :]                     # (B, C, d)

        # FIX 2: Link-quality signal for each merge candidate pair.
        # attn_weights[b, n, r] = how strongly task n attends to UAV r.
        # dot(attn_i, attn_j) is HIGH when both tasks attend to the SAME UAV
        #   → no cross-UAV transfer needed → merge less urgent
        # dot(attn_i, attn_j) is LOW when tasks attend to DIFFERENT UAVs
        #   → cross-link exists → merging eliminates that transfer → more urgent
        attn_i = attn_weights[:, self.pair_i, :]                     # (B, C, Nr)
        attn_j = attn_weights[:, self.pair_j, :]                     # (B, C, Nr)
        link_quality = (attn_i * attn_j).sum(dim=-1, keepdim=True)  # (B, C, 1)

        merge_logits = self.merge_head(
            torch.cat([emb_i, emb_j, link_quality], dim=-1)
        ).squeeze(-1)                                                  # (B, C)

        all_logits = torch.cat(
            [noop_logit, split_logits, merge_logits], dim=-1
        )                                                              # (B, ACTION_DIM)
        value = self.value_head(global_state)                         # (B, 1)
        return all_logits, value

    # ------------------------------------------------------------------
    def compute_action_mask(
        self, obs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Build (B, ACTION_DIM) boolean mask where True = action is valid.

        Split(i) valid  : node i is a real (non-padding) node AND
                          current graph size < MAX_TASK_NODES.
        Merge(i,j) valid: both i and j are real LEAF nodes
                          (out-degree == 0 in the padded graph).
        """
        task_x    = obs["task_x"]     # (B, N, 2)
        task_edge = obs["task_edge"]  # (B, 2, E)
        B, N, _   = task_x.shape
        device    = task_x.device

        # --- real nodes: non-zero feature rows ---
        is_real  = task_x.abs().sum(-1) > 0                          # (B, N)
        n_real   = is_real.sum(dim=1)                                 # (B,)

        # --- leaf nodes: real AND not a source of any real edge ---
        # Real edges exclude zero-padded (0,0) self-loops
        real_edge_flag = (task_edge[:, 0, :] != task_edge[:, 1, :])  # (B, E)
        src_idx   = task_edge[:, 0, :].clamp(0, N - 1)               # (B, E)
        src_hot   = F.one_hot(src_idx, num_classes=N).float()        # (B, E, N)
        src_hot   = src_hot * real_edge_flag.unsqueeze(-1).float()
        is_source = src_hot.sum(dim=1) > 0                           # (B, N)
        is_leaf   = is_real & ~is_source                              # (B, N)

        # --- NOOP: always valid ---
        noop_mask  = torch.ones(B, 1, dtype=torch.bool, device=device)

        # --- SPLIT: real node AND room to grow AND fleet has compute capacity ---
        # FIX 3: Reject split if all UAVs are depleted (eff_cpu_ratio == 0).
        # res_x[:, :, 0] = eff_cpu_ratio; max > 0 means at least one active UAV.
        res_x      = obs["res_x"]
        if res_x.dim() == 2:
            res_x = res_x.unsqueeze(0)
        fleet_has_cpu = (res_x[:, :, 0].max(dim=1).values > 0.0).unsqueeze(1)  # (B, 1)
        can_split  = (n_real < N).unsqueeze(1) & fleet_has_cpu       # (B, 1)
        split_mask = is_real & can_split                              # (B, N)

        # --- MERGE: both endpoints are real leaves ---
        merge_mask = is_leaf[:, self.pair_i] & is_leaf[:, self.pair_j]  # (B, C)

        return torch.cat([noop_mask, split_mask, merge_mask], dim=-1)  # (B, ACTION_DIM)

    # ------------------------------------------------------------------
    def get_action_and_value(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_logits, value = self.forward(obs)
        mask          = self.compute_action_mask(obs)
        masked_logits = all_logits.masked_fill(~mask, float("-inf"))
        dist          = torch.distributions.Categorical(logits=masked_logits)
        action        = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    # ------------------------------------------------------------------
    @staticmethod
    def decode(action_idx) -> Tuple[int, int, int]:
        """Convenience wrapper around configs.params.decode_node_action."""
        return decode_node_action(action_idx)
