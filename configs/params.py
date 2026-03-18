# configs/params.py
# Simulation and training hyperparameters

# --- UAV Fleet ---
NUM_UAVS = 6

# Each UAV is defined by (cpu_capacity, bandwidth_mbps, battery_pct)
# Heterogeneous by design: mix of high-compute, high-bandwidth, balanced nodes
UAV_PROFILES = [
    {"id": 0, "cpu": 8.0,  "bw": 100.0, "battery": 1.0},   # edge-server class
    {"id": 1, "cpu": 4.0,  "bw": 50.0,  "battery": 0.9},   # mid-tier
    {"id": 2, "cpu": 4.0,  "bw": 80.0,  "battery": 0.85},  # high-bandwidth
    {"id": 3, "cpu": 2.0,  "bw": 30.0,  "battery": 0.95},  # lightweight
    {"id": 4, "cpu": 2.0,  "bw": 20.0,  "battery": 0.7},   # low-resource
    {"id": 5, "cpu": 6.0,  "bw": 60.0,  "battery": 0.8},   # balanced
]

# --- Task Graph ---
MAX_TASK_NODES = 10          # maximum nodes in the DAG
MIN_TASK_NODES = 3
TASK_CPU_DEMAND_RANGE = (0.5, 4.0)   # CPU units demanded per task
TASK_DATA_SIZE_RANGE  = (0.5, 5.0)   # MB per task edge (UAV sensing payloads)

# --- Task Shaping ---
SHAPING_SPLIT_THRESHOLD   = 0.8   # latency ratio that triggers node splitting
SHAPING_MERGE_THRESHOLD   = 0.3   # success-rate below which nodes are merged
MAX_SHAPING_STEPS         = 3     # max topology edits per episode step

# --- Scheduling / Matching ---
DEADLINE_MS           = 1000.0  # end-to-end latency deadline (ms)
TRANSFER_LATENCY_BASE = 5.0     # base transfer ms per MB (static fallback)

# --- RL Training ---
MAX_EPISODE_STEPS   = 50
GAMMA               = 0.99
LR                  = 3e-4
CLIP_EPS            = 0.2      # PPO clip epsilon
ENTROPY_COEF        = 0.01
VALUE_LOSS_COEF     = 0.5
PPO_EPOCHS          = 4
BATCH_SIZE          = 64
ROLLOUT_STEPS       = 256
TOTAL_TIMESTEPS     = 200_000

# --- UAV Mobility (Random Waypoint Model) ---
AREA_SIZE_M      = 600.0    # simulation arena (m × m) — keeps most pairs within COMM_RANGE
UAV_SPEED_MPS    = 15.0     # cruise speed (m/s)
SIM_STEP_S       = 0.5      # wall-clock seconds per RL step
COMM_RANGE_M     = 400.0    # maximum communication range (m)
BW_HALF_DIST_M   = 150.0    # distance at which link BW halves (m)
# BW values are calibrated to TRANSFER_LATENCY_BASE (5 ms/MB → ~1600 Mbps effective).
# BW_MAX matches the static baseline; BW_MIN creates a ~8× latency penalty at range.
BW_MAX_MBPS      = 1600.0   # peak link bandwidth (Mbps) — consistent with static baseline
BW_MIN_MBPS      = 200.0    # floor bandwidth for in-range links (Mbps)

# --- Heterogeneous Energy Constraints ---
# Battery is tracked as a fraction in [0, 1]; cost per step is subtracted each RL step.
ENERGY_CPU_COEF   = 0.003   # battery fraction per CPU-unit consumed
ENERGY_TX_COEF    = 0.0002  # battery fraction per MB transmitted
BATTERY_LOW_RATIO = 0.20    # below this, UAV enters low-power mode (50 % CPU)
BATTERY_DEAD_RATIO= 0.0     # UAV is offline when battery reaches this value

# --- Node-level Action Space ---
# Layout: 0=noop | 1..MAX_TASK_NODES=split(node_i) | rest=merge(node_i, node_j)
N_SPLIT_ACTIONS = MAX_TASK_NODES
N_MERGE_ACTIONS = MAX_TASK_NODES * (MAX_TASK_NODES - 1) // 2   # C(N,2) = 45
ACTION_DIM      = 1 + N_SPLIT_ACTIONS + N_MERGE_ACTIONS         # = 56

# Pre-compute the flat pair list for O(1) action decoding
_PAIR_LIST: list = [
    (i, j)
    for i in range(MAX_TASK_NODES)
    for j in range(i + 1, MAX_TASK_NODES)
]


def decode_node_action(action_idx: int) -> tuple:
    """
    Decode a flat action index to (op_type, padded_node_i, padded_node_j).

    op_type : 0 = noop
              1 = split(node at padded position padded_node_i)
              2 = merge(node at padded_node_i, node at padded_node_j)

    'padded position' means the index into sorted(task_graph.nodes),
    consistent with how _get_obs() builds task_x.
    """
    action_idx = int(action_idx)
    if action_idx == 0:
        return 0, -1, -1
    elif action_idx <= MAX_TASK_NODES:
        return 1, action_idx - 1, -1
    else:
        i, j = _PAIR_LIST[action_idx - 1 - MAX_TASK_NODES]
        return 2, i, j


# --- GNN ---
GNN_HIDDEN_DIM  = 64
GNN_NUM_LAYERS  = 2
EMBED_DIM       = 32           # final per-node embedding size
