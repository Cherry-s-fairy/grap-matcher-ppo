"""
Plot 1 – 2×2 bar chart: converged performance of 3 methods, error bars = std over 5 seeds.
  (a) Scheduling Latency      (b) Task Success Rate
  (c) Reschedule Count        (d) Resource Utilization

Plot 2 – standalone reward convergence line plot (3 algorithms).
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter1d

# ── Style ──────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "legend.fontsize":  9.5,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

SEEDS         = 5
EPISODE_STEPS = 50
TAIL          = 200     # last N checkpoints treated as converged performance
SMOOTH        = 50      # for reward convergence line plot only
ALPHA         = 0.18

CONFIGS = {
    "rl_node":   {"color": "#2166ac", "label": "PPO-base",  "zorder": 3},
    "rsdqn":     {"color": "#4dac26", "label": "RSDQN",       "zorder": 4},
    "rl_global": {"color": "#d6604d", "label": "RTGS-PPO (Ours)", "zorder": 5},
}

# metric definitions: (field, y_label, title, as_pct, y_margin_frac)
METRICS = [
    ("latency",              "Scheduling Latency (ms)", "(a) Scheduling Latency",   False, 0.08),
    ("success",              "Task Success Rate (%)",   "(b) Task Success Rate",    True,  0.08),
    ("reschedule_count",     "Reschedule Count",        "(c) Reschedule Count",     False, 0.15),
    ("resource_utilization", "Resource Utilization (%)", "(d) Resource Utilization", True, 0.10),
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_metric(algo: str, field: str):
    """Return (episodes_array, values_matrix [seeds × T])."""
    all_vals = []
    for seed in range(SEEDS):
        with open(f"results_old/{algo}_seed{seed}_trainlog.json") as f:
            records = json.load(f)
        all_vals.append([r[field] for r in records])
    steps    = np.array([r["step"] for r in records])
    episodes = steps / EPISODE_STEPS
    return episodes, np.array(all_vals)


def converged_stats(algo: str, field: str, as_pct: bool):
    """Mean and std of per-seed converged values (tail average)."""
    seed_means = []
    for seed in range(SEEDS):
        with open(f"results_old/{algo}_seed{seed}_trainlog.json") as f:
            records = json.load(f)
        vals = np.array([r[field] for r in records[-TAIL:]])
        seed_means.append(vals.mean())
    arr = np.array(seed_means)
    if as_pct:
        arr *= 100
    return arr.mean(), arr.std()


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    return uniform_filter1d(x, size=w, mode="nearest")


# ── Figure 1 – 2×2 bar chart ───────────────────────────────────────────────────
algos  = list(CONFIGS.keys())
labels = [cfg["label"] for cfg in CONFIGS.values()]
colors = [cfg["color"] for cfg in CONFIGS.values()]

n      = len(algos)
x_pos  = np.arange(n)
width  = 0.52

fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.4))
axes_flat = axes.flatten()

for ax, (field, ylabel, title, as_pct, margin) in zip(axes_flat, METRICS):
    means, stds = [], []
    for algo in algos:
        m, s = converged_stats(algo, field, as_pct)
        means.append(m)
        stds.append(s)

    bars = ax.bar(x_pos, means, width,
                  color=colors, zorder=3,
                  edgecolor="white", linewidth=0.6)

    ax.errorbar(x_pos, means, yerr=stds,
                fmt="none", ecolor="black",
                elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)

    # y-axis range: zoom in to show differences clearly
    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    span = hi - lo
    ax.set_ylim(lo - margin * span, hi + margin * span)

    ax.set_title(title, pad=4)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.55, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)   # hide x tick marks

fig.tight_layout(h_pad=2.5, w_pad=2.0)

for ext in ("pdf", "png"):
    fig.savefig(f"convergence_metrics.{ext}")
    print(f"Saved → convergence_metrics.{ext}")

plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – standalone reward convergence (3 algorithms)
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(5.5, 3.6))

for algo, cfg in CONFIGS.items():
    episodes, vals = load_metric(algo, "reward")

    mean_s = smooth(vals.mean(axis=0), SMOOTH)
    std_s  = smooth(vals.std(axis=0),  SMOOTH)

    x = episodes / 1_000

    ax2.plot(x, mean_s,
             color=cfg["color"], lw=1.6,
             label=cfg["label"], zorder=cfg["zorder"])
    ax2.fill_between(x,
                     mean_s - std_s,
                     mean_s + std_s,
                     color=cfg["color"], alpha=ALPHA,
                     linewidth=0, zorder=cfg["zorder"] - 1)

ax2.set_xlabel("Training Episodes (×10³)")
ax2.set_ylabel("Average Episode Reward")
ax2.set_xlim(x[0], x[-1])
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
ax2.legend(loc="lower right", framealpha=0.9, edgecolor="0.8")
ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=1)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig2.tight_layout()

for ext in ("pdf", "png"):
    fig2.savefig(f"convergence_reward.{ext}")
    print(f"Saved → convergence_reward.{ext}")

plt.show()
