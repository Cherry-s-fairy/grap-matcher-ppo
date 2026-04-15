# plot_uav_scalability.py
"""
Figure: Performance vs. Number of UAVs  (2×2 subplots)
Methods: Greedy | GA | PPO-base | RSDQN | RTGS-PPO
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── data ───────────────────────────────────────────────────────────────────────
uavs = [8, 10, 12, 15]

data = {
    "Greedy":   {
        "success":  [60.2, 63.5, 65.1, 66.2],
        "latency":  [520,  480,  450,  425 ],
        "miss":     [39.5, 36.8, 34.2, 32.8],
        "util":     [58.1, 60.2, 61.5, 62.5],
    },
    "GA":       {
        "success":  [65.3, 67.8, 69.4, 70.5],
        "latency":  [540,  500,  470,  445 ],
        "miss":     [34.8, 32.1, 30.5, 29.4],
        "util":     [62.0, 64.3, 65.5, 66.8],
    },
    "PPO-base": {
        "success":  [70.1, 73.2, 75.6, 77.8],
        "latency":  [430,  400,  380,  365 ],
        "miss":     [29.6, 26.4, 24.1, 22.7],
        "util":     [68.5, 71.2, 73.6, 74.9],
    },
    "RSDQN":    {
        "success":  [74.5, 78.0, 80.3, 82.1],
        "latency":  [390,  360,  345,  338 ],
        "miss":     [25.3, 21.9, 19.8, 18.5],
        "util":     [72.8, 76.5, 79.0, 81.3],
    },
    "RTGS-PPO": {
        "success":  [78.8, 82.6, 84.9, 86.4],
        "latency":  [360,  330,  315,  305 ],
        "miss":     [21.2, 17.6, 15.3, 14.2],
        "util":     [76.9, 81.0, 84.2, 88.1],
    },
}

# ── style config ───────────────────────────────────────────────────────────────
METHOD_STYLE = {
    #  method       color        marker  lw    ms    zorder  alpha
    "Greedy":   ("#7f7f7f", "o",  1.8,  6,   2,    0.85),
    "GA":       ("#ff7f0e", "s",  1.8,  6,   2,    0.85),
    "PPO-base": ("#1f77b4", "^",  1.8,  7,   3,    0.90),
    "RSDQN":    ("#2ca02c", "D",  1.8,  6,   3,    0.90),
    "RTGS-PPO": ("#d62728", "*",  2.6, 10,   4,    1.00),   # ← emphasized
}

SUBPLOT_CFG = [
    ("success", "(a) Task Success Rate",    "Success Rate (%)"),
    ("latency", "(b) Scheduling Latency",   "Latency (ms)"),
    ("miss",    "(c) Deadline Miss Rate",   "Miss Rate (%)"),
    ("util",    "(d) Resource Utilization", "Utilization (%)"),
]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))

    # ── draw each subplot ──────────────────────────────────────────────────────
    for ax, (key, title, ylabel) in zip(axes.flat, SUBPLOT_CFG):
        for method, (color, marker, lw, ms, zo, alpha) in METHOD_STYLE.items():
            is_rtgs = (method == "RTGS-PPO")
            ax.plot(
                uavs, data[method][key],
                color=color,
                marker=marker,
                linewidth=lw,
                markersize=ms,
                zorder=zo,
                alpha=alpha,
                markerfacecolor=color if not is_rtgs else "white",
                markeredgewidth=2.0 if is_rtgs else 1.2,
                markeredgecolor=color,
                label=method,
            )

        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Number of UAVs", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(uavs)
        ax.tick_params(labelsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, color="gray")
        ax.spines[["top", "right"]].set_visible(False)

        # nudge y-limits for visual breathing room
        ymin, ymax = ax.get_ylim()
        margin = (ymax - ymin) * 0.08
        ax.set_ylim(ymin - margin, ymax + margin)

    # ── shared legend above subplots ───────────────────────────────────────────
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=5,
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
        bbox_to_anchor=(0.5, 1.01),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = "uav_scalability_4metrics.png"
    plt.savefig(out,                    dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"[saved] {out}  +  .pdf")
    plt.close()


if __name__ == "__main__":
    main()
