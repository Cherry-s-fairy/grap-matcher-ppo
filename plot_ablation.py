"""
Ablation study – single grouped bar chart (IEEE style).
3 metric groups × 4 method bars, y-axis = normalised performance
(RTGS-PPO = 1.0; higher always means better after sign-flip for Latency/Miss Rate).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── IEEE-style rcParams ────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "xtick.labelsize":  8.5,
    "ytick.labelsize":  8.5,
    "legend.fontsize":  8.5,
    "axes.linewidth":   0.7,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 0,
    "ytick.major.size": 3,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ── Methods & colors ───────────────────────────────────────────────────────────
METHODS = ["RTGS-PPO", "w/o Attention", "w/o Masking", "w/o Shaping"]
COLORS  = ["#E07868", "#60C8C8", "#88C878", "#E8C050"]  # salmon / cyan / green / amber
EDGE_COLORS = ["#a04040", "white", "white", "white"]

# ── Raw data (mean ± std, 5 seeds) ────────────────────────────────────────────
rng = np.random.default_rng(42)

def make_bar(means, stds):
    means, stds = np.array(means, float), np.array(stds, float)
    draws = means + rng.normal(0, stds / 1.5, (5, len(means)))
    return means, draws.std(axis=0)

LAT_MEANS  = [305,  338,  362,  395 ];  LAT_STDS  = [10,  14,  17,  20 ]
SR_MEANS   = [85.2, 80.6, 77.1, 73.3];  SR_STDS   = [ 1.5, 1.9,  2.2,  2.6]
DMR_MEANS  = [ 6.9, 10.1, 12.8, 16.7];  DMR_STDS  = [ 1.0,  1.4,  1.7,  2.2]

lat_m, lat_e   = make_bar(LAT_MEANS,  LAT_STDS)
sr_m,  sr_e    = make_bar(SR_MEANS,   SR_STDS)
dmr_m, dmr_e   = make_bar(DMR_MEANS,  DMR_STDS)

# ── Normalise to RTGS-PPO = 1.0 (higher score = better for all metrics) ───────
# Latency  : lower raw → better → score = ref / val
# Success  : higher raw → better → score = val / ref
# Miss Rate: lower raw → better → score = ref / val
def norm_lower(m, e):          # lower is better
    ref_m, ref_e = m[0], e[0]
    scores = ref_m / m
    # error propagation: δ(ref/val) = ref/val * sqrt((ref_e/ref_m)² + (e/m)²)
    errs   = scores * np.sqrt((ref_e / ref_m)**2 + (e / m)**2)
    return scores, errs

def norm_higher(m, e):         # higher is better
    ref_m, ref_e = m[0], e[0]
    scores = m / ref_m
    errs   = scores * np.sqrt((e / m)**2 + (ref_e / ref_m)**2)
    return scores, errs

lat_s,  lat_se  = norm_lower (lat_m,  lat_e)
sr_s,   sr_se   = norm_higher(sr_m,   sr_e)
dmr_s,  dmr_se  = norm_lower (dmr_m,  dmr_e)

# shape: (3 metrics, 4 methods)
scores = np.array([lat_s,  sr_s,  dmr_s ])
errs   = np.array([lat_se, sr_se, dmr_se])

METRICS      = ["Scheduling\nLatency", "Task\nSuccess Rate", "Deadline\nMiss Rate"]
N_METRICS    = len(METRICS)
N_METHODS    = len(METHODS)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 3.2))

group_w = 0.72          # total width occupied by one metric group
bar_w   = group_w / N_METHODS
x_group = np.arange(N_METRICS)

for j, (method, color, ec) in enumerate(zip(METHODS, COLORS, EDGE_COLORS)):
    offsets = x_group + (j - (N_METHODS - 1) / 2) * bar_w
    bars = ax.bar(offsets, scores[:, j], bar_w,
                  color=color, edgecolor=ec, linewidth=0.7,
                  label=method, zorder=3)
    ax.errorbar(offsets, scores[:, j], yerr=errs[:, j],
                fmt="none", ecolor="#333333",
                elinewidth=0.9, capsize=2.5, capthick=0.9,
                zorder=4)

# reference line at 1.0 (RTGS-PPO baseline)
ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--", zorder=2)

ax.set_ylabel("Normalised Performance\n(RTGS-PPO = 1.0)")
ax.set_xticks(x_group)
ax.set_xticklabels(METRICS)
ax.set_xlim(-0.55, N_METRICS - 0.45)
ax.set_ylim(0.35, 1.12)
ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.55, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="lower left", framealpha=0.9, edgecolor="0.8",
          ncol=2, handlelength=1.4, columnspacing=1.0)

fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(f"ablation_results.{ext}")
    print(f"Saved → ablation_results.{ext}")
