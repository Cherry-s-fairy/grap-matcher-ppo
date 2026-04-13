"""
Ablation Study – IEEE style.
X-axis  : metric names (Success Rate / Miss Rate / Resource Util.)
Legend  : model names  (PPO-base, RTGS w/o Split, RTGS w/o Merge, RTGS-PPO)
Left  Y : percentage bars (3 metrics)
Right Y : Schedule Latency line, scaled to float ABOVE the bars
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines

# ── rcParams ───────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  10.5,
    "axes.linewidth":   0.8,
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 0,
    "ytick.major.size": 3.5,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ── Data ───────────────────────────────────────────────────────────────────────
MODEL_LABELS = [
    "PPO-base (No Shape)",
    "RTGS w/o Split",
    "RTGS w/o Merge",
    "RTGS-PPO (Ours)",
]
N = len(MODEL_LABELS)

SUCCESS  = np.array([77.8, 80.5, 82.1, 86.4])
MISS     = np.array([22.7, 19.6, 17.3, 14.2])
RESOURCE = np.array([74.9, 78.6, 81.2, 88.1])
LATENCY  = np.array([365.7, 345.2, 328.4, 305.2])

# bar groups (percentage metrics only)
BAR_GROUPS  = ["Success Rate (%)", "Deadline Miss Rate (%)", "Resource Util. (%)"]
BAR_DATA    = [SUCCESS, MISS, RESOURCE]
X_GROUPS    = np.array([0, 1, 2])          # group centres on x-axis

# ── Visual encoding (one colour/hatch per MODEL) ────────────────────────────────
COLORS  = ["#2166ac", "#74add1", "#f46d43", "#a50026"]
HATCHES = ["//",      "\\\\",    "xx",      ""      ]
ECOLORS = ["#0d4a82", "#4a8aab", "#c04020", "#780018"]
LWS     = [0.8,        0.8,       0.8,       1.6    ]

LINE_COLOR = "#6a0dad"     # deep purple for latency

BAR_W   = 0.18
offsets = np.array([-0.27, -0.09, 0.09, 0.27])   # 4 model offsets within each group

# ── Axes setup ─────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
ax2 = ax1.twinx()

# ── Draw bars ──────────────────────────────────────────────────────────────────
for g, (group_vals) in enumerate(BAR_DATA):
    for k in range(N):
        xi  = X_GROUPS[g] + offsets[k]
        v   = group_vals[k]
        best = (k == N - 1)
        ax1.bar(xi, v, BAR_W,
                color=COLORS[k], hatch=HATCHES[k],
                edgecolor=ECOLORS[k], linewidth=LWS[k],
                zorder=3)
        ax1.text(xi, v + 0.7, f"{v:.1f}",
                 ha="center", va="bottom",
                 fontsize=7.5,
                 fontweight="bold" if best else "normal",
                 color="#111111")

# ── Left y-axis ────────────────────────────────────────────────────────────────
LEFT_MAX = 105          # percentage axis ceiling
ax1.set_ylim(0, LEFT_MAX)
ax1.set_ylabel("Rate (%)", color="#333333", labelpad=6)
ax1.tick_params(axis="y", colors="#333333")

# ── Right y-axis: scale so latency line floats ABOVE bars ─────────────────────
# We want LATENCY.min() → left-y ≈ 91  and  LATENCY.max() → left-y ≈ 101
# Solve: (lat - r_lo)/(r_hi - r_lo) = left_y/LEFT_MAX
#   lat_min at left_y=91 → (lat_min - r_lo)/(r_hi - r_lo) = 91/105
#   lat_max at left_y=101 → (lat_max - r_lo)/(r_hi - r_lo) = 101/105
f_lo, f_hi = 91 / LEFT_MAX, 101 / LEFT_MAX
lat_lo, lat_hi = LATENCY.min(), LATENCY.max()
r_span = (lat_hi - lat_lo) / (f_hi - f_lo)   # total right-axis span
r_lo   = lat_lo - f_lo * r_span
r_hi   = r_lo + r_span

ax2.set_ylim(r_lo, r_hi)
# show only ticks in the actual latency range
tick_step = 20
t_start = int(np.ceil(lat_lo / tick_step)) * tick_step
t_end   = int(np.floor(lat_hi / tick_step)) * tick_step
ax2.set_yticks(np.arange(t_start, t_end + 1, tick_step))
ax2.set_ylabel("Schedule Latency (ms)", color=LINE_COLOR, labelpad=6)
ax2.tick_params(axis="y", colors=LINE_COLOR)
ax2.spines["right"].set_edgecolor(LINE_COLOR)
ax2.spines["right"].set_linewidth(0.9)

# ── Latency line: x-positions = model offsets, centred at mean group (x=1) ───
lat_x = 1.0 + offsets      # [0.73, 0.91, 1.09, 1.27]
ax2.plot(lat_x, LATENCY,
         color=LINE_COLOR, marker="D", ms=6.5,
         lw=2.2, ls="--", zorder=5)
for xi, v, k in zip(lat_x, LATENCY, range(N)):
    best = (k == N - 1)
    dx = 0.03 if k < N - 1 else -0.03
    ha = "left" if k < N - 1 else "right"
    ax2.text(xi + dx, v + 2, f"{v:.1f}",
             ha=ha, va="bottom",
             fontsize=8,
             fontweight="bold" if best else "normal",
             color=LINE_COLOR)

# ── X-axis ────────────────────────────────────────────────────────────────────
ax1.set_xticks(X_GROUPS)
ax1.set_xticklabels(BAR_GROUPS, fontsize=11)
ax1.set_xlim(-0.52, 2.52)

# ── Grid & spines ─────────────────────────────────────────────────────────────
ax1.yaxis.grid(True, linestyle="--", linewidth=0.5,
               color="#bbbbbb", alpha=0.45, zorder=0)
ax1.set_axisbelow(True)
for sp in ("top",): ax1.spines[sp].set_visible(False)
for sp in ("top",): ax2.spines[sp].set_visible(False)
ax1.spines["left"].set_linewidth(0.7)
ax1.spines["bottom"].set_linewidth(0.7)

# ── Legend ────────────────────────────────────────────────────────────────────
bar_handles = [
    mpatches.Patch(facecolor=COLORS[k], hatch=HATCHES[k],
                   edgecolor=ECOLORS[k], linewidth=0.8,
                   label=MODEL_LABELS[k])
    for k in range(N)
]
line_handle = mlines.Line2D([], [],
                             color=LINE_COLOR, marker="D", ms=6,
                             lw=2.2, ls="--", label="Schedule Latency (ms)")
fig.legend(handles=bar_handles + [line_handle],
           loc="lower center",
           ncol=5, framealpha=0.92, edgecolor="0.75",
           fontsize=9.5, handlelength=2.0, handleheight=1.3,
           columnspacing=1.0,
           bbox_to_anchor=(0.5, -0.08))

fig.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    fig.savefig(f"ablation_4metrics.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved → ablation_4metrics.{ext}")
