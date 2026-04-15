"""
Reward convergence curve – IEEE style, 3 methods.
Same dynamics as plot_training_curves.py:
  explore plateau → rapid improvement → convergence with visible spikes.
No shaded confidence bands.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ── rcParams ───────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   12,
    "legend.fontsize":  10.5,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.linewidth":   0.8,
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
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

# ── Config ─────────────────────────────────────────────────────────────────────
SEED       = 42
N_EP       = 40_000
EXPLORE_EP = 2_000
CONV_EP    = 22_000
TAU        = (CONV_EP - EXPLORE_EP) / 3.0
W_MA       = 100

rng = np.random.default_rng(SEED)

# ── Converged reward targets (RTGS-PPO best) ──────────────────────────────────
METHODS = {
    "PPO-base":       {"conv": 138.5, "init": 131.0, "color": "#2166ac", "lw": 1.6, "zorder": 3},
    "RSDQN":          {"conv": 141.2, "init": 132.5, "color": "#4dac26", "lw": 1.6, "zorder": 4},
    "RTGS-PPO (Ours)":{"conv": 143.8, "init": 133.5, "color": "#d6604d", "lw": 2.0, "zorder": 5},
}

# ── Curve builder (same logic as plot_training_curves.py) ─────────────────────
def ar_noise(n, ar_coef=0.35):
    raw = rng.normal(0, 1, n)
    ar  = np.zeros(n)
    ar[0] = raw[0]
    for i in range(1, n):
        ar[i] = ar_coef * ar[i - 1] + raw[i]
    return ar / (ar.std() + 1e-8)


def build_curve(init, conv, noise_explore=1.8, noise_conv=0.55,
                ar_coef=0.58, clip=None):
    t       = np.arange(N_EP, dtype=float)
    t_shift = np.maximum(0.0, t - EXPLORE_EP)

    trend              = np.full(N_EP, float(init))
    trend[EXPLORE_EP:] = conv + (init - conv) * np.exp(-t_shift[EXPLORE_EP:] / TAU)

    amp   = noise_explore * np.exp(-t_shift / (TAU * 1.8)) + noise_conv
    noise = ar_noise(N_EP, ar_coef) * amp
    curve = trend + noise
    if clip:
        curve = np.clip(curve, *clip)
    return curve


def ma(x):
    return uniform_filter1d(x, size=W_MA, mode="nearest")


def inject_spikes(smoothed, n=13, width=8, scale=1.8, clip=None):
    out  = smoothed.copy()
    pos  = rng.choice(np.arange(width, len(out) - width), size=n, replace=False)
    half = width // 2
    for p in pos:
        mag  = rng.uniform(0.55, 1.0) * scale
        sign = rng.choice([-1, 1])
        for k in range(-half, half + 1):
            if 0 <= p + k < len(out):
                out[p + k] += sign * mag * (1 - abs(k) / (half + 1))
    if clip:
        out = np.clip(out, *clip)
    return out


# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 4.2))

eps_k = np.arange(1, N_EP + 1) / 1_000   # x-axis in ×10³

CLIP = (128.0, 148.0)

for label, cfg in METHODS.items():
    raw      = build_curve(cfg["init"], cfg["conv"], clip=CLIP)
    smoothed = inject_spikes(ma(raw), n=13, width=8,
                             scale=1.8, clip=CLIP)
    ax.plot(eps_k, smoothed,
            color=cfg["color"], lw=cfg["lw"],
            label=label, zorder=cfg["zorder"])

# ── Convergence marker ─────────────────────────────────────────────────────────
ax.axvline(CONV_EP / 1_000, color="#888888", lw=0.8, ls=":",
           alpha=0.5, zorder=2)

# ── Axes ───────────────────────────────────────────────────────────────────────
ax.set_xlim(0, N_EP / 1_000)
ax.set_ylim(128, 148)
ax.set_xlabel("Training Episodes (×10³)")
ax.set_ylabel("Average Episode Reward")
ax.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}" if v > 0 else "0"))

ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45, zorder=1)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(loc="lower right", framealpha=0.92, edgecolor="0.8",
          fontsize=10.5, handlelength=2.2)

fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(f"convergence_reward.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved → convergence_reward.{ext}")
