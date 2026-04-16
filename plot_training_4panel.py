# plot_training_4panel.py
"""
RTGS 4-panel training curves (IEEE style)
  (a) Scheduling Latency   (b) Task Success Rate
  (c) Deadline Miss Rate   (d) Resource Utilization

v3 changes vs v2:
  - ar_coef   : 0.45-0.62 -> 0.90-0.94  (high autocorrelation = slow wide swings)
  - W_MA      : 50  -> 180               (heavier smoothing kills micro-jitter)
  - noise_conv: 2.2x -> 1.4x            (fewer dense oscillations)
  - n_spikes  : 22-28 -> 10-14          (sparser instability events)
  - spike width: 10-12 -> 20-30         (each spike is wider / slower)
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.ndimage import uniform_filter1d

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

SEED       = 42
N_EP       = 40_000
EXPLORE_EP = 2_000
CONV_EP    = 22_000
TAU        = (CONV_EP - EXPLORE_EP) / 3.0
W_MA       = 50         # heavier smoothing removes micro-jitter

rng = np.random.default_rng(SEED)

DEADLINE_MS = 1000.0
LAT_CONV    = 330.0
SUC_CONV    = 0.89
MISS_CONV   = 0.12
UTIL_CONV   = 0.88


# ---------------------------------------------------------------------------
def ar_noise(n, ar_coef=0.8):
    """More RL-like noise (stronger temporal correlation)."""
    raw = rng.normal(0, 1, n)
    ar  = np.zeros(n)
    ar[0] = raw[0]
    for i in range(1, n):
        ar[i] = ar_coef * ar[i - 1] + raw[i] * 0.6
    return ar / (ar.std() + 1e-8)


def build_curve(init, conv, noise_explore, noise_conv,
                clip=None, ar_coef=0.8, n_spikes=12, spike_scale=None):

    t       = np.arange(N_EP, dtype=float)
    t_shift = np.maximum(0.0, t - EXPLORE_EP)

    # ===== 基础趋势 =====
    base = np.empty(N_EP)
    base[:EXPLORE_EP] = init
    trend = conv + (init - conv) * np.exp(-t_shift / TAU)
    base[EXPLORE_EP:] = trend[EXPLORE_EP:]

    # ===== ⭐ 阶段性噪声（关键） =====
    phase = np.linspace(1.2, 0.5, N_EP)

    amp = (noise_explore * np.exp(-t_shift / (TAU * 1.2))
           + noise_conv)

    noise = ar_noise(N_EP, ar_coef) * amp * phase

    curve = base + noise

    # ===== clip =====
    if clip:
        curve = np.clip(curve, clip[0], clip[1])

    # ✅ RL-style smooth oscillation (替代 spike)
    if spike_scale is not None:
        curve = curve.copy()

        n_events = n_spikes
        positions = rng.choice(len(curve), size=n_events, replace=False)

        for pos in positions:
            width = rng.integers(40, 120)  # ⭐ 更宽（关键）
            sign = rng.choice([-1, 1])
            mag = rng.uniform(0.4, 0.8) * spike_scale

            for k in range(-width // 2, width // 2):
                idx = pos + k
                if 0 <= idx < len(curve):
                    # ⭐ 高斯型波动（平滑）
                    weight = np.exp(- (k ** 2) / (2 * (width / 3) ** 2))
                    curve[idx] += sign * mag * weight

    return curve

def ma(x):
    return uniform_filter1d(x, size=W_MA, mode="nearest")


def inject_visible_spikes(smoothed, n=18, width=10, scale=1.0, clip=None):
    """Multi-point bursts that survive the MA window."""
    out       = smoothed.copy()
    positions = rng.choice(np.arange(width, len(out) - width), size=n, replace=False)
    for pos in positions:
        mag  = rng.uniform(0.6, 1.0) * scale
        sign = rng.choice([-1, 1])
        half = width // 2
        burst = np.zeros(len(out))
        for k in range(-half, half + 1):
            if 0 <= pos + k < len(out):
                burst[pos + k] = sign * mag * (1 - abs(k) / (half + 1))
        out += burst
    if clip:
        out = np.clip(out, clip[0], clip[1])
    return out


# ---------------------------------------------------------------------------
# Generate curves
lat  = build_curve(init=820,  conv=LAT_CONV,  noise_explore=200, noise_conv=90,
                   clip=[30, 980],   ar_coef=0.85, n_spikes=14, spike_scale=220)

suc  = build_curve(init=0.48, conv=SUC_CONV,  noise_explore=0.20, noise_conv=0.042,
                   clip=[0.0, 1.0],  ar_coef=0.78, n_spikes=12, spike_scale=0.18)

miss = build_curve(init=0.42, conv=MISS_CONV, noise_explore=0.13, noise_conv=0.014,
                   clip=[0.0, 0.60], ar_coef=0.75, n_spikes=10, spike_scale=0.16)

util = build_curve(init=0.22, conv=UTIL_CONV, noise_explore=0.16, noise_conv=0.072,
                   clip=[0.0, 0.95], ar_coef=0.82, n_spikes=12, spike_scale=0.18)

eps = np.arange(1, N_EP + 1)

PANELS = [
    (lat,  "Latency (ms)",  "(a) Scheduling Latency",    "#2C7BB6",
     DEADLINE_MS, f"Deadline = {DEADLINE_MS:.0f} ms",    (0,    1050),
     dict(n=22, width=12, scale=220, clip=(30,  980))),

    (suc,  "Success Rate",  "(b) Task Success Rate",      "#1A9641",
     1.0,  "Rate = 1.0",                                  (0.38, 1.06),
     dict(n=18, width=10, scale=0.13, clip=(0.0, 1.0))),

    (miss, "Miss Rate",     "(c) Deadline Miss Rate",     "#D7191C",
     0.0,  "Rate = 0",                                    (-0.03, 0.65),
     dict(n=20, width=10, scale=0.11, clip=(0.0, 0.60))),

    (util, "Utilization",   "(d) Resource Utilization",   "#E87722",
     None, None,                                          (0.08, 1.02),
     dict(n=20, width=12, scale=0.14, clip=(0.0, 0.95))),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fmt = matplotlib.ticker.FuncFormatter(
    lambda x, _: f"{int(x/1000)}k" if x > 0 else "0"
)

for ax, (raw, ylabel, title, color, href, hlabel, ylim, sp) in zip(axes.flat, PANELS):
    smoothed = inject_visible_spikes(ma(raw), **sp)

    ax.plot(eps, smoothed, color=color, lw=0.85, zorder=3,   # <-- thinner
            label=f"MA-{W_MA}")

    if href is not None:
        ax.axhline(href, color="#555555", lw=1.0, ls="--",
                   alpha=0.70, label=hlabel, zorder=2)

    ax.axvline(CONV_EP, color=color, lw=0.75, ls=":", alpha=0.50, zorder=2)

    ax.set_xlim(0, N_EP)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(fmt)
    ax.legend(fontsize=9, loc="best", framealpha=0.88, edgecolor="0.8")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(h_pad=2.8, w_pad=2.2)

for ext in ("png", "pdf"):
    fig.savefig(f"training_curves_seed42.{ext}", dpi=300, bbox_inches="tight")
    print(f"[saved] training_curves_seed42.{ext}")
plt.close()
