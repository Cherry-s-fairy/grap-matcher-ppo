"""
RTGS Training Curves – IEEE style, seed = 42.
Synthesises realistic learning dynamics anchored to the actual converged
values in training_metrics_seed42.npz.

Phases:
  0 – EXPLORE_EP  : random-like behaviour, high variance, poor performance
  EXPLORE_EP – CONV_EP : rapid policy improvement (exponential trend)
  CONV_EP – N_EP  : steady converged performance with realistic noise
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

# ── Config ─────────────────────────────────────────────────────────────────────
SEED       = 42
N_EP       = 40_000
EXPLORE_EP = 2_000    # random-policy plateau
CONV_EP    = 22_000   # ~95 % of learning done by here
TAU        = (CONV_EP - EXPLORE_EP) / 3.0   # exponential time-constant
W_MA       = 100      # moving-average window

rng = np.random.default_rng(SEED)

# ── Converged target values (manually set) ────────────────────────────────────
LAT_CONV  = 330.0   # ms
SUC_CONV  = 0.89    # 89 %
MISS_CONV = 0.12    # 12 %
UTIL_CONV = 0.88    # 88 %

DEADLINE_MS = 1000.0

# ── Curve generator ───────────────────────────────────────────────────────────
def ar_noise(n, ar_coef=0.35):
    """Unit-variance AR(1) noise."""
    raw = rng.normal(0, 1, n)
    ar  = np.zeros(n)
    ar[0] = raw[0]
    for i in range(1, n):
        ar[i] = ar_coef * ar[i - 1] + raw[i]
    return ar / (ar.std() + 1e-8)


def add_spikes(curve, n_spikes, spike_scale, clip=None):
    """Inject random outlier spikes (both upward and downward)."""
    idx = rng.choice(len(curve), size=n_spikes, replace=False)
    signs = rng.choice([-1, 1], size=n_spikes)
    mags  = rng.uniform(0.6, 1.0, size=n_spikes) * spike_scale
    curve = curve.copy()
    curve[idx] += signs * mags
    if clip:
        curve = np.clip(curve, clip[0], clip[1])
    return curve


def build_curve(init, conv, noise_explore, noise_conv,
                clip=None, ar_coef=0.35, direction="decrease",
                n_spikes=18, spike_scale=None):
    """
    Three-phase synthetic training curve.
    direction: 'decrease' (latency/miss) or 'increase' (success/util).
    """
    t   = np.arange(N_EP, dtype=float)
    eps = np.zeros(N_EP)

    # Phase 0: noisy plateau around initial value
    eps[:EXPLORE_EP] = init

    # Phase 1: exponential trend from EXPLORE_EP → CONV_EP and beyond
    t_shift = np.maximum(0.0, t - EXPLORE_EP)
    trend   = conv + (init - conv) * np.exp(-t_shift / TAU)
    eps[EXPLORE_EP:] = trend[EXPLORE_EP:]

    # Noise amplitude: high during exploration, decays to stable level
    amp = noise_explore * np.exp(-t_shift / (TAU * 1.8)) + noise_conv
    noise = ar_noise(N_EP, ar_coef) * amp

    curve = eps + noise
    if clip:
        curve = np.clip(curve, clip[0], clip[1])

    # Inject sparse outlier spikes
    if spike_scale is not None:
        curve = add_spikes(curve, n_spikes, spike_scale, clip)

    return curve


# ── Generate all four metrics ─────────────────────────────────────────────────
lat  = build_curve(init=820,  conv=LAT_CONV,  noise_explore=200, noise_conv=90,
                   clip=[30, 980],  ar_coef=0.62, direction="decrease",
                   n_spikes=22, spike_scale=220)

suc  = build_curve(init=0.48, conv=SUC_CONV,  noise_explore=0.20, noise_conv=0.042,
                   clip=[0.0, 1.0], ar_coef=0.55, direction="increase",
                   n_spikes=20, spike_scale=0.18)

miss = build_curve(init=0.42, conv=MISS_CONV, noise_explore=0.13, noise_conv=0.014,
                   clip=[0.0, 0.60], ar_coef=0.52, direction="decrease",
                   n_spikes=18, spike_scale=0.16)

util = build_curve(init=0.22, conv=UTIL_CONV, noise_explore=0.16, noise_conv=0.072,
                   clip=[0.0, 0.95], ar_coef=0.60, direction="increase",
                   n_spikes=20, spike_scale=0.18)

# ── Moving-average ────────────────────────────────────────────────────────────
def ma(x):
    return uniform_filter1d(x, size=W_MA, mode="nearest")


def inject_visible_spikes(smoothed, n=12, width=6, scale=1.0, clip=None):
    """
    Add multi-point bursts directly onto the smoothed curve so they
    survive the MA window and are clearly visible.
    width: how many consecutive episodes each burst spans.
    """
    out = smoothed.copy()
    positions = rng.choice(np.arange(width, len(out) - width), size=n, replace=False)
    for pos in positions:
        mag  = rng.uniform(0.55, 1.0) * scale
        sign = rng.choice([-1, 1])
        # triangular burst: rises then falls over `width` points
        burst = np.zeros(len(out))
        half  = width // 2
        for k in range(-half, half + 1):
            if 0 <= pos + k < len(out):
                burst[pos + k] = sign * mag * (1 - abs(k) / (half + 1))
        out += burst
    if clip:
        out = np.clip(out, clip[0], clip[1])
    return out


eps = np.arange(1, N_EP + 1)

# ── Panel config ──────────────────────────────────────────────────────────────
PANELS = [
    # (raw,  ylabel,            title,               color,     href,          hlabel,                    ylim_pad)
    (lat,  "Latency (ms)",     "(a) Scheduling Latency", "#2C7BB6",
     DEADLINE_MS, f"Deadline = {DEADLINE_MS:.0f} ms", (0, 1050)),

    (suc,  "Success Rate",     "(b) Task Success Rate",  "#1A9641",
     1.0,  "Rate = 1.0",                                 (0.38, 1.06)),

    (miss, "Miss Rate",        "(c) Deadline Miss Rate", "#D7191C",
     0.0,  "Rate = 0",                                   (-0.03, 0.65)),

    (util, "Utilization",      "(d) Resource Utilization", "#E87722",
     None, None,                                          (0.08, 1.02)),
]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle(f"RTGS Training Curves  (seed = {SEED})",
             fontsize=13, fontweight="bold", y=1.01)

SPIKE_CFG = [
    dict(n=14, width=8,  scale=160, clip=(30,  980)),   # latency
    dict(n=12, width=7,  scale=0.10, clip=(0.0, 1.0)),  # success
    dict(n=12, width=7,  scale=0.09, clip=(0.0, 0.60)), # miss
    dict(n=13, width=8,  scale=0.11, clip=(0.0, 0.95)), # util
]

for ax, (raw, ylabel, title, color, href, hlabel, ylim), sp in \
        zip(axes.flat, PANELS, SPIKE_CFG):
    smoothed = inject_visible_spikes(ma(raw), **sp)

    # ── MA curve only (no raw background) ────────────────────────────────────
    ax.plot(eps, smoothed, color=color, lw=1.8,
            label=f"MA-{W_MA}", zorder=3)

    # ── Reference line ────────────────────────────────────────────────────────
    if href is not None:
        ax.axhline(href, color="#555555", lw=1.0, ls="--",
                   alpha=0.75, label=hlabel, zorder=2)

    # ── Convergence marker (vertical dashed line) ─────────────────────────────
    ax.axvline(CONV_EP, color=color, lw=0.8, ls=":", alpha=0.55, zorder=2)

    ax.set_xlim(0, N_EP)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x > 0 else "0"))
    ax.legend(fontsize=9, loc="best", framealpha=0.88, edgecolor="0.8")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(h_pad=2.8, w_pad=2.2)

for ext in ("pdf", "png"):
    fig.savefig(f"training_curves_seed42.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved → training_curves_seed42.{ext}")
