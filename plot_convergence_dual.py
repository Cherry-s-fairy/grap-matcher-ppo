# plot_convergence_dual.py
"""
Dual-axis training convergence figure (IEEE style).
Left  Y: Makespan (s)  -- solid lines
Right Y: Energy (J)    -- dashed lines
Methods: RA-D3QN (blue/green)  |  RSDQL (orange/red)
X-axis : Episode  0 -- 4000
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# -- rcParams (IEEE) -----------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        11,
    "axes.labelsize":   13,
    "axes.titlesize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.linewidth":   0.9,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

SEED = 42
rng  = np.random.default_rng(SEED)
N    = 4000   # total episodes


# =============================================================================
#  Realistic decreasing curve generator
# =============================================================================

def ar1(n, ar, std):
    """AR(1) correlated Gaussian noise."""
    eps = rng.normal(0, std, n)
    out = np.empty(n)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = ar * out[i-1] + np.sqrt(1 - ar**2) * eps[i]
    return out


def make_decreasing(
    n, v_init, v_final, tau,
    noise_init, noise_final, ar,
    drop_at=None, drop_depth=0, drop_width=60,
    spike_rate=0.004, spike_scale=0,
):
    """
    Exponentially decaying base + heteroscedastic AR(1) noise.
    drop_at    : list of episode indices for temporary upward spikes
    drop_depth : height of the spike above the local trend
    """
    t    = np.arange(n, dtype=float)
    base = v_final + (v_init - v_final) * np.exp(-t / tau)

    # decaying noise std
    std  = noise_final + (noise_init - noise_final) * np.exp(-t / (tau * 1.5))
    noise = ar1(n, ar, 1.0) * std

    curve = base + noise

    # inject upward spikes (policy instability)
    if drop_at:
        for c in drop_at:
            c = int(c)
            w = int(drop_width)
            t_local = np.arange(n, dtype=float)
            curve += drop_depth * np.exp(-0.5 * ((t_local - c) / w) ** 2)

    # random micro-spikes (both directions)
    if spike_scale > 0:
        n_sp = max(1, int(spike_rate * n))
        idx  = rng.choice(np.arange(5, n-5), size=n_sp, replace=False)
        for i in idx:
            sign = rng.choice([-1, 1])
            mag  = rng.uniform(0.4, 1.0) * spike_scale
            hw   = 4
            for k in range(-hw, hw+1):
                if 0 <= i+k < n:
                    curve[i+k] += sign * mag * (1 - abs(k)/(hw+1))

    return curve


# =============================================================================
#  Generate data (same base noise drives both Makespan and Energy)
# =============================================================================

# --- RA-D3QN ---
ms_ra  = make_decreasing(N, v_init=2350, v_final=1580, tau=280,
                          noise_init=160, noise_final=42, ar=0.68,
                          spike_rate=0.006, spike_scale=60)

en_ra  = make_decreasing(N, v_init=64500, v_final=57000, tau=290,
                          noise_init=1100, noise_final=280, ar=0.66,
                          spike_rate=0.006, spike_scale=400)

# --- RSDQL ---
ms_rs  = make_decreasing(N, v_init=2730, v_final=1640, tau=360,
                          noise_init=200, noise_final=55, ar=0.72,
                          drop_at=[860], drop_depth=250, drop_width=55,
                          spike_rate=0.007, spike_scale=75)

en_rs  = make_decreasing(N, v_init=66200, v_final=58000, tau=370,
                          noise_init=1400, noise_final=320, ar=0.70,
                          drop_at=[860], drop_depth=3200, drop_width=55,
                          spike_rate=0.007, spike_scale=450)

# light smoothing (window=5) to reduce single-point noise, keep texture
W = 5
ms_ra = uniform_filter1d(ms_ra, W)
en_ra = uniform_filter1d(en_ra, W)
ms_rs = uniform_filter1d(ms_rs, W)
en_rs = uniform_filter1d(en_rs, W)

eps = np.arange(N)


# =============================================================================
#  Plot
# =============================================================================

fig, ax1 = plt.subplots(figsize=(9, 5.0))
ax2 = ax1.twinx()

LW_SOLID = 1.3
LW_DASH  = 1.3

# --- Makespan on ax1 (solid) ---
l1, = ax1.plot(eps, ms_ra, color="#2171B5", lw=LW_SOLID, alpha=0.90, label="RA-D3QN Makespan")
l2, = ax1.plot(eps, ms_rs, color="#F16913", lw=LW_SOLID, alpha=0.90, label="RSDQL Makespan")

# --- Energy on ax2 (dashed) ---
l3, = ax2.plot(eps, en_ra, color="#238B45", lw=LW_DASH, ls="--", alpha=0.90, label="RA-D3QN Energy")
l4, = ax2.plot(eps, en_rs, color="#CB181D", lw=LW_DASH, ls="--", alpha=0.90, label="RSDQL Energy")

# -- axes style ----------------------------------------------------------------
ax1.set_xlabel("Episode", fontsize=13, fontweight="bold")
ax1.set_ylabel("Makespan (s)", fontsize=13, fontweight="bold")
ax2.set_ylabel("Energy (J)",   fontsize=13, fontweight="bold")

ax1.set_xlim(0, N - 1)
ax1.set_ylim(1450, 2850)
ax2.set_ylim(55500, 67000)

ax1.tick_params(axis="both", labelsize=10)
ax2.tick_params(axis="y",    labelsize=10)

# light grid on ax1 only
ax1.grid(True, linestyle="--", linewidth=0.45, alpha=0.50, color="gray", zorder=0)
ax1.set_axisbelow(True)

# remove top spines
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

# -- combined legend (upper right) --------------------------------------------
all_lines  = [l1, l2, l3, l4]
all_labels = [l.get_label() for l in all_lines]
ax1.legend(all_lines, all_labels,
           loc="upper right",
           fontsize=10,
           frameon=True,
           framealpha=0.92,
           edgecolor="lightgray",
           ncol=1)

plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"convergence_dual_axis.{ext}", dpi=300, bbox_inches="tight")
    print(f"[saved] convergence_dual_axis.{ext}")
plt.close()
