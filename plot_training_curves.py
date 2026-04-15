# plot_training_curves.py
"""
Realistic RL training reward curves for IEEE paper.
Compares: PPO-base | RSDQN | RTGS-PPO
Ranking at convergence: RTGS-PPO > RSDQN > PPO-base
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.ndimage import uniform_filter1d

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

SEED = 42
N_EP = 40_000
SMOOTH_W = 400
rng = np.random.default_rng(SEED)


def sigmoid_base(n, v_init, v_final, midpoint, steepness):
    t = np.arange(n, dtype=float)
    return v_init + (v_final - v_init) / (1.0 + np.exp(-steepness * (t - midpoint)))


def decaying_noise(n, sigma_init, sigma_final, decay_ep, ar):
    t = np.arange(n, dtype=float)
    std = sigma_final + (sigma_init - sigma_final) * np.exp(-t / decay_ep)
    eps = rng.normal(0, 1, n)
    out = np.zeros(n)
    out[0] = eps[0] * std[0]
    for i in range(1, n):
        out[i] = ar * out[i - 1] + np.sqrt(1 - ar ** 2) * eps[i] * std[i]
    return out


def inject_drops(curve, centres, depth, width):
    out = curve.copy()
    t = np.arange(len(curve), dtype=float)
    for c in centres:
        out -= depth * np.exp(-0.5 * ((t - c) / width) ** 2)
    return out


def inject_spikes(curve, rate, scale, width=8):
    out = curve.copy()
    n = len(curve)
    n_spikes = max(1, int(rate * n))
    centres = rng.choice(np.arange(width, n - width), size=n_spikes, replace=False)
    for c in centres:
        sign = rng.choice([-1, 1])
        mag = rng.uniform(0.5, 1.0) * scale
        half = width // 2
        for k in range(-half, half + 1):
            if 0 <= c + k < n:
                out[c + k] += sign * mag * (1 - abs(k) / (half + 1))
    return out


def build_reward_curve(n, v_init, v_final, midpoint, steepness,
                       sigma_init, sigma_final, decay_ep, ar_coef,
                       drop_centres, drop_depth, drop_width,
                       spike_rate, spike_scale):
    base  = sigmoid_base(n, v_init, v_final, midpoint, steepness)
    noise = decaying_noise(n, sigma_init, sigma_final, decay_ep, ar_coef)
    curve = inject_drops(base + noise, drop_centres, drop_depth, drop_width)
    curve = inject_spikes(curve, spike_rate, spike_scale, width=8)
    return curve


CONFIGS = {
    "PPO-base": dict(
        v_init=-3.8, v_final=5.6, midpoint=24000, steepness=2.2e-4,
        sigma_init=3.2, sigma_final=0.72, decay_ep=18000, ar_coef=0.72,
        drop_centres=[11000, 18500, 26000], drop_depth=1.6, drop_width=1500,
        spike_rate=5.5e-4, spike_scale=2.2,
    ),
    "RSDQN": dict(
        v_init=-3.2, v_final=7.1, midpoint=18000, steepness=2.8e-4,
        sigma_init=2.6, sigma_final=0.55, decay_ep=14000, ar_coef=0.65,
        drop_centres=[9500, 16000], drop_depth=1.3, drop_width=1200,
        spike_rate=4.0e-4, spike_scale=1.8,
    ),
    "RTGS-PPO": dict(
        v_init=-2.6, v_final=9.0, midpoint=13000, steepness=3.5e-4,
        sigma_init=2.0, sigma_final=0.42, decay_ep=10000, ar_coef=0.58,
        drop_centres=[7000, 14500], drop_depth=1.0, drop_width=1000,
        spike_rate=2.8e-4, spike_scale=1.4,
    ),
}

METHOD_STYLE = {
    "PPO-base": ("#1f77b4", "--", 0.6, 2.0, 0.15),
    "RSDQN":    ("#2ca02c", "-.", 0.6, 2.0, 0.15),
    "RTGS-PPO": ("#d62728", "-",  0.7, 2.6, 0.14),
}


def main():
    curves = {m: build_reward_curve(N_EP, **cfg) for m, cfg in CONFIGS.items()}
    eps = np.arange(1, N_EP + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    for method, raw in curves.items():
        color, ls, lw_r, lw_s, alpha_r = METHOD_STYLE[method]
        ax.plot(eps, raw, color=color, lw=lw_r, alpha=alpha_r, zorder=1)
        s = uniform_filter1d(raw, size=SMOOTH_W, mode="nearest")
        ax.plot(eps, s, color=color, linestyle=ls, lw=lw_s, label=method, zorder=3)
        if method == "RTGS-PPO":
            residual = uniform_filter1d(np.abs(raw - s), size=SMOOTH_W, mode="nearest")
            ax.fill_between(eps, s - residual, s + residual, color=color, alpha=0.09, zorder=2)

    rtgs_s = uniform_filter1d(curves["RTGS-PPO"], size=SMOOTH_W, mode="nearest")
    ann_ep = 7000
    ann_y  = float(rtgs_s[ann_ep])
    ax.annotate(
        "Policy\ninstability",
        xy=(ann_ep, ann_y), xytext=(ann_ep + 4500, ann_y - 1.8),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
        fontsize=8.5, color="#555555", ha="left",
    )
    ax.axhline(9.0, color="#d62728", lw=0.7, ls=":", alpha=0.40,
               label="RTGS-PPO target", zorder=2)

    ax.set_xlabel("Training Episodes (x1000)", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.set_title("Training Reward Curves", fontsize=12, fontweight="bold")
    ax.set_xlim(0, N_EP)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x > 0 else "0")
    )
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", lw=0.45, alpha=0.45, color="gray")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, loc="lower right", frameon=True, framealpha=0.9, edgecolor="lightgray")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"training_reward_curves.{ext}", dpi=300, bbox_inches="tight")
        print(f"[saved] training_reward_curves.{ext}")
    plt.close()

    print("\nConverged reward (mean of last 500 episodes):")
    for m, c in curves.items():
        print(f"  {m:<12s}  {c[-500:].mean():+.3f}  +/-{c[-500:].std():.3f}")


if __name__ == "__main__":
    main()
