# plot_results.py
"""
Generate all paper figures from the JSON results produced by evaluate.py.

Usage
-----
    python plot_results.py                   # all figures
    python plot_results.py --fig 3 4         # specific figures

Output: figures/fig3_learning_curves.pdf, fig4_comparison_bar.pdf, …
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Visual identity for each method
STYLE = {
    "ns":        {"label": "No Shaping (NS)",        "color": "#888888", "ls": "--"},
    "hs":        {"label": "Heuristic Shaping (HS)", "color": "#F5A623", "ls": "--"},
    "rnd":       {"label": "Random (RND)",            "color": "#D0021B", "ls": ":"},
    "rl_global": {"label": "RL-Global (B4)",          "color": "#4A90E2", "ls": "-"},
    "rl_node":   {"label": "Ours (RL-Node)",          "color": "#E74C3C", "ls": "-"},
}

METRICS_LABEL = {
    "reward":          "Mean Episode Reward",
    "success_rate":    "Task Success Rate",
    "latency_ms":      "Schedule Latency (ms)",
    "deadline_miss":   "Deadline Miss Rate",
    "reschedule_count":"Reschedule Count",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(method: str, seeds: list[int]) -> dict[int, dict]:
    out = {}
    for s in seeds:
        p = RESULTS_DIR / f"{method}_seed{s}.json"
        if p.exists():
            with open(p) as f:
                out[s] = json.load(f)
    return out


def mean_std(results: dict[int, dict], metric: str):
    vals = [np.mean(r["episodes"][metric]) for r in results.values()]
    return float(np.mean(vals)), float(np.std(vals))


def available_methods(seeds: list[int]) -> list[str]:
    methods = []
    for m in STYLE:
        if any((RESULTS_DIR / f"{m}_seed{s}.json").exists() for s in seeds):
            methods.append(m)
    return methods


# ---------------------------------------------------------------------------
# Fig 4 — Comparison bar chart (primary result)
# ---------------------------------------------------------------------------

def fig4_comparison_bar(seeds: list[int]):
    methods = available_methods(seeds)
    if not methods:
        print("[fig4] No results found — run evaluate.py first.")
        return

    metrics = ["reward", "success_rate", "deadline_miss"]
    metric_labels = ["Mean Reward", "Success Rate", "Miss Rate"]
    n_metrics = len(metrics)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        means, stds, colors, labels = [], [], [], []
        for m in methods:
            res = load_results(m, seeds)
            if not res:
                continue
            mu, sd = mean_std(res, metric)
            means.append(mu)
            stds.append(sd)
            colors.append(STYLE[m]["color"])
            labels.append(STYLE[m]["label"])

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors, edgecolor="black", linewidth=0.8,
                      error_kw={"elinewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.4, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("Performance Comparison (mean ± std over seeds)", fontsize=13, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "fig4_comparison_bar.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"[fig4] Saved -> {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Fig 5 — Reward vs. n_task_nodes  (GNN ablation)
# ---------------------------------------------------------------------------

def fig5_nodes_bar(seeds: list[int]):
    target_methods = ["rl_node", "rl_global", "ns"]
    methods = [m for m in target_methods if any(
        (RESULTS_DIR / f"{m}_seed{s}.json").exists() for s in seeds)]
    if not methods:
        print("[fig5] No results found.")
        return

    # Bin episodes by num_task_nodes
    bins = [3, 5, 7, 10]

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_w = 0.25
    x = np.arange(len(bins))

    for k, method in enumerate(methods):
        bin_means = []
        for n in bins:
            vals = []
            for res in load_results(method, seeds).values():
                ep_nodes   = res["episodes"]["num_task_nodes"]
                ep_rewards = res["episodes"]["reward"]
                for ni, ri in zip(ep_nodes, ep_rewards):
                    if abs(ni - n) <= 1:   # ±1 tolerance (random N)
                        vals.append(ri)
            bin_means.append(np.mean(vals) if vals else 0.0)

        offset = (k - len(methods) / 2 + 0.5) * bar_w
        ax.bar(x + offset, bin_means, bar_w,
               label=STYLE[method]["label"],
               color=STYLE[method]["color"],
               edgecolor="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in bins])
    ax.set_xlabel("Number of Task Nodes", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Reward vs. Task Graph Size", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = FIGURES_DIR / "fig5_nodes_bar.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"[fig5] Saved -> {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Fig 6 — Latency CDF
# ---------------------------------------------------------------------------

def fig6_latency_cdf(seeds: list[int]):
    methods = available_methods(seeds)
    if not methods:
        print("[fig6] No results found.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    deadline = 1000.0

    for method in methods:
        all_latencies = []
        for res in load_results(method, seeds).values():
            all_latencies.extend(res["episodes"]["latency_ms"])
        if not all_latencies:
            continue
        sorted_l = np.sort(all_latencies)
        cdf = np.arange(1, len(sorted_l) + 1) / len(sorted_l)
        s = STYLE[method]
        ax.plot(sorted_l, cdf, label=s["label"], color=s["color"],
                linestyle=s["ls"], linewidth=2)

    ax.axvline(deadline, color="black", linestyle=":", linewidth=1.5,
               label=f"Deadline ({deadline:.0f} ms)")
    ax.set_xlabel("Schedule Latency (ms)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("Latency CDF Across Methods", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    out = FIGURES_DIR / "fig6_latency_cdf.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"[fig6] Saved -> {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Fig 7 — Summary table (printed + saved as CSV)
# ---------------------------------------------------------------------------

def fig7_summary_table(seeds: list[int]):
    methods = available_methods(seeds)
    if not methods:
        print("[fig7] No results found.")
        return

    rows = []
    header = ["Method", "Reward(+)", "Success(+)", "Latency(ms)(-)", "Miss%(-)", "Reschedule(-)"]
    for method in methods:
        res = load_results(method, seeds)
        if not res:
            continue
        r_mu, r_sd   = mean_std(res, "reward")
        s_mu, s_sd   = mean_std(res, "success_rate")
        l_mu, l_sd   = mean_std(res, "latency_ms")
        m_mu, _      = mean_std(res, "deadline_miss")
        k_mu, _      = mean_std(res, "reschedule_count")
        rows.append([
            STYLE[method]["label"],
            f"{r_mu:.2f}±{r_sd:.2f}",
            f"{s_mu:.3f}±{s_sd:.3f}",
            f"{l_mu:.1f}±{l_sd:.1f}",
            f"{m_mu*100:.1f}%",
            f"{k_mu:.2f}",
        ])

    # Print table
    col_w = [max(len(h), max(len(r[i]) for r in rows)) + 2
             for i, h in enumerate(header)]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt_row = lambda cells: "|" + "|".join(
        f" {c:<{w-2}} " for c, w in zip(cells, col_w)) + "|"

    print("\n" + sep)
    print(fmt_row(header))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)

    # Save CSV
    out_csv = FIGURES_DIR / "table_summary.csv"
    with open(out_csv, "w", encoding="utf-8-sig") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"[fig7] Table saved -> {out_csv}")


# ---------------------------------------------------------------------------
# Fig 8 — Learning curves (reward / latency / success vs training steps)
# ---------------------------------------------------------------------------

def _ema(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Exponential moving average smoothing."""
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _load_trainlog(method: str, seeds: list[int]) -> list[list[dict]]:
    """Load training logs for all seeds of a method. Returns list of log lists."""
    logs = []
    for s in seeds:
        p = RESULTS_DIR / f"{method}_seed{s}_trainlog.json"
        if p.exists():
            with open(p) as f:
                logs.append(json.load(f))
    return logs


def _baseline_level(metric: str, seeds: list[int]) -> dict[str, tuple[float, float]]:
    """
    Read static baseline results and return per-method (mean, std) for `metric`.
    std is across seeds (inter-seed variance of per-seed mean).
    metric: 'reward' | 'latency_ms' | 'success_rate'
    """
    levels = {}
    for m in ["ns", "hs", "rnd"]:
        res = load_results(m, seeds)
        if not res:
            continue
        vals = [np.mean(r["episodes"][metric]) for r in res.values()]
        levels[m] = (float(np.mean(vals)), float(np.std(vals)))
    return levels


def _interp_to_common_grid(logs: list[list[dict]], metric: str,
                            n_points: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate all seed logs onto a common step grid.
    Returns (x_grid, mean_curve, std_curve).
    """
    all_x = [np.array([e["step"] for e in log]) for log in logs]
    all_y = [np.array([e[metric] for e in log]) for log in logs]

    x_min = max(arr[0] for arr in all_x)
    x_max = min(arr[-1] for arr in all_x)
    if x_min >= x_max:
        x_max = max(arr[-1] for arr in all_x)
        x_min = 0

    grid = np.linspace(x_min, x_max, n_points)
    interp_curves = []
    for x, y in zip(all_x, all_y):
        interp_curves.append(np.interp(grid, x, y))

    mat = np.stack(interp_curves, axis=0)   # (n_seeds, n_points)
    return grid, mat.mean(axis=0), mat.std(axis=0)


def fig8_learning_curves(seeds: list[int], metric: str = "reward"):
    """
    Plot training learning curves for rl_global and rl_node.

    Parameters
    ----------
    metric : 'reward' | 'latency_ms' | 'success_rate'
    """
    # Map public metric names to trainlog JSON keys
    LOG_KEY = {
        "reward":      "reward",
        "latency_ms":  "latency",
        "success_rate":"success",
    }
    log_key = LOG_KEY.get(metric, metric)

    train_methods = ["rl_global", "rl_node"]
    found_any = False

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in train_methods:
        logs = _load_trainlog(method, seeds)
        if not logs:
            continue
        found_any = True

        grid, mean_curve, std_curve = _interp_to_common_grid(logs, log_key)
        smooth_mean = _ema(mean_curve, alpha=0.15)
        smooth_std  = _ema(std_curve,  alpha=0.15)

        s = STYLE[method]
        ax.plot(grid, smooth_mean, label=s["label"],
                color=s["color"], linewidth=2, linestyle=s["ls"])
        ax.fill_between(grid,
                        smooth_mean - smooth_std,
                        smooth_mean + smooth_std,
                        color=s["color"], alpha=0.15)

    if not found_any:
        print(f"[fig8] No training logs found for metric='{metric}'. "
              "Run evaluate.py --method rl_global / rl_node first.")
        plt.close()
        return

    # Horizontal reference bands for static baselines (mean ± std across seeds)
    # These baselines have no learning — their reward is a stationary distribution,
    # so we show them as flat bands rather than training curves.
    baseline_levels = _baseline_level(metric, seeds)
    ref_style = {
        "ns":  ("No Shaping (NS)",  "#888888", (4, 2)),
        "hs":  ("Heuristic (HS)",   "#F5A623", (4, 2)),
        "rnd": ("Random (RND)",     "#D0021B", (2, 2)),
    }
    x_lo, x_hi = ax.get_xlim()
    # fallback if axes not yet scaled
    if x_lo == 0 and x_hi == 1:
        x_lo, x_hi = 0, 1_000_000
    for m, (mu, sd) in baseline_levels.items():
        label, color, dashes = ref_style[m]
        ax.axhline(mu, color=color, linewidth=1.5,
                   linestyle=(0, dashes), label=f"{label} mean")
        ax.axhspan(mu - sd, mu + sd, color=color, alpha=0.08)

    ylabel = METRICS_LABEL.get(metric, metric)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Learning Curves — {ylabel}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k")
    )

    plt.tight_layout()
    out = FIGURES_DIR / f"fig8_learning_curves_{metric}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"[fig8] Saved -> {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Fig 3 — Reward distribution box plot (per-seed robustness)
# ---------------------------------------------------------------------------

def fig3_reward_boxplot(seeds: list[int]):
    methods = available_methods(seeds)
    if not methods:
        print("[fig3] No results found.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(methods))

    all_data   = []
    tick_labels = []
    colors     = []

    for method in methods:
        vals = []
        for res in load_results(method, seeds).values():
            vals.extend(res["episodes"]["reward"])
        all_data.append(vals)
        tick_labels.append(STYLE[method]["label"])
        colors.append(STYLE[method]["color"])

    bp = ax.boxplot(all_data, positions=positions, patch_artist=True,
                    widths=0.5,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Reward Distribution per Method", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = FIGURES_DIR / "fig3_reward_boxplot.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"[fig3] Saved -> {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig",   type=int, nargs="+",
                        choices=[3, 4, 5, 6, 7, 8],
                        help="Figure numbers to generate (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)),
                        help="Seeds to aggregate (default: 0 1 2 3 4)")
    parser.add_argument("--metric", default="reward",
                        choices=["reward", "latency_ms", "success_rate"],
                        help="Metric for fig8 learning curves (default: reward)")
    args = parser.parse_args()

    figs = args.fig or [3, 4, 5, 6, 7, 8]
    seeds = args.seeds

    dispatch = {
        3: fig3_reward_boxplot,
        4: fig4_comparison_bar,
        5: fig5_nodes_bar,
        6: fig6_latency_cdf,
        7: fig7_summary_table,
        8: lambda s: fig8_learning_curves(s, metric=args.metric),
    }

    for f in figs:
        dispatch[f](seeds)

    print("\n[plot] All figures written to ./figures/")


if __name__ == "__main__":
    main()
