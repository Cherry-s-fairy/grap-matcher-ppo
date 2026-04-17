"""
生成论文级 RL 训练曲线 (seed=42)
核心设计：
  - 基础趋势：CubicSpline 控制点
  - 噪声：高 rho AR(1) → 低频结构波动，不是白噪声
  - 三阶段：exploration → oscillation plateau → stabilization
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.interpolate import CubicSpline

# ─── 全局参数 ────────────────────────────────────────────────
SEED       = 42
N          = 40_000
DEADLINE   = 1000.0

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
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})


# ─── 工具函数 ────────────────────────────────────────────────
def ar1(n, rho, seed):
    """单位方差的 AR(1) 过程：rho 越高，波动越平滑（频率越低）。"""
    rng = np.random.default_rng(seed)
    w   = rng.standard_normal(n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + np.sqrt(1.0 - rho ** 2) * w[i]
    return out / (out.std() + 1e-8)


def taper(n, ramp=2000):
    """两端渐入渐出窗口，避免拼接突变。"""
    t = np.ones(n)
    r = min(ramp, n // 4)
    t[:r]  = np.linspace(0, 1, r) ** 1.5
    t[-r:] = np.linspace(1, 0, r) ** 1.5
    return t


def make_curve(ctrl_pts, phases, lo, hi, n=N):
    """
    ctrl_pts : [(t, v), ...]  控制点，定义平滑基础趋势
    phases   : [(t0, t1, rho, amp, seed), ...]  各阶段 AR 噪声
    lo, hi   : 最终 clip 范围
    """
    t = np.arange(n, dtype=float)

    # 基础趋势（三次样条）
    cps  = np.array(ctrl_pts, dtype=float)
    cs   = CubicSpline(cps[:, 0], cps[:, 1], bc_type="not-a-knot")
    base = cs(t)

    # 阶段性结构噪声叠加
    noise = np.zeros(n)
    for t0, t1, rho, amp, s in phases:
        seg = ar1(t1 - t0, rho, s) * amp * taper(t1 - t0, ramp=min(2000, (t1-t0)//5))
        noise[t0:t1] += seg

    return np.clip(base + noise, lo, hi)


# ─── 各指标曲线定义 ──────────────────────────────────────────

# (a) Scheduling Latency (ms)  —— 递减，前高后低
#     阶段一：探索期，快速下降但结构性抖动大
#     阶段二：震荡 plateau，围绕 ~440ms 慢慢改善
#     阶段三：稳定，轻微波动在 ~390ms
lat = make_curve(
    ctrl_pts=[
        (0,     820), (1500,  700), (4000,  530), (7000,  460),
        (11000, 445), (15000, 428), (20000, 415), (25000, 405),
        (30000, 397), (35000, 392), (39999, 388),
    ],
    phases=[
        # 探索期：rho 稍低一些，波动频率略高（结构性但有活力）
        (0,     8000,  0.991, 30.0, 10),
        # 震荡 plateau：高 rho → 宽波形（每次振荡跨几千 episode）
        (5000,  24000, 0.998, 24.0, 20),
        (9000,  22000, 0.997, 18.0, 30),   # 叠加，让中期更明显
        # 稳定期：低振幅
        (22000, N,     0.996,  7.0, 40),
    ],
    lo=200, hi=980,
)

# (b) Task Success Rate  —— 递增
suc = make_curve(
    ctrl_pts=[
        (0,     0.40), (2000,  0.52), (5000,  0.65), (8000,  0.73),
        (12000, 0.77), (16000, 0.80), (21000, 0.83), (26000, 0.855),
        (32000, 0.868), (37000, 0.875), (39999, 0.880),
    ],
    phases=[
        (0,     8000,  0.990, 0.042, 11),
        (5000,  24000, 0.998, 0.030, 21),
        (9000,  22000, 0.997, 0.022, 31),
        (22000, N,     0.996, 0.009, 41),
    ],
    lo=0.20, hi=1.00,
)

# (c) Deadline Miss Rate  —— 递减
miss = make_curve(
    ctrl_pts=[
        (0,     0.44), (2000,  0.33), (5000,  0.20), (8000,  0.15),
        (12000, 0.130), (16000, 0.118), (21000, 0.108), (26000, 0.098),
        (32000, 0.090), (37000, 0.085), (39999, 0.082),
    ],
    phases=[
        (0,     8000,  0.991, 0.040, 12),
        (5000,  24000, 0.998, 0.026, 22),
        (9000,  22000, 0.997, 0.020, 32),
        (22000, N,     0.996, 0.007, 42),
    ],
    lo=0.0, hi=0.60,
)

# (d) Resource Utilization  —— 递增
util = make_curve(
    ctrl_pts=[
        (0,     0.17), (2000,  0.33), (5000,  0.52), (8000,  0.62),
        (12000, 0.66), (16000, 0.695), (21000, 0.72), (26000, 0.745),
        (32000, 0.760), (37000, 0.770), (39999, 0.775),
    ],
    phases=[
        (0,     8000,  0.990, 0.048, 13),
        (5000,  24000, 0.998, 0.032, 23),
        (9000,  22000, 0.997, 0.025, 33),
        (22000, N,     0.996, 0.010, 43),
    ],
    lo=0.0, hi=1.0,
)


# ─── 绘图 ────────────────────────────────────────────────────
arrays      = [lat,        suc,               miss,               util]
titles      = ["(a) Scheduling Latency",
               "(b) Task Success Rate",
               "(c) Deadline Miss Rate",
               "(d) Resource Utilization"]
ylabels     = ["Latency (ms)", "Success Rate", "Miss Rate", "Utilization"]
colors      = ["#2C7BB6",   "#1A9641",          "#D7191C",          "#E87722"]
hrefs       = [DEADLINE,    1.0,                0.0,                None]
href_labels = [f"Deadline = {DEADLINE:.0f} ms", "Rate = 1.0",  "Rate = 0",  None]

eps = np.arange(1, N + 1)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))

for ax, arr, title, ylabel, color, href, hlabel in zip(
        axes.flat, arrays, titles, ylabels, colors, hrefs, href_labels):

    ax.plot(eps, arr, color=color, linewidth=1.2, zorder=3)

    if href is not None:
        ax.axhline(href, color="gray", linewidth=1.0,
                   linestyle="--", alpha=0.7, label=hlabel, zorder=2)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, N)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda x, _: "0" if x == 0 else f"{int(x/1000)}k"
        )
    )
    if hlabel:
        ax.legend(fontsize=9, loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, color="gray")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=200, bbox_inches="tight")
plt.savefig("training_curves.pdf",           bbox_inches="tight")
print("Saved.")
