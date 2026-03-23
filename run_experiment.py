"""
run_experiment.py — One-command full experiment runner.

Usage
-----
    # Full experiment (5 seeds, 100 eval eps, generate all figures):
    python run_experiment.py

    # Quick smoke-test (1 seed, 10 eval eps, 50k timesteps):
    python run_experiment.py --quick

    # Only re-plot (no training/eval):
    python run_experiment.py --plot-only

    # Resume: skips any method/seed whose result file already exists
    python run_experiment.py --resume

    # Custom:
    python run_experiment.py --seeds 0 1 2 --n_eval 50 --device cpu
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

ALL_METHODS    = ["ns", "hs", "rnd", "rl_global", "rl_node"]
RL_METHODS     = {"rl_global", "rl_node"}
DEFAULT_SEEDS  = [0, 1, 2, 3, 4]
DEFAULT_N_EVAL = 100


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as 'Xh Ym Zs' or shorter forms."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _run(cmd: list[str]) -> int:
    """Run a subprocess, stream output, return exit code."""
    print(f"\n{'='*60}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _result_exists(method: str, seed: int) -> bool:
    return (RESULTS_DIR / f"{method}_seed{seed}.json").exists()


def _trainlog_exists(method: str, seed: int) -> bool:
    return (RESULTS_DIR / f"{method}_seed{seed}_trainlog.json").exists()


def run_all(
    seeds: list[int],
    n_eval: int,
    device: str,
    resume: bool,
    quick: bool,
    timesteps: int | None,
):
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    python = sys.executable
    t_start = time.time()
    skipped = 0
    failed  = []

    job_times: dict[str, float] = {}

    for method in ALL_METHODS:
        needs_train = method in RL_METHODS
        for seed in seeds:
            tag = f"{method}/seed{seed}"

            # Resume: skip if result already exists (and trainlog for RL methods)
            if resume:
                result_ok   = _result_exists(method, seed)
                trainlog_ok = (not needs_train) or _trainlog_exists(method, seed)
                if result_ok and trainlog_ok:
                    print(f"[skip] {tag} — result already exists")
                    skipped += 1
                    continue

            cmd = [
                python, "evaluate.py",
                "--method", method,
                "--seeds",  str(seed),
                "--n_eval", str(n_eval),
                "--device", device,
            ]
            if needs_train:
                cmd.append("--train")
            if quick and needs_train and timesteps is None:
                timesteps = 200_000   # 50k → 200k: enough for ~97 PPO updates to show convergence
            if timesteps is not None and needs_train:
                cmd += ["--timesteps", str(timesteps)]

            t_job = time.time()
            rc = _run(cmd)
            job_elapsed = time.time() - t_job
            job_times[tag] = job_elapsed
            status = "OK" if rc == 0 else f"FAILED(rc={rc})"
            print(f"[{status}] {tag} — {_fmt_elapsed(job_elapsed)}")
            if rc != 0:
                failed.append(tag)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Training & evaluation finished in {_fmt_elapsed(elapsed)}")
    if job_times:
        print(f"  Per-job breakdown:")
        for tag, t in job_times.items():
            print(f"    {tag:30s} {_fmt_elapsed(t)}")
    if skipped:
        print(f"  Skipped (--resume): {skipped} job(s)")
    if failed:
        print(f"  FAILED jobs: {failed}")
    print(f"{'='*60}\n")


def plot_all(seeds: list[int]):
    python = sys.executable
    seed_args = [str(s) for s in seeds]

    # All standard figures
    rc = _run([python, "plot_results.py", "--seeds"] + seed_args)
    if rc != 0:
        print("[WARN] plot_results.py exited with non-zero code")

    # Learning curves for all three metrics
    for metric in ["reward", "success_rate", "latency_ms"]:
        _run([python, "plot_results.py",
              "--fig", "8",
              "--metric", metric,
              "--seeds"] + seed_args)

    print("\n[plot] All figures saved to ./figures/")


def main():
    parser = argparse.ArgumentParser(description="Full experiment runner")
    parser.add_argument("--seeds",      type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--n_eval",     type=int, default=DEFAULT_N_EVAL,
                        help="Evaluation episodes per method/seed")
    parser.add_argument("--device",     default="auto",
                        help="cuda / cpu / auto")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip method/seed if result file already exists")
    parser.add_argument("--quick",      action="store_true",
                        help="Smoke-test mode: 1 seed, 10 eval eps, 50k timesteps")
    parser.add_argument("--timesteps",  type=int, default=None,
                        help="Override TOTAL_TIMESTEPS for RL training")
    parser.add_argument("--plot-only",  action="store_true",
                        help="Skip training/eval, only regenerate figures")
    parser.add_argument("--no-plot",    action="store_true",
                        help="Skip figure generation after training")
    args = parser.parse_args()

    if args.quick:
        args.seeds  = [0]
        args.n_eval = 10

    t_main_start = time.time()

    if not args.plot_only:
        run_all(
            seeds     = args.seeds,
            n_eval    = args.n_eval,
            device    = args.device,
            resume    = args.resume,
            quick     = args.quick,
            timesteps = args.timesteps,
        )

    if not args.no_plot:
        t_plot = time.time()
        plot_all(args.seeds)
        print(f"[plot] Plotting took {_fmt_elapsed(time.time() - t_plot)}")

    total = time.time() - t_main_start
    print(f"\n{'='*60}")
    print(f"  Total experiment time: {_fmt_elapsed(total)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
