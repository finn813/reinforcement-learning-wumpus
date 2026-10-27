"""Summarise runs/*.json into results_table.md and comparison figures in plots/."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
PERIOD_COLOR = {100: "#2a78d6", 500: "#eb6834", 1000: "#1baf7a"}   # validated slots 1-3
WIN, DEATH, TIMEOUT = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
KB_TEST_WIN = 0.459   # KB agent on the same 1,000 test maps (seed 12345)


def load():
    runs = defaultdict(list)
    for f in sorted((HERE / "runs").glob("n*_seed*.json")):
        r = json.loads(f.read_text())
        if r["num_training_maps"] != "100" or r.get("goal_reward", 100) != 100:
            # map-count and reward runs are summarised by maps_analysis.py
            continue
        runs[(r["episodes"], r["period"], r["num_training_maps"])].append(r)
    return dict(sorted(runs.items()))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


def cell(values, fmt):
    """mean (seed values) for a short list of per-seed values."""
    m = np.mean(values)
    return fmt(m) + (" (" + " / ".join(fmt(v) for v in values) + ")" if len(values) > 1 else "")


def table(runs):
    pct = lambda x: f"{x:.1%}"  # noqa: E731
    num = lambda x: f"{x:.1f}"  # noqa: E731
    rows = ["| n_episodes | target_update_period | target syncs | seeds | Win, training maps (100) | Win, test maps (1,000) "
            "| Death, test | Timeout, test | Avg. reward, test | Win, benchmark test maps (100) | Minutes |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (n, p, maps), rs in runs.items():
        f = [r["final"]["test1000"] for r in rs]
        rows.append(" | ".join([
            f"| {n:,}", str(p), f"~{int(np.mean([r['target_syncs'] for r in rs])):,}", str(len(rs)),
            cell([x["training"]["win_rate"] for x in f], pct) if "training" in f[0] else "–",
            cell([x["test"]["win_rate"] for x in f], pct),
            cell([x["test"]["death_rate"] for x in f], pct),
            cell([x["test"]["timeout_rate"] for x in f], pct),
            cell([x["test"]["average_reward"] for x in f], num),
            cell([r["final"]["test100"]["test"]["win_rate"] for r in rs], pct),
            num(np.mean([r["seconds"] for r in rs]) / 60)]) + " |")
    return "\n".join(rows)


def curves_figure(runs, path):
    lengths = sorted({n for n, _, _ in runs})
    fig, axs = plt.subplots(2, len(lengths), figsize=(6.2 * len(lengths), 8.4), facecolor=SURFACE,
                            sharey=True, squeeze=False)
    for col, n in enumerate(lengths):
        for row, (key, label) in enumerate((("train", "training maps (100)"), ("test", "unseen test maps (1,000)"))):
            ax = axs[row][col]
            style(ax)
            ends = []
            for (n2, p, maps), rs in runs.items():
                if n2 != n:
                    continue
                eps = [pt["episode"] for pt in rs[0]["curve"]]
                win = np.array([[pt[key]["win_rate"] for pt in r["curve"]] for r in rs])
                color = PERIOD_COLOR.get(p, INK2)
                ax.plot(eps, win.mean(0), color=color, linewidth=2, marker="o", markersize=4,
                        label=f"p = {p}  (n={len(rs)} seeds)")
                if len(rs) > 1:
                    ax.fill_between(eps, win.min(0), win.max(0), color=color, alpha=0.13, linewidth=0)
                ends.append([win.mean(0)[-1], p, eps[-1]])
            ends.sort()
            for i in range(1, len(ends)):   # keep direct labels at least 3.5 points apart
                ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.035)
            for y, p, x in ends:
                ax.text(x + n * 0.015, y, f"p={p}", color=INK2, fontsize=8, va="center")
            if key == "test":
                ax.axhline(KB_TEST_WIN, color=INK2, linestyle="--", linewidth=1.1)
                ax.text(n * 0.02, KB_TEST_WIN + 0.012, f"KB agent (no learning) {KB_TEST_WIN:.1%}",
                        color=INK2, fontsize=8)
            ax.axvline(n / 2, color=GRID, linewidth=1.2)
            ax.text(n / 2, 0.02, " ε reaches 0.01", color=INK2, fontsize=8)
            ax.set_xlim(0, n * 1.12)
            ax.set_ylim(0, 0.8)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.set_title(f"{n:,} episodes - greedy win rate on {label}", loc="left", color=INK, fontsize=10.5)
            ax.set_xlabel("Training episode", color=INK2, fontsize=9)
            if col == 0:
                ax.set_ylabel("Win rate", color=INK2, fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("DRQN: target_update_period p and number of episodes  (line = mean, band = min-max over seeds)",
                 x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rewards_figure(runs, path, window=500):
    lengths = sorted({n for n, _, _ in runs})
    fig, axs = plt.subplots(1, len(lengths), figsize=(6.2 * len(lengths), 4.2), facecolor=SURFACE,
                            sharey=True, squeeze=False)
    for col, n in enumerate(lengths):
        ax = axs[0][col]
        style(ax)
        for (n2, p, maps), rs in runs.items():
            if n2 != n:
                continue
            ma = np.mean([np.convolve(r["returns"], np.ones(window), "valid") / window for r in rs], axis=0)
            ax.plot(np.arange(len(ma)) + window, ma, color=PERIOD_COLOR.get(p, INK2), linewidth=2,
                    label=f"p = {p}  (n={len(rs)} seeds)")
        ax.axvline(n / 2, color=GRID, linewidth=1.2)
        ax.text(n / 2, 0.03, " ε reaches 0.01", color=INK2, fontsize=8, transform=ax.get_xaxis_transform())
        ax.set_title(f"{n:,} episodes - training episode reward ({window}-episode moving average)",
                     loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("Training episode", color=INK2, fontsize=9)
        ax.set_ylabel("Average reward", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def outcomes_figure(runs, path):
    keys = list(runs)
    fig, axs = plt.subplots(1, 2, figsize=(13, 0.55 * len(keys) + 1.8), facecolor=SURFACE, sharey=True)
    for ax, (part, title) in zip(axs, (("training", "Training maps (100)"), ("test", "Unseen test maps (1,000)"))):
        style(ax)
        ax.grid(False)
        for i, k in enumerate(keys):
            f = [r["final"]["test1000"][part] for r in runs[k]]
            left = 0.0
            for metric, color in (("win_rate", WIN), ("death_rate", DEATH), ("timeout_rate", TIMEOUT)):
                share = float(np.mean([x[metric] for x in f]))
                ax.barh(i, share, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
                if share >= 0.07:
                    ax.text(left + share / 2, i, f"{share:.0%}", ha="center", va="center", fontsize=8.5,
                            color="white", fontweight="bold")
                left += share
        ax.set_yticks(range(len(keys)), [f"n={n // 1000}k, p={p}  ({len(runs[(n, p, m)])} seeds)" for n, p, m in keys],
                      color=INK, fontsize=9)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    axs[0].invert_yaxis()  # first configuration on top (shared y-axis)
    handles = [Patch(color=c, label=lab) for c, lab in ((WIN, "Win"), (DEATH, "Death"), (TIMEOUT, "Timeout"))]
    fig.legend(handles=handles, ncol=3, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Final greedy outcomes (mean over seeds)", x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    runs = load()
    md = table(runs)
    (HERE / "results_table.md").write_text(md + "\n")
    print(md)
    curves_figure(runs, HERE / "plots" / "learning_curves.png")
    rewards_figure(runs, HERE / "plots" / "training_rewards.png")
    outcomes_figure(runs, HERE / "plots" / "outcomes.png")


if __name__ == "__main__":
    main()
