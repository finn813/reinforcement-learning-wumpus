"""Aggregate final_experiments/runs into per-agent result files, one summary table and plots.

Run from the repository root:  python final_experiments/summarize.py
Writes final_experiments/results/{q_learning,dqn,drqn}.json, summary.md, summary.csv and
final_experiments/results/plots/*.png (random.json and kb.json come from `pipeline.py baselines`).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS, RESULTS = HERE / "runs", HERE / "results"
PLOTS = RESULTS / "plots"
AGENTS = {"q_learning": "Tabular Q-learning", "dqn": "DQN", "drqn": "DRQN"}
POOLS = (1, 10, 100, 1000)
SEEDS = (0, 1, 2)
METRICS = ("win_rate", "death_rate", "timeout_rate", "average_reward")

# Categorical slots 1-3 of the validated reference palette (pass all-pairs CVD checks)
COLORS = {"q_learning": "#2a78d6", "dqn": "#eb6834", "drqn": "#1baf7a"}
WIN, DEATH, TIMEOUT = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def mean_std(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0  # sample std over seeds


def load_runs(agent):
    runs = []
    for pool in POOLS:
        for seed in SEEDS:
            r = json.loads((RUNS / f"{agent}_pool{pool}_seed{seed}.json").read_text())
            r.pop("training_returns"), r.pop("training_lengths")
            runs.append(r)
    return runs


def aggregate(runs):
    out = {}
    for pool in POOLS:
        rs = [r for r in runs if r["pool_size"] == pool]
        assert sorted(r["seed"] for r in rs) == list(SEEDS)
        agg = {"seeds": len(rs)}
        for part in ("train", "test"):
            for m in METRICS:
                agg[f"{part}_{m}"] = mean_std([r[part][m] for r in rs])
        agg["generalization_gap"] = mean_std([r["generalization_gap"] for r in rs])
        out[str(pool)] = agg
    return out


def fmt(ms, kind):
    m, s = ms
    return f"{m:.1%} ± {s:.1%}" if kind == "pct" else f"{m:.1f} ± {s:.1f}"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


def pool_axis(ax):
    ax.set_xticks(range(len(POOLS)), [f"{p:,}" for p in POOLS])
    ax.set_xlabel("Training maps (pool size)", color=INK2, fontsize=9)
    ax.set_xlim(-0.3, len(POOLS) - 0.7)


def line_by_pool(ax, runs, key, agent, label=None, linestyle="-", scale=1.0, hollow=False):
    """Mean over seeds as a line, every seed as a dot."""
    xs = range(len(POOLS))
    vals = [[key(r) * scale for r in runs if r["pool_size"] == p] for p in POOLS]
    ax.plot(xs, [np.mean(v) for v in vals], color=COLORS[agent], linewidth=2, marker="o", markersize=6,
            linestyle=linestyle, label=label or AGENTS[agent], markerfacecolor=SURFACE if hollow else COLORS[agent],
            markeredgewidth=1.8)
    for x, v in zip(xs, vals):
        ax.scatter([x] * len(v), v, color=COLORS[agent], s=13, alpha=0.4, linewidths=0)


def baseline_lines(ax, baselines, key, scale=1.0):
    for name, ls in (("kb", "--"), ("random", ":")):
        y = np.mean([key(r) * scale for r in baselines[name]["runs"]])
        ax.axhline(y, color=INK2, linestyle=ls, linewidth=1.1)
        label = "KB agent" if name == "kb" else "Random agent"
        ax.text(len(POOLS) - 0.75, y, f" {label}", color=INK2, fontsize=8, va="bottom", ha="right")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=150)
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    runs = {a: load_runs(a) for a in AGENTS}
    baselines = {b: json.loads((RESULTS / f"{b}.json").read_text()) for b in ("random", "kb")}

    # one results file per learning agent: every run plus mean/std per pool size
    for a, rs in runs.items():
        (RESULTS / f"{a}.json").write_text(json.dumps(
            {"agent": a, "episodes": rs[0]["episodes"], "pool_sizes": POOLS, "seeds": SEEDS,
             "std": "sample standard deviation over the 3 seeds (ddof=1)",
             "runs": rs, "aggregate": aggregate(rs)}, indent=1))

    # combined summary table
    header = ["Agent", "Training maps", "Seeds", "Train win", "Train death", "Train timeout", "Train return",
              "Test win", "Test death", "Test timeout", "Test return", "Generalisation gap (win)"]
    rows = []
    for a in AGENTS:
        agg = aggregate(runs[a])
        for p in POOLS:
            g = agg[str(p)]
            rows.append([AGENTS[a], f"{p:,}", "3"]
                        + [fmt(g[f"train_{m}"], "num" if m == "average_reward" else "pct") for m in METRICS]
                        + [fmt(g[f"test_{m}"], "num" if m == "average_reward" else "pct") for m in METRICS]
                        + [fmt(g["generalization_gap"], "pct")])
    for b, label in (("random", "Random agent (baseline)"), ("kb", "Knowledge-based agent (baseline)")):
        tests = [r["test"] for r in baselines[b]["runs"]]
        cells = [fmt(mean_std([t[m] for t in tests]), "num" if m == "average_reward" else "pct") for m in METRICS]
        if b == "kb":
            cells = [c.split(" ±")[0] for c in cells]  # deterministic: a single value
        rows.append([label, "–", str(len(tests))] + ["–"] * 4 + cells + ["–"])
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]
    note = ("All agents evaluated greedily (epsilon = 0) on the same 1,000 unseen test maps (map_sets.py). "
            "Learning agents: 10,000 training episodes, agent seeds 0-2, mean ± sample standard deviation over "
            "seeds; generalisation gap = training-map win rate - test-map win rate. Random agent: policy seeds "
            "0-2. Knowledge-based agent: deterministic.\n\n")
    (RESULTS / "summary.md").write_text(note + "\n".join(md) + "\n")
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print(note + "\n".join(md))

    # 1. test win rate
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    style(ax)
    for a in AGENTS:
        line_by_pool(ax, runs[a], lambda r: r["test"]["win_rate"], a)
    baseline_lines(ax, baselines, lambda r: r["test"]["win_rate"])
    pool_axis(ax)
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Win rate on 1,000 unseen test maps (line = mean, dots = seeds)", loc="left", color=INK, fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, "test_win_rate.png")

    # 2. mean return
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    style(ax)
    for a in AGENTS:
        line_by_pool(ax, runs[a], lambda r: r["test"]["average_reward"], a)
    baseline_lines(ax, baselines, lambda r: r["test"]["average_reward"])
    ax.axhline(0, color=GRID, linewidth=1)
    pool_axis(ax)
    ax.set_title("Mean return on 1,000 unseen test maps (line = mean, dots = seeds)", loc="left", color=INK,
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 0.9))
    save(fig, "mean_return.png")

    # 3. training vs test win rate, one panel per agent
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE, sharey=True)
    for ax, a in zip(axs, AGENTS):
        style(ax)
        line_by_pool(ax, runs[a], lambda r: r["train"]["win_rate"], a, "own training maps (dashed)", linestyle="--",
                     hollow=True)
        line_by_pool(ax, runs[a], lambda r: r["test"]["win_rate"], a, "unseen test maps (solid)")
        pool_axis(ax)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(AGENTS[a], loc="left", color=INK, fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right", handlelength=4)
    fig.suptitle("Win rate on the training maps vs. unseen test maps", x=0.01, ha="left", color=INK, fontsize=11.5)
    save(fig, "train_vs_test_win_rate.png")

    # 4. generalisation gap
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    style(ax)
    for a in AGENTS:
        line_by_pool(ax, runs[a], lambda r: r["generalization_gap"], a, scale=100)
    ax.axhline(0, color=INK2, linewidth=1)
    pool_axis(ax)
    ax.set_ylabel("Training win rate - test win rate (percentage points)", color=INK2, fontsize=9)
    ax.set_title("Generalisation gap (line = mean, dots = seeds)", loc="left", color=INK, fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    save(fig, "generalisation_gap.png")

    # 5. outcome shares on the test maps
    labels, shares = [], []
    for a in AGENTS:
        for p in POOLS:
            rs = [r for r in runs[a] if r["pool_size"] == p]
            labels.append(f"{AGENTS[a]}, {p:,} map{'' if p == 1 else 's'}")
            shares.append([np.mean([r["test"][m] for r in rs]) for m in METRICS[:3]])
    for b, label in (("random", "Random agent"), ("kb", "KB agent")):
        labels.append(label)
        shares.append([np.mean([r["test"][m] for r in baselines[b]["runs"]]) for m in METRICS[:3]])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(labels) + 1.4), facecolor=SURFACE)
    style(ax)
    ax.grid(False)
    for i, sh in enumerate(shares):
        left = 0.0
        for share, color in zip(sh, (WIN, DEATH, TIMEOUT)):
            ax.barh(i, share, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
            if share >= 0.07:
                ax.text(left + share / 2, i, f"{share:.0%}", ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold")
            left += share
    ax.set_yticks(range(len(labels)), labels, color=INK, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(handles=[Patch(color=c, label=l) for c, l in ((WIN, "Win"), (DEATH, "Death"), (TIMEOUT, "Timeout"))],
              ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9)
    ax.set_title("Outcomes on the 1,000 unseen test maps (mean over seeds)", loc="left", color=INK, fontsize=10.5,
                 pad=24)
    save(fig, "test_outcomes.png")
    print("plots written to", PLOTS)


if __name__ == "__main__":
    main()
