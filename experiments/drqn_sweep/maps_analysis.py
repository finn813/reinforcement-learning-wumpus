"""Effect of the number of training maps (10k episodes, p = 100) for gold reward +100 and +150 (training only).

Writes maps_table.md / maps_table_g150.md, reward_comparison.md and plots/training_maps*.png,
plots/reward_comparison.png. All test numbers are scored with the standard rewards (gold = +100).
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from environment import WumpusEnv  # noqa: E402

HERE = Path(__file__).resolve().parent
POOLS = ["10", "100", "500", "1000", "inf"]
POOL_LABEL = {"10": "10", "100": "100", "500": "500", "1000": "1,000", "inf": "new map\nevery episode"}
WIN, DEATH, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TRAIN = "#1baf7a"
KB_TEST_WIN = 0.459
TEST = WumpusEnv(num_training_maps=1000, seed=12345)


def key(pits, wumpus, gold):
    return (frozenset(pits), tuple(wumpus), tuple(gold))


TEST_KEYS = [key(*m) for m in TEST._map_pool]


def seen_test_ids(run):
    if "test_maps_seen_in_training" in run:
        return set(run["test_maps_seen_in_training"])
    pool = {key(*m) for m in WumpusEnv(num_training_maps=int(run["num_training_maps"]), seed=42)._map_pool}
    return {i for i, k in enumerate(TEST_KEYS) if k in pool}


def per_run(pool, seed, goal=100):
    name = f"n10k_p100_maps{pool}{'' if goal == 100 else f'_g{goal}'}_seed{seed}"
    run = json.loads((HERE / "runs" / f"{name}.json").read_text())
    diag = json.loads((HERE / "runs" / f"diag_{name}.json").read_text())
    test = run["final"]["test1000"]["test"]
    seen = seen_test_ids(run)
    unseen = [e for e in diag if e["map_id"] not in seen]
    start = [e for e in diag if e["start_percept"]]
    clean = [e for e in diag if not e["start_percept"]]
    left = [e for e in start if any(m["kind"] != "bump" for m in e["moves"])]
    share = lambda eps, o: sum(e["outcome"] == o for e in eps) / len(eps)  # noqa: E731
    return {
        "train_win": run["final"]["test1000"]["training"]["win_rate"] if "training" in run["final"]["test1000"] else None,
        "distinct_maps": run.get("distinct_training_maps", int(pool) if pool != "inf" else None),
        "test_win": test["win_rate"], "test_death": test["death_rate"], "test_timeout": test["timeout_rate"],
        # evaluate_train_test scores with the training gold reward (finite pools); rescore every win with +100.
        # evaluate_agent (new map every episode) already uses the default +100.
        "test_reward": test["average_reward"] - ((run.get("goal_reward", 100) - 100) * test["win_rate"]
                                                 if "training" in run["final"]["test1000"] else 0),
        "n_seen": len(seen), "unseen_win": share(unseen, "win"),
        "start_leave": len(left) / len(start), "start_win": share(start, "win"), "start_death": share(start, "death"),
        "clean_win": share(clean, "win"), "clean_death": share(clean, "death"),
    }


def fmt(values, kind):
    vals = [v for v in values if v is not None]
    if not vals:
        return "–"
    f = (lambda x: f"{x:.1%}") if kind == "pct" else (lambda x: f"{x:.1f}") if kind == "num" else (lambda x: f"{x:,.0f}")
    m = np.mean(vals)
    return f(m) + (f" ({f(min(vals))}–{f(max(vals))})" if len(vals) > 1 else "")


def report(goal):
    data = {p: [per_run(p, s, goal) for s in range(3)] for p in POOLS}
    suffix = "" if goal == 100 else f"_g{goal}"
    cols = [("distinct_maps", "Distinct training maps", "int"), ("train_win", "Win, own training maps", "pct"),
            ("test_win", "Win, 1,000 test maps", "pct"), ("test_death", "Death, test", "pct"),
            ("test_timeout", "Timeout, test", "pct"), ("test_reward", "Avg. reward, test", "num"),
            ("n_seen", "Test maps also seen in training", "int"), ("unseen_win", "Win, test maps never seen", "pct"),
            ("start_leave", "Leaves a start with breeze/stench", "pct"), ("start_win", "Win, start with breeze/stench", "pct"),
            ("start_death", "Death, start with breeze/stench", "pct"), ("clean_win", "Win, clean start", "pct"),
            ("clean_death", "Death, clean start", "pct")]
    lines = ["| Training maps | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for p in POOLS:
        lines.append(f"| {POOL_LABEL[p].replace(chr(10), ' ')} | "
                     + " | ".join(fmt([r[c] for r in data[p]], k) for c, _, k in cols) + " |")
    md = (f"Gold reward during training: +{goal}. All runs: 10,000 episodes, target_update_period 100, 3 seeds; "
          "mean (min–max over seeds). Test rewards are scored with gold = +100.\n\n"
          + "\n".join(lines) + "\n")
    (HERE / f"maps_table{suffix}.md").write_text(md)
    print(md)

    x = np.arange(len(POOLS))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    for ax in axs:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_xticks(x, [POOL_LABEL[p] for p in POOLS])
        ax.set_xlabel("Number of training maps", color=INK2, fontsize=9)
    ax = axs[0]
    for metric, color, label in (("train_win", TRAIN, "Win, own training maps"),
                                 ("test_win", WIN, "Win, unseen test maps"),
                                 ("test_death", DEATH, "Death, unseen test maps")):
        xs = [i for i, p in enumerate(POOLS) if data[p][0][metric] is not None]
        means = [np.mean([r[metric] for r in data[POOLS[i]]]) for i in xs]
        ax.plot(xs, means, color=color, linewidth=2, marker="o", markersize=6, label=label)
        for i in xs:
            ax.scatter([i] * 3, [r[metric] for r in data[POOLS[i]]], color=color, s=12, alpha=0.45, linewidths=0)
    ax.axhline(KB_TEST_WIN, color=INK2, linestyle="--", linewidth=1.1)
    ax.text(len(POOLS) - 0.35, KB_TEST_WIN + 0.015, f"KB agent on the test maps {KB_TEST_WIN:.1%}", color=INK2,
            fontsize=8, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.3, len(POOLS) - 0.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Win and death rate (line = mean, dots = seeds)", loc="left", color=INK, fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axs[1]
    means = [np.mean([r["test_reward"] for r in data[p]]) for p in POOLS]
    ax.plot(x, means, color=INK, linewidth=2, marker="o", markersize=6)
    for i, p in enumerate(POOLS):
        ax.scatter([i] * 3, [r["test_reward"] for r in data[p]], color=INK, s=12, alpha=0.4, linewidths=0)
        ax.text(i + 0.1, means[i], f"{means[i]:.1f}", ha="left", va="center", color=INK2, fontsize=8.5)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_title("Average reward on the 1,000 unseen test maps (gold = +100)", loc="left", color=INK, fontsize=10.5)
    fig.suptitle(f"DRQN trained for 10,000 episodes on pools of different size (p = 100, 3 seeds, gold +{goal})",
                 x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(HERE / "plots" / f"training_maps{suffix}.png", dpi=150)
    plt.close(fig)
    return data


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


def compare(data):
    goals = sorted(data)
    colors = {100: "#2a78d6", 150: "#eb6834"}
    metrics = [("test_win", "Win, unseen test maps", "pct"), ("test_death", "Death, unseen test maps", "pct"),
               ("test_reward", "Avg. reward, unseen test maps (gold = +100)", "num"),
               ("clean_death", "Death on clean starts", "pct"), ("start_leave", "Leaves a start with breeze/stench", "pct"),
               ("train_win", "Win, own training maps", "pct")]
    lines = ["| Training maps | " + " | ".join(f"{label}: +{g}" for _, label, _ in metrics for g in goals) + " |",
             "|---" * (len(metrics) * len(goals) + 1) + "|"]
    for p in POOLS:
        lines.append(f"| {POOL_LABEL[p].replace(chr(10), ' ')} | " + " | ".join(
            fmt([r[m] for r in data[g][p]], k) for m, _, k in metrics for g in goals) + " |")
    md = ("Gold reward during training +100 vs +150 (hazard -100, step -1 unchanged); 10k episodes, p = 100, "
          "3 seeds, mean (min–max). All test numbers scored with gold = +100.\n\n" + "\n".join(lines) + "\n")
    (HERE / "reward_comparison.md").write_text(md)
    print(md)
    x = np.arange(len(POOLS))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=SURFACE)
    for ax, (m, label, kind) in zip(axs, metrics[:3]):
        style(ax)
        for g in goals:
            means = [np.mean([r[m] for r in data[g][p]]) for p in POOLS]
            ax.plot(x, means, color=colors[g], linewidth=2, marker="o", markersize=6, label=f"gold +{g} in training")
            for i, p in enumerate(POOLS):
                ax.scatter([i] * 3, [r[m] for r in data[g][p]], color=colors[g], s=12, alpha=0.4, linewidths=0)
        ax.set_xticks(x, [POOL_LABEL[p] for p in POOLS])
        ax.set_xlabel("Number of training maps", color=INK2, fontsize=9)
        if kind == "pct":
            ax.set_ylim(0, 0.6)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        else:
            ax.axhline(0, color=GRID, linewidth=1)
        ax.set_title(label, loc="left", color=INK, fontsize=10.5)
        ax.legend(frameon=False, fontsize=8, loc="upper right" if m != "test_reward" else "lower right")
    axs[0].axhline(KB_TEST_WIN, color=INK2, linestyle="--", linewidth=1.1)
    axs[0].text(len(POOLS) - 1, KB_TEST_WIN + 0.012, f"KB agent {KB_TEST_WIN:.1%}", color=INK2, fontsize=8, ha="right")
    fig.suptitle("Gold reward +100 vs +150 during training (line = mean, dots = 3 seeds; scored with +100)",
                 x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "reward_comparison.png", dpi=150)
    plt.close(fig)


def main():
    data = {100: report(100)}
    if all((HERE / "runs" / f"diag_n10k_p100_maps{p}_g150_seed{s}.json").exists() for p in POOLS for s in range(3)):
        data[150] = report(150)
        compare(data)


if __name__ == "__main__":
    main()
