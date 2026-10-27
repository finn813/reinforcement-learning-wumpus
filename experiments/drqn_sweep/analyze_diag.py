"""Turn runs/diag_*.json into diagnosis tables (diagnosis.md) and example-map figures."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from environment import WumpusEnv  # noqa: E402

HERE = Path(__file__).resolve().parent
DRQN_C, KB_C, SAFE_C = "#2a78d6", "#52514e", "#1baf7a"
INK, INK2, GRID, SURFACE, HAZARD_FILL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#d9d8d3"
TEST = WumpusEnv(num_training_maps=1000, seed=12345)
KB = json.loads((HERE / "runs" / "kb_reference.json").read_text())


def group(ep):
    if ep["start_percept"]:
        return "A"
    return "B" if KB[ep["map_id"]]["outcome"] == "win" else "C"


GROUPS = {"A": "Breeze/stench at the start (no risk-free move exists)",
          "B": "Clean start, KB agent wins (winnable without risk)",
          "C": "Clean start, KB agent times out (a risk is needed later)"}


def death_cause(ep):
    last = ep["moves"][-1]
    if last["kind"] == "known_hazard":
        return "walked into a provable hazard"
    if ep["start_percept"] and all(p == [0, 0] for p in ep["path"][:-1]):
        return "first move out of a start with breeze/stench"
    if last["safe_unvisited_available"] == 0:
        return "risk taken after all provably safe cells were used up"
    return "risk taken although provably safe cells were still unvisited"


def timeout_pattern(ep):
    kinds = [m["kind"] for m in ep["moves"]]
    if all(k == "bump" for k in kinds):
        return "never left the start"
    last_new = max((i for i, k in enumerate(kinds) if k in ("safe_new", "risk")), default=-1)
    if ep["safe_unvisited_at_end"]:
        if len(kinds) - 1 - last_new >= 6:
            return "stopped exploring: looped or bumped although safe cells were left"
        return "still exploring when the 20 steps ran out"
    return "no provably safe cell left"


def load():
    models = defaultdict(list)
    for f in sorted((HERE / "runs").glob("diag_n*_maps100_seed*.json")):
        cfg = re.match(r"diag_(n\d+k_p\d+)_maps\w+_seed\d+", f.stem).group(1)
        models[cfg].append(json.loads(f.read_text()))
    return dict(sorted(models.items(), key=lambda kv: tuple(map(int, re.findall(r"\d+", kv[0])))))


def pct(x):
    return f"{x:.1%}"


def report(models):
    lines = []
    for cfg, seeds in models.items():
        n, p = re.match(r"n(\d+)k_p(\d+)", cfg).groups()
        lines.append(f"## n_episodes = {n}k, target_update_period = {p}  ({len(seeds)} seeds, mean over seeds)\n")

        # 1. outcome by map group
        lines += ["| Test maps | Maps | DRQN win | DRQN death | DRQN timeout | KB agent win |", "|---|---|---|---|---|---|"]
        for g, label in GROUPS.items():
            per_seed = [[e for e in eps if group(e) == g] for eps in seeds]
            size = len(per_seed[0])
            rates = {o: np.mean([sum(e["outcome"] == o for e in s) / size for s in per_seed])
                     for o in ("win", "death", "timeout")}
            kb_win = sum(KB[e["map_id"]]["outcome"] == "win" for e in per_seed[0]) / size
            lines.append(f"| {label} | {size} | {pct(rates['win'])} | {pct(rates['death'])} | "
                         f"{pct(rates['timeout'])} | {pct(kb_win)} |")
        totals = {o: np.mean([sum(e["outcome"] == o for e in eps) / len(eps) for eps in seeds])
                  for o in ("win", "death", "timeout")}
        lines.append(f"| **All** | 1000 | {pct(totals['win'])} | {pct(totals['death'])} | {pct(totals['timeout'])} | "
                     f"{pct(sum(r['outcome'] == 'win' for r in KB) / len(KB))} |\n")

        # 2. deaths by cause
        causes = Counter()
        for eps in seeds:
            causes.update(death_cause(e) for e in eps if e["outcome"] == "death")
        total = sum(causes.values())
        lines += ["| Cause of death (fatal move) | Share of deaths | Per 1,000 maps |", "|---|---|---|"]
        for cause in ("first move out of a start with breeze/stench",
                      "risk taken after all provably safe cells were used up",
                      "risk taken although provably safe cells were still unvisited",
                      "walked into a provable hazard"):
            lines.append(f"| {cause} | {pct(causes[cause] / total)} | {causes[cause] / len(seeds):.0f} |")
        lines.append("")

        # 3. timeouts on maps the KB agent wins
        pats, gold_safe, safe_left = Counter(), 0, []
        for eps in seeds:
            for e in eps:
                if e["outcome"] == "timeout" and group(e) == "B":
                    pats[timeout_pattern(e)] += 1
                    gold_safe += e["gold_on_safe_unvisited"]
                    safe_left.append(len(e["safe_unvisited_at_end"]))
        total = sum(pats.values())
        if total:
            lines += ["| Timeouts on maps the KB agent wins | Share | Per 1,000 maps |", "|---|---|---|"]
            for pat in ("stopped exploring: looped or bumped although safe cells were left",
                        "still exploring when the 20 steps ran out", "never left the start",
                        "no provably safe cell left"):
                lines.append(f"| {pat} | {pct(pats[pat] / total)} | {pats[pat] / len(seeds):.0f} |")
            lines.append(f"\nIn {pct(gold_safe / total)} of these timeouts the gold lay on a cell the DRQN's own "
                         f"percepts had already proven safe; on average {np.mean(safe_left):.1f} provably safe cells "
                         f"were left unvisited.\n")

        # 4. waiting at a start with breeze/stench
        waits, left = [], 0
        for eps in seeds:
            for e in eps:
                if e["start_percept"]:
                    kinds = [m["kind"] for m in e["moves"]]
                    first_move = next((i for i, k in enumerate(kinds) if k != "bump"), None)
                    if first_move is not None:
                        waits.append(first_move)
                        left += 1
        n_a = sum(e["start_percept"] for e in seeds[0]) * len(seeds)
        lines.append(f"On maps with breeze/stench at the start the DRQN leaves the start in {pct(left / n_a)} of the "
                     f"episodes, after bumping into the wall for a median of {np.median(waits):.0f} steps "
                     f"(min {min(waits)}, max {max(waits)}).\n")
    return "\n".join(lines)


def draw_map(ax, ep, title):
    pits, wumpus, gold = TEST._map_pool[ep["map_id"]]
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    visited = {tuple(c) for c in ep["path"]}
    safe_left = {tuple(c) for c in ep["safe_unvisited_at_end"]}
    for x in range(4):
        for y in range(4):
            face = HAZARD_FILL if ((x, y) in pits or (x, y) == wumpus) else SURFACE
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=face, edgecolor=GRID, linewidth=1.2))
            if (x, y) in safe_left:
                ax.add_patch(Rectangle((x - 0.42, y - 0.42), 0.84, 0.84, facecolor="none", edgecolor=SAFE_C,
                                       linewidth=2.2, linestyle="-"))
            label = "P" if (x, y) in pits else "W" if (x, y) == wumpus else "G" if (x, y) == gold else ""
            if label:
                ax.text(x + 0.3, y + 0.3, label, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
            if (x, y) == (0, 0):
                ax.text(x - 0.32, y - 0.36, "start", fontsize=6.5, color=INK2)
    kb_path = np.array(KB[ep["map_id"]]["path"], dtype=float) + 0.12
    ax.plot(kb_path[:, 0], kb_path[:, 1], color=KB_C, linestyle="--", linewidth=1.3)
    path = np.array(ep["path"], dtype=float) - 0.08
    ax.plot(path[:, 0], path[:, 1], color=DRQN_C, linewidth=2.2, marker="o", markersize=3.5)
    ax.plot(path[-1, 0], path[-1, 1], marker="X", markersize=11, color=DRQN_C, markeredgecolor=SURFACE)
    visits = Counter(map(tuple, ep["path"]))
    for (x, y), k in visits.items():
        if k > 1:
            ax.text(x - 0.32, y + 0.33, f"{k}×", fontsize=7.5, color=DRQN_C)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")


def examples_figure(eps, path, picks, suptitle):
    fig, axs = plt.subplots(2, 3, figsize=(11, 8.4), facecolor=SURFACE)
    for ax, (ep, title) in zip(axs.flat, picks):
        draw_map(ax, ep, title)
    for ax in list(axs.flat)[len(picks):]:
        ax.axis("off")
    handles = [Line2D([], [], color=DRQN_C, linewidth=2.2, marker="o", markersize=4, label="DRQN path (X = end)"),
               Line2D([], [], color=KB_C, linestyle="--", label="KB agent path"),
               Patch(facecolor="none", edgecolor=SAFE_C, linewidth=2, label="provably safe from the DRQN's own percepts, never visited"),
               Patch(facecolor=HAZARD_FILL, edgecolor=GRID, label="hazard (P = pit, W = Wumpus), G = gold")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(suptitle, x=0.02, ha="left", fontsize=11.5, color=INK)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.11, wspace=0.08, hspace=0.22)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    models = load()
    md = report(models)
    (HERE / "diagnosis.md").write_text(md + "\n")
    print(md)
    # example maps from the first seed of the best configuration
    best = max(models, key=lambda c: np.mean([sum(e["outcome"] == "win" for e in eps) for eps in models[c]]))
    eps = models[best][0]
    n, p = re.match(r"n(\d+)k_p(\d+)", best).groups()
    pretty = f"{n}k episodes, p = {p}, seed 0"
    timeouts = [e for e in eps if e["outcome"] == "timeout" and group(e) == "B"]
    stuck = [e for e in timeouts if timeout_pattern(e).startswith("stopped")]
    picks = [(e, f"map {e['map_id']}: timeout, {len(e['safe_unvisited_at_end'])} safe cell(s) left"
                 + (", gold among them" if e["gold_on_safe_unvisited"] else "")) for e in stuck[:6]]
    examples_figure(eps, HERE / "plots" / "examples_timeouts.png", picks,
                    f"DRQN ({pretty}) times out on maps the KB agent wins without risk")
    deaths = [e for e in eps if e["outcome"] == "death" and group(e) != "A"]
    short = {"walked into a provable hazard": "walked into a provable hazard",
             "risk taken although provably safe cells were still unvisited": "risk although safe cells were left",
             "risk taken after all provably safe cells were used up": "risk after safe cells were used up",
             "first move out of a start with breeze/stench": "gamble at the start"}
    picks = []
    for cause, k in (("walked into a provable hazard", 2),
                     ("risk taken although provably safe cells were still unvisited", 2),
                     ("risk taken after all provably safe cells were used up", 2)):
        picks += [(e, f"map {e['map_id']}: death, {short[cause]}") for e in deaths if death_cause(e) == cause][:k]
    examples_figure(eps, HERE / "plots" / "examples_deaths.png", picks,
                    f"DRQN ({pretty}) deaths on maps with a clean start (green = safe cells it still had)")
    print("best config:", best)


if __name__ == "__main__":
    main()
