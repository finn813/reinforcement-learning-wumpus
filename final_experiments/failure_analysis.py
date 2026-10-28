"""Failure analysis of the DRQN agents on the unseen test maps.

Every trained DRQN model (pools 100 and 1,000, seeds 0-2) is replayed greedily on the
1,000 test maps while a knowledge-based agent receives exactly the same percepts. Before
each DRQN move we can therefore say whether the target cell was provably safe, unknown
(a risk) or provably a hazard, and whether provably safe cells were still unvisited.

Run from the repository root:  python final_experiments/failure_analysis.py
Writes runs/diag_drqn_pool*_seed*.json, runs/kb_test_episodes.json,
results/failure_analysis.{json,md} and results/plots/failure_examples_{timeouts,deaths}.png
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dqrn_agent import DRQNN  # noqa: E402
from kb_agent import MOVES, KBWumpusAgent  # noqa: E402
from map_sets import make_env, test_maps  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS, RESULTS = HERE / "runs", HERE / "results"
POOLS, SEEDS = (100, 1000), (0, 1, 2)
GROUPS = {"A": "Breeze/stench at the start (no risk-free move exists)",
          "B": "Clean start, KB agent wins (winnable without risk)",
          "C": "Clean start, KB agent times out (a risk is needed later)"}
DRQN_C, KB_C, SAFE_C = "#2a78d6", "#52514e", "#1baf7a"
INK, INK2, GRID, SURFACE, HAZARD_FILL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#d9d8d3"


def test_env():
    return make_env(test_maps())


def outcome(terminated, reward):
    return "win" if terminated and reward > 0 else "death" if terminated else "timeout"


def kb_episodes():
    """KB agent outcome and path on every test map."""
    env, kb, episodes = test_env(), None, []
    kb = KBWumpusAgent(env)
    for map_id in range(len(env._map_pool)):
        obs, _ = env.reset(options={"map_id": map_id})
        kb.reset()
        path, done = [(int(obs[0]), int(obs[1]))], False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(kb.get_action(obs))
            path.append((int(obs[0]), int(obs[1])))
            done = terminated or truncated
        episodes.append({"outcome": outcome(terminated, reward), "path": path})
    return episodes


def diagnose(pt_path):
    """Replay one DRQN model with a shadow knowledge base; label every move."""
    torch.set_num_threads(1)
    env = test_env()
    net = DRQNN(state_space=4, action_space=4, obs_high=env.observation_space.high)
    net.load_state_dict(torch.load(pt_path))
    net.eval()
    knowledge = KBWumpusAgent(env)  # only its knowledge base is used, not its actions
    episodes = []
    for map_id in range(len(env._map_pool)):
        obs, _ = env.reset(options={"map_id": map_id})
        knowledge.reset()
        h, c = net.init_hidden_state(batch_size=1, training=False)
        pos = (int(obs[0]), int(obs[1]))
        ep = {"map_id": map_id, "start_percept": bool(obs[2] or obs[3]), "path": [pos], "moves": []}
        done = False
        while not done:
            knowledge.get_action(obs)  # TELL the percept and update the provably safe cells
            with torch.no_grad():
                q, h, c = net(torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1), h, c)
            action = int(q.argmax().item())
            dx, dy = MOVES[action]
            target = (min(max(pos[0] + dx, 0), env.width - 1), min(max(pos[1] + dy, 0), env.height - 1))
            if target == pos:
                kind = "bump"
            elif target in knowledge.visited:
                kind = "revisit"
            elif target in knowledge.safe:
                kind = "safe_new"
            elif knowledge.ask({knowledge.P(target), knowledge.W(target)}):
                kind = "known_hazard"
            else:
                kind = "risk"
            ep["moves"].append({"kind": kind, "safe_unvisited_available": len(knowledge.safe - knowledge.visited)})
            obs, reward, terminated, truncated, _ = env.step(action)
            pos = (int(obs[0]), int(obs[1]))
            ep["path"].append(pos)
            done = terminated or truncated
        ep["outcome"] = outcome(terminated, reward)
        if ep["outcome"] == "timeout":
            knowledge.get_action(obs)  # include the last percept
        ep["safe_unvisited_at_end"] = sorted(knowledge.safe - knowledge.visited)
        ep["gold_on_safe_unvisited"] = tuple(env._gold) in ep["safe_unvisited_at_end"]
        episodes.append(ep)
    return episodes


def death_cause(ep):
    last = ep["moves"][-1]
    if last["kind"] == "known_hazard":
        return "walked into a provable hazard"
    if ep["start_percept"] and all(tuple(p) == (0, 0) for p in ep["path"][:-1]):
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


def summarise(seeds_eps, kb):
    group = lambda e: "A" if e["start_percept"] else ("B" if kb[e["map_id"]]["outcome"] == "win" else "C")  # noqa
    res = {"groups": {}}
    for g, label in GROUPS.items():
        per = [[e for e in eps if group(e) == g] for eps in seeds_eps]
        res["groups"][g] = {"label": label, "maps": len(per[0]),
                            **{o: float(np.mean([sum(e["outcome"] == o for e in s) / len(s) for s in per]))
                               for o in ("win", "death", "timeout")}}
    causes, patterns, gold, left = Counter(), Counter(), 0, []
    for eps in seeds_eps:
        causes.update(death_cause(e) for e in eps if e["outcome"] == "death")
        for e in eps:
            if e["outcome"] == "timeout" and group(e) == "B":
                patterns[timeout_pattern(e)] += 1
                gold += e["gold_on_safe_unvisited"]
                left.append(len(e["safe_unvisited_at_end"]))
    res["death_causes"] = {k: v / sum(causes.values()) for k, v in causes.most_common()}
    res["timeouts_on_risk_free_maps"] = {k: v / sum(patterns.values()) for k, v in patterns.most_common()}
    res["timeouts_on_risk_free_maps_gold_on_safe_cell"] = gold / sum(patterns.values())
    res["timeouts_on_risk_free_maps_mean_safe_cells_left"] = float(np.mean(left))
    return res, group


def draw_map(ax, ep, kb_path, title):
    pits, wumpus, gold = test_maps()[ep["map_id"]]
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    safe_left = {tuple(c) for c in ep["safe_unvisited_at_end"]}
    for x in range(4):
        for y in range(4):
            face = HAZARD_FILL if ((x, y) in pits or (x, y) == wumpus) else SURFACE
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=face, edgecolor=GRID, linewidth=1.2))
            if (x, y) in safe_left:
                ax.add_patch(Rectangle((x - 0.42, y - 0.42), 0.84, 0.84, facecolor="none", edgecolor=SAFE_C,
                                       linewidth=2.2))
            label = "P" if (x, y) in pits else "W" if (x, y) == wumpus else "G" if (x, y) == gold else ""
            if label:
                ax.text(x + 0.3, y + 0.3, label, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
            if (x, y) == (0, 0):
                ax.text(x - 0.32, y - 0.36, "start", fontsize=6.5, color=INK2)
    kp = np.array(kb_path, dtype=float) + 0.12
    ax.plot(kp[:, 0], kp[:, 1], color=KB_C, linestyle="--", linewidth=1.3)
    path = np.array(ep["path"], dtype=float) - 0.08
    ax.plot(path[:, 0], path[:, 1], color=DRQN_C, linewidth=2.2, marker="o", markersize=3.5)
    ax.plot(path[-1, 0], path[-1, 1], marker="X", markersize=11, color=DRQN_C, markeredgecolor=SURFACE)
    for (x, y), k in Counter(map(tuple, ep["path"])).items():
        if k > 1:
            ax.text(x - 0.32, y + 0.33, f"{k}×", fontsize=7.5, color=DRQN_C)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")


def examples_figure(picks, kb, path, suptitle):
    fig, axs = plt.subplots(2, 3, figsize=(11, 8.4), facecolor=SURFACE)
    for ax, (ep, title) in zip(axs.flat, picks):
        draw_map(ax, ep, kb[ep["map_id"]]["path"], title)
    for ax in list(axs.flat)[len(picks):]:
        ax.axis("off")
    handles = [Line2D([], [], color=DRQN_C, linewidth=2.2, marker="o", markersize=4, label="DRQN path (X = end)"),
               Line2D([], [], color=KB_C, linestyle="--", label="knowledge-based agent path"),
               Patch(facecolor="none", edgecolor=SAFE_C, linewidth=2,
                     label="provably safe from the DRQN's own percepts, not visited"),
               Patch(facecolor=HAZARD_FILL, edgecolor=GRID, label="hazard (P = pit, W = Wumpus), G = gold")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(suptitle, x=0.02, ha="left", fontsize=11.5, color=INK)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.11, wspace=0.08, hspace=0.22)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    kb = kb_episodes()
    (RUNS / "kb_test_episodes.json").write_text(json.dumps(kb))
    jobs = [(p, s, str(RUNS / f"drqn_pool{p}_seed{s}.pt")) for p in POOLS for s in SEEDS]
    with Pool(2) as pool:
        all_eps = pool.map(diagnose, [j[2] for j in jobs])
    eps_by = {}
    for (p, s, _), eps in zip(jobs, all_eps):
        (RUNS / f"diag_drqn_pool{p}_seed{s}.json").write_text(json.dumps(eps))
        eps_by[(p, s)] = eps

    results, lines = {}, []
    for p in POOLS:
        res, group = summarise([eps_by[(p, s)] for s in SEEDS], kb)
        results[str(p)] = res
        lines += [f"## DRQN trained on {p:,} maps (mean over seeds 0-2)\n",
                  "| Test maps | Maps | Win | Death | Timeout |", "|---|---|---|---|---|"]
        lines += [f"| {g['label']} | {g['maps']} | {g['win']:.1%} | {g['death']:.1%} | {g['timeout']:.1%} |"
                  for g in res["groups"].values()]
        lines += ["", "| Cause of death | Share of deaths |", "|---|---|"]
        lines += [f"| {k} | {v:.1%} |" for k, v in res["death_causes"].items()]
        lines += ["", "| Timeouts on risk-free maps | Share |", "|---|---|"]
        lines += [f"| {k} | {v:.1%} |" for k, v in res["timeouts_on_risk_free_maps"].items()]
        lines += ["", f"Gold on a provably safe, unvisited cell in "
                      f"{res['timeouts_on_risk_free_maps_gold_on_safe_cell']:.1%} of these timeouts; "
                      f"{res['timeouts_on_risk_free_maps_mean_safe_cells_left']:.1f} such cells left on average.", ""]
    (RESULTS / "failure_analysis.json").write_text(json.dumps(results, indent=1))
    (RESULTS / "failure_analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # example maps: DRQN trained on 100 maps, seed 0
    eps = eps_by[(100, 0)]
    _, group = summarise([eps], kb)
    stuck = [e for e in eps if e["outcome"] == "timeout" and group(e) == "B"
             and timeout_pattern(e).startswith("stopped")]
    picks = [(e, f"map {e['map_id']}: timeout, {len(e['safe_unvisited_at_end'])} safe cell(s) left"
              + (", gold among them" if e["gold_on_safe_unvisited"] else "")) for e in stuck[:6]]
    examples_figure(picks, kb, RESULTS / "plots" / "failure_examples_timeouts.png",
                    "DRQN (100 training maps, seed 0) times out on maps the knowledge-based agent wins without risk")
    deaths = [e for e in eps if e["outcome"] == "death" and not e["start_percept"]]
    short = {"walked into a provable hazard": "walked into a provable hazard",
             "risk taken although provably safe cells were still unvisited": "risk although safe cells were left",
             "risk taken after all provably safe cells were used up": "risk after safe cells were used up"}
    picks = []
    for cause in short:
        picks += [(e, f"map {e['map_id']}: death, {short[cause]}") for e in deaths if death_cause(e) == cause][:2]
    examples_figure(picks, kb, RESULTS / "plots" / "failure_examples_deaths.png",
                    "DRQN (100 training maps, seed 0): deaths on maps with a clean start")


if __name__ == "__main__":
    main()
