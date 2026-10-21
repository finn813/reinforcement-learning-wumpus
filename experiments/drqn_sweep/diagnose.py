"""Why does the DRQN not win every map that is winnable without risk?

Replays every trained DRQN (runs/*.pt) greedily on the 1,000 test maps (seed 12345)
and, next to it, keeps a KB agent's knowledge base fed with exactly the percepts the
DRQN has seen. Before every DRQN move we can therefore say whether the target cell was
provably safe, unknown (a risk) or provably a hazard, and whether provably safe
unvisited cells were still available.

Usage: python diagnose.py            -> runs/diag_<run>.json for every model
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import torch

from dqrn_agent import DRQNN
from environment import WumpusEnv
from kb_agent import KBWumpusAgent, MOVES

HERE = Path(__file__).resolve().parent
N_MAPS = 1000


def test_env():
    return WumpusEnv(num_training_maps=N_MAPS, seed=12345)


def layout(env, map_id):
    pits, wumpus, gold = env._map_pool[map_id]
    return {"pits": sorted(map(list, pits)), "wumpus": list(wumpus), "gold": list(gold)}


def run_kb(env, kb, map_id):
    obs, _ = env.reset(options={"map_id": map_id})
    kb.reset()
    path, done = [(int(obs[0]), int(obs[1]))], False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(kb.get_action(obs))
        path.append((int(obs[0]), int(obs[1])))
        done = terminated or truncated
    outcome = "win" if terminated and reward > 0 else "death" if terminated else "timeout"
    return outcome, path


def kb_reference():
    """KB-agent outcome and path on every test map (cached)."""
    cache = HERE / "runs" / "kb_reference.json"
    if cache.exists():
        return json.loads(cache.read_text())
    env = test_env()
    kb = KBWumpusAgent(env)
    ref = []
    for i in range(N_MAPS):
        outcome, path = run_kb(env, kb, i)
        ref.append({"outcome": outcome, "path": path})
    cache.write_text(json.dumps(ref))
    return ref


def diagnose_model(pt_path):
    env = test_env()
    net = DRQNN(state_space=4, action_space=4, obs_high=env.observation_space.high)
    net.load_state_dict(torch.load(pt_path))
    net.eval()
    knowledge = KBWumpusAgent(env)   # only used as a knowledge base, its own actions are ignored
    episodes = []
    for map_id in range(N_MAPS):
        obs, _ = env.reset(options={"map_id": map_id})
        knowledge.reset()
        h, c = net.init_hidden_state(batch_size=1, training=False)
        pos = (int(obs[0]), int(obs[1]))
        ep = {"map_id": map_id, "start_percept": bool(obs[2] or obs[3]), "path": [pos], "moves": []}
        done = False
        while not done:
            knowledge.get_action(obs)                       # TELL the percept, update the safe set
            with torch.no_grad():
                q, h, c = net(torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1), h, c)
            action = int(q.argmax().item())
            dx, dy = MOVES[action]
            target = (min(max(pos[0] + dx, 0), env.width - 1), min(max(pos[1] + dy, 0), env.height - 1))
            safe_left = sorted(knowledge.safe - knowledge.visited)
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
            ep["moves"].append({"kind": kind, "safe_unvisited_available": len(safe_left)})
            obs, reward, terminated, truncated, _ = env.step(action)
            pos = (int(obs[0]), int(obs[1]))
            ep["path"].append(pos)
            done = terminated or truncated
        ep["outcome"] = "win" if terminated and reward > 0 else "death" if terminated else "timeout"
        if ep["outcome"] == "timeout":
            knowledge.get_action(obs)  # include the last percept
        ep["safe_unvisited_at_end"] = sorted(knowledge.safe - knowledge.visited)
        ep["gold_on_safe_unvisited"] = tuple(env._gold) in ep["safe_unvisited_at_end"]
        episodes.append(ep)
    out = HERE / "runs" / f"diag_{Path(pt_path).stem}.json"
    out.write_text(json.dumps(episodes))
    return out.name


def main():
    kb_reference()
    models = sorted(str(p) for p in (HERE / "runs").glob("*.pt"))
    if len(sys.argv) > 1:
        models = [m for m in models if any(a in m for a in sys.argv[1:])]
    models = [m for m in models if not (HERE / "runs" / f"diag_{Path(m).stem}.json").exists()]
    torch.set_num_threads(1)
    with Pool(2) as pool:
        for name in pool.imap_unordered(diagnose_model, models):
            print("written", name, flush=True)


if __name__ == "__main__":
    main()
