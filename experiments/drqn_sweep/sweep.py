"""DRQN sweep over n_episodes and target_update_period.

Training is identical to dqrn_agent.ipynb (comparison-tests branch): 100 training maps
(seed 42), Adam lr 1e-3, epsilon 1.0 -> 0.01 decayed linearly over the first half of
training. Evaluation uses evaluation.py (greedy, fixed unseen test maps, seed 12345).

Usage: python sweep.py --episodes 10000 --period 500 --seed 0
"""
import argparse
import contextlib
import io
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dqrn_agent import DQRNWumpusAgent  # noqa: E402
from environment import WumpusEnv  # noqa: E402
from evaluation import _evaluate_on_maps, evaluate_train_test  # noqa: E402

HERE = Path(__file__).resolve().parent
CURVE_EVERY = 1000          # episodes between learning-curve evaluations


def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode) / window


def notebook_plot(env, agent, title, path, rolling_length=500):
    """The same three panels as the notebook, saved to a file."""
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    ma = get_moving_avgs(env.return_queue, rolling_length, "valid")
    axs[0].plot(range(len(ma)), ma)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")
    axs[1].set_title("Episode lengths")
    ma = get_moving_avgs(env.length_queue, rolling_length, "valid")
    axs[1].plot(range(len(ma)), ma)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")
    axs[2].set_title("Training Error")
    ma = get_moving_avgs(agent.training_error, rolling_length, "same")
    axs[2].plot(range(len(ma)), ma)
    axs[2].set_ylabel("Loss (Huber)")
    axs[2].set_xlabel("Training step")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--period", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maps", default="100")
    ap.add_argument("--goal-reward", type=float, default=100.0, help="gold reward during training only")
    args = ap.parse_args()
    num_training_maps = args.maps if args.maps == "inf" else int(args.maps)
    name = f"n{args.episodes // 1000}k_p{args.period}_maps{args.maps}_seed{args.seed}"
    if args.goal_reward != 100:
        name = name.replace("_seed", f"_g{args.goal_reward:g}_seed")

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- identical to the notebook -------------------------------------------------
    learning_rate = 1e-3
    n_episodes = args.episodes
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.01
    training_seed = 42
    target_update_period = args.period

    env = gym.wrappers.RecordEpisodeStatistics(
        WumpusEnv(num_training_maps=num_training_maps, seed=training_seed, goal_reward=args.goal_reward),
        buffer_length=n_episodes)
    agent = DQRNWumpusAgent(env=env, learning_rate=learning_rate, initial_epsilon=start_epsilon,
                            epsilon_decay=epsilon_decay, final_epsilon=final_epsilon,
                            target_update_period=target_update_period)
    # --------------------------------------------------------------------------------

    finite = num_training_maps != "inf"
    train_eval_env = WumpusEnv(num_training_maps=num_training_maps, seed=training_seed) if finite else None
    test_env = WumpusEnv(num_training_maps=1000, seed=12345)

    def layout_key(pits, wumpus, gold):
        return (frozenset(pits), tuple(wumpus), tuple(gold))

    base = env.unwrapped
    seen = {layout_key(*m) for m in base._map_pool} if finite else set()
    curve, t0 = [], time.time()
    for episode in range(n_episodes):
        obs, info = env.reset()
        agent.reset_hidden()
        if not finite:
            seen.add(layout_key(base._pits, base._wumpus, base._gold))
        done = False
        episode_obs, episode_actions, episode_rewards, episode_terminated = [obs], [], [], []
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_obs.append(next_obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_terminated.append(terminated)
            agent.train_step()
            done = terminated or truncated
            obs = next_obs
        agent.memory.put(episode_obs, episode_actions, episode_rewards, episode_terminated)
        agent.decay_epsilon()

        if (episode + 1) % CURVE_EVERY == 0:
            point = {"episode": episode + 1, "epsilon": agent.epsilon,
                     "train_reward_ma1000": float(np.mean(list(env.return_queue)[-1000:])),
                     "test": _evaluate_on_maps(agent, test_env, range(1000), policy_seed=54321)}
            if finite:
                point["train"] = _evaluate_on_maps(agent, train_eval_env, range(num_training_maps),
                                                   policy_seed=54321)
            curve.append(point)
            print(f"{name} ep {episode + 1}: test win {point['test']['win_rate']:.1%}"
                  + (f", train win {point['train']['win_rate']:.1%}" if finite else "")
                  + f" ({time.time() - t0:.0f}s)", flush=True)

    seconds = time.time() - t0
    final = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for n_test in (100, 1000):
            if finite:
                final[f"test{n_test}"] = evaluate_train_test(agent, training_env=env, num_test_maps=n_test,
                                                             test_seed=12345)
            else:
                from evaluation import evaluate_agent
                final[f"test{n_test}"] = {"test": evaluate_agent(agent, num_test_maps=n_test, test_seed=12345)}

    out = {"name": name, "episodes": n_episodes, "period": target_update_period, "seed": args.seed,
           "goal_reward": args.goal_reward,
           "num_training_maps": args.maps, "train_updates": agent.train_steps,
           "target_syncs": agent.train_steps // target_update_period, "seconds": seconds,
           "distinct_training_maps": len(seen),
           "test_maps_seen_in_training": [i for i, m in enumerate(test_env._map_pool) if layout_key(*m) in seen],
           "curve": curve, "final": final,
           "returns": [float(r) for r in env.return_queue], "lengths": [int(x) for x in env.length_queue]}
    (HERE / "runs" / f"{name}.json").write_text(json.dumps(out))
    torch.save(agent.drqnn.state_dict(), HERE / "runs" / f"{name}.pt")
    notebook_plot(env, agent, f"DRQN  n_episodes={n_episodes}, target_update_period={target_update_period}, "
                              f"maps={args.maps}, seed={args.seed}", HERE / "plots" / f"{name}_training.png")
    t = final["test1000"]
    print(f"DONE {name}: test1000 win {t['test']['win_rate']:.1%}"
          + (f", train win {t['training']['win_rate']:.1%}" if finite else "") + f", {seconds / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
