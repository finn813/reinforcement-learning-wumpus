"""Behavioural tests for :class:`wumpus_env.WumpusEnv`.

Runs either way::

    pytest wumpus_env/test_environment.py     # in an env with pytest
    python wumpus_env/test_environment.py      # plain python, no pytest needed

Percept/reward tests set the map directly on the instance (white-box) so the
outcome is deterministic and independent of random map generation.
"""

from collections import deque

import numpy as np

from environment import DOWN, LEFT, RIGHT, UP, WumpusEnv


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _set_map(env, pits, wumpus, gold, agent=(0, 0)):
    """Force a specific layout onto an already-reset env (white-box)."""
    env._pits = set(pits)
    env._wumpus = wumpus
    env._gold = gold
    env._agent_pos = agent
    env._steps = 0


def _independent_solvable(width, height, pits, wumpus, gold):
    """A second, independent BFS used to cross-check generated maps."""
    blocked = set(pits) | {wumpus}
    if (0, 0) in blocked or gold in blocked:
        return False
    seen = {(0, 0)}
    q = deque([(0, 0)])
    while q:
        x, y = q.popleft()
        if (x, y) == gold:
            return True
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in blocked | seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return False


# --------------------------------------------------------------------------- #
# spaces & reset                                                               #
# --------------------------------------------------------------------------- #
def test_action_and_observation_spaces():
    env = WumpusEnv(width=4, height=4, seed=0)
    assert env.action_space.n == 4
    assert env.observation_space.shape == (4,)


def test_reset_returns_valid_obs_and_info():
    env = WumpusEnv(seed=1)
    obs, info = env.reset(seed=1)
    assert obs.shape == (4,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert "map_id" in info and "steps" in info


def test_agent_starts_at_origin():
    env = WumpusEnv(seed=2)
    obs, _ = env.reset(seed=2)
    assert (obs[0], obs[1]) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# percepts                                                                     #
# --------------------------------------------------------------------------- #
def test_breeze_fires_on_adjacent_pit():
    env = WumpusEnv(width=4, height=4, seed=3)
    env.reset(seed=3)
    _set_map(env, pits={(1, 0)}, wumpus=(3, 3), gold=(2, 2), agent=(0, 0))
    obs = env._get_obs()
    assert obs[2] == 1.0  # breeze
    assert obs[3] == 0.0  # stench


def test_stench_fires_on_adjacent_wumpus():
    env = WumpusEnv(width=4, height=4, seed=4)
    env.reset(seed=4)
    _set_map(env, pits={(3, 3)}, wumpus=(0, 1), gold=(2, 2), agent=(0, 0))
    obs = env._get_obs()
    assert obs[2] == 0.0
    assert obs[3] == 1.0


def test_no_false_percepts_on_safe_cell():
    env = WumpusEnv(width=4, height=4, seed=5)
    env.reset(seed=5)
    _set_map(env, pits={(3, 3)}, wumpus=(3, 0), gold=(2, 2), agent=(0, 0))
    obs = env._get_obs()
    assert obs[2] == 0.0 and obs[3] == 0.0


def test_diagonal_hazard_is_not_perceived():
    env = WumpusEnv(width=4, height=4, seed=6)
    env.reset(seed=6)
    # pit diagonally at (1, 1) is NOT orthogonally adjacent to (0, 0).
    _set_map(env, pits={(1, 1)}, wumpus=(3, 3), gold=(2, 2), agent=(0, 0))
    obs = env._get_obs()
    assert obs[2] == 0.0 and obs[3] == 0.0


# --------------------------------------------------------------------------- #
# movement                                                                     #
# --------------------------------------------------------------------------- #
def test_each_action_moves_correctly():
    env = WumpusEnv(width=5, height=5, seed=7)
    env.reset(seed=7)
    # keep hazards/gold out of the working area around (2, 2)
    _set_map(env, pits=set(), wumpus=(4, 4), gold=(4, 0), agent=(2, 2))
    cases = {UP: (2, 3), DOWN: (2, 1), LEFT: (1, 2), RIGHT: (3, 2)}
    for action, expected in cases.items():
        env._agent_pos = (2, 2)
        obs, reward, term, trunc, _ = env.step(action)
        assert (obs[0], obs[1]) == expected
        assert reward == env.step_penalty and not term and not trunc


def test_walls_clamp_the_agent():
    env = WumpusEnv(width=4, height=4, seed=8)
    env.reset(seed=8)
    _set_map(env, pits=set(), wumpus=(2, 2), gold=(3, 3), agent=(0, 0))
    for action in (LEFT, DOWN):  # both walk into a wall from (0, 0)
        env._agent_pos = (0, 0)
        obs, reward, term, trunc, _ = env.step(action)
        assert (obs[0], obs[1]) == (0.0, 0.0)
        assert reward == env.step_penalty and not term


# --------------------------------------------------------------------------- #
# rewards & termination                                                        #
# --------------------------------------------------------------------------- #
def test_reaching_gold_terminates_with_reward():
    env = WumpusEnv(width=4, height=4, seed=9)
    env.reset(seed=9)
    _set_map(env, pits=set(), wumpus=(3, 3), gold=(1, 0), agent=(0, 0))
    obs, reward, term, trunc, _ = env.step(RIGHT)  # (0,0) -> (1,0) = gold
    assert reward == env.goal_reward and term and not trunc


def test_pit_terminates_with_penalty():
    env = WumpusEnv(width=4, height=4, seed=10)
    env.reset(seed=10)
    _set_map(env, pits={(1, 0)}, wumpus=(3, 3), gold=(2, 2), agent=(0, 0))
    obs, reward, term, trunc, _ = env.step(RIGHT)  # (0,0) -> (1,0) = pit
    assert reward == env.hazard_penalty and term and not trunc


def test_wumpus_terminates_with_penalty():
    env = WumpusEnv(width=4, height=4, seed=11)
    env.reset(seed=11)
    _set_map(env, pits={(3, 3)}, wumpus=(1, 0), gold=(2, 2), agent=(0, 0))
    obs, reward, term, trunc, _ = env.step(RIGHT)  # (0,0) -> (1,0) = wumpus
    assert reward == env.hazard_penalty and term and not trunc


def test_ordinary_step_incurs_step_penalty():
    env = WumpusEnv(width=4, height=4, seed=12)
    env.reset(seed=12)
    _set_map(env, pits=set(), wumpus=(3, 3), gold=(3, 0), agent=(0, 0))
    obs, reward, term, trunc, _ = env.step(UP)
    assert reward == env.step_penalty and not term and not trunc


def test_truncation_after_max_steps():
    env = WumpusEnv(width=5, height=5, max_steps=3, seed=13)
    env.reset(seed=13)
    _set_map(env, pits=set(), wumpus=(4, 4), gold=(4, 0), agent=(0, 0))
    seq = [RIGHT, LEFT, RIGHT]  # 3 safe moves, none terminal
    results = [env.step(a) for a in seq]
    # first two steps: still running
    assert not results[0][2] and not results[0][3]
    assert not results[1][2] and not results[1][3]
    # third step hits the limit -> truncated, not terminated
    _, reward, term, trunc, _ = results[2]
    assert trunc and not term and reward == env.step_penalty


# --------------------------------------------------------------------------- #
# map generation, pools & seeding                                              #
# --------------------------------------------------------------------------- #
def test_generated_maps_are_always_solvable():
    env = WumpusEnv(width=4, height=4, num_pits=2, num_training_maps="inf", seed=14)
    for i in range(300):
        env.reset(seed=1000 + i)
        pits, wumpus, gold = env._pits, env._wumpus, env._gold
        assert (0, 0) not in pits and (0, 0) != wumpus and (0, 0) != gold
        assert len(pits) == 2
        assert _independent_solvable(4, 4, pits, wumpus, gold)


def test_single_map_pool_is_stable_across_resets():
    env = WumpusEnv(num_training_maps=1, seed=15)
    layouts = set()
    for _ in range(10):
        env.reset()
        layouts.add((frozenset(env._pits), env._wumpus, env._gold))
    assert len(layouts) == 1  # always the same single map


def test_same_seed_reproduces_the_same_pool():
    a = WumpusEnv(num_training_maps=5, seed=42)
    b = WumpusEnv(num_training_maps=5, seed=42)
    assert a._map_pool == b._map_pool


def test_map_id_option_selects_pool_entry():
    env = WumpusEnv(num_training_maps=4, seed=16)
    for map_id in range(4):
        _, info = env.reset(options={"map_id": map_id})
        assert info["map_id"] == map_id
        expected = env._map_pool[map_id]
        assert (env._pits, env._wumpus, env._gold) == (set(expected[0]),) + expected[1:]


def test_infinite_mode_rejects_map_id():
    env = WumpusEnv(num_training_maps="inf", seed=17)
    try:
        env.reset(options={"map_id": 0})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for map_id in infinite mode")


# --------------------------------------------------------------------------- #
# validation                                                                   #
# --------------------------------------------------------------------------- #
def test_invalid_action_raises():
    env = WumpusEnv(seed=18)
    env.reset()
    for bad in (-1, 4, 99):
        try:
            env.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for action {bad}")


def test_invalid_configuration_raises():
    for kwargs in (
        {"width": 1},                       # grid too small
        {"num_pits": 20},                   # more hazards than cells
        {"num_training_maps": 0},           # invalid pool size
        {"num_training_maps": "banana"},    # invalid string
        {"max_steps": 0},                   # non-positive horizon
    ):
        try:
            WumpusEnv(seed=19, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


# --------------------------------------------------------------------------- #
# plain-python runner (used when pytest is unavailable)                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)
