# wumpus_env

A clean, self-contained simplified Wumpus World for reinforcement-learning
experiments. Only `numpy` is required; `gymnasium` is optional — when installed,
`WumpusEnv` is a real `gymnasium.Env` (wrappers/registration work), otherwise it
still runs and follows the same `reset`/`step` API.

## Usage

```python
from wumpus_env import WumpusEnv

env = WumpusEnv(width=4, height=4, num_training_maps=1, seed=0)
obs, info = env.reset(seed=0)          # obs = np.array([x, y, breeze, stench])

for _ in range(100):
    action = env.action_space.sample()  # 0=up, 1=left, 2=down, 3=right
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

For tabular methods, make the observation hashable with `tuple(obs)`.

## Train a Q-learning agent

`agent.py` adapts Gymnasium's introductory tabular Q-learning agent to this
environment. By default it trains against the single fixed map, evaluates a
greedy policy, and saves the learned Q-table:

```bash
python agent.py --episodes 1000 --evaluation-episodes 1000
```

Evaluation reports both the average number of steps across all episodes and the
average for successful episodes. Compare those together with goal, hazard, and
truncation rates: average steps alone can misleadingly reward an agent that
fails quickly. Use the same maps, seeds, step limit, and evaluation episode
count when comparing tabular Q-learning with DQN or recurrent DQN agents.

Useful options include `--seed`, `--maps`, and `--output`. A saved policy can be
loaded with `QLearningAgent.load("wumpus_q_table.pkl")`.

The fixed-map default is intentional. The observation contains position and
local percepts, but not the map identity or observation history. With multiple
maps, the same observation may require different actions, so ordinary tabular
Q-learning learns a compromise rather than a policy that remembers or infers a
map. Generalisation across changing maps needs a richer observation/state (for
example map identity during training or a history/belief state) and commonly a
function approximator or recurrent agent.

## Specification

**Observation** — `float32` vector `[x, y, breeze, stench]`. `x, y` are raw
coordinates; `breeze`/`stench` are `1.0` when a pit / the wumpus is orthogonally
adjacent. The agent perceives only its current cell (no global map, no memory).

**Actions** — `Discrete(4)`: `0=UP (y+1)`, `1=LEFT (x-1)`, `2=DOWN (y-1)`,
`3=RIGHT (x+1)`. Moving into a wall leaves the agent in place but still costs a step.

**Rewards** — gold `+100` (terminated), pit/wumpus `-100` (terminated), any other
step `-1`. Surviving `max_steps` (default 20) without reaching gold ends the
episode as `truncated` (not `terminated`) with the ordinary step penalty. All
reward values are constructor parameters.

**Maps** (`num_training_maps`) — `1` (default): one fixed map reused every
episode; `N > 1`: a fixed pool of N maps sampled each reset; `None`/`"inf"`: a
fresh random map each reset. Every generated map is guaranteed solvable (BFS
from start to gold avoiding hazards), and generation is driven by one seeded RNG
for reproducibility. With a finite pool, `reset(options={"map_id": k})` selects a
specific map.

## Tests

```bash
pytest wumpus_env/test_environment.py     # with pytest
python wumpus_env/test_environment.py      # plain python, no pytest needed
```

Covers spaces, percepts (including that diagonals are ignored), movement and wall
clamping, all reward/termination paths, truncation, map solvability (cross-checked
with an independent BFS), pool stability, seed reproducibility, and input validation.
