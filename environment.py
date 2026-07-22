"""A clean, self-contained implementation of a simplified Wumpus World.

This environment is intended as a stable substrate for experimenting with
reinforcement-learning algorithms. It deliberately keeps the state small and the
dynamics deterministic (given a fixed map), so the learning code is the only
moving part.

--------------------------------------------------------------------------------
Grid & coordinates
--------------------------------------------------------------------------------
The world is a ``width x height`` grid. A cell is addressed as ``(x, y)`` where

    x -> column, increasing to the right, range [0, width - 1]
    y -> row,    increasing upward,      range [0, height - 1]

The agent always starts in the bottom-left corner ``(0, 0)``.

Each map contains exactly one gold cell (the goal), one wumpus, and
``num_pits`` pit cells. None of these ever occupies the start cell, and every map
is guaranteed to have a hazard-free path from the start to the gold.

--------------------------------------------------------------------------------
Observation  (what the agent perceives)
--------------------------------------------------------------------------------
A length-4 ``float32`` vector::

    [ x, y, breeze, stench ]

    x, y     agent's current coordinates (raw, un-normalised)
    breeze   1.0 if a pit is orthogonally adjacent, else 0.0
    stench   1.0 if the wumpus is orthogonally adjacent, else 0.0

The agent perceives only its current cell (there is no global map view and no
accumulated memory - that is the learner's job). For tabular methods the
observation can be made hashable with ``tuple(obs)``.

--------------------------------------------------------------------------------
Actions
--------------------------------------------------------------------------------
Discrete(4)::

    0 = UP     (y + 1)
    1 = LEFT   (x - 1)
    2 = DOWN   (y - 1)
    3 = RIGHT  (x + 1)

Moving into a wall leaves the agent in place (the move still counts as a step).

--------------------------------------------------------------------------------
Rewards & episode end
--------------------------------------------------------------------------------
    reaching the gold      -> ``goal_reward``    (default +100), terminated
    entering a pit/wumpus   -> ``hazard_penalty`` (default -100), terminated
    any other step          -> ``step_penalty``   (default   -1)

If the agent survives ``max_steps`` without reaching the gold, the episode is
``truncated`` (not ``terminated``) and the final step yields the ordinary
``step_penalty``. Truncation is kept distinct from death on purpose, so a
learner can treat "ran out of time" differently from "died".

--------------------------------------------------------------------------------
Map modes (``num_training_maps``)
--------------------------------------------------------------------------------
    1 (default)      one fixed map, reused every episode (good for tabular Q).
    N (int > 1)      a fixed pool of N maps; each ``reset`` samples one.
    None / "inf"     a fresh random map every ``reset`` (tests generalisation).

Everything is driven by a single seeded RNG, so runs are reproducible.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# gymnasium is an *optional* dependency. When it is installed the environment
# is a genuine ``gymnasium.Env`` (so gym wrappers, vectorisation and
# registration all work). When it is absent, the tiny fall-backs below keep the
# module importable and testable with numpy alone.
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium.spaces import Box, Discrete

    _BaseEnv = gym.Env
    _HAS_GYM = True
except ModuleNotFoundError:  # pragma: no cover - only hit without gymnasium
    _HAS_GYM = False
    _BaseEnv = object

    class Discrete:  # minimal stand-in for gymnasium.spaces.Discrete
        def __init__(self, n: int):
            self.n = int(n)
            self._rng = random.Random()

        def sample(self) -> int:
            return self._rng.randrange(self.n)

        def contains(self, x) -> bool:
            return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    class Box:  # minimal stand-in for gymnasium.spaces.Box (float boxes only)
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.dtype = dtype
            self.shape = self.low.shape if shape is None else tuple(shape)

        def contains(self, x) -> bool:
            arr = np.asarray(x, dtype=self.dtype)
            return (
                arr.shape == self.shape
                and bool(np.all(arr >= self.low))
                and bool(np.all(arr <= self.high))
            )


# Action constants and their (dx, dy) effect on an (x, y) position.
UP, LEFT, DOWN, RIGHT = 0, 1, 2, 3
_ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    UP: (0, 1),
    LEFT: (-1, 0),
    DOWN: (0, -1),
    RIGHT: (1, 0),
}

Position = Tuple[int, int]
# A generated map: (set of pit cells, wumpus cell, gold cell).
MapLayout = Tuple[Set[Position], Position, Position]

_MAX_MAP_ATTEMPTS = 1000  # guard against impossible configurations


class WumpusEnv(_BaseEnv):
    """Simplified Wumpus World following the gymnasium step/reset API."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        width: int = 4,
        height: int = 4,
        num_pits: int = 2,
        num_training_maps: Union[int, str, None] = 1,
        max_steps: int = 20,
        step_penalty: float = -1.0,
        goal_reward: float = 100.0,
        hazard_penalty: float = -100.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self._validate_config(width, height, num_pits, max_steps)

        self.width = width
        self.height = height
        self.num_pits = num_pits
        self.max_steps = max_steps
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.hazard_penalty = float(hazard_penalty)
        self.render_mode = render_mode

        # A single RNG governs map generation and map sampling -> reproducible.
        self.rng = random.Random(seed)

        self.action_space = Discrete(4)
        # Observation bounds: [x, y, breeze, stench].
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([width - 1, height - 1, 1, 1], dtype=np.float32)
        self.observation_space = Box(low, high, shape=(4,), dtype=np.float32)

        # Decide between a fixed pool of maps and an endless stream of maps.
        self._infinite_maps, self._num_training_maps = self._resolve_map_mode(
            num_training_maps
        )
        self._map_pool: List[MapLayout] = (
            [] if self._infinite_maps else self._build_map_pool()
        )

        # Runtime state (populated by reset()).
        self._agent_pos: Position = (0, 0)
        self._pits: Set[Position] = set()
        self._wumpus: Optional[Position] = None
        self._gold: Optional[Position] = None
        self._steps = 0
        self._current_map_id: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Configuration helpers                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_config(width: int, height: int, num_pits: int, max_steps: int) -> None:
        if width < 2 or height < 2:
            raise ValueError("width and height must both be >= 2.")
        if num_pits < 0:
            raise ValueError("num_pits must be >= 0.")
        # Need room for the pits, the wumpus and the gold on non-start cells.
        free_cells = width * height - 1
        if num_pits + 2 > free_cells:
            raise ValueError(
                f"not enough free cells: {num_pits} pits + wumpus + gold need "
                f"{num_pits + 2} cells but only {free_cells} are available."
            )
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

    @staticmethod
    def _resolve_map_mode(
        num_training_maps: Union[int, str, None]
    ) -> Tuple[bool, Optional[int]]:
        """Return (is_infinite, pool_size)."""
        if num_training_maps is None:
            return True, None
        if isinstance(num_training_maps, str):
            if num_training_maps.strip().lower() in {"inf", "infinite"}:
                return True, None
            raise ValueError("string num_training_maps must be 'inf' or 'infinite'.")
        if isinstance(num_training_maps, int) and num_training_maps >= 1:
            return False, num_training_maps
        raise ValueError("num_training_maps must be a positive int, 'inf', or None.")

    # ------------------------------------------------------------------ #
    # Map generation                                                     #
    # ------------------------------------------------------------------ #
    def _orthogonal_neighbors(self, pos: Position) -> Iterable[Position]:
        """Yield the in-bounds cells directly up/down/left/right of ``pos``."""
        x, y = pos
        for dx, dy in _ACTION_DELTAS.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                yield (nx, ny)

    def _is_solvable(self, pits: Set[Position], wumpus: Position, gold: Position) -> bool:
        """True if a hazard-free path exists from the start to the gold (BFS)."""
        blocked = set(pits) | {wumpus}
        start = (0, 0)
        if start in blocked or gold in blocked:
            return False

        seen = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur == gold:
                return True
            for nb in self._orthogonal_neighbors(cur):
                if nb not in blocked and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        return False

    def _generate_map(self) -> MapLayout:
        """Randomly place pits, wumpus and gold, retrying until solvable."""
        cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) != (0, 0)
        ]
        for _ in range(_MAX_MAP_ATTEMPTS):
            chosen = self.rng.sample(cells, self.num_pits + 2)
            pits = set(chosen[: self.num_pits])
            wumpus = chosen[self.num_pits]
            gold = chosen[self.num_pits + 1]
            if self._is_solvable(pits, wumpus, gold):
                return pits, wumpus, gold
        raise RuntimeError(
            "Failed to generate a solvable map; use fewer pits or a larger grid."
        )

    def _build_map_pool(self) -> List[MapLayout]:
        assert self._num_training_maps is not None
        return [self._generate_map() for _ in range(self._num_training_maps)]

    # ------------------------------------------------------------------ #
    # gymnasium API: reset / step / render                               #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode and return ``(observation, info)``."""
        if seed is not None:
            self.rng.seed(seed)
        options = options or {}

        self._steps = 0
        self._agent_pos = (0, 0)

        if self._infinite_maps:
            if "map_id" in options:
                raise ValueError("map_id is only valid with a finite map pool.")
            pits, wumpus, gold = self._generate_map()
            self._current_map_id = None
        else:
            map_id = options.get("map_id")
            if map_id is None:
                map_id = self.rng.randrange(len(self._map_pool))
            elif not (0 <= map_id < len(self._map_pool)):
                raise ValueError(
                    f"map_id must be in [0, {len(self._map_pool) - 1}]."
                )
            pits, wumpus, gold = self._map_pool[map_id]
            self._current_map_id = map_id

        # Copy so per-episode state never mutates the stored pool entry.
        self._pits = set(pits)
        self._wumpus = wumpus
        self._gold = gold

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """Apply ``action`` and return ``(obs, reward, terminated, truncated, info)``."""
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}; expected an int in 0..3.")

        dx, dy = _ACTION_DELTAS[int(action)]
        x, y = self._agent_pos
        # Walls clamp the agent in place instead of wrapping or erroring.
        x = min(max(x + dx, 0), self.width - 1)
        y = min(max(y + dy, 0), self.height - 1)
        self._agent_pos = (x, y)
        self._steps += 1

        terminated = False
        truncated = False

        if self._agent_pos == self._gold:
            reward = self.goal_reward
            terminated = True
        elif self._agent_pos in self._pits or self._agent_pos == self._wumpus:
            reward = self.hazard_penalty
            terminated = True
        else:
            reward = self.step_penalty
            if self._steps >= self.max_steps:
                truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """Render the grid. ``human`` prints it; ``ansi`` returns the string."""
        text = self._render_text()
        if self.render_mode == "ansi":
            return text
        print(text)
        print()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _compute_percepts(self) -> Tuple[bool, bool]:
        """Return ``(breeze, stench)`` for the agent's current cell."""
        neighbors = set(self._orthogonal_neighbors(self._agent_pos))
        breeze = any(pit in neighbors for pit in self._pits)
        stench = self._wumpus in neighbors
        return breeze, stench

    def _get_obs(self) -> np.ndarray:
        x, y = self._agent_pos
        breeze, stench = self._compute_percepts()
        return np.array([x, y, float(breeze), float(stench)], dtype=np.float32)

    def _get_info(self) -> dict:
        return {"map_id": self._current_map_id, "steps": self._steps}

    def _render_text(self) -> str:
        rows = []
        for y in range(self.height - 1, -1, -1):  # top row first
            cells = []
            for x in range(self.width):
                cell = (x, y)
                if cell == self._agent_pos:
                    cells.append("A")
                elif cell in self._pits:
                    cells.append("P")
                elif cell == self._wumpus:
                    cells.append("W")
                elif cell == self._gold:
                    cells.append("G")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return "\n".join(rows)

    @property
    def unwrapped(self) -> "WumpusEnv":
        # Lets caller code use env.unwrapped.width/height uniformly, with or
        # without gymnasium's wrapper machinery.
        return self
