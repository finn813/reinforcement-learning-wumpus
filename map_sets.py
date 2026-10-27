"""Fixed map sets shared by all agents and experiments.

Training maps
    The master pool holds the first 1,000 *distinct* solvable layouts drawn by the
    environment's generator with seed 42 (``WumpusEnv(..., seed=42)``). A training
    pool of size N is the first N maps of the master pool, so the pools are nested
    (1 in 10 in 100 in 1,000). For N <= 100 this is exactly
    ``WumpusEnv(num_training_maps=N, seed=42)._map_pool``; the plain 1,000-map pool
    of the environment contains duplicates (only 968 distinct maps), which is why
    the master pool skips repeated layouts.

Test maps
    Layouts drawn by the generator with seed 12345, skipping repeated layouts and
    every layout of the master pool, until the requested number is reached. Test
    maps are therefore distinct and never occur in any training pool.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from environment import WumpusEnv

TRAINING_SEED = 42
TEST_SEED = 12345
MASTER_POOL_SIZE = 1000
NUM_TEST_MAPS = 1000


def layout_key(layout) -> Tuple:
    """Hashable identity of a map layout ``(pits, wumpus, gold)``."""
    pits, wumpus, gold = layout
    return (frozenset(map(tuple, pits)), tuple(wumpus), tuple(gold))


def _distinct_layouts(seed: int, count: int, exclude=frozenset()) -> Tuple:
    """First ``count`` distinct layouts of the generator seeded with ``seed``.

    Uses the environment's own generator in the same order as
    ``WumpusEnv(num_training_maps=n, seed=seed)`` builds its pool.
    """
    generator = WumpusEnv(num_training_maps="inf", seed=seed)
    seen, layouts = set(exclude), []
    while len(layouts) < count:
        layout = generator._generate_map()
        key = layout_key(layout)
        if key not in seen:
            seen.add(key)
            layouts.append(layout)
    return tuple(layouts)


@lru_cache(maxsize=None)
def master_pool() -> Tuple:
    return _distinct_layouts(TRAINING_SEED, MASTER_POOL_SIZE)


def training_pool(size: int) -> List:
    """The first ``size`` maps of the master pool."""
    if not 1 <= size <= MASTER_POOL_SIZE:
        raise ValueError(f"training pool size must be in [1, {MASTER_POOL_SIZE}]")
    return [(set(p), w, g) for p, w, g in master_pool()[:size]]


@lru_cache(maxsize=None)
def _test_layouts(count: int, seed: int) -> Tuple:
    exclude = frozenset(layout_key(m) for m in master_pool())
    return _distinct_layouts(seed, count, exclude)


def test_maps(count: int = NUM_TEST_MAPS, seed: int = TEST_SEED) -> List:
    """``count`` distinct test maps that never occur in the master training pool."""
    return [(set(p), w, g) for p, w, g in _test_layouts(count, seed)]


def make_env(maps, seed: int = 0, **env_kwargs) -> WumpusEnv:
    """Environment whose fixed pool is exactly ``maps``."""
    env = WumpusEnv(num_training_maps=len(maps), seed=seed, **env_kwargs)
    env._map_pool = [(set(p), w, g) for p, w, g in maps]
    return env


def training_env(size: int, **env_kwargs) -> WumpusEnv:
    """Training environment for a pool of ``size`` maps.

    Built like the notebooks (``WumpusEnv(num_training_maps=size, seed=42)``) so map
    sampling during training is unchanged; only the pool contents are replaced by the
    de-duplicated master pool (a no-op for size <= 100).
    """
    env = WumpusEnv(num_training_maps=size, seed=TRAINING_SEED, **env_kwargs)
    env._map_pool = training_pool(size)
    return env
