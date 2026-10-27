## n_episodes = 10k, target_update_period = 100  (3 seeds, mean over seeds)

| Test maps | Maps | DRQN win | DRQN death | DRQN timeout | KB agent win |
|---|---|---|---|---|---|
| Breeze/stench at the start (no risk-free move exists) | 350 | 13.6% | 52.8% | 33.6% | 0.0% |
| Clean start, KB agent wins (winnable without risk) | 459 | 65.3% | 10.5% | 24.2% | 100.0% |
| Clean start, KB agent times out (a risk is needed later) | 191 | 13.3% | 35.1% | 51.7% | 0.0% |
| **All** | 1000 | 37.3% | 30.0% | 32.7% | 45.9% |

| Cause of death (fatal move) | Share of deaths | Per 1,000 maps |
|---|---|---|
| first move out of a start with breeze/stench | 54.8% | 164 |
| risk taken after all provably safe cells were used up | 10.7% | 32 |
| risk taken although provably safe cells were still unvisited | 13.3% | 40 |
| walked into a provable hazard | 21.2% | 64 |

| Timeouts on maps the KB agent wins | Share | Per 1,000 maps |
|---|---|---|
| stopped exploring: looped or bumped although safe cells were left | 83.8% | 93 |
| still exploring when the 20 steps ran out | 16.2% | 18 |
| never left the start | 0.0% | 0 |
| no provably safe cell left | 0.0% | 0 |

In 52.0% of these timeouts the gold lay on a cell the DRQN's own percepts had already proven safe; on average 2.3 provably safe cells were left unvisited.

On maps with breeze/stench at the start the DRQN leaves the start in 87.9% of the episodes, after bumping into the wall for a median of 2 steps (min 1, max 12).

## n_episodes = 10k, target_update_period = 500  (3 seeds, mean over seeds)

| Test maps | Maps | DRQN win | DRQN death | DRQN timeout | KB agent win |
|---|---|---|---|---|---|
| Breeze/stench at the start (no risk-free move exists) | 350 | 7.2% | 36.6% | 56.2% | 0.0% |
| Clean start, KB agent wins (winnable without risk) | 459 | 56.4% | 5.2% | 38.4% | 100.0% |
| Clean start, KB agent times out (a risk is needed later) | 191 | 7.2% | 30.2% | 62.7% | 0.0% |
| **All** | 1000 | 29.8% | 21.0% | 49.3% | 45.9% |

| Cause of death (fatal move) | Share of deaths | Per 1,000 maps |
|---|---|---|
| first move out of a start with breeze/stench | 59.0% | 124 |
| risk taken after all provably safe cells were used up | 9.4% | 20 |
| risk taken although provably safe cells were still unvisited | 16.1% | 34 |
| walked into a provable hazard | 15.6% | 33 |

| Timeouts on maps the KB agent wins | Share | Per 1,000 maps |
|---|---|---|
| stopped exploring: looped or bumped although safe cells were left | 76.2% | 134 |
| still exploring when the 20 steps ran out | 23.8% | 42 |
| never left the start | 0.0% | 0 |
| no provably safe cell left | 0.0% | 0 |

In 49.9% of these timeouts the gold lay on a cell the DRQN's own percepts had already proven safe; on average 2.6 provably safe cells were left unvisited.

On maps with breeze/stench at the start the DRQN leaves the start in 66.7% of the episodes, after bumping into the wall for a median of 6 steps (min 2, max 17).

## n_episodes = 10k, target_update_period = 1000  (3 seeds, mean over seeds)

| Test maps | Maps | DRQN win | DRQN death | DRQN timeout | KB agent win |
|---|---|---|---|---|---|
| Breeze/stench at the start (no risk-free move exists) | 350 | 8.8% | 37.9% | 53.3% | 0.0% |
| Clean start, KB agent wins (winnable without risk) | 459 | 62.4% | 6.7% | 30.9% | 100.0% |
| Clean start, KB agent times out (a risk is needed later) | 191 | 5.9% | 13.1% | 81.0% | 0.0% |
| **All** | 1000 | 32.8% | 18.8% | 48.3% | 45.9% |

| Cause of death (fatal move) | Share of deaths | Per 1,000 maps |
|---|---|---|
| first move out of a start with breeze/stench | 64.8% | 122 |
| risk taken after all provably safe cells were used up | 4.4% | 8 |
| risk taken although provably safe cells were still unvisited | 16.3% | 31 |
| walked into a provable hazard | 14.5% | 27 |

| Timeouts on maps the KB agent wins | Share | Per 1,000 maps |
|---|---|---|
| stopped exploring: looped or bumped although safe cells were left | 70.7% | 100 |
| still exploring when the 20 steps ran out | 29.3% | 42 |
| never left the start | 0.0% | 0 |
| no provably safe cell left | 0.0% | 0 |

In 48.1% of these timeouts the gold lay on a cell the DRQN's own percepts had already proven safe; on average 2.1 provably safe cells were left unvisited.

On maps with breeze/stench at the start the DRQN leaves the start in 63.7% of the episodes, after bumping into the wall for a median of 2 steps (min 1, max 7).

## n_episodes = 20k, target_update_period = 100  (3 seeds, mean over seeds)

| Test maps | Maps | DRQN win | DRQN death | DRQN timeout | KB agent win |
|---|---|---|---|---|---|
| Breeze/stench at the start (no risk-free move exists) | 350 | 9.9% | 39.7% | 50.4% | 0.0% |
| Clean start, KB agent wins (winnable without risk) | 459 | 66.5% | 12.7% | 20.8% | 100.0% |
| Clean start, KB agent times out (a risk is needed later) | 191 | 15.0% | 36.6% | 48.3% | 0.0% |
| **All** | 1000 | 36.9% | 26.7% | 36.4% | 45.9% |

| Cause of death (fatal move) | Share of deaths | Per 1,000 maps |
|---|---|---|
| first move out of a start with breeze/stench | 45.6% | 122 |
| risk taken after all provably safe cells were used up | 11.3% | 30 |
| risk taken although provably safe cells were still unvisited | 17.7% | 47 |
| walked into a provable hazard | 25.3% | 68 |

| Timeouts on maps the KB agent wins | Share | Per 1,000 maps |
|---|---|---|
| stopped exploring: looped or bumped although safe cells were left | 84.3% | 80 |
| still exploring when the 20 steps ran out | 15.7% | 15 |
| never left the start | 0.0% | 0 |
| no provably safe cell left | 0.0% | 0 |

In 52.1% of these timeouts the gold lay on a cell the DRQN's own percepts had already proven safe; on average 2.3 provably safe cells were left unvisited.

On maps with breeze/stench at the start the DRQN leaves the start in 63.7% of the episodes, after bumping into the wall for a median of 3 steps (min 2, max 3).

## n_episodes = 20k, target_update_period = 500  (3 seeds, mean over seeds)

| Test maps | Maps | DRQN win | DRQN death | DRQN timeout | KB agent win |
|---|---|---|---|---|---|
| Breeze/stench at the start (no risk-free move exists) | 350 | 4.5% | 19.2% | 76.3% | 0.0% |
| Clean start, KB agent wins (winnable without risk) | 459 | 64.3% | 9.7% | 26.0% | 100.0% |
| Clean start, KB agent times out (a risk is needed later) | 191 | 15.0% | 34.0% | 51.0% | 0.0% |
| **All** | 1000 | 33.9% | 17.7% | 48.4% | 45.9% |

| Cause of death (fatal move) | Share of deaths | Per 1,000 maps |
|---|---|---|
| first move out of a start with breeze/stench | 35.2% | 62 |
| risk taken after all provably safe cells were used up | 14.7% | 26 |
| risk taken although provably safe cells were still unvisited | 31.3% | 55 |
| walked into a provable hazard | 18.8% | 33 |

| Timeouts on maps the KB agent wins | Share | Per 1,000 maps |
|---|---|---|
| stopped exploring: looped or bumped although safe cells were left | 81.8% | 98 |
| still exploring when the 20 steps ran out | 18.2% | 22 |
| never left the start | 0.0% | 0 |
| no provably safe cell left | 0.0% | 0 |

In 50.0% of these timeouts the gold lay on a cell the DRQN's own percepts had already proven safe; on average 2.0 provably safe cells were left unvisited.

On maps with breeze/stench at the start the DRQN leaves the start in 33.3% of the episodes, after bumping into the wall for a median of 2 steps (min 1, max 2).

## n_episodes = 20k, target_update_period = 1000  (3 seeds, mean over seeds)

| Test maps | Maps | DRQN win | DRQN death | DRQN timeout | KB agent win |
|---|---|---|---|---|---|
| Breeze/stench at the start (no risk-free move exists) | 350 | 4.4% | 18.9% | 76.8% | 0.0% |
| Clean start, KB agent wins (winnable without risk) | 459 | 65.3% | 9.0% | 25.7% | 100.0% |
| Clean start, KB agent times out (a risk is needed later) | 191 | 13.4% | 30.4% | 56.2% | 0.0% |
| **All** | 1000 | 34.1% | 16.5% | 49.4% | 45.9% |

| Cause of death (fatal move) | Share of deaths | Per 1,000 maps |
|---|---|---|
| first move out of a start with breeze/stench | 37.1% | 61 |
| risk taken after all provably safe cells were used up | 16.1% | 27 |
| risk taken although provably safe cells were still unvisited | 19.8% | 33 |
| walked into a provable hazard | 27.0% | 45 |

| Timeouts on maps the KB agent wins | Share | Per 1,000 maps |
|---|---|---|
| stopped exploring: looped or bumped although safe cells were left | 88.1% | 104 |
| still exploring when the 20 steps ran out | 11.9% | 14 |
| never left the start | 0.0% | 0 |
| no provably safe cell left | 0.0% | 0 |

In 55.4% of these timeouts the gold lay on a cell the DRQN's own percepts had already proven safe; on average 2.3 provably safe cells were left unvisited.

On maps with breeze/stench at the start the DRQN leaves the start in 33.3% of the episodes, after bumping into the wall for a median of 4 steps (min 2, max 4).

