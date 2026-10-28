## DRQN trained on 100 maps (mean over seeds 0-2)

| Test maps | Maps | Win | Death | Timeout |
|---|---|---|---|---|
| Breeze/stench at the start (no risk-free move exists) | 356 | 13.1% | 52.6% | 34.3% |
| Clean start, KB agent wins (winnable without risk) | 451 | 65.3% | 10.8% | 23.9% |
| Clean start, KB agent times out (a risk is needed later) | 193 | 14.0% | 34.9% | 51.1% |

| Cause of death | Share of deaths |
|---|---|
| first move out of a start with breeze/stench | 54.6% |
| walked into a provable hazard | 21.8% |
| risk taken although provably safe cells were still unvisited | 13.2% |
| risk taken after all provably safe cells were used up | 10.4% |

| Timeouts on risk-free maps | Share |
|---|---|
| stopped exploring: looped or bumped although safe cells were left | 83.3% |
| still exploring when the 20 steps ran out | 16.7% |

Gold on a provably safe, unvisited cell in 50.8% of these timeouts; 2.3 such cells left on average.

## DRQN trained on 1,000 maps (mean over seeds 0-2)

| Test maps | Maps | Win | Death | Timeout |
|---|---|---|---|---|
| Breeze/stench at the start (no risk-free move exists) | 356 | 0.0% | 0.0% | 100.0% |
| Clean start, KB agent wins (winnable without risk) | 451 | 59.8% | 4.7% | 35.6% |
| Clean start, KB agent times out (a risk is needed later) | 193 | 6.9% | 12.6% | 80.5% |

| Cause of death | Share of deaths |
|---|---|
| walked into a provable hazard | 64.0% |
| risk taken after all provably safe cells were used up | 26.5% |
| risk taken although provably safe cells were still unvisited | 9.6% |

| Timeouts on risk-free maps | Share |
|---|---|
| stopped exploring: looped or bumped although safe cells were left | 75.7% |
| still exploring when the 20 steps ran out | 24.3% |

Gold on a provably safe, unvisited cell in 42.8% of these timeouts; 2.2 such cells left on average.
