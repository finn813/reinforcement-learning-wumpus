# Final experiments

Five agents on the same 1,000 unseen test maps:

| Agent | Trained? | Implementation |
|---|---|---|
| Random agent | no | `RandomAgent` in `pipeline.py` (uniform actions, own seeded generator) |
| Tabular Q-learning | yes | `q_agent.py` |
| DQN | yes | `deep_q_agent.py` |
| DRQN | yes | `dqrn_agent.py` |
| Knowledge-based agent | no | `kb_agent.py` |

## Design

* **Training pools:** 1, 10, 100 and 1,000 maps.
* **Seeds:** agent seeds 0, 1, 2, so 3 learning agents × 4 pool sizes × 3 seeds = **36 training runs**.
* **Training length:** 10,000 episodes each.
* **Map sets** (`map_sets.py` in the repository root):
  * **Master training pool:** the first 1,000 *distinct* layouts drawn by the environment's generator with seed 42. A pool of size N is its first N maps, so the pools are nested. For N = 1, 10, 100 this is exactly the old `WumpusEnv(num_training_maps=N, seed=42)` pool; the old 1,000-map pool contained only 968 distinct maps.
  * **Test set:** 1,000 distinct layouts drawn with seed 12345, skipping every layout of the master pool. No test map occurs in any training pool.
* **Evaluation:** greedy (epsilon = 0, policy seed 54321 as in `evaluation.py`) on every map of the agent's own training pool and on all 1,000 test maps.
  * **Recorded:** win, death and timeout rate, plus mean return (standard rewards: +100 gold, -100 pit/Wumpus, -1 per step, 20 steps).
  * **Generalisation gap:** training-map win rate minus test-map win rate.
* **Aggregation:** mean ± *sample* standard deviation over the three seeds. Every per-seed value is kept in the result files.
* **Baselines:** the random agent is evaluated with policy seeds 0, 1, 2; the knowledge-based agent is deterministic. Both are evaluated on the test maps only.

## Hyperparameters (unchanged, taken from the notebooks)

All agents use epsilon 1.0 -> 0.01, decayed linearly over the first 5,000 episodes, and gamma = 0.95.

| Agent | Settings |
|---|---|
| Tabular Q-learning | learning rate 0.01 |
| DQN | Adam lr 1e-3, batch 64, replay buffer 10,000 transitions, target update every 100 updates, MSE loss, raw rewards |
| DRQN | Adam lr 1e-3, episode replay (1,000 episodes, batch 8, 20 warm-up episodes), target update every 100 updates, Huber loss, rewards / 100, inputs scaled to [0, 1] |

The DQN and DRQN implementations differ in more than the recurrent layer (loss, reward and input scaling, replay unit). They are compared as two implementations, not as a controlled ablation of memory.

## Changes to the existing code

* **`map_sets.py` (new):** shared training pools and test set (see Design).
* **`evaluation.py`:**
  * test maps now come from `map_sets.test_maps` instead of `WumpusEnv(num_training_maps=n, seed=test_seed)`, so the old test maps that also occurred in training are gone;
  * agents without `epsilon` (random, knowledge-based) are supported;
  * agents with a `reset()` method (the knowledge-based agent) are reset before every map. Before this change the KB agent would have kept its knowledge base from one map to the next.

  The 100-map benchmark used in the notebooks is now the first 100 of the new test maps.
* **`test_map_sets.py` (new):** tests for the map sets and the reset hook. All existing tests still pass.
* **Seeding:** every training run seeds Python's `random`, NumPy, PyTorch and the environment's action space (`env.action_space.seed`), which is where random exploration actions come from. Runs are reproducible; running the same configuration twice gives identical training returns.
* No changes to the environment, the rewards or the agents.

## Reused runs

The DRQN runs for pools 10 and 100 (seeds 0–2) are reused from `experiments/drqn_sweep` (`n10k_p100_maps{10,100}_seed{0,1,2}`).

* **Same training setup:** same training pool, notebook hyperparameters, 10,000 episodes, and the same seeding of `random`, NumPy and PyTorch.
* **One difference:** those runs did not seed the action space, so they cannot be reproduced bit by bit.
* **Re-evaluated:** their saved weights were evaluated again with the protocol above.

Everything else was trained here:

* DRQN on pools 1 and 1,000 (the 1,000 pool changed, because it is now de-duplicated);
* all Q-learning and DQN runs.

## Results (`results/summary.md`)

Test-map results, mean over 3 seeds.

| Agent | Best win rate (pool) | Best mean return (pool) |
|---|---|---|
| Knowledge-based agent | 45.1% (no training) | 31.8 |
| DRQN | 36.8% (100 maps) | 8.2 (1,000 maps) |
| Random agent | 24.9% | -49.4 |
| Tabular Q-learning | 20.6% (1 map) | -3.8 (1,000 maps) |
| DQN | 14.4% (10 maps) | -7.4 (100 maps) |

* **Win rate:** Q-learning and DQN stay below the random agent at every pool size. DRQN is the only learner above it, and only with 100 or 1,000 maps.
* **Mean return:** every learner beats the random agent from 10 maps on (Q-learning with 1 map does not).
* **Memorisation:**
  * With 1 map, every learner wins its training map on every seed but only 4–21% of unseen maps.
  * With 10 maps, they win 33–83% of their training maps but only 14–22% of unseen maps.
* **Generalisation gap:** it shrinks with the pool size and is about 0 at 1,000 maps for all three learners.
* **Pool size 1:** Q-learning and DQN learn the same fixed route on their single map for every seed. Their test results are therefore identical across seeds (standard deviation 0), although the networks/tables differ.
* **Comparison with `experiments/drqn_sweep`:** numbers differ slightly, because the test maps and the 1,000-map pool changed.

## Failure analysis (`failure_analysis.py`, `results/failure_analysis.md`)

The DRQN models trained on 100 and 1,000 maps are replayed on the test maps while a knowledge-based agent receives the same percepts, so every DRQN move can be labelled as provably safe, a risk, or a step into a provable hazard.

* **Test-map groups:** 356 maps with a breeze or stench at the start (no risk-free first move), 451 maps the KB agent wins without risk, 193 maps that need a risk later.
* **DRQN, 100 maps, on the 451 risk-free maps:** win 65.3%, death 10.8%, timeout 23.9%.
  * Deaths (all maps): 54.6% on the first move out of a start with a percept, 21.8% into a provable hazard.
  * Timeouts on risk-free maps: 83.3% stop exploring although on average 2.3 provably safe cells are left; the gold lies on one of them in 50.8%.
* **DRQN, 1,000 maps:** times out on every map with a percept at the start; 64.0% of its deaths are steps into provable hazards.
* **Example maps:** `results/plots/failure_examples_timeouts.png`, `results/plots/failure_examples_deaths.png` (100 maps, seed 0).

## Checks

* The same configuration trained twice gives identical training returns (all three learners).
* One run (DQN, 10 maps, seed 1) was re-evaluated with an independent loop: its win rate and mean return match on both the training and the test maps.
* The aggregated means and standard deviations were re-computed by hand for one cell.

## Files

* `pipeline.py`: training + evaluation of one run (`train`), re-evaluation of reused DRQN models (`reuse-drqn`), baselines (`baselines`)
* `summarize.py`: result files, summary table, plots
* `failure_analysis.py`: shadow-KB diagnosis of the DRQN models (pools 100 and 1,000) and the example-map figures
* `jobs.txt`: the 30 training runs done here
* `runs/`: one JSON per run (metrics, hyperparameters, source, training returns and lengths) plus the trained model (`.pt` or `.pkl`)
* `results/q_learning.json`, `results/dqn.json`, `results/drqn.json`: all 12 runs of each agent plus mean/std per pool size
* `results/random.json`, `results/kb.json`: baselines
* `results/summary.md`, `results/summary.csv`: combined table
* `results/plots/`:
  * `test_win_rate.png`, `mean_return.png`, `train_vs_test_win_rate.png`, `generalisation_gap.png` (the four planned plots)
  * `test_outcomes.png`: win/death/timeout shares, which explain why win rate and return can disagree
  * `failure_examples_timeouts.png`, `failure_examples_deaths.png`: example episodes from the failure analysis
* `results/failure_analysis.json`, `results/failure_analysis.md`: failure-analysis numbers
* `runs/diag_drqn_pool*_seed*.json`, `runs/kb_test_episodes.json`: per-episode diagnosis and KB-agent paths on the test maps

## Reproduce (from the repository root)

```bash
xargs -P 2 -L 1 python final_experiments/pipeline.py < final_experiments/jobs.txt
python final_experiments/pipeline.py reuse-drqn experiments/drqn_sweep/runs
python final_experiments/pipeline.py baselines
python final_experiments/summarize.py
python final_experiments/failure_analysis.py
```

To retrain the reused DRQN runs instead:
`python final_experiments/pipeline.py train --agent drqn --pool 10 --seed 0` (and likewise for pool 100 and seeds 1, 2).
