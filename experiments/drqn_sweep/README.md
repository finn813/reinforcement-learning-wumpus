# DRQN: n_episodes x target_update_period sweep, and why the DRQN loses winnable maps

Training is identical to `dqrn_agent.ipynb` on the `comparison-tests` branch: 100 training maps
(`training_seed = 42`), Adam lr 1e-3, epsilon 1.0 -> 0.01 decayed linearly over the first half of
training. Only `n_episodes` and `target_update_period` (p) change. Three seeds per setting (seed =
torch / numpy / random seed of the agent; the training maps are the same for every run).

Evaluation uses `evaluation.py` (greedy, one attempt per map): the 100 training maps, the shared
100-map benchmark (`test_seed = 12345`) and, for less noise, 1,000 test maps with the same seed
(the benchmark's 100 maps are the first 100 of these). 5 of the 100 training maps also appear
among the 1,000 test maps.

## Results (mean over seeds, individual seeds in brackets)

| n_episodes | target_update_period | target syncs | seeds | Win, training maps (100) | Win, test maps (1,000) | Death, test | Timeout, test | Avg. reward, test | Win, benchmark test maps (100) | Minutes |
|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 100 | ~1,079 | 3 | 56.0% (53.0% / 57.0% / 58.0%) | 37.3% (37.9% / 35.9% / 38.0%) | 30.0% (30.6% / 23.5% / 35.9%) | 32.7% (31.5% / 40.6% / 26.1%) | -2.7 (-2.7 / 1.5 / -7.0) | 35.7% (39.0% / 33.0% / 35.0%) | 7.8 |
| 10,000 | 500 | ~226 | 3 | 40.7% (32.0% / 40.0% / 50.0%) | 29.8% (26.1% / 32.2% / 31.0%) | 21.0% (14.5% / 16.8% / 31.6%) | 49.3% (59.4% / 51.0% / 37.4%) | -4.2 (-2.6 / 1.7 / -11.8) | 30.7% (26.0% / 35.0% / 31.0%) | 7.2 |
| 10,000 | 1000 | ~112 | 3 | 48.0% (49.0% / 47.0% / 48.0%) | 32.8% (34.9% / 32.4% / 31.2%) | 18.8% (15.4% / 22.6% / 18.5%) | 48.3% (49.7% / 45.0% / 50.3%) | 1.7 (6.7 / -2.2 / 0.6) | 32.3% (31.0% / 31.0% / 35.0%) | 7.1 |
| 20,000 | 100 | ~2,019 | 3 | 59.0% (59.0% / 55.0% / 63.0%) | 36.9% (36.6% / 36.5% / 37.5%) | 26.7% (30.5% / 24.1% / 25.6%) | 36.4% (32.9% / 39.4% / 36.9%) | -0.3 (-3.7 / 1.5 / 1.2) | 35.0% (34.0% / 35.0% / 36.0%) | 12.8 |
| 20,000 | 500 | ~424 | 3 | 49.0% (55.0% / 47.0% / 45.0%) | 33.9% (35.5% / 32.8% / 33.5%) | 17.7% (32.7% / 9.7% / 10.7%) | 48.4% (31.8% / 57.5% / 55.8%) | 3.8 (-6.8 / 9.1 / 9.1) | 30.7% (34.0% / 30.0% / 28.0%) | 13.8 |
| 20,000 | 1000 | ~214 | 3 | 47.3% (43.0% / 53.0% / 46.0%) | 34.1% (32.5% / 36.6% / 33.1%) | 16.5% (10.6% / 22.8% / 16.2%) | 49.4% (56.9% / 40.6% / 50.7%) | 5.1 (8.2 / 2.7 / 4.5) | 31.0% (30.0% / 31.0% / 32.0%) | 14.8 |

* p barely matters; none of the settings beats p = 100 on unseen maps (all between 30 % and 37 %).
* The training reward keeps rising mainly because epsilon is still decaying (it reaches 0.01 at
  n/2); the greedy win rate on unseen maps ends between 30 % and 37 % and is flat over the last
  third of the 20k runs.
* The gap between training maps (41-59 %) and unseen maps (30-37 %) is memorisation of the 100
  training maps.
* Larger p has fewer deaths mainly because those agents gamble less at a start with breeze/stench
  (see `diagnosis.md`), not because they avoid hazards better.

## Diagnosis (`diagnose.py`, `analyze_diag.py` -> `diagnosis.md`)

Every trained model is replayed on the 1,000 test maps while a KB agent's knowledge base is fed
exactly the percepts the DRQN sees. Each DRQN move is labelled as provably safe, revisit, wall
bump, risk (cell not provably safe) or walk into a provable hazard. Maps are grouped into
A) breeze/stench at the start (35 %, no risk-free move exists), B) clean start and the KB agent
wins (46 %, winnable without risk), C) clean start but the KB agent times out (19 %).

## Number of training maps (`maps_analysis.py` -> `maps_table.md`, `plots/training_maps.png`)

10k episodes, p = 100, 3 seeds each for 10 / 100 / 500 / 1,000 training maps and a new random map
every episode (pools are nested: the first 100 maps of the 500-pool are the 100-pool; pools are
drawn with replacement, so 500 and 1,000 contain 486 and 968 distinct maps).

* Win rate on unseen maps peaks at 100 maps (37 %); 500 / 1,000 / new map every episode reach 32-34 %.
* Deaths fall steadily with more maps (49 % -> 30 % -> 19-21 % -> 11 %) and the average reward on
  unseen maps rises (-33 -> -3 -> about 0 -> +7). By the reward, which Q-learning optimises,
  a new map every episode is best.
* The extra wins at 100 maps come with much bolder play: on clean starts 18 % deaths vs. 4-8 %.
* It is not memorisation of the test maps: win rates on test maps never seen in training are the same.
* Memorisation of the training pool is clear at 10 and 100 maps (83 % / 56 % on own maps) and gone
  from 500 maps on (own maps = unseen maps).

## Gold reward +150 during training (`reward_comparison.md`, `plots/reward_comparison.png`)

Same map-count experiment with the gold reward raised from +100 to +150 during training only
(`sweep.py --goal-reward 150`; pit/Wumpus -100 and step -1 unchanged). All test numbers are scored
with the standard +100 so they are comparable (`maps_table_g150.md` has the full table).

* No clear effect: win rates on unseen maps change by at most 2 points (e.g. 100 maps 37.3 % -> 36.3 %,
  new map every episode 32.4 % -> 33.9 %), deaths by at most 7 points, all within the seed spread.
* The average reward on unseen maps is a few points higher for 1,000 maps and a new map every
  episode (-0.4 -> 4.0, 7.1 -> 12.4), but the seed ranges overlap.
* A bigger reward for the gold does not make the agent find it more often: the bottleneck is
  finding the gold (exploration and memory), not the willingness to go for it.

## Files

* `plots/learning_curves.png` greedy win rate on training / unseen maps during training
* `plots/training_rewards.png` training episode reward (the notebook's left panel) per setting
* `plots/outcomes.png` final win / death / timeout shares
* `plots/training_maps.png` effect of the number of training maps (`training_maps_g150.png` for +150)
* `plots/reward_comparison.png` gold reward +100 vs +150
* `plots/examples_timeouts.png`, `plots/examples_deaths.png` example maps with DRQN and KB paths
* `plots/<run>_training.png` the notebook's three panels for every single run
* `runs/` per-run results (json), trained weights (pt), diagnosis per model (diag_*.json)

## Reproduce

```bash
export PYTHONPATH=../..                                   # repo root: environment, agents, evaluation
python sweep.py --episodes 10000 --period 500 --seed 0   # one run (jobs*.txt list all runs)
python summarize_sweep.py                                 # results_table.md + summary plots
python diagnose.py && python analyze_diag.py              # diagnosis.md + example maps
python maps_analysis.py                                   # maps / reward tables (needs jobs3/jobs4 runs + diagnose)
```
