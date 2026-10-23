# Simulation-first bimanual SO-101 benchmark

A reproducible robotics benchmark in Isaac Sim: **the left SO-101 picks a 40 mm
cube, transfers it to the right SO-101 in mid-air, and the right arm places it in
a basket.** Both arms retreat; an independent evaluator checks the physical task.
Repeated runs reset automatically between episodes.

The deterministic expert works under simulated gravity and contact. The completed
pilot has **12/12 physical task successes, 12/12 automated recording-QA passes and
3/3 selected physical action replay successes**. Human acceptance of the pilot
videos is pending. There is **no accepted training dataset or trained policy in
the reset pipeline**. See [current status and approval gates](docs/STATUS.md) and
[measured pilot results](docs/PILOT_RESULTS.md).

## Quick start

Use the existing workstation environments. Do not reinstall or upgrade the locked
simulation stack. Each terminal needs its own activation:

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
export LEISAAC_ASSETS_ROOT=/home/rynem8/robotics/leisaac/assets
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python scripts/so101.py --help
python scripts/so101.py scene --frames
```

`scene` opens the stationary task and a panel showing the actual overhead and
both wrist RGB feeds. It only settles the initial physics state; it starts no
expert, collection or task motion. Close Isaac Sim before launching another run.

Run the validated expert, without recording demonstrations:

```bash
python scripts/so101.py expert
```

GUI is the default. Expect `[EPISODE] nominal/1: passed` and
`[RESULT] passed_requested_stage`; verify `full_task_success` and the independent
evaluation in the saved report. The final view stays open until you close Isaac.
Use `--dry-run` to inspect a command, `--case yaw_minus_1deg` for a named scenario,
`--episodes 3` for reset regression, or explicit `--headless --exit-after` for
unattended evaluation. [Getting started](docs/GETTING_STARTED.md) explains expected
views, outputs, environment checks and troubleshooting.

Inspect the existing local pilot:

```bash
python scripts/so101.py dataset inspect \
  --quality-report outputs/protocol_v1_pilot_qa/quality.json \
  --ledger outputs/protocol_v1_pilot_run/attempts.jsonl
python scripts/so101.py dataset review \
  --episode datasets/protocol_v1_pilot/pilot_008_20261023T171324135324Z \
  --output outputs/reviews/nominal.html
```

Open the printed local HTML address to scrub synchronized images, joint positions,
velocities and action targets. These commands review existing data; they do not
collect, accept or train on it. [Dataset instructions](docs/DATASET.md) cover QA,
physical replay, collection gates and future conversion.

## One command interface

| Operation | Command | Environment | Default behaviour |
| --- | --- | --- | --- |
| Scene and camera inspection | `python scripts/so101.py scene` | `leisaac-sim` | Stationary GUI and three actual feeds |
| Deterministic expert | `python scripts/so101.py expert` | `leisaac-sim` | Full task, continuous motion, 20 Hz targets, grip feedback |
| Manifest planning / collection | `python scripts/so101.py collect --help` | `leisaac-sim` | `plan` writes manifests; `run` only prints launches unless explicitly authorized |
| Data inspection / QA / conversion | `python scripts/so101.py dataset --help` | Simulation for raw review/QA; learning for conversion/viz | Conversion previews; execution still requires accepted QA |
| Official ACT / Diffusion training | `python scripts/so101.py train --help` | `so101-il-py312` | Config/data preflight only; separate authorization required for updates |
| Checkpoint serving / simulation evaluation | `python scripts/so101.py evaluate --help` | Learning for `serve`; simulation for `run` | Real checkpoint and inference socket required |
| Recorded-action physical replay | `python scripts/so101.py replay --help` | `leisaac-sim` | Original reset pose, shared physics and evaluator |

For future training preflight, activate the existing learning environment:

```bash
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate so101-il-py312
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/so101.py train --policy act --config matched-v1 \
  --dataset datasets/lerobot_train --output checkpoints/act_seed17 --seed 17
```

This prints the official LeRobot command and checks prerequisites. It currently
stops because an accepted converted training dataset does not exist. Change
`--policy diffusion` and output name for the other official policy. No optimizer
runs without explicit execution, session authorization and a hash-bound approval.
Read [ACT, Diffusion and inference](docs/POLICIES.md) before training anything.

## Learn and navigate

| Guide | What it answers |
| --- | --- |
| [Getting started and environment locks](docs/GETTING_STARTED.md) | Which Python? What do I run and expect? |
| [Simulation, cameras and coordinate frames](docs/SIMULATION.md) | Where do the robots/cameras come from? What is calibrated? |
| [Deterministic controller](docs/CONTROLLER.md) | Which FK, IK, collision and trajectory methods are actually used? |
| [Dataset procedure and field reference](docs/DATASET.md) | What is recorded, reviewed, replayed and converted? |
| [Provisional demonstration protocol](docs/DEMONSTRATION_PROTOCOL.md) | How many attempted cases, which variations and why? |
| [Official policies and evaluation](docs/POLICIES.md) | Inputs, targets, horizons, training, checkpoints and fair comparison |
| [Architecture and complete file inventory](docs/ARCHITECTURE.md) | What does each file/function do? |
| [Repository audit and proposed cleanup](docs/REPOSITORY_AUDIT.md) | Which files/branches are historical? What could be removed after approval? |
| [Consolidation verification](docs/VERIFICATION.md) | What was tested, with which commands and limitations? |

Core flow: `so101.py` → `run_manipulation.py` (expert/replay/policy client) →
`benchmark_env.py` (shared physics) → `benchmark_core.py` (independent evaluation).
`so101_model.py` and `so101_planning.py` supply geometry/IK/paths. `policy_cameras.py`
supplies sensors/inspection; `dataset_io.py` records transitions;
`demonstration_pipeline.py` plans and audits attempts; `policy_pipeline.py` adapts
them to official LeRobot. Tests live in `tests/`; approved cameras and the active
protocol are in `configs/`. Historical diagnostics are inventoried separately.

## Reproducibility and limits

The existing stack is Isaac Sim 5.1, Isaac Lab 2.3, LeIsaac 0.4 and a separate
LeRobot 0.6.1 environment; [exact versions](docs/GETTING_STARTED.md#environment-locks)
and [the learning package lock](configs/il_requirements.lock.txt) are recorded.
Camera parameters are frozen in [policy_cameras.json](configs/policy_cameras.json).
The active provisional design is [demonstration_protocol.json](configs/demonstration_protocol.json),
not the historical `task_distribution.json`.

Source, configuration, tests and measured result summaries are versioned. Raw
recordings, generated reports/videos, caches and checkpoints remain local and
ignored by Git. GitHub-only reviewers can read all guides and tracked results;
local artifact paths in code examples require the workstation's preserved data.

Most existing `outputs/` files are generated evidence, plans or diagnostics, not
code needed to launch the expert. The pilot's many PNGs are its three synchronized
lossless camera streams. Verified stale caches and disposable previews were
cleaned; [the output inventory and retention rules](docs/REPOSITORY_AUDIT.md#generated-output-cleanup-and-retention)
explain what remains and why.

The pilot is a purpose-selected set of corners/reference poses, not a statistical
estimate of broad task reliability. Camera extrinsics are simulation estimates,
not a real-hardware calibration. No learned controller or sim-to-real transfer is
validated. Earlier failed learning experiments are preserved as explicitly
[historical evidence](docs/POLICY_REVIEW.md), outside the operating instructions.
