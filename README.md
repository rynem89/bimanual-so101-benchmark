# Bimanual SO-101 benchmark

The left SO-101 picks up a 40 mm cube, hands it to the right SO-101 in mid-air,
and the right arm places it in a basket. Both arms retreat. Isaac Sim provides
physics and contact; an independent evaluator checks the result.

**Start here and follow steps 1–6 in order.** You can run the scene and expert
now. Demonstration collection and policy training require separate approval.

## Table of contents

1. [Setup](#1-setup)
2. [Scene and cameras](#2-scene-and-cameras)
3. [Expert IK and task execution](#3-expert-ik-and-task-execution)
4. [Demonstrations](#4-demonstrations)
5. [ACT and Diffusion](#5-act-and-diffusion)
6. [Evaluation](#6-evaluation)
7. [Project status](#project-status)
8. [Code, detailed guides and historical records](#code-detailed-guides-and-historical-records)
9. [Outputs and repository integration](#outputs-and-repository-integration)

## 1. Setup

These commands use the existing workstation. They do not install anything.
Activate the environment in **each new terminal**:

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
export LEISAAC_ASSETS_ROOT=/home/rynem8/robotics/leisaac/assets
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python --version
python scripts/so101.py --help
```

Expect Python **3.11.16**. The locked simulation stack is Isaac Sim 5.1, Isaac
Lab 2.3 and LeIsaac 0.4. The learning environment uses Python 3.12 and LeRobot
0.6.1. Keep both environments: their Python/NumPy requirements differ. Both ACT
and Diffusion use the same official LeRobot environment, not two custom models.
[Exact versions and asset paths](docs/SIMULATION.md#environment-locks).

## 2. Scene and cameras

```bash
python scripts/so101.py scene --frames
```

This opens two robots, the cube, basket and a panel with the actual overhead,
left wrist and right wrist RGB images. Use the selector to choose a camera.
Yellow guides show camera directions; axes show coordinate frames. You can orbit
the viewport and expand `/World/InspectionGuides` in the Stage panel.

The scene settles for three simulated seconds, then freezes physics. The camera
panel shows that settled state; the free viewport continues rendering. No expert
motion, recording or collection starts. **Close the whole Isaac window before
running the next command.** [Camera calibration and frames](docs/SIMULATION.md#cameras-and-coordinate-frames).

## 3. Expert IK and task execution

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_manipulation.py --stage full
```

The expert executes pickup → lift → handover → basket placement → retreat.
GUI is the default; the final state stays open for inspection. Close Isaac after
reviewing it. It runs once by default. An episode means one complete task attempt,
not a recorded demonstration. `python scripts/so101.py expert` is the equivalent
high-level entry point; the direct command above uses the same coordinated recipe.

The right starts moving **while the left opens and approaches the cube**. It now
waits lower, about 4 cm back along the final insertion direction. The donor brings
the cube in front of that waiting gripper; the receiver uses a direct insertion
instead of going around to a different handover pose. After verified transfer
and finger separation, the receiver makes one continuous move to the basket
while the donor returns HOME. The collision checker validates both moving arms
and the carried cube together. Existing checked fallback routes remain available.
**Your GUI approval of this revision is still required.** Historical pilot
successes describe the earlier sequential controller.

The improved nominal task passed **two non-recording visible Isaac runs**: all four
physical stages succeeded with zero forbidden contacts in **114.88 simulated
seconds each**, including the 3 s reset. The previous coordinated version took
**134.12 s** and the sequential reference **153.18 s**. That is **14.3% shorter
than the previous version**. Excluding reset, the shorter motion would mean about
14.7% fewer 20 Hz observation/action rows if later recorded. No demonstration
dataset is currently retained for this revision; the prior pilot batches were
rejected and deleted at the owner’s request.

The cold run took **232.24 s wall time**, including **64.81 s of stage planning**.
The aligned receiver insertion needed **6.61 s planning**; the new return/carry
needed **28.43 s**. First-run calculation is still noticeable. Planning pauses
physics, so those pauses add no dataset frames. Cached plans are reused only
when source, geometry and measured joint/cube inputs match.
The verified cached run took **168.69 s wall time**, with only **0.78 s
of stage planning**. Changing source, cube pose or measured grasp can require
fresh computation.
[Timing evidence and limits](docs/archive/VERIFICATION.md#aligned-handover-and-continuous-bin-carry--2026-10-24).

**What calculates the movement?** Our code uses SciPy bounded inverse kinematics
(IK) to find five arm-joint angles for a requested fingertip pose. It checks the
whole path against both robots, table, cube and basket. Joint transits use quintic
timing; Cartesian waypoint approaches use a cubic spline with quintic timing.
Those position targets are held
at 20 Hz while Isaac executes 60 Hz physics. Contact feedback controls gripping;
the cube is moved by simulated contact, not teleported.
[Equations and source references](docs/SIMULATION.md#expert-ik-and-motion).

Planning can take time: `[PLAN]` messages mean CPU collision checks are running
while physics pauses. Expect `[EPISODE] nominal/1: passed` and
`[RESULT] passed_requested_stage`. Check `full_task_success`, independent stage
flags and failures in `outputs/controller/simulation.json`; exit code alone
is insufficient. The timestamped `full_*.json` preserves the run. `plans/` stores
validated planning caches.

Preview the command without opening Isaac, or run a named regression case:

```bash
python scripts/so101.py expert --dry-run
python scripts/so101.py expert --case yaw_minus_1deg --output outputs/expert_yaw
```

## 4. Demonstrations

A demonstration is a **recorded physical expert episode**, not an IK solution or
a video alone. Each 20 Hz row pairs three RGB images and measured joint state
with the absolute joint target applied over the next three physics ticks.

### Generate your own demonstrations

Use **`scripts/collect_demos.py`**. It calls the existing coordinated expert,
recorder and `so101.py dataset review`; it does not implement another controller.
You choose the ranges and run collection yourself. Activate `leisaac-sim` in each
new terminal using [step 1](#1-setup), including the asset path and thread settings.
No packages need installing.

- `--count 10` asks for **10 successful physical demonstrations**, not 10 attempts.
- `--max-attempts 20` caps simulation attempts, including failures. Default is
  three times `--count`. Collection stops at the success target or the cap/available
  distinct poses. An unmet target returns a nonzero exit code and the actual count.
- `--x-offset-cm MIN MAX` and `--y-offset-cm MIN MAX` are **centimetres relative
  to nominal cube centre X = −0.34 m, Y = −0.04 m**. For example, X offset +2 cm
  means world X = −0.32 m. Negative/positive offsets follow the world axes.
- `--yaw-deg MIN MAX` controls cube rotation about the vertical axis in degrees.
- `--seed 42` reproduces the candidate ordering with the same ranges, count,
  attempt cap and unchanged model. Keep all these arguments identical between
  preview and recording.

The examples below request a **6 cm × 4 cm** region and yaw ±6°. These are
user-selected sampling bounds, not a claim that every pose will work. The sampler
stratifies the requested region, filters candidates, then chooses spread-out
positions at least **5 mm apart in XY**. It never substitutes the old small range.
Preview prints requested bounds, proposed world X/Y/yaw and actual feasible
coverage, plus rejection counts. If insufficient distinct poses are found, it
fails before simulation; change the ranges, seed or count explicitly.

Before recording, the geometry filter checks the **whole rotated 40 mm cube** is
supported by the table with a 10 mm edge margin, clear of the bin and both HOME
robots, and has valid open-gripper pregrasp/grasp IK under the existing joint
limits and collision checks. This establishes initial geometric feasibility;
**it does not prove a full collision-free task or physical success**. The expert
still checks complete motion and applies all existing contact/drop/grasp guards.

**1. Preview only — no simulator, recording or output files:**

```bash
python scripts/collect_demos.py --count 10 --max-attempts 20 \
  --x-offset-cm -3 3 --y-offset-cm -2 2 --yaw-deg -6 6 --seed 42 \
  --preview-only
```

**2. Record visibly in Isaac Sim:**

```bash
python scripts/collect_demos.py --count 10 --max-attempts 20 \
  --x-offset-cm -3 3 --y-offset-cm -2 2 --yaw-deg -6 6 --seed 42 \
  --gui
```

**3. Record without the GUI, with the same three cameras and HTML reviews:**

```bash
python scripts/collect_demos.py --count 10 --max-attempts 20 \
  --x-offset-cm -3 3 --y-offset-cm -2 2 --yaw-deg -6 6 --seed 42 \
  --headless
```

GUI is the default. Each attempt launches the existing expert, resets the scene,
records one pose and closes Isaac on completion; the next attempt starts fresh.
Use **Ctrl+C in the terminal** to stop the whole collection. Closing one attempt's
window early can count as a failure and advance to the next candidate.

The script prints a new run directory under
`datasets/generated_demos/<UTC timestamp>_seed42/`:

| Location inside that run | What it contains |
| --- | --- |
| `successful/<episode>/` | Finalized recordings with independent pickup/lift/handover/placement success; existing raw schema, three RGB streams, measured states and held actions |
| `failed/<episode>/` | Failed or interrupted recordings, kept separately and never counted toward `--count` |
| `reviews/attempt_0001.html`, etc. | One interactive three-camera/state/action review per recording, where enough data exists; images stay in the corresponding episode |
| `attempts/attempt_0001/`, etc. | The exact cube-case input, expert log and normal controller outputs; startup failures can have logs without a recording |
| `summary.json` | Cube XY/yaw/seed, physical and collection result, duration, failure reason, episode and review paths, actual successful/failed counts |

A failed grasp, drop, contact guard or other failure remains recorded; the script
tries another feasible position up to the cap. It does not relax safety checks.
HTML generation uses the existing review command and its built-in format checks;
no separate dataset QA, plots, physical replay or training runs are added.
Physical success is **not human acceptance or permission to train**. Review the
images and actions yourself; conversion/training still require their existing
separate approvals. The old pilot directories and their review links were
superseded and deleted at the owner's request.

**Implementation verification is limited to argument handling and geometric pose
previews. GUI/headless collection with this new helper has not been run by the
assistant.** [Raw format and learning procedure](docs/DATASET.md#raw-attempt-layout).

## 5. ACT and Diffusion

There is **no accepted training dataset or trained policy in the reset pipeline**.
Both future policies will use official LeRobot implementations and the same
approved three-camera/joint observations. They predict joint-action chunks;
they do not run expert IK during inference.

Switch this terminal to the existing learning environment:

```bash
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate so101-il-py312
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/so101.py train --policy act --config matched-v1 --dataset datasets/lerobot_train --output checkpoints/act_seed17 --seed 17
python scripts/so101.py train --policy diffusion --config matched-v1 --dataset datasets/lerobot_train --output checkpoints/diffusion_seed17 --seed 17
```

These are **preflight commands**, not training. They show the official command and
check configuration/data. They currently reject the missing accepted dataset.
After approved collection, QA and conversion, actual training needs explicit
execution flags and a genuine hash-bound approval record. See the
[complete policy procedure](docs/POLICIES.md) for inputs, horizons, reasons for the
recipe, training, checkpoints and resume. Do not treat config parsing as evidence
of convergence or manipulation success.

## 6. Evaluation

After you generate a recording, you can replay its actual actions in a fresh
`leisaac-sim` terminal activated as in step 1. Replace the capitalized path parts
with a successful episode from your run’s `summary.json`:

```bash
python scripts/so101.py replay --episode datasets/generated_demos/RUN_ID/successful/EPISODE_DIRECTORY --output outputs/replay_manual
```

Replay sends recorded targets to real simulated actuators and the independent
evaluator. It does not write recorded robot/cube states into the scene. It stops
at independent success, so the final recorded settling tail may be omitted.

**For a future trained policy**, use two terminals: the learning environment
serves the real checkpoint; the simulation environment runs the shared task.
There is no trained checkpoint to run today.

Terminal A, `so101-il-py312`; add `--execute` only when this checkpoint exists:

```bash
python scripts/so101.py evaluate serve --policy act --checkpoint checkpoints/act_seed17/checkpoints/100000/pretrained_model --socket /tmp/so101-act.sock
```

Terminal B, `leisaac-sim`, after the real server prints `[READY]`:

```bash
python scripts/so101.py evaluate run --socket /tmp/so101-act.sock --manifest outputs/demonstration_protocol_v1/validation.json --split validation --output outputs/evaluation/act_seed17_validation
```

Use the same declared cases, reset, physics, 20 Hz action interface and evaluator
for expert, ACT and Diffusion. Report all failures and stage/full-task success.
[Inference details and fair comparison](docs/POLICIES.md#validation-and-fair-evaluation).

## Project status

This section is the current status and approval authority. Historical logs do
not override it.

| Established by evidence | Still unverified / not approved |
| --- | --- |
| Sequential reference pickup, lift, handover, placement and automatic reset | Coordinated Stage 1 controller GUI approval; broad-distribution reliability and hardware transfer |
| Improved coordinated nominal controller: 2/2 final-revision physics successes, zero forbidden contacts | Owner GUI approval and other starting-pose validation |
| Approved frozen overhead + two wrist cameras; owner accepted scene/HTML workflow and guide fix | Physical camera-mount registration and hardware calibration |
| Historical pilot/reference results remain documented; rejected pilot data was deleted by owner approval | No retained or human-accepted demonstration corpus; no new helper collection run by the assistant |
| Official ACT/Diffusion command/config preflight and guarded inference interface | Accepted training corpus, optimizer updates, trained inference and learned task success |

The owner now controls dataset generation through `scripts/collect_demos.py`.
The former 12-case pilot and the rejected `coordinated_pilot_10` batch and its
outputs were deleted with explicit approval. Reference benchmark evidence,
source/configurations, approved cameras and recovery tags are preserved.

The assistant is authorized to implement this helper and verify arguments/pose
previews only, **not generate another dataset**. No accepted training corpus or
trained policy exists. Final protocol collection, augmentation, conversion and
training retain their separate approval gates. The earlier 100/20/100 design is
historical context, not authorization to collect. Controller, cameras, scene,
physics, evaluator and action semantics remain unchanged by this helper.

## Code, detailed guides and historical records

After the six README steps, open only the guide needed for your question:

| Guide | Purpose |
| --- | --- |
| [Simulation and expert](docs/SIMULATION.md) | Environment locks, USD assets, camera transforms, FK/IK, collision checks, trajectories and physical evaluation |
| [Demonstrations](docs/DATASET.md) | Dataset design, exact record fields, acceptance, conversion and measured pilot results |
| [Policies and evaluation](docs/POLICIES.md) | Official ACT/Diffusion recipe, training/resume, inference and comparison limits |
| [Code map](docs/ARCHITECTURE.md) | Every maintained file/API, output retention and completed consolidation |

Core flow: `so101.py` → `run_manipulation.py` → shared `benchmark_env.py` and
`benchmark_core.py`. Model/planner compute expert motion; dataset and policy
modules record, validate, convert and serve actions. Imported modules are not
additional commands you need to run.

Historical engineering, failed-policy reviews, pilot report and old verification
logs live separately under [docs/archive](docs/archive). They preserve evidence;
they are not a second operating procedure.

## Outputs and repository integration

Source/configs/tests are versioned. Raw data, weights and generated outputs stay
local. Superseded pilot recordings and their generated reviews/reports were
deleted with the owner's explicit approval. Preserve the reference benchmark,
camera-approval evidence, source/configurations, Git history and recovery tags.
New user-generated runs live under `datasets/generated_demos/` with successful
and failed recordings separated. Human acceptance is still required.

The current tree has **42 tracked files, four active detailed guides, 14 Python
scripts, four configurations and seven tests** after adding this one collector.
The sole branch is `main`; no new branch was created. The historical
[consolidation record](docs/ARCHITECTURE.md#consolidation-record) describes the
previous eight-file retirement and its then-current counts. Existing recovery
tags are preserved.
