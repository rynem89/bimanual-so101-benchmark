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
14.7% fewer 20 Hz observation/action rows if later recorded. No demonstrations
have been recorded for this revision.

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

Inspect the preserved pilot without recording anything:

```bash
python scripts/so101.py dataset inspect --quality-report outputs/protocol_v1_pilot_qa/quality.json --ledger outputs/protocol_v1_pilot_run/attempts.jsonl
python scripts/so101.py dataset review --episode datasets/protocol_v1_pilot/pilot_008_20261023T171324135324Z --output outputs/reviews/nominal.html
```

Open the printed HTML address locally. Scrub the three synchronized images with
the measured joints and next action. Use a new HTML output name when repeating.
For all 12 attempts, open `outputs/protocol_v1_pilot_report/index.html` locally.
These recordings are local and intentionally absent from GitHub.

**No final collection is authorized.** The consumed pilot is excluded from
training. Review all 12 videos/data first; future train/validation attempts need
separate approval. Failed attempts must remain counted.
[Design, fields, QA, replay and conversion procedure](docs/DATASET.md).

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

You can replay an existing recording now, in a fresh `leisaac-sim` terminal
activated as in step 1:

```bash
python scripts/so101.py replay --episode datasets/protocol_v1_pilot/pilot_008_20261023T171324135324Z --output outputs/replay_nominal_manual
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
| 12/12 physical pilot successes, 12/12 automated QA passes, 3/3 selected physical replays | Human acceptance of all 12 demonstrations; the other nine physical replays |
| Official ACT/Diffusion command/config preflight and guarded inference interface | Accepted training corpus, optimizer updates, trained inference and learned task success |

The provisional design proposes 100 train /20 validation attempts and 100
held-out evaluation cases. **Collection, augmentation and all policy training
remain unauthorized.** Repository consolidation and the exact eight-file/four-branch
cleanup received separate owner approval; it does not authorize learning work.
Camera/protocol JSON bytes, scene, physics, IK solver, evaluator and action
semantics remain fixed. Stage 1 authorizes the coordinated controller and your
manual GUI inspection only. Ten new demonstrations and their HTML reviews wait
for your explicit Stage 2 approval. Your subsequent request authorizes focused
non-recording physics verification and timing, not collection, dataset QA, replay
or training. [Earlier pilot details](docs/DATASET.md#measured-pilot-results).

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
local. The 107,394 pilot PNGs are three original lossless image streams plus
terminal diagnostics, not redundant scripts. Preserve the recordings, QA reports,
reference benchmark and camera-approval evidence. Disposable caches removed in
the previous cleanup are **not counted again**; this pass found no identical
stationary PNG previews to delete. [What remains and why](docs/ARCHITECTURE.md#local-outputs).

The approved consolidation is complete: **41 tracked files, four active detailed
guides, 13 Python scripts, four configurations and seven tests**. The sole branch
is `main`; the four incorporated feature branches have been retired locally and
on GitHub. The [consolidation record](docs/ARCHITECTURE.md#consolidation-record)
lists the eight exact file removals and before/after counts. Git history and all
existing recovery tags are preserved. No new branch or tag was created.
