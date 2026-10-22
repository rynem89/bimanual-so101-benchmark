# Bimanual SO-101 reference benchmark

A simulation-first robotics benchmark: the left five-axis SO-101 picks up a
40 mm cube, transfers it in mid-air to the right SO-101, and the right arm
releases it into a basket. Both arms retreat, the cube settles on the basket
floor, and the scene resets automatically. The reference policy uses geometry,
IK and contact feedback. The isolated LeRobot pipeline records, trains and evaluates official ACT and
Diffusion policies; no learned manipulation success is claimed.

The original complete sequence passed **3/3 repeated physics episodes** at
`9af7dc7`; the user visually confirmed it in Isaac Sim. The benchmark-hardening
revision adds an independent evaluator, transfer instrumentation, faster
collision rejection and fixed regression cases. **Staged receiver preload
passed 3/3 complete independently evaluated episodes**, including a final run
with strict reset verification,
automatic reset and zero collision violations. The fixed perturbation suite
passed **3/4 perturbed poses** (4/5 including nominal); −1° yaw was rejected
before receiver motion. Both deliberate grasp/transfer omissions were detected
with zero task successes. Detailed measurements are
recorded in the [engineering handoff](docs/ROBOTICS_CODEX_HANDOFF.md#engineering-audit--benchmark-hardening-2026-10-22).
The user confirmed the complete physical sequence and the continuous-motion
replay. The newer RRT transit and measured donor-clearance move still need a visual check. Repeated identical starts establish
repeatability, not general robustness or real-robot success.

## Run the simulator

Use the existing environment in **every new shell**. Do not reinstall the stack.

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
export LEISAAC_ASSETS_ROOT=/home/rynem8/robotics/leisaac/assets
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python --version  # Python 3.11.16

# Full task with simulator window; final scene stays open for inspection.
python scripts/run_manipulation.py --stage full --motion-mode continuous --expert-hz 20 --grip-feedback-from-lift

# Automated complete episodes with the same physics/controller, no renderer.
python scripts/run_manipulation.py --headless --stage full --handover-mode staged --episodes 3

# Five fixed cube poses, each replanned and independently evaluated.
python scripts/run_manipulation.py --headless --stage full --handover-mode staged --suite perturb

# Audit actual reset poses and the 20 Hz interface for all five cases.
python scripts/run_manipulation.py --headless --stage model --suite perturb

# Deliberately omit donor/receiver closure; both failures must be detected.
python scripts/run_manipulation.py --headless --stage full --handover-mode staged --suite negative
```

GUI is the default; `--headless` is an explicit per-run choice. Add `--exit-after`
to close a GUI run automatically. CPU planning pauses physics while keeping the
viewer rendering. `[PLAN DONE]` reports fresh planning duration; `[CACHE]`
confirms checked-segment reuse. Preserve `outputs/controller/plans/` for warm
runs. Cold simulator startup measured about 5.3 s with existing local Kit/assets
caches. The first complete run must validate many paths and is slower.

| `--stage` | Evaluation stops after |
| --- | --- |
| `model` | Reset and measured FK consistency |
| `approach` (default) | Open left hand at the cube, before closing |
| `pickup` | Left pickup, lift and three-second hold |
| `transport` | Left holds at the handover region |
| `handover` | Three seconds of right-only support after donor release |
| `full` | Released cube settled in basket, both arms withdrawn |

`staged` is now the default after the 3/3 regression. Use
`--handover-mode baseline` for the original receiving-force schedule.
`--case nominal/x_plus_1mm/x_minus_1mm/yaw_plus_1deg/yaw_minus_1deg` selects one
fixed case; `--suite perturb` runs all five. Seeds 101–105 and exact initial
poses are in `benchmark_core.EVALUATION_CASES`. `--episodes` repeats each case.
These are small diagnostic perturbations, not a sampled task distribution.
The current full perturbation command exits 1 because the −1° yaw case has no
validated receiver route; its failure is an expected documented limitation,
not an allowed-contact or success-threshold exception.

Each run writes `outputs/controller/simulation.json` checkpoints and a unique
`full_<UTC timestamp>.json` (or the requested stage name). Full timestamped history persists at episode boundaries; the two-second
checkpoint retains the active phase and 12 recent samples. Reports contain source
revision **and file hashes**, dirty status, configuration/seeds, stage outcomes,
independent metrics, timing, failure observations and physics-step transfer
traces. Outputs are ignored by Git. Exit 0 means the requested stage passed;
negative-suite exit 0 means both deliberate failures were detected, **zero task
successes**. Unexpected failure exits 1 and attempts a verified reset. Do not
interpret a shorter-stage pass as complete-task success.

## Architecture and objective evaluation

```text
USD model → FK / constrained IK → checked task waypoints → expert joint targets
                                                         ↓
policy targets → BenchmarkEnv → Isaac Sim / PhysX → measured Frame → TaskEvaluator
```

`BenchmarkEnv` provides `reset`, `observe`, `step` and the expert's one-tick
`advance`. Physics runs at **60 Hz**. The policy interface holds absolute joint
position targets of shape `(2,6)` in radians for three ticks (**20 Hz**). Shared
implicit PD and robot gravity feed-forward remain part of the environment.
Default observations are joint positions, velocities and the previous action.
Object poses, contacts and geometry are privileged expert/evaluator inputs.
The RGB learning track receives only 24 proprioceptive values (12 positions,
12 velocities) and two RGB images. The reference expert proposes grip
corrections at 60 Hz; `--record-dir` selects exact 20 Hz sample-and-hold actuator
commands for demonstrations. Internal expert contact feedback remains privileged.
Learned policies have no expert grip servo, IK, collision planner, future
actions, object state or success labels at inference. All policies share the
implicit PD actuators, gravity feed-forward and independent evaluator.

The evaluator imports no expert/planner and accepts no expert phase/success
flags. It derives success from physics observations:

- Left bilateral support with cube centre ≥60 mm for three continuous seconds.
- Right bilateral support, cube centre ≥150 mm and no donor contact for three
  continuous seconds.
- Entire cube inside basket interior, 0.3–0.7 N floor support, linear speed
  <0.01 m/s, angular speed <0.1 rad/s, no hand contact and both arms HOME within
  3° for three continuous seconds.
- Forbidden contacts ≥0.05 N, sole-holder slip >5 mm / 10°, nonfinite states,
  joint/effort/speed violations or 300 s timeout fail the episode.

Reset checks cube pose within 0.1 mm and orientation within 0.1°, velocities,
arm HOME and measured cube
weight before control. Direct joint/object state writes occur only in reset.
There are no task-time cube teleports, attachment constraints or cube gravity
cancellation. Bin placement currently releases the cube from above the basket
(about 106 mm centre height); gravity settles it at 30 mm centre height. It does
not perform controlled lowering into floor contact.

## How motion is calculated

FK composes the USD parent/child joint frames and axis rotations. The calibrated
TCP is `(0.0141000004, 0, -0.09)` m in the gripper link; X closes jaws, -Z is
approach. Full-footprint triangle checks clear a 42 mm test box at 45° opening;
authored triangles are not identical to cooked PhysX contact hulls.

IK uses bounded SciPy Trust Region Reflective least squares for five arm joints,
with deterministic multiple starts. Position plus approach direction has five
independent constraints; arbitrary full six-dimensional poses are not generally
feasible. Accepted solutions need <1 mm position and <1° orientation/direction
error. Handover jointly solves both arms' ten joints on orthogonal cube faces.

The task planner validates every connecting segment, with 16 short Cartesian
IK legs near the receiving grasp and donor withdrawal. AABBs reject distant
pairs; separating axes of inflated convex hulls avoid unnecessary LP calls;
inconclusive cases retain definitive HiGHS half-space feasibility checks.
Triangle/box plus containment tests preserve the gripper's empty grasp region.
Payload geometry follows a measured cube/TCP transform for planning only.
Sample spacing uses a 0.6 m downstream-motion bound, unchanged 1–3 mm margins
and an explicit 0.1 mm reserve for cache state tolerance. Fingerprints include
resolved geometry, calibration, limits, scene/model/planner source and relevant
measured context. Runtime contacts and tracking are checked separately.

Joint segments use `q(u)=q0+(10u³−15u⁴+6u⁵)(q1−q0)` with
`T=max(1 s, 1.875 max|q1−q0| / 0.35 rad/s)`. This bounds nominal speed at
0.35 rad/s and gives zero endpoint velocity/acceleration. Exact acceleration
and jerk peaks are recorded, as are actual executed derivatives. The robot
configuration has 10 rad/s and 10 N m limits, **no acceleration/jerk limits**;
those cannot be certified. Phase changes and contact feedback can create
transients despite smooth nominal segments.

`--motion-mode continuous` joins the receiver approach and donor withdrawal
with clamped cubic joint splines, parameterized by the same quintic time law.
Position, velocity and acceleration are continuous; contact boundaries retain
90 settling ticks. The planner checks the actual spline, its exact polynomial
extrema against joint limits, and intermediate configurations at a conservative
motion-bound spacing. `--motion-mode legacy` preserves the successful stopped
waypoint reference. Three continuous full-task runs passed independently:
150.65 vs 209.17 simulated seconds (28% shorter), settling 24 vs 70.5 s,
warm headless execution ~48 vs ~69 s. Peak speed/effort were 2.275 rad/s /
1.427 N m, below the authored 10/10 limits; measured acceleration/jerk increased
and have no authored limits. Handover dip was 10.30 vs 9.98 mm. GUI pacing now
defaults to zero; `--viewer-delay 0.008` restores the old artificial pause.

Measured bilateral forces regulate bounded jaw targets. Staged receiving
preload delays the stronger squeeze until the donor unloads, then ramps it over
one second. Original gains, masses, gravity, friction and actuator limits remain
unchanged. Detailed formulas, frames, failure analysis and official API sources
are in [Learning notes / code map](docs/ROBOTICS_CODEX_HANDOFF.md#learning-notes--code-map-current-algorithms).

## Tests and measured performance

CPU tests use the USD libraries already bundled with Isaac Sim, without Kit:

```bash
# After activation above; use a fresh shell for simulator runs afterwards.
USD_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
export PYTHONPATH="scripts:$USD_LIB"
export LD_LIBRARY_PATH="$USD_LIB/bin:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m unittest discover -s tests -v
python scripts/so101_planning.py --profile --output outputs/controller/profile_final.json
```

Current CPU regression: **31 tests passed**, including independent evaluator
false-positive/negative cases, timeout/reset, FK/IK, triangle containment,
rotated/acute convex envelopes, ambiguous LP status, trajectory derivatives and
cache invalidation. Complete approach tests compare every optimized hull query
with the original LP; all intermediate poses remain validated.

| Measurement on this workstation | Before | Optimized |
| --- | ---: | ---: |
| Profiled cold approach, same 3,794 samples | 32.53 s | 17.84 s |
| Convex-query inclusive time | 22.76 s | 2.65 s |
| LP calls on that approach | 1,958 | 0 (all separations proved) |
| Final cold approach, **4,215 samples** with cache reserve | — | 13.03 s |
| Final geometry-warm revalidation, no plan cache | — | 11.55 s |
| Complete cold planning, historical / hardened task | ~548 s | ~134 s |
| Warm headless episode, excluding planning | 94.17 s | 69.02 s |
| Release cube / right-TCP dip, initial paired trial | 17.33 / 17.18 mm | 9.98 / 10.07 mm |
| Pose convergence after release, initial paired trial | 1.85 s | 1.25 s |

These are local timing samples and initial paired transfer measurements, not
universal performance claims. Final sampling is denser than the 3,794-sample
comparison because cache tolerance now has an explicit motion reserve. Full
nominal/perturbed results and cold/warm execution timings are in the handoff.
The initial three-repeat checkpoint cost was 43.51 s; compact checkpoints cost
4.15 s across the later five-case suite (different workloads, no matched
speedup claim).
PhysX reports a residual vertical velocity during grasp even when pose finite
differences converge; reports preserve both and do not call it velocity-settled.

## Environment and dependencies

Validated workstation: Ryzen 9 9950X3D2, RTX 5090 32 GB, Ubuntu 24.04.5,
NVIDIA 580.178.04, CUDA 12.8.1, Python 3.11.16, torch 2.7.0+cu128,
Isaac Sim 5.1.0, Isaac Lab 2.3.0 (`3c6e67bb5c7ada942a6d1884ab69338f57596f77`),
LeIsaac 0.4.0 (`24d3bcd`). NumPy 1.26.0 handles geometry; SciPy 1.15.3 supplies
bounded optimization, rotations, convex hulls and LP; bundled USD composes the
actual robot asset. PhysX supplies dynamics and measured contacts. No packages were changed in `leisaac-sim`. The isolated training environment
below is new. Installation history and pinned-path details stay in the
handoff. cuRobo and MoveIt are assessed there, not installed dependencies.



The expanded distribution exposed two additional expert limitations. The optional
`--grip-feedback-from-lift` applies the existing bounded contact servo during
lift/hold: **4/4 held-out pickup/lift/hold checks passed**, versus 1/4 full-task
attempts reaching those stages with the fixed jaw target. New receiver transits
use a bounded, fixed-seed **RRT-Connect** fallback when the direct joint line
crosses the cube. Every edge uses the original dense collision checks; only
collinear subdivisions are removed. RRT turns execute as separate quintic moves;
the checked insertion retains its C2 spline. No collision margins were reduced.
After jaw opening, retained donor contact triggers a checked 4–12 mm move away
along the fixed pad's negative TCP-X normal, followed by the unchanged three-second
right-only support gate. **Three full regressions passed**, all independent gates true and zero forbidden
contacts, each 156.18 simulated seconds. The original fixed-jaw expert remains
available by omitting the feedback flag; baseline `05a0c13` is preserved in Git.

The launcher previously replaced the shell's `OPENBLAS_NUM_THREADS=1` with 32.
CPU-only NumPy/SciPy initialization now precedes Kit startup, and public Kit
arguments set per-process worker limits (`--cpu-threads 1`, default). Reports
verify actual Carbonite/TBB counts and the numerical environment. No OS setting,
driver or installed simulator package changed. `--cpu-threads 32` restores the
original Kit worker count; numerical threads follow the pre-launch environment.

## Demonstrations and official LeRobot policies

The end-to-end **smoke pipeline works**: physical 20 Hz expert recording,
independent replay, local LeRobotDataset **v3.0** conversion, official ACT and
Diffusion GPU training, saved/reloaded checkpoints and real simulator evaluation.
This is not yet a successful learned manipulation benchmark. Initial 20-update
ACT: **0/1**, 300 s timeout, no pickup. Diffusion: **0/1**, actuator-limit failure
at 0.05 s with two collision violations. These tiny training runs test plumbing;
their losses are not comparable success metrics. Larger-data results remain pending. A subsequent 1,000-update run on two varied
demonstrations also scored **0/1 nominal + 0/4 held-out for both methods**:
ACT timed out, Diffusion violated actuator/contact limits. This is a small-data
diagnostic, not a credible method ranking.
The original Diffusion pilot did not seed denoising at episode reset. The current
versioned inference protocol seeds Python, NumPy and Torch, records checkpoint
and adapter hashes, and rejects an obsolete server. Repeated reset with seed
813000 produced exactly equal actions; simulator evaluation must use this protocol
before claiming a reproducible comparison.

Two fixed 224×224 RGB cameras use Isaac Lab 2.3 `CameraCfg`: overhead eye
`(0,-.12,.92)` m and front eye `(0,.72,.56)` m, with targets in
`benchmark_env.CAMERA_VIEWS`. Focal length 18 mm, horizontal aperture 20.955 mm,
clipping .01–10 m. Capture renders the current physical state before the next
20 Hz action. Raw metadata stores measured intrinsics/extrinsics. A stale camera
pose-buffer bug in the first smoke recording was found and corrected; that
recording remains diagnostic, not a production calibration reference. RGB
projection checks against measured cube positions are recorded in the handoff.

Each successful raw episode contains `trajectory.npz` with `(N,24)` state,
`(N,12)` absolute target actions in radians, `(N,3,12)` actual substep targets,
physics indices and timestamps; PNGs; `episode.json`; and a separate
`privileged.json` for object/contact diagnostics. Left six joints precede right
six: pan, lift, elbow, wrist flex, wrist roll, jaw. Velocities use the same order.
Frames describe state **before** the action held for the next three ticks.
Attempt metadata is saved immediately and partial trajectories every 200 frames
so interrupted generation is visible. Partial terminal transitions are excluded
and explicitly counted; final physical
state is retained separately. Failed attempts are kept and counted, but excluded
from successful demonstration conversion. Validation rejects missing RGB,
nonfinite/misaligned targets and invalid camera calibration. Raw schema 1.1 also
stores authored radian joint limits and rejects out-of-range actions, with only
float32 rounding tolerance. Process-start source hashes, revision and dirty status
identify the generating controller. Legacy 1.0 files remain readable and explicitly
report whether action bounds were verified. A local v3 dataset
embeds images in parquet; no video codec or Hub upload is required.

The original symmetric ±10 mm X / ±7 mm Y / ±6° candidate distribution produced
**2/4 full successes**, with two pre-grasp collision rejections. IK continuation
found nearby solutions but correctly retained unsafe-path rejection. Reducing the
pre-grasp height and receiver orientation alternatives did not resolve the
failures; these were not adopted. The versioned inward candidate distribution
uses X offsets ±10 mm, Y offsets −12 to −4 mm, yaw ±6°. **12/12 independent offline
approach draws passed complete path checks**; the first physical source-frozen four-case
batch had one success, two recorder-clock failures and one receiver-plan
rejection. All remain counted. The recorder clock now restarts relative to each
reset, including resets at global ticks not divisible by three.
The manifest fixes 32 training, 8 validation and 30 held-out cases with separate
seeds. A geometric pass is not a manipulation success; all physical failures
must remain in the denominator. The fixed nominal and −1° regressions remain
separate, including the unresolved receiver limitation.

Use separate shells/environments:

```bash
# Original simulator environment; GUI remains the default.
python scripts/run_manipulation.py --stage full --motion-mode continuous \
  --cases-file configs/task_distribution.json --split train \
  --expert-hz 20 --grip-feedback-from-lift --record-dir datasets/raw_train_v2

# A different shell: isolated training, never activate over a running simulator.
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate so101-il-py312
python scripts/policy_pipeline.py convert --raw datasets/raw_train_v2 --output datasets/lerobot_train_v2
python scripts/policy_pipeline.py train --dataset datasets/lerobot_train_v2 \
  --output checkpoints/act_v2 --method act --steps 2000
python scripts/policy_pipeline.py serve --checkpoint checkpoints/act_v2/pretrained_model \
  --socket /tmp/so101_act.sock

# Original simulator environment, separate shell; no expert plans are used.
python scripts/run_manipulation.py --stage full --policy-socket /tmp/so101_act.sock
# The same interface accepts --cases-file ... --split heldout.
# Recorded replay: --stage full --replay datasets/raw_train_v2/<episode_directory>
```

Choose a **new** output directory for each conversion/training run; existing
artifacts are preserved. Record validation/test splits into separate directories
and never mix their statistics into training. Generated data, caches and weights
are ignored by Git. `--headless` changes rendering/UI visibility, not physics or
policy frequency. Expensive inference pauses simulated time: 20 Hz is the
simulated interface, not a claim that every action meets the 50 ms wall budget.

Training lock: Python 3.12.15, LeRobot 0.6.1, NumPy 2.2.6, torch 2.7.1+cu128,
torchvision 0.22.1+cu128, torchcodec 0.5.0, PyAV 15.1.0; full dependency versions in
`configs/il_requirements.lock.txt`. CUDA matrix multiplication and both official
policy imports passed on RTX 5090 sm_120; `pip check` passed. Recreate only in
an isolated environment using the CUDA 12.8 PyTorch index and that lock. The
original simulation stack and NVIDIA driver remain unchanged.

ACT uses official LeRobot ResNet18 image encoding, a conditional VAE transformer
and 40-target chunks, executing 20 actions per query (1 s). The first configuration
has model width 256, 8 heads, 4 encoder layers and one decoder. Diffusion uses the
official conditional U-Net `(128,256,512)`, two observation frames, horizon 32,
16 executed actions and 10 DDIM inference steps. Both normalize observations and
unnormalize targets through saved official processors; neither computes online
IK or a separate robot trajectory. Their 12 targets go directly to the common
20 Hz actuator interface. Seed, architecture, losses, compute and finite-action
checks are saved in each checkpoint's `training.json`. Checkpoints every 500
updates protect longer runs. Initial varied dataset: two successes from four
attempts, **5,858 frames**, 292.9 s at 20 Hz; this remains a small diagnostic set.

Official pinned sources: [LeRobot v0.6.1](https://github.com/huggingface/lerobot/tree/v0.6.1),
[ACT](https://github.com/huggingface/lerobot/tree/v0.6.1/src/lerobot/policies/act),
[Diffusion](https://github.com/huggingface/lerobot/tree/v0.6.1/src/lerobot/policies/diffusion),
[Isaac Lab 2.3 cameras](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/core-concepts/sensors/camera.html).

## File inventory

All commands below assume the activated environment; offline commands also need
the bundled USD paths above. Imported modules have no separate run command.

| File | Purpose; inputs → outputs; status / entry point |
| --- | --- |
| `scripts/run_manipulation.py` | Expert orchestration; scene/model/case → JSON outcomes. Active simulator CLI above. |
| `scripts/benchmark_env.py` | Shared reset, observations, joint actions, stepping/contact acquisition. Imported after AppLauncher. |
| `scripts/benchmark_core.py` | Independent task specification/evaluator, fixed cases, trajectory and transfer metrics. Pure CPU module. |
| `scripts/so101_model.py` | USD → FK, TCP/mesh geometry and bounded IK. `python scripts/so101_model.py` writes offline model evidence. |
| `scripts/so101_planning.py` | Model + measured poses → sampled collision/path checks and plans. `--profile` produces timing JSON. |
| `scripts/inspect_task_layout.py` | Shared scene/configuration; importing launches Kit. Standalone frame diagnostic: `python scripts/inspect_task_layout.py`. |
| `scripts/audit_so101_finger_geometry_v2.py` | USD/live jaw poses → geometry CSV/JSON. `python scripts/audit_so101_finger_geometry_v2.py --offline` or GUI without flag; maintained diagnostic. |
| `tests/test_so101_model.py` | Geometry/FK/IK/collision/cache regression, real USD and analytic fixtures → unittest results. |
| `tests/test_benchmark.py` | Independent success/failure/reset/timeout, trajectory and reset-only state writes → unittest results. |
| `tests/test_finger_geometry.py` | Analytic triangle sections and missing prerequisites → unittest results. |
| `scripts/verify_motion_gripper_reset.py` | Preserved independent motion/jaw/reset regression; run as simulator script. |
| `scripts/inspect_bimanual_scene.py`, `scripts/inspect_fk_frames.py` | Historical simulator scene/frame diagnostics; standalone scripts, preserved. |
| `scripts/dataset_io.py` | Versioned raw demonstration writer/validator; measured frames + actual held targets + RGB → NPZ/PNG/JSON. Imported; no Kit/LeRobot dependency. |
| `scripts/policy_pipeline.py` | Isolated LeRobot v0.6.1 `convert`, `train`, `serve`; raw episodes → v3 images/parquet → official checkpoint → Unix-socket actions. |
| `configs/task_distribution.json` | Seeded candidate train/validation/held-out poses, disjoint configurations; physical coverage must be reported. |
| `configs/il_requirements.lock.txt` | Exact installed Python 3.12 training packages; never install into the simulation environment. |
| `tests/test_dataset.py` | Actual-file chronology, held targets, RGB presence, privilege isolation and terminal-transition validation. |
| `docs/ROBOTICS_CODEX_HANDOFF.md` | Environment locks, dated commands/results, audit and deeper learning notes. |
| `AGENTS.md` | Operating contract; required configuration. |
| `.gitignore`, `.vscode/settings.json` | Generated/secret exclusions and existing Conda editor integration. |

Known limits: task-specific local planning, single simulated scene/hardware,
compliant transfer, authored/cooked geometry differences, no certified tracking
tube or continuous-contact guarantee between physics ticks, no real-robot trials. Future ACT, diffusion, RL and compact VLA comparisons
share the evaluator and disclose observation/action rate and privileged expert
access. Substantial data collection, trained-policy success, held-out statistical
comparison, cuRobo experiments and a publishable comparison demo remain incomplete.
