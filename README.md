# Bimanual SO-101 reference benchmark

A simulation-first robotics benchmark: the left five-axis SO-101 picks up a
40 mm cube, transfers it in mid-air to the right SO-101, and the right arm
releases it into a basket. Both arms retreat, the cube settles on the basket
floor, and the scene resets automatically. The reference policy uses geometry,
IK and contact feedback. No learned policy, training or dataset generation is
implemented.

The original complete sequence passed **3/3 repeated physics episodes** at
`9af7dc7`; the user visually confirmed it in Isaac Sim. The benchmark-hardening
revision adds an independent evaluator, transfer instrumentation, faster
collision rejection and fixed regression cases. Its final measurements are
recorded in the [engineering handoff](docs/ROBOTICS_CODEX_HANDOFF.md#engineering-audit--benchmark-hardening-2026-10-22).
Repeated identical starts establish repeatability, not general robustness or
real-robot success.

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
python scripts/run_manipulation.py --stage full --handover-mode staged

# Automated complete episodes with the same physics/controller, no renderer.
python scripts/run_manipulation.py --headless --stage full --handover-mode staged --episodes 3

# Five fixed cube poses, each replanned and independently evaluated.
python scripts/run_manipulation.py --headless --stage full --handover-mode staged --suite perturb

# Deliberately omit donor/receiver closure; both failures must be detected.
python scripts/run_manipulation.py --headless --stage full --handover-mode staged --suite negative
```

GUI is the default; `--headless` is an explicit per-run choice. Add `--exit-after`
to close a GUI run automatically. CPU planning pauses physics while keeping the
viewer rendering. `[PLAN DONE]` reports fresh planning duration; `[CACHE]`
confirms checked-segment reuse. Preserve `outputs/controller/plans/` for warm
runs. The first complete run must validate many paths and is slower.

| `--stage` | Evaluation stops after |
| --- | --- |
| `model` | Reset and measured FK consistency |
| `approach` (default) | Open left hand at the cube, before closing |
| `pickup` | Left pickup, lift and three-second hold |
| `transport` | Left holds at the handover region |
| `handover` | Three seconds of right-only support after donor release |
| `full` | Released cube settled in basket, both arms withdrawn |

Use `--handover-mode baseline` for the original receiving-force schedule.
`--case nominal/x_plus_1mm/x_minus_1mm/yaw_plus_1deg/yaw_minus_1deg` selects one
fixed case; `--suite perturb` runs all five. Seeds 101–105 and exact initial
poses are in `benchmark_core.EVALUATION_CASES`. `--episodes` repeats each case.
These are small diagnostic perturbations, not a sampled task distribution.

Each run writes `outputs/controller/simulation.json` checkpoints and a unique
`full_<UTC timestamp>.json` (or the requested stage name). Reports contain source
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
Object poses, contacts and geometry are an explicitly privileged track used by
the expert/evaluator; cameras are not configured. The expert updates grip at
60 Hz, so a future 20 Hz learned comparison must disclose or remove this
frequency/privilege advantage. This is an incremental adapter, not a registered
Gym environment or an implemented IL/RL/VLA pipeline.

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

Reset checks cube pose/orientation, velocities, arm HOME and measured cube
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

Current CPU regression: **23 tests passed**, including independent evaluator
false-positive/negative cases, timeout/reset, FK/IK, triangle containment,
rotated/acute convex envelopes, ambiguous LP status, trajectory derivatives and
cache invalidation. Complete approach tests compare every optimized hull query
with the original LP; all intermediate poses remain validated.

| Measurement on this workstation | Before | Optimized |
| --- | ---: | ---: |
| Profiled cold approach, same 3,794 samples | 32.53 s | 17.84 s |
| Convex-query inclusive time | 22.76 s | 2.65 s |
| LP calls on that approach | 1,958 | 0 (all separations proved) |
| Geometry-warm approach revalidation, no plan cache | — | 13.11 s |
| Release cube / right-TCP dip, initial paired trial | 17.33 / 17.18 mm | 9.98 / 10.07 mm |
| Pose convergence after release, initial paired trial | 1.85 s | 1.25 s |

These are local timing samples and initial paired transfer measurements, not
universal performance claims. Final sampling is denser than the 3,794-sample
comparison because cache tolerance now has an explicit motion reserve. Full
nominal/perturbed results and cold/warm execution timings are in the handoff.
PhysX reports a residual vertical velocity during grasp even when pose finite
differences converge; reports preserve both and do not call it velocity-settled.

## Environment and dependencies

Validated workstation: Ryzen 9 9950X3D2, RTX 5090 32 GB, Ubuntu 24.04.5,
NVIDIA 580.178.04, CUDA 12.8.1, Python 3.11.16, torch 2.7.0+cu128,
Isaac Sim 5.1.0, Isaac Lab 2.3.0 (`3c6e67bb5c7ada942a6d1884ab69338f57596f77`),
LeIsaac 0.4.0 (`24d3bcd`). NumPy 1.26.0 handles geometry; SciPy 1.15.3 supplies
bounded optimization, rotations, convex hulls and LP; bundled USD composes the
actual robot asset. PhysX supplies dynamics and measured contacts. No new
packages were added. Installation history and pinned-path details stay in the
handoff. cuRobo and MoveIt are assessed there, not installed dependencies.

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
| `scripts/probe_workspace.py`, `scripts/probe_coordinated_workspace.py` | Historical unsafe joint-grid explorations, outside active workflow; proposed removal awaits approval. |
| `scripts/calibrate_so101_fingertips.py` | Pre-existing untracked failed collider discovery, preserved; superseded by maintained triangle audit. |
| `docs/ROBOTICS_CODEX_HANDOFF.md` | Environment locks, dated commands/results, audit and deeper learning notes. |
| `AGENTS.md` | Operating contract; required configuration. |
| `.gitignore`, `.vscode/settings.json` | Generated/secret exclusions and existing Conda editor integration. |

Known limits: task-specific local planning, single simulated scene/hardware,
compliant transfer, authored/cooked geometry differences, no certified tracking
tube, no real-robot trials. Future ACT, diffusion, RL and compact VLA comparisons
must share the evaluator, declare observations/action rate, use held-out scene
conditions and report uncertainty. Training remains outside this assignment.
