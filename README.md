# Bimanual SO-101 Benchmark

Simulation-first bimanual manipulation benchmark using NVIDIA Isaac Sim,
Isaac Lab, and LeIsaac.

## Task

Left SO-101 grasps a cube -> moves it to a handover region ->
right SO-101 grasps the cube -> left releases ->
right SO-101 places the cube into a target bin.

## Current milestone

Deterministic physical pickup and lift are simulation-verified in three repeated
episodes. Bimanual handover, placement and full-task success remain unverified.
Training and dataset generation remain gated on the complete deterministic task.

## Validated simulation stack

- Ubuntu 24.04.5 LTS
- NVIDIA driver 580.178.04 open
- CUDA 12.8.1
- Python 3.11.16
- PyTorch 2.7.0+cu128
- Isaac Sim 5.1.0
- Isaac Lab v2.3.0
- LeIsaac 0.4.0

The simulator environment is treated as locked unless a demonstrated
compatibility problem requires a change.

## Current evidence (2026-10-21)

- Human verified the intended finger surfaces, jaw motion and stationary cube.
- Full-footprint triangle geometry defines the TCP relative to the gripper link:
  translation `(0.0141000004, 0, -0.09)` m, identity rotation. X closes the jaws,
  Y spans pad width, and -Z is approach. A 40 mm cube plus a 1 mm-per-side test
  margin clears the authored finger meshes at a 45-degree opening.
- Multi-start bounded IK respects five-arm-joint limits. A downward pregrasp at
  120 mm height is infeasible; the validated pregrasp is 55 mm above the table.
- HOME → open → pregrasp → grasp passes 3,794 nominal collision samples, followed
  by measured force and cube-drift checks at every physics tick. Three approaches
  had zero cube drift before contact and final TCP error below 0.001 mm.
- Physical closure at approximately 21.71 degrees develops 3.39/3.75 N pad forces.
  Three pickup episodes lifted the cube to at least 73.27 mm centre height and
  held it for 3 seconds, with both pad forces above 1.57 N. No attachment or
  task-time object state writes are used. These are deterministic repeated
  episodes, not randomized robustness statistics or full-task success.

Authored triangles differ from cooked PhysX hulls. Conservative envelopes reject
unsafe nominal paths; actual tracking and contacts are checked independently.
A 1 N preload failed to lift and was correctly rejected. Gravity feed-forward
was necessary to prevent arm sag and unintended cube contact during approach.
See [handoff](docs/ROBOTICS_CODEX_HANDOFF.md) for failures and measured evidence.

## Architecture and controller

`so101_model.py` reads the pinned USD and provides transforms, mesh geometry and
IK. `so101_planning.py` validates nominal paths and a predicted carried-cube
volume. `run_manipulation.py` reuses `inspect_task_layout.py`, applies smooth
actuator commands plus bounded gravity torque, monitors physical contacts, and
records episode outcomes. Only reset writes robot/cube state directly.

The controller gives fixed table/bin shapes kinematic rigid-body representation
so GPU contact filters can observe them; their shapes and poses remain fixed.
It checks measured cube support against its 50 g weight before proceeding.
The original shared scene and historical diagnostics remain unchanged.

After the environment activation below:

```bash
# GUI replay of the verified milestones. These commands open the simulator.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_manipulation.py --stage approach
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_manipulation.py --stage pickup
# Automated regression runs the same physics/controller with rendering disabled.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_manipulation.py --headless --stage pickup --episodes 3
```

GUI mode is the default. After a successful evaluation the final scene stays
open, frozen for inspection, until you close its window. Add `--exit-after` for
a finite GUI run. Both arms and the bin are visible in the default camera.
Headless testing does not change any script's default into headless mode.
Each run records a timestamped stage report with source hashes as well as the
latest `simulation.json`; generated evidence stays in ignored `outputs/`.

Transport and receiver work is in progress on the feature branch. One earlier
transport trial reached the central handover region with maximum relative cube
slip 0.0462 mm and rotation 0.185 degrees; this does **not** prove transfer.
The receiving approach is being corrected because a joint-linear path collided
with the cube despite feasible endpoints. Experimental GUI command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_manipulation.py --stage handover
```

This command must report `handover_success: true` before transfer can be claimed.
Bin placement and full-task evaluation are implemented but still experimental:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_manipulation.py --stage full
```

The full stage requires right-only support after donor release, both arms to
retreat, the entire cube to stay inside the bin for three seconds, floor support
of 0.3–0.7 N, linear speed below 0.01 m/s and angular speed below 0.1 rad/s.
It cannot report full success from merely reaching the bin or opening a jaw.
`simulation.json` also checkpoints the active phase and partial measurements
every two simulated seconds, so interrupted runs can be diagnosed.

The first nominal-path validation takes about 3.5 minutes; valid generated plans
are cached using asset/model/planner fingerprints. Outputs are ignored under
`outputs/controller/`: `simulation.json` contains stage outcomes, per-episode
measurements, failure/reset evidence and sampled state traces. Exit 0 means the
**requested stage** passed; full-task success remains false until transfer and
placement are implemented. Exit 1 means a prerequisite or objective failed.

## Run in the existing environment

Every new shell needs activation; do not install or upgrade dependencies:

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
export LEISAAC_ASSETS_ROOT=/home/rynem8/robotics/leisaac/assets
python --version
```

Headless physics audit (120-second timeout is a failure guard; verified exit 0):

```bash
mkdir -p outputs/finger_audit
timeout --signal=TERM --kill-after=10s 120s python scripts/audit_so101_finger_geometry_v2.py --headless > outputs/finger_audit/headless.log 2>&1
```

Visual checkpoint:

```bash
python scripts/audit_so101_finger_geometry_v2.py
```

Watch the left jaw at 0, 45, 90, 0 degrees. A yellow line marks the triangle
surface section at gripper-frame y=0, z=-80 mm; it disappears when the moving
jaw no longer intersects that section. Check whether its ends lie on the
intended inner finger surfaces and whether the cube stays still. This line is
not a TCP marker. Close the simulator after inspection.

CPU-only asset audit and focused tests use the USD libraries already bundled
with this exact Isaac Sim installation (no Kit/GPU launch):

```bash
USD_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
export PYTHONPATH="$USD_LIB${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$USD_LIB/bin:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python scripts/audit_so101_finger_geometry_v2.py --offline --output-dir outputs/finger_audit_offline
python tests/test_finger_geometry.py
python tests/test_so101_model.py
```

Use a fresh activated shell for the simulation commands after this CPU-only
setup. Offline output never substitutes for live physics measurements.

## File inventory / learning notes

All script entry points below run with `python scripts/<name>.py` after the
activation above. Simulator scripts read the existing LeIsaac USD; older
interactive diagnostics keep their viewer open until closed.

| File | Purpose, inputs / outputs, status |
| --- | --- |
| `scripts/so101_model.py` | Reusable CPU USD geometry, FK and bounded multi-start IK; USD → model and `outputs/controller/offline.json`. Run `python scripts/so101_model.py` with bundled USD paths below. |
| `scripts/so101_planning.py` | Imported collision/path checks, lift and coupled handover planning, Cartesian receiver/withdrawal legs, loaded bin planning. Conservative envelopes + triangle/box tests; no simulator or CLI side effects. Physical gates are separate. |
| `scripts/run_manipulation.py` | `--stage model/approach/pickup/transport/handover/full`, `--episodes`, `--output-dir`; GUI by default, optional `--headless` / `--exit-after`. Scene + plans → checkpointed and timestamped JSON outcomes. Pickup/transport verified; transfer/placement experimental. |
| `tests/test_so101_model.py` | Five CPU tests for containment, independent USD FK, FK/IK round trips, unreachable targets and opening geometry. Run with the bundled USD paths below. |
| `scripts/inspect_task_layout.py` | Shared task scene and live FK inspection; USD + fixed layout → console poses. Active scene foundation; importing launches Kit. |
| `scripts/audit_so101_finger_geometry_v2.py` | Maintained geometry/jaw diagnostic; USD and optionally measured body transforms → hierarchy, geometry CSV, surface sections and JSON in `outputs/finger_audit`. `--headless`, `--offline`, `--output-dir` supported. Not a calibrated TCP. |
| `tests/test_finger_geometry.py` | `python tests/test_finger_geometry.py` with USD paths above; analytic triangle inputs → three unittest checks for face-interior intersections, degenerate/parallel triangles and failing prerequisites. |
| `scripts/verify_motion_gripper_reset.py` | Existing regression; scene → console motion/gripper/reset checks. Preserved unchanged; deliberately displaces cube only in reset test. |
| `scripts/inspect_bimanual_scene.py` | Earlier scene, independent motion and stability diagnostic; USD → console. Preserved baseline. |
| `scripts/inspect_fk_frames.py` | Earlier world/base/link transform experiment; USD → console. Preserved baseline. |
| `scripts/probe_workspace.py` | Historical joint-grid exploration; scene → console distances. Known collision risk; not an active planner. |
| `scripts/probe_coordinated_workspace.py` | Historical coordinated joint-grid exploration; scene → console distances. Known collision risk; not an active planner. |
| `scripts/calibrate_so101_fingertips.py` | Pre-existing untracked calibration attempt, preserved unchanged. Misses instance-proxy colliders; its proxy distance is not a TCP. Not an active validation entry point. |
| `docs/ROBOTICS_CODEX_HANDOFF.md` | Environment locks, mission gates, measured results and next checkpoint. |
| `AGENTS.md` | User-provided operating contract, preserved. |
| `.gitignore`, `.vscode/settings.json` | Repository hygiene and existing editor configuration. |

The maintained controller includes the complete task sequence behind measured
stage gates; transfer and placement are still under validation. Historical diagnostics
are retained in place; nothing has been deleted. `outputs/` and Python caches are
ignored generated evidence, not source. No remaining unknown source files were
found in the initial inventory.

For self-study: `mesh_evidence` shows USD row-vector transforms and triangle
planes; `x_surface_intersections` uses barycentric coordinates to intersect
face interiors; `finger_sections` applies measured jaw rotation/translation.
Keep authored geometry, actual body pose and effective physical contact distinct.

API references: [OpenUSD bounds cache](https://openusd.org/release/api/class_usd_geom_b_box_cache.html)
and [Isaac Sim 5.1 application lifecycle](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html).


## Learning notes / code map

Start with `SO101Model.fk`: each child transform composes the parent joint frame,
axis rotation and inverse child joint frame. `ik` adds bounds and checks the
achieved pose rather than trusting solver convergence alone. `CollisionWorld`
separates conservative geometry rejection from runtime force evidence.
`Episode.run` shows the reset/approach/closure/lift/hold sequence and the measured
conditions required for success. The handover solver is a feasibility study only.
