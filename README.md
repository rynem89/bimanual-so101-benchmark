# Bimanual SO-101 Benchmark

Simulation-first bimanual manipulation benchmark using NVIDIA Isaac Sim,
Isaac Lab, and LeIsaac.

## Task

Left SO-101 grasps a cube -> moves it to a handover region ->
right SO-101 grasps the cube -> left releases ->
right SO-101 places the cube into a target bin.

## Current milestone

Deterministic physical pickup and lift are simulation-verified in three repeated
episodes. The complete pickup → handover → basket placement → retreat sequence
now passes **3/3 full physics episodes**, including automatic reset between
episodes. Final GUI regression is in progress.
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

## Current evidence (2026-10-22)

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
- Force-regulated transport reached the handover region with maximum relative
  slip 0.110 mm and rotation 0.177 degrees. A light receiving preload followed
  by donor release established three seconds of right-only support: minimum
  forces 4.445/3.976 N, zero left-pad force, cube height >=173.407 mm. These
  measurements are saved in timestamped stage reports.
- Three complete episodes: both arms withdrew, the entire cube stayed inside the
  basket for three seconds, minimum floor support was 0.490499 N, maximum linear
  speed 0.000475 m/s and angular speed 0.01098 rad/s. Final centre was
  `(0.346719, -0.032673, 0.030000)` m. Carried-cube environment force remained
  zero; peak measured joint speed/torque were 2.151 rad/s / 1.514 N m.
  This baseline releases above the basket (centre about 106 mm) and lets the cube
  settle on its floor (centre 30 mm); it does not lower the cube into floor contact.
- Final regression: `full_20261022T085453054801Z.json`, exit 0, all seven stage
  counts 3/3. Minimum bin support >=0.490498 N, maximum linear speed <0.000475 m/s,
  angular speed <0.01098 rad/s. Episode three reused all stage/segment plans with
  no new collision solves. Ten focused CPU tests passed. These are repeated
  deterministic conditions, not randomized robustness or real-robot trials.

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

Choose the stage explicitly; the startup `[RUN]` line confirms what is running:

| Stage | Stops after | Example GUI command |
| --- | --- | --- |
| `approach` | Open left hand at the cube, before closing | `python scripts/run_manipulation.py --stage approach` |
| `pickup` | Left pickup, lift and hold; right arm stays HOME | `python scripts/run_manipulation.py --stage pickup --episodes 3` |
| `transport` | Left holds the cube at the handover region | `python scripts/run_manipulation.py --stage transport` |
| `handover` | Right-only hold after left releases | `python scripts/run_manipulation.py --stage handover` |
| `full` | Cube settled in basket, both arms withdrawn | `python scripts/run_manipulation.py --stage full --episodes 3` |

Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` before these commands as above.
Only `full` attempts donor withdrawal and basket placement. A saved failed
`full` report must not be interpreted as a failed `pickup` run.

GUI mode is the default. After a successful evaluation the final scene stays
open, frozen for inspection, until you close its window. Add `--exit-after` for
a finite GUI run. Both arms and the bin are visible in the default camera.
Lengthy CPU planning runs in a worker while the GUI renders without advancing
physics, so the viewer remains available during collision validation.
Headless testing does not change any script's default into headless mode.
Each run records a timestamped stage report with source hashes as well as the
latest `simulation.json`; generated evidence stays in ignored `outputs/`.

Transport and receiver development is recorded on the feature branch. One earlier
transport trial reached the central handover region with maximum relative cube
slip 0.0462 mm and rotation 0.185 degrees; this does **not** prove transfer.
The receiving approach now uses 16 checked Cartesian legs because a joint-linear
path collided with the cube despite feasible endpoints. Handover GUI command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_manipulation.py --stage handover
```

This command requires `handover_success: true` and measured right-only support.
Complete task, with the simulator window:

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
| `scripts/run_manipulation.py` | `--stage model/approach/pickup/transport/handover/full`, `--episodes`, `--output-dir`; GUI by default, optional `--headless` / `--exit-after`. Scene + plans → checkpointed and timestamped JSON outcomes. Complete task 3/3 in physics; GUI regression in progress. |
| `tests/test_so101_model.py` | Seven CPU tests for containment, rotated convex envelopes, ambiguous LP failure handling, independent USD FK, FK/IK round trips, unreachable targets and opening geometry. Run with the bundled USD paths above. |
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
stage gates; three repeated full episodes passed. Historical diagnostics
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

### What the controller calculates

This is a deterministic, measured state machine:
`reset → approach → close → lift → transport → receive → release left → withdraw left → carry right → release in bin → retreat → evaluate`.
Every motion is an actuator command. Object/robot state is written directly only
at episode reset. Planning uses CPU NumPy/SciPy; Isaac Sim/PhysX executes the
rigid-body dynamics and supplies measured joint states and contact forces.

**Frames and forward kinematics.** Positions are metres, angles radians, and
4×4 matrices map column vectors from a local frame into its parent/world frame.
`SO101Model` reads joint axes, joint-frame transforms and limits from the actual
USD. For each joint it calculates

```text
T_world_child(q) = T_world_parent · T_parent_joint0 · R_axis(q) · inverse(T_child_joint1)
T_world_TCP(q)   = T_world_gripper(q) · T_gripper_TCP
```

The calibrated TCP offset is `(0.0141000004, 0, -0.09)` m in the gripper link.
TCP X is the closing direction; -Z is the approach direction. Measured simulator
link poses independently check the FK model at motion endpoints. These checks
measure simulator consistency, not real-world calibration accuracy.

**Inverse kinematics.** `SO101Model.ik` uses SciPy 1.15.3
[`least_squares`](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html),
with its Trust Region Reflective (`trf`) algorithm, joint bounds and numerical
two-point derivatives. It minimizes `0.5 × ||r(q)||²` for the five arm joints;
the sixth actuator opens/closes the gripper separately. The residual is:

```text
r_position = (p_TCP(q) - p_target) / 0.1 metres
r_direction = -R_TCP(q)[:, 2] - desired_approach_unit_vector
# Or, when a full orientation is requested:
r_orientation = rotation_vector(inverse(R_target) · R_TCP(q))
r = concatenate(r_position, r_direction or r_orientation)
```

Position plus approach direction imposes five independent constraints and leaves
tool roll free. A complete orientation asks this five-joint arm to satisfy six
pose constraints, so many requests are infeasible. Solver termination alone is
insufficient: accepted solutions must achieve <1 mm position error and <1 degree
orientation/direction error, inside all joint limits. Several deterministic
starting guesses (random seed 101, plus the current/reference pose) reduce local
solver failures; this is a bounded local search, not a proof of reachability.

`handover_candidates` jointly solves the ten arm joints of both robots. It aligns
the measured cube pose with the receiver on an orthogonal pair of cube faces,
instead of independently prescribing two incompatible six-dimensional poses.
The cube-to-TCP transform is measured after grasping; it is never assumed to be
identity or used to create a physics attachment.

**Path selection and collision checking.** The task planner chooses staged joint
waypoints and checks every connecting segment. Close receiver approaches and
donor withdrawals use 16 short Cartesian legs, solving IK along the tool axis.
The donor first separates laterally, then retracts, turns away and returns HOME.
This is a task-specific waypoint planner; it does not guarantee a solution in
arbitrary clutter.

Axis-aligned bounds cheaply reject separated mesh pairs. Remaining robot/bin
pairs use conservative convex envelopes: SciPy
[`linprog(method='highs')`](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.linprog-highs.html)
tests whether their inflated half-space inequalities have a common point.
Gripper/cube tests use authored triangles, separating-axis tests and containment
checks because convexifying the whole gripper would fill its empty grasp region.
The carried cube is transformed along the path and checked against obstacles.
These geometric checks predict clearance; measured forces and slip remain
independent execution gates.

For joint-linear segments, the number of samples is
`ceil(0.6 metres × sum(abs(q_end-q_start)) / margin) + 1`, at least two.
The 0.6 m downstream radius bounds nominal point movement to at most half the
collision margin from the nearest sample. Typical margins are 1–3 mm depending
on the check. This bound does not cover servo tracking error or differences
between authored meshes and PhysX collision hulls; runtime checks cover those
failure modes. Intended finger/cube contact and adjacent mechanical interfaces
have explicit exceptions; donor contact is forbidden again after transfer.

**Trajectory timing.** Once a joint segment is accepted, `move` / `move_right`
apply the quintic interpolation below, with zero commanded velocity and
acceleration at each endpoint:

```text
u = elapsed_time / T,  0 <= u <= 1
s(u) = 10u^3 - 15u^4 + 6u^5
q_desired(u) = q_start + s(u) · (q_end - q_start)
T = max(1 second, 1.875 · max(abs(q_end-q_start)) / (0.35 radians/second))
```

`max(ds/du)=1.875`, which gives the duration formula and nominal joint command
speed limit. Actual joint speeds are measured separately. The existing implicit
PD actuators track these commands, with PhysX gravity-compensation torque added
as feed-forward. Original gains, gravity, mass, friction and 10 rad/s / 10 N m
actuator limits remain enabled. This is smooth interpolation with collision
validation, not an optimized global time/jerk trajectory.

**Grasp and objective evaluation.** The left hand closes until both pad forces
exceed 3 N, then proves lift and hold. Loaded grip feedback keeps the smaller
pad force near 3–4 N with bounded jaw-command changes. The receiver first reaches
1 N bilateral preload while both compliant arms hold the cube, then increases
preload as the donor releases. Right-only hold requires zero donor contact and
three seconds of sustained receiver support. Slip >5 mm or rotation >10 degrees,
forbidden contact >=0.05 N, or actuator limit violations stop the episode.
Placement requires the entire cube inside the bin, both arms withdrawn, three
seconds of floor support and low measured linear/angular speeds. Failed episodes
attempt reset and retain the failure evidence; repeated full runs test reset
from a cube already transferred or placed.

An indeterminate HiGHS result is retried with dual simplex and presolve disabled.
Only definite feasible/infeasible status is accepted; an unresolved retry still
fails planning. This handles a numerical failure seen on a nearly tangent pair
after reset without silently treating an unknown status as clear space.

### Why collision planning takes time

The checks are CPU geometry and optimization, separate from GPU physics/rendering.
Thousands of intermediate poses are tested; checking endpoints alone missed real
cube collisions during development. Vectorized mesh bounds reduced a measured
approach plan from about 205 s to 32.23 s on this workspace. A comparison corpus
of 1,080 checks (both bases, rotated cubes, two margins, contact exceptions and
random/approach poses) produced identical fault lists: 76.55 s before versus
30.06 s after, a 2.55× improvement on that mixed corpus.

Accepted plans are cached in `outputs/controller/`. The cache checks asset/model/
planner fingerprints and the actual held state, so subsequent matching runs
skip expensive validation. Changing code, assets or the grasp state can trigger
replanning. Keep the output directory between runs. `OPENBLAS_NUM_THREADS=1`
and `OMP_NUM_THREADS=1` avoid unnecessary thread overhead for these small solves;
they do not change simulation physics. GUI planning renders without stepping
physics while CPU checks run; completed GUI trials keep their final scene open.
Closure, release and post-placement retreat segments are cached too, including
their cube pose, opposite-arm transforms, joint endpoints and contact/margin
options. `[CACHE]` confirms reuse; `[PLAN DONE]` reports fresh computation time.
A regression moved the cube 6 mm into a finger: the cache invalidated and the
new collision was rejected. Cached failed checks remain failures.
Matching tolerates `1e-5` per joint angle, translation or rotation-matrix entry.
With the same 0.6 m radius bound, combined moving-arm/opposite-arm/payload changes
stay below 0.1 mm, inside the remaining sampling-clearance allowance of the
>=1 mm margins. This avoids replanning for measured reset differences of about
`1e-6` while retaining every physical contact/slip gate. Code or asset changes
still invalidate the cache. Fresh retreat validation measured 60.22 s; cached
retreat skips its 2,661 collision samples entirely.

Start with `SO101Model.fk`: each child transform composes the parent joint frame,
axis rotation and inverse child joint frame. `ik` adds bounds and checks the
achieved pose rather than trusting solver convergence alone. `CollisionWorld`
separates conservative geometry rejection from runtime force evidence.
`Episode.run` owns reset/approach/closure/lift/transport. `handover` brings the
receiver onto orthogonal cube faces and proves right-only support after release.
`place` validates withdrawal and loaded travel, then checks the released cube's
containment, velocity and floor support after both arms retreat. `active_grips`
regulates measured pad force as gravity loading changes; it does not attach the
object. `carried_segment` predicts payload geometry using the measured cube/TCP
transform, while physical slip and contact checks independently reject execution.
