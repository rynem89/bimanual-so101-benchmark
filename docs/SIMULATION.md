# Simulation, cameras and deterministic expert

[Start with README](../README.md) · [Demonstrations](DATASET.md) · [Code map](ARCHITECTURE.md)

Use README steps 1–3 to operate the project. This guide explains the existing
stack, robot/camera geometry and the actual controller implementation.

## Contents

- [Environment locks](#environment-locks)
- [Assets and task construction](#assets-and-task-construction)
- [Cameras and coordinate frames](#cameras-and-coordinate-frames)
- [Expert IK and motion](#expert-ik-and-motion)

## Environment locks


This is the authoritative human-readable environment table. Installed packages
and pinned external checkouts remain the source of truth if they disagree.

| Component | Simulation | Learning/conversion/serving |
| --- | --- | --- |
| Conda environment | `leisaac-sim` | `so101-il-py312` |
| Python | 3.11.16 | 3.12.15 |
| NumPy | 1.26.0 | 2.2.6 |
| SciPy | 1.15.3 | Not required by the project learning adapter |
| PyTorch | 2.7.0+cu128 | 2.7.1+cu128 |
| Framework | Isaac Sim 5.1.0, Isaac Lab 2.3.0, LeIsaac 0.4.0 | LeRobot 0.6.1 |
| GPU/driver | RTX 5090 32 GB / NVIDIA 580.178.04 | Same workstation GPU |

Isaac Lab checkout: `/home/rynem8/robotics/IsaacLab`, commit
`3c6e67bb5c7ada942a6d1884ab69338f57596f77` (tag `v2.3.0`). Its internal
Python package/extension version is **0.47.2**; this is a different version field
from the Isaac Lab release tag, not a reason to reinstall. LeIsaac checkout:
`/home/rynem8/robotics/leisaac`, commit
`24d3bcd3f1e4585740fc79921782c41617237812`. Existing compatibility
patch: Isaac Lab's `flatdict` requirement is 4.1.0. Preserve it.
The complete installed learning packages are in
[il_requirements.lock.txt](../configs/il_requirements.lock.txt).

For conversion, training preflight and policy serving:

```bash
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate so101-il-py312
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

LeRobot and the locked simulator require incompatible Python/NumPy versions.
Separate environments preserve both; both policies use the **same official
LeRobot implementation/environment**, not separate custom implementations.
Inference exchanges RGB bytes, lists and targets through a local Unix socket.
The installed optional TorchCodec decoder may warn about shared FFmpeg libraries;
use the existing PyAV backend. No dependency repair is part of this assignment.

## Assets and task construction

The runtime loads the **existing LeIsaac USD**:
`$LEISAAC_ASSETS_ROOT/robots/so101_follower.usd`, normally
`/home/rynem8/robotics/leisaac/assets/robots/so101_follower.usd`.
LeIsaac's `SO101_FOLLOWER_CFG` supplies the articulation/actuators;
[inspect_task_layout.py](../scripts/inspect_task_layout.py) supplies the two-arm
scene, table, dynamic cube, fixed basket, lights and object/joint lookups.
The controller and stationary viewer use this same layout.

URDF describes links, joints and geometry references. USD holds the already
converted scene/articulation consumed by Isaac, including authored joint frames,
collision meshes and physics properties. There is no per-run URDF conversion.
No upstream asset was changed for this viewer.

Each arm has five arm joints plus an actuated jaw. Base positions, world metres:
left `(-0.25,-0.30,-0.030081)`, right `(0.25,-0.30,-0.030081)`; both face +Y.
The table is 1.2 ×0.8 m with top z=0. Cube edge is 0.04 m, mass 0.05 kg, nominal
XY `(-0.34,-0.04)`. Reset spawns its centre at z=0.025 m, then lets it settle near
z=0.020 m. The basket is centred at `(0.34,-0.04)` with a 0.14 ×0.14 m interior.

Our runtime makes table/basket boxes kinematic for pairwise force sensing;
robots/cube expose contact sensors. These are the established benchmark settings.
The viewer reuses them and performs only the established reset settling, then
renders with task physics frozen.

## Cameras and coordinate frames

[policy_cameras.json](../configs/policy_cameras.json) is the sole calibration
configuration. Approval: user visual review, 2026-10-23. Its exact hash is in
[README](../README.md#project-status). The original overview variants remain historical diagnostics;
they are not the policy configuration. `configure_cameras` dynamically adds camera
prims to the scene; it does not change robot mass/collision or the upstream USD.

| Sensor | Parent | Position and aim, metres in that parent frame |
| --- | --- | --- |
| `overview` | World | Eye `(0,-0.20,0.78)`, target `(0,-0.20,0.02)` |
| `left_wrist` | `Left_Robot/gripper` | Position `(0.0141,-0.045,0.005)`, target `(0.0141,0,-0.09)` |
| `right_wrist` | `Right_Robot/gripper` | Same local transform on the right gripper housing |

`gripper` is the rigid wrist-roll housing; `jaw` is the moving finger. The camera
must follow the former, not the opening jaw. The local wrist optical quaternion
is **wxyz** `(0,0,0.9756376988650908,0.21938796811409378)`. The overview ROS optical
quaternion is `(0,1,0,0)`.

Both wrists and overhead use RGB uint8 **640×480**, policy sampling **20 Hz**,
focal length **12 mm**, horizontal aperture **20.955 mm**, near/far **0.01/10 m**.
The pinhole intrinsic matrix, pixels, is:

```text
K = [[366.4996338, 0,           320],
     [0,           366.4996338, 240],
     [0,           0,             1]]
```

All three streams share the current physics boundary before the next action;
they are not separately timed viewport screenshots. Their tensors are rendered
and checked explicitly even while planning pauses physics. The stationary viewer
captures the actual sensors once at the settled, frozen state, before creating
display guides. Its viewport keeps rendering, but it does not repeatedly hide/show
USD geometry or recapture identical stationary RGB. This removes the repeated
stage edits that disrupted inspection; the camera panel is labelled as frozen.
Expert/collection sampling continues through the unchanged policy-boundary
capture path.

### Frames and attachment implementation

A transform `T_A_B` maps B-frame coordinates into A. Metres, radians and seconds
are used internally. Quaternion storage in metadata is wxyz; SciPy conversion
uses xyzw explicitly at the boundary.

- **World**: table and basket frame. Base translations above place both robots.
- **Base/link**: authored USD joint and rigid-link frames, not mesh AABB centres.
- **Gripper/TCP**: `T_world_TCP = T_world_gripper @ T_gripper_TCP`.
  Mesh-derived TCP translation is approximately `(0.0141,0,-0.09)` m, rotation
  identity in the gripper frame. [Expert notes](#expert-ik-and-motion) explain its limits.
- **Optical camera**: ROS +X right, +Y down, +Z forward.
  `T_world_camera = T_world_gripper @ T_gripper_camera` for wrists.
  OpenGL uses −Z forward/+Y up; the adapter performs this convention conversion.

Pinned Isaac Sim's camera path can read stale USD transforms after physics motion.
Our `ArticulatedCamera` reads Fabric, and `BenchmarkEnv.capture_images()` synchronizes
**sensor-only** world transforms from measured PhysX gripper poses while preserving
local extrinsics. It never moves the robot or cube. `camera_metadata()` compares
the actual rendered wrist pose against the independently measured parent:
translation tolerance 20 µm; rotation-matrix element tolerance 2×10⁻⁴. Dataset QA
also checks every capture, intrinsic matrix, boundary tick and sensor frame ID.

### Inspect the actual images

```bash
python scripts/so101.py scene --frames --snapshot-dir outputs/scene_review
```

This opens the actual settled-state sensor panel and optical guides; RGB is
captured before the guides exist, so guides cannot contaminate those images.
The selector changes which feed panels are visible. Stage prim paths
include `/World/envs/env_0/overview` and
`/World/envs/env_0/Left_Robot/gripper/left_wrist` (right analogous). The free viewport
is an additional presentation view. Use `expert --camera-preview <new-directory>`
for sparse phase images during task motion, or `dataset review` for an existing
fully synchronized recording. A static HOME image does not establish visibility
throughout grasp/handover/placement.

The optional [camera mount STL](../assets/camera_mount/README.md) is a licensed
mechanical reference. It is **not instantiated, physically registered, weighted
or collidable** in the current robot. There is no simulated camera body/cable
collision model. Do not infer physical mount feasibility from sensor poses.

### Hardware transfer remains work

Measure actual lenses/distortion, camera-to-housing transforms, robot base/TCP
frames, joint offsets/limits, latency and timestamp alignment. Register mounts,
check cable routes/clearances and match actuation/compliance. Frozen simulation
parameters are an approved benchmark arrangement, not hardware calibration.

API references checked for the pinned stack:
[Isaac Lab 2.3 camera concepts](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/core-concepts/sensors/camera.html)
and the installed `Camera`, `CameraCfg`, `InteractiveScene`, `SimulationContext`
and `omni.ui.ByteImageProvider` implementations. Current upstream documentation
may describe APIs newer than the installed versions.

## Expert IK and motion

The expert is project code. Isaac Lab/LeIsaac provide assets, actuators,
rendering and physics; SciPy provides optimization and interpolation. The task
controller supplies geometry, targets, collision checks, timing and coordination.
It uses neither a learned policy nor MoveIt/ROS/Isaac IK.

### Geometry, forward kinematics and TCP

`SO101Model` reads existing USD joint frames/axes/limits and collision triangles,
including instance proxies. Forward kinematics composes fixed link transforms
and joint rotations. With q the joint angles:

```text
T_world_TCP(q) = T_world_base × product(T_joint_fixed × R_axis(q_i)) × T_gripper_TCP
```

`fk(q, base)` returns link transforms; `tcp_pose(q, base)` returns a 4×4 world TCP
transform. Position is metres; angles are radians. Each six-value arm vector is
shoulder pan/lift, elbow flex, wrist flex/roll, gripper.

`calibrate(model)` intersects opposing authored finger triangles over a 40 mm
cube footprint at gripper-local z≈−0.09 m. It places the TCP near
`(0.0141,0,-0.09)` m, with identity gripper-relative rotation and initial fixed-pad
clearance. Example geometric aperture estimates are 36.23 mm at 20°, 40.67 mm at
25°, 45.42 mm at 30° and 62.96 mm at 45°. These are footprint-specific triangle
estimates, not PhysX contact measurements or guarantees at every approach pose.
A 90° jaw angle has no opposing surface in that footprint; it is not an infinite
usable aperture. Actual gripping is verified dynamically below.

### Bounded inverse kinematics

`SO101Model.ik` calls SciPy `least_squares` with bounds and the default **TRF
(trust-region reflective)** method. Only the **five arm joints** are optimization
variables; the gripper angle is supplied separately. Multiple deterministic
starting seeds seek distinct candidates within authored joint limits.

The usual residual combines scaled position and approach-axis error:

```text
r(q) = [(p_TCP(q) − p_target) / 0.1 m,
        −R_TCP(q)[:,2] − desired_approach_direction]
minimize ||r(q)||² subject to USD lower ≤ q_arm ≤ USD upper
```

The approach axis contributes two independent orientation constraints; axial roll
is free. Where a full rotation is requested, the residual uses its rotation
vector and rejects infeasible results. Five joints cannot generally satisfy an
arbitrary six-dimensional pose. Accepted candidates require <1 mm position and
<1° orientation/approach error; candidate order favours the reference joint pose.
IK convergence alone never means a path is collision-free or a cube was grasped.

### Collision-aware path planning

`CollisionWorld` checks joint limits, table, cube, basket, opposite arm and
non-adjacent self-link pairs. AABB comparisons cheaply reject distant pairs.
Remaining robot/obstacle pairs use conservative convex envelopes: half-space
inflation, sufficient separating-axis tests, and SciPy HiGHS linear-program
feasibility as the unresolved-pair fallback. An unresolved solver result fails;
overlap rejects a candidate rather than certifying a physical collision/contact.
Cube/finger checks use triangle/box geometry. Authored adjacent mechanical joints
and the fixed base/table mounting interface have explicit exceptions.

Every accepted straight joint edge is sampled with a conservative displacement
bound based on a 0.6 m downstream radius and the requested clearance margin.
Curved trajectories are checked on the **actual spline**, including bulges and
joint extrema. Payload-aware checks transform the held cube using its measured
TCP-relative pose; endpoint-only checks are insufficient. Tracking error and
PhysX hull approximation remain reasons to validate actual motion independently.

Blocked open-hand transits use the existing bounded, seeded **RRT-Connect**
fallback over five arm joints. Its edges use the same intermediate checker;
there is no unchecked shortcut/smoothing. Plan-cache validity binds source/model
fingerprints and measured start/payload states. It is not permission to skip
checks on a different pose.

### Trajectory calculation and execution

The current interface selects `continuous` mode. `JointPath` builds a clamped
SciPy cubic spline S(s) through joint waypoints, parameterized by cumulative
joint distance. It applies rest-to-rest quintic time scaling:

```text
u = clip(t / T, 0, 1)
h(u) = 10u³ − 15u⁴ + 6u⁵
q_target(t) = S(h(u))
```

The spline is C² across internal knots; whole-path endpoint velocity is zero.
Exact polynomial derivative extrema bound joint position/speed/acceleration/jerk.
`JointPath` chooses duration at least 1 s using its requested speed and derivative
bound. For a simple joint-linear quintic segment of displacement Δq, peak nominal
speed is `1.875 |Δq| / T`, acceleration `(10/√3)|Δq|/T²`, jerk `60|Δq|/T³`.
These describe requested motion, not measured compliant actuator motion.

The 20 Hz expert selects an **absolute 12-joint position target**, held for three
60 Hz physics ticks. Isaac's existing joint actuators execute it with measured
gravity compensation. This is not torque-learning or a new dynamics controller.
Both arms share the clock; planning can pause physical time while the viewer
renders. Dense collision checks are intentionally costly; they were preserved
in this consolidation.

### Gripping, handover and placement

The state machine opens, reaches pregrasp, descends, closes, lifts and holds.
It uses actual bilateral pad force and cube motion. The bounded gripper feedback
adjusts jaw position to maintain preload, with the validated lift feedback enabled.
A commanded closed jaw or geometric overlap cannot establish possession.

The left transports the held cube; the right follows checked approach waypoints,
establishes light bilateral contact, then acquires preload. The left opens from
its measured loaded pose; a checked outward clearance can remove residual fixed
finger contact. Right-only support must persist for three seconds before transfer
is credited. Donor withdrawal is checked against the receiver and held cube.
The right then moves above/into the basket, opens, retreats, and the cube must
settle inside the interior on its floor. The small observed handover dip is
measured in transfer diagnostics; it is not a commanded teleport or regrasp trick.

### Reset and independent evaluation

Only `BenchmarkEnv.reset` directly writes robot/cube physical state; it seeds,
zeros requested velocities, settles and checks reset error. During task motion,
all commands go through actuators. Expert, replay and future policies share
`BenchmarkEnv`, contact sensors, gravity compensation and `TaskEvaluator`.

The evaluator receives measured `Frame` objects, not expert phase-success claims.
It checks stage order, sustained pickup/lift/right-only support and resting basket
placement, forbidden contacts (0.05 N threshold), joint/effort limits, finite
physics, stable grasp slip and a 300 s timeout. Stage holds use three seconds;
stable grasp limits are 5 mm /10°. Placement requires interior cube geometry,
floor support, low cube velocity, released pads and retreated arms.
Failures, resets and timing are saved per attempt. See [pilot evidence](DATASET.md#measured-pilot-results)
for what has actually passed, not just what these guards are intended to test.

Start studying at `SO101Model.ik`, `CollisionWorld.segment/curve/joint_transit`,
`JointPath.position`, `Episode.run/handover/place`, `BenchmarkEnv.advance/step`
and `TaskEvaluator.update`; all are linked in [the architecture map](ARCHITECTURE.md).
