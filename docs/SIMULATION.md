# Simulation and cameras

[Getting started](GETTING_STARTED.md) · [Controller](CONTROLLER.md) · [Architecture](ARCHITECTURE.md)

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

## Three frozen policy sensors

[policy_cameras.json](../configs/policy_cameras.json) is the sole calibration
configuration. Approval: user visual review, 2026-10-23. Its exact hash is in
[STATUS](STATUS.md). The original overview variants remain historical diagnostics;
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

## Frames and attachment implementation

A transform `T_A_B` maps B-frame coordinates into A. Metres, radians and seconds
are used internally. Quaternion storage in metadata is wxyz; SciPy conversion
uses xyzw explicitly at the boundary.

- **World**: table and basket frame. Base translations above place both robots.
- **Base/link**: authored USD joint and rigid-link frames, not mesh AABB centres.
- **Gripper/TCP**: `T_world_TCP = T_world_gripper @ T_gripper_TCP`.
  Mesh-derived TCP translation is approximately `(0.0141,0,-0.09)` m, rotation
  identity in the gripper frame. [Controller notes](CONTROLLER.md) explain its limits.
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

## Inspect the actual images

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

## Hardware transfer remains work

Measure actual lenses/distortion, camera-to-housing transforms, robot base/TCP
frames, joint offsets/limits, latency and timestamp alignment. Register mounts,
check cable routes/clearances and match actuation/compliance. Frozen simulation
parameters are an approved benchmark arrangement, not hardware calibration.

API references checked for the pinned stack:
[Isaac Lab 2.3 camera concepts](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/core-concepts/sensors/camera.html)
and the installed `Camera`, `CameraCfg`, `InteractiveScene`, `SimulationContext`
and `omni.ui.ByteImageProvider` implementations. Current upstream documentation
may describe APIs newer than the installed versions.
