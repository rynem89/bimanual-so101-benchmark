# How the deterministic expert works

[Run instructions](GETTING_STARTED.md#3-run-the-expert) · [Source map](ARCHITECTURE.md)

The expert is project code using Isaac Lab/LeIsaac for assets, actuators, rendering
and physics. It does **not** use learned control, MoveIt, ROS planning or Isaac's
IK solver. SciPy supplies numerical optimization/interpolation; our code supplies
the task-specific geometry, targets, checks, timing and coordination.

## 1. Geometry, forward kinematics and TCP

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

## 2. Bounded inverse kinematics

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

## 3. Collision-aware path planning

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

## 4. Trajectory calculation and execution

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

## 5. Gripping, handover and placement

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

## 6. Reset and independent evaluation

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
Failures, resets and timing are saved per attempt. See [pilot evidence](PILOT_RESULTS.md)
for what has actually passed, not just what these guards are intended to test.

Start studying at `SO101Model.ik`, `CollisionWorld.segment/curve/joint_transit`,
`JointPath.position`, `Episode.run/handover/place`, `BenchmarkEnv.advance/step`
and `TaskEvaluator.update`; all are linked in [the architecture map](ARCHITECTURE.md).
