# Project handoff — Simulation-first bimanual SO-101
Updated 2026-10-23. Verify local files and current Git state before acting.

**Current learning gate:** `feat/il-camera-dataset-v2`. Approved cameras frozen.
The user authorized only the 12-attempt pilot: all 12 have now completed physically
and passed automated QA; 3 selected physical action replays also passed. Stop for
human review. No further collection, augmentation or training. The user separately
authorized publishing source/docs on this feature branch for GitHub review on
2026-10-23; `main` and local datasets/weights/generated outputs remain untouched.
Protocol remains provisional, not frozen. See `docs/PILOT_RESULTS.md` and the final
dated section below for current evidence and exact local artifact paths.
The older branch/state paragraph below is historical deterministic-reference evidence.

**Latest maintained state:** branch `feat/fingertip-geometry-calibration`;
geometry/TCP, approach, pickup, transport, handover and basket placement/retreat
have passed measured simulator gates in **3/3 complete episodes**, including
automatic reset. The user subsequently visually confirmed the complete physical
sequence (mission brief dated 2026-10-22). Benchmark hardening results below
supersede earlier evaluation and timing claims. The dated execution
updates below supersede the original snapshot's missing bounds/unverified TCP
and early Git-state descriptions. Use README commands and current reports.

## Purpose and portfolio goal
Build a defensible robotics/embodied-AI portfolio demonstrating simulation, forward/inverse kinematics, manipulation, physics, collision-aware planning, control, evaluation, imitation learning and eventual sim-to-real. Target roles in robotics software, embodied AI, machine learning/RL and automation. Produce a strong **first LinkedIn post** when there is an actual reliably functioning, filmed and objectively measured deterministic milestone (preferably complete bimanual handover; smaller pickup milestone only if meaningful). Do not advertise unverified capability, fabricated benchmark numbers, or imply trained AI when using deterministic control. Later create a clean GitHub README, architecture figure, demo video and measured success rate; keep repository private until user explicitly approves publication. Explain engineering tradeoffs and why Isaac Lab was selected; MoveIt 2/ROS 2 integration is a later optional comparison, not a reason to restart.

## Hard mission and gates
Left SO-101 physically picks 40 mm cube, lifts, hands to right SO-101 in mid-air, right places cube in fixed bin, both retreat. Simulator gravity/contacts active. **Milestone #1: entire loop repeated, with automatic reset and objective success detection, before ACT / Diffusion Policy / SmolVLA.** No cheating via teleporting the cube during task execution; state write/reset permitted only at episode reset, clearly separated from grasp/manipulation.

## Environment: persistent vs per-process
OS Ubuntu 24.04.5, most recent observed kernel 7.0.0-38-generic (older environment document 7.0.0-34). Ryzen 9 9950X3D2, RTX 5090 32GB; NVIDIA 580.178.04 (open); Miniforge conda env `leisaac-sim`, Python 3.11.16, PyTorch 2.7.0+cu128, CUDA toolkit 12.8.1, Isaac Sim 5.1.0, Isaac Lab v2.3.0 pinned commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77`, LeIsaac 0.4.0 pinned `24d3bcd`. Editable installations already present.

Project directory `~/robotics/bimanual-so101-benchmark`.
External checkouts `~/robotics/IsaacLab` and `~/robotics/leisaac` (outside project repository). USD robot asset `~/robotics/leisaac/assets/robots/so101_follower.usd`. Additional LeIsaac asset `~/robotics/leisaac/assets/scenes/table_with_cube/scene.usd`.

**New terminal setup** (only when needed; no repeated installation/URDF conversion/import):
```bash
cd ~/robotics/bimanual-so101-benchmark
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate leisaac-sim
export LEISAAC_ASSETS_ROOT="$HOME/robotics/leisaac/assets"
python --version
```
The export is a **process/shell environment setting** for scripts that rely on LeIsaac assets root, not a repeated import. Inspect current code to confirm exact variable expectations before relying on it. Avoid `conda init` modifications without need. When Codex invokes a new non-interactive shell, it may need to source conda.sh and activate the environment **in that shell**. The working scene imports its already-existing USD reference; it is not necessary to import URDF per run. Conda package installation, cloned source files, and USD assets persist on disk. Check `which python`, `sys.executable`, and `nvidia-smi` as needed. Never assume Codex cloud environment can use local RTX 5090 or run Isaac GUI: prefer local Codex on the workstation for physics tests. If GUI access isn't viable, run headless and ask human to visually inspect only the milestone candidate.

Known compatibility: Nvidia driver R595 previously caused an Isaac Sim crash; R580.178.04 worked. Intentional editable Isaac Lab metadata patch `flatdict==4.0.1` to `flatdict==4.1.0` in `~/robotics/IsaacLab/source/isaaclab/setup.py`; don't revert. Do NOT upgrade stack or touch OS/driver/BIOS/secure boot/disks. ROS 2, LeRobot training and MoveIt 2 deliberately not installed for this milestone.

## Git and evidence
Private remote: `rynem89/bimanual-so101-benchmark`. Observed working branch `feat/gripper-characterization`. Earlier committed baseline: `8b598d9` task layout/FK, `e887eb1` workspace probes (intentionally collision-prone). User confirms later verification script committed/pushed; exact hash to determine from `git log`. `main` protected conceptually: NEVER merge/change it without approval. Current working tree and untracked files unknown; **audit first**. Initial scripts known: `inspect_task_layout.py`, `inspect_fk_frames.py`, `probe_workspace.py`, `probe_coordinated_workspace.py`, `verify_motion_gripper_reset.py`, `calibrate_so101_fingertips.py` (failed discovery), `audit_so101_finger_geometry_v2.py` (successful discovery), `verify_task_reset.py` (invalid superseded test; may already be removed). Do not delete by filename alone.

## Physics scene details
2 x SO-101; each 5 arm joints (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`) + `gripper`. Bodies `base`, `shoulder`, `upper_arm`, `lower_arm`, `wrist`, `gripper`, `jaw`.
Left base = (-0.25,-0.30,-0.030081) m; right base = (+0.25,-0.30,-0.030081) m; arms face +Y.
Table 1.2 x 0.8 m, tabletop Z=0; cube edge .04 m, mass .05 kg, settled center approx (-.34,-.04,.02) m. Fixed bin center XY=(+.34,-.04) m; inner opening .14 x .14 m, top rim Z~.085m. Candidate handover (0,+.015,+.18) m. Candidate pregrasp TCP location (-.34,-.04,+.12) m, **orientation not chosen, reachability unproven**. Normal gravity on. The original joint grid tests struck the cube and bin; these are not motion planners.

## Verified trials vs remaining unknowns
User visually confirmed 2 shoulders moved +15/-15 deg, both jaws open/close, and 120mm cube sideways displacement/reset. Diagnostic measured gripper link displacement 67.4mm each; 3 reset trials displaced 118.5mm and restored within 1.47mm (<2mm threshold). Good but not evidence of grasp. Cube drift 0 in jaw inspection.
Latest finger audit `audit_so101_finger_geometry_v2.py` successfully found collision shapes and user visually saw both jaws open/close 0,45,90,0; joint measured approx 0/44.99/90/0 deg; jaw link relative to gripper origin remains ~(+0.0202,+0.0188,-0.0234)m, but rotates by 45/90deg. Outputs at `~/Downloads/so101_finger_audit/{hierarchy.txt,geometry.csv,measurements.json}`.

Critical finding: collision hierarchy uses USD **instance proxies** and inherited CollisionAPI on `/collisions` Xform. Old `calibrate_so101_fingertips.py` using GetChildren failed to find geometry. New audit correctly enumerated 105 prims; collision shapes:
- `gripper/collisions/sts3215_03a_v1/mesh` (servo housing, not necessarily finger)
- `gripper/collisions/wrist_roll_follower_so101_v1/mesh` (candidate stationary finger structure)
- `jaw/collisions/moving_jaw_so101_v1/mesh` (moving finger)
But `geometry.csv` has blank collision mesh bounds (`problem=no bound`), whereas visual counterparts have bounds; **not calibrated**. Do not claim verified 40mm aperture, contact surfaces or TCP. Root likely from bounds extraction of instance proxies—investigate, not assume. Avoid claiming visual AABB equals PhysX contact.

## Mandated milestone definitions
**M1 Geometry + TCP:** infer contact surfaces from USD instance/reference composition and mesh triangles, validate contact gap vs jaw angle using direct geometric measurements and if possible a safe physical cube-contact test. Annotate TCP transform relative to gripper rigid-body link (units/frame). Fail clearly if unresolved.
**M2 IK / safe pregrasp:** select physically achievable approach orientation for 5-axis arm, calculate FK/IK with joint limits, check full planned path with collision geometry against table/cube/bin/opposite arm; execute controlled travel to pregrasp and measured pose error; cube does not move (e.g. <2mm) and no forbidden contacts.
**M3 Pickup:** controlled approach, close on cube, lift and verify cube follows gripper for a sustained interval; monitor object pose/contact; reset on failure.
**M4 Handover:** left delivers cube to feasible transfer zone, right approaches and physically closes, left releases only after right controls cube; verify transfer with object trajectory and contacts.
**M5 Place + repeated episodes:** right moves into bin, opens jaw, object settles inside interior, retreat; objective full-task success incl no unintended collisions, reproducible start reset, repeated seeds/episodes and quantitative metrics (report real counts). Use staged deterministic state machine. Instrument failures.
**M6 Learning ONLY after M5:** demonstration capture, dataset spec, ACT, diffusion policy, compact VLA (SmolVLA), common metrics, model comparisons, sim-to-real roadmap.
These are results to drive toward, not claims they are complete. If M4 handover is infeasible at current robot base spacing/orientation, measure feasibility and propose minimal geometry changes with physical rationale before modifying task definition.

## Repository cleanliness and learning
For every *maintained* file, document purpose/entry point/dependencies/run command/status in project `README.md`. Keep compact coherent project tree, no proliferation. Cleanups: inspect all tracked/untracked; classify; propose removals with reason, wait for human approval before tracked/ambiguous deletes; prefer removing stale debug probes from *active workflow*, retain traceability in Git history. Don't rewrite good code merely to change style. Update docs only when useful, avoid duplicate progress documents. Add focused comments for frames/units and non-obvious code; normal black/ruff/pytest only if already present and compatible, don't install unapproved tools.
At each milestone hand off concise RESULT / EVIDENCE / FILES / RUN COMMAND / RISKS / HUMAN VERIFY / NEXT. Include a short "Learn later" explaining where to study algorithms; no massive tutorial in main agent run. Preserve human control on scene visual checks and architecture.

## First agent task scope
Read-only inventory and environment verification, THEN autonomously implement and test M1 and move toward M2 without new user prompts for routine edits. Branch/worktree protection first. If cannot run Isaac locally or inspect 3D geometry, stop at one narrow human checkpoint with precise instructions and captured output; never bluff. Do not delete files or merge to main. At milestone end update README/status. Use official pinned NVIDIA/IsaacLab docs for API details.

## Local execution update — 2026-10-21

Environment verified with both commands, each returning Python 3.11.16:

```bash
/home/rynem8/miniforge3/envs/leisaac-sim/bin/python --version
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
python --version
```

Baseline `c121012` on `feat/gripper-characterization`; created branch
`feat/fingertip-geometry-calibration`. Existing untracked AGENTS, handoff and both
calibration scripts were inspected before edits. Original scripts copied to
`/tmp/so101-original-scripts` as a task-local precaution. No deletion, reset,
remote push, dependency installation or stack modification. Isaac Lab commit
`3c6e67bb5c7ada942a6d1884ab69338f57596f77` and LeIsaac `24d3bcd` match the lock.
Sandbox GPU query failed (no NVIDIA device nodes); approved host-side query
returned RTX 5090 / driver 580.178.04. Host execution can run the physics audit.

### M1 progress, not completion

**Correction to prior diagnosis:** bounds were blank because collider purpose is
`guide`, excluded by the audit's BBoxCache purpose list. `ignoreVisibility=True`
alone did not fix them. Instance proxies were already correctly traversed by v2.
Read-only offline checks used the existing bundled OpenUSD 0.24.5 libraries.
Including `guide` recovers all bounds; full transformed-vertex bounds agree with
USD bounds to <=1e-6 m. Asset metresPerUnit=1.0. Offline asset has 104 composed
prims; spawned scene has 105 (different roots/composition, not a failed count).

Collision triangle counts:
- Servo housing: 57,240 vertices / 19,080 triangles.
- Fixed finger structure: 86,388 vertices / 28,796 triangles.
- Moving jaw: 84,810 vertices / 28,270 triangles.

All three inherit authored `physics:approximation=convexDecomposition` from the
collision Xform. Triangle sections are not cooked-hull clearance or contact.
The audit exports the largest coplanar triangle groups as candidate surfaces,
without selecting housing faces as pads automatically.

Live tests measured both grippers at approximately -0.01178, 44.99006,
89.99744, -0.01178 degrees for commands 0/45/90/0. Jaw rotations relative to
closed were 0, 45.00178, 90.00920, 0 degrees. Cube displacement was 0.0 mm in
all four samples. At y=0, z=-0.080 m in gripper coordinates, triangle-section
spacing was 23.4884 mm closed and 73.8164 mm at 45 degrees. At 90 degrees there
is no moving-jaw intersection at this section; null is correct, not a huge gap.
No 3-D cube fit, effective contact gap, calibrated TCP or grasp is proven.

Exact test commands and visual command are in README. Generated results:
`outputs/finger_audit/{geometry.csv,measurements.json,hierarchy.txt,headless.log}`
and `outputs/finger_audit_offline/`. CPU audit exited 0. Three focused unittest
checks passed (face-interior hit and outside miss, parallel/degenerate rejection,
missing/invalid geometry failure). Live measurement checks completed, but the
first runs hung after final reporting and were stopped by PID; clean unattended
process shutdown must be distinguished from completed numerical checks.

Next human checkpoint: run the README visual command. Observe the yellow
non-collidable section line at closed/45-degree poses; confirm whether its ends
land on the intended opposing inner surfaces, and that the cube remains still.
A view of those two poses plus the final console result is the narrowest useful
evidence. M1 must then validate effective contact/cube fit and define a documented
TCP before M2 offline IK/orientation feasibility and full-path collision checks.
No pre-grasp motion was attempted while this prerequisite remains unresolved.


Final lifecycle diagnosis and verification: a bounded diagnostic exited 124 and
captured the stack in Isaac Lab `SimulationContext._app_control_on_stop_handle_fn`
→ `render` when `SimulationApp.close` closes the stage. Disabling Replicator wait
alone did not fix it. The audit now calls the public `clear_all_callbacks()` and
`clear_instance()` before app close, matching the pinned library's
`build_simulation_context` teardown. The final README headless command exited 0,
including construction/update of the yellow non-collidable section marker.
No upstream library changes were made. On assertion/exception the audit prints
the traceback and explicitly exits 1, avoiding Kit 5.1 fast shutdown masking a
failed test with status 0. Unit tests (3), py_compile and git diff --check passed.
GUI appearance still needs human inspection; headless marker construction is
not evidence of visual correctness. The superseded untracked calibration
script's SHA-256 was unchanged after this work.

## Controller development — 2026-10-21 (in progress)

Human checkpoint accepted: user reports both jaws move correctly, yellow marker
ends touch intended inner surfaces at 0/45 degrees, cube stays still. Marker
visibility does not enter the triangle/pose measurements; no render-only repair
is needed for measurement validity.

New reusable CPU model reads all six joint frames/limits and 17 collision meshes
from the existing USD. Five arm joints use deterministic multi-start bounded
least-squares IK, with either position + approach direction or a requested full
orientation that is rejected if infeasible. The original downward pregrasp at
z=0.12 m is infeasible. The selected pregrasp is (-0.34,-0.04,0.055) m; grasp is
(-0.34,-0.04,0.022) m. Rotation is identity in world coordinates, aligning cube
faces. TCP is (0.0141000004,0,-0.09) m in the gripper rigid link, rotation identity:
X closure, Y pad width, -Z approach. This uses the fixed finger's full cube YZ
footprint, including ridges, with 2 mm initial fixed-side clearance.

Authored footprint aperture at 0/15/25/35/45 degrees: 15.800/31.994/40.672/50.597/
62.962 mm. A 42 mm test box (40 mm cube plus 1 mm per side) intersects the moving
finger through 35 degrees and clears both meshes at 45 degrees. Width alone is
insufficient because the jaw is inclined. At 90 degrees no opposing surface
remains in the footprint. Triangle tests include containment, not only surface
crossings. Physical closure/contact remains a separate gate.

Offline path validation: 473 opening samples, 3,040 HOME/open-to-pregrasp samples,
281 descent samples; all accepted. Triangle/box checks for the cube, conservative
convex envelopes for self/opposite/bin, actual vertex/table height. Adjacent USD
joint interfaces and fixed base/table mounting are excluded explicitly. A 0.6 m
lever-arm bound controls sample spacing with 1 mm nominal collision margin.
This bounds nominal interpolation only; tracking error and cooked-hull contact
must be checked in physics. Planning took 205 seconds with one BLAS/OpenMP thread;
valid plans are cached with asset/model/approach-planner fingerprints.

Physics instrumentation requires kinematic rigid actors for the fixed table/bin:
purely static colliders produced unsupported GPU contact-filter warnings. Their
poses/dimensions are unchanged; the shared baseline scene source is unchanged.
A reset prerequisite verifies the 50 g cube has 0.3–0.7 N table support (expected
0.49 N), preventing missing force data from being interpreted as clear contact.
Both robots have pairwise force matrices against the cube, table/bin and all
robot links; forbidden forces >=0.05 N fail execution.

First uncompensated approach failed on 0.1421 N unintended fixed-finger/cube
contact during pregrasp travel. Added actuator gravity feed-forward using the
pinned PhysX API, leaving gravity and original PD gains/limits enabled. Corrected
approach and then three repeated reset+approach episodes exited 0: cube drift
0.0 mm, final TCP position error 0.000329 mm, maximum moving TCP tracking error
1.302 mm, gravity torque <=0.633 N m. Live FK versus measured link transforms
agreed within 0.000206 mm / 0.000078 degrees throughout waypoint checks. Report
precision reflects simulator arithmetic, not real-world calibration accuracy.

Evidence: `outputs/controller/approach_3episodes.json`,
`outputs/controller/approach_repeated.log`, generated `plan.json` and
`calibration.json`. Five model tests passed: closed-box containment/separation,
independent USD-rest FK, three FK/IK round trips and limits, unreachable-target
rejection, and open-versus-closed cube geometry. Earlier three audit tests remain.
Pickup implementation is being tested; no handover or placement result exists.

### Verified pickup milestone

After the 1 N preload failure (maximum cube height only 20.46 mm, lost both pad
contacts), a measured >=3 N preload succeeded without changing friction, gravity,
cube mass, solver settings or actuator gains. Final command:

```bash
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
export LEISAAC_ASSETS_ROOT=/home/rynem8/robotics/leisaac/assets
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 300s python scripts/run_manipulation.py --headless --stage pickup --episodes 3
```

Final evaluator run: exit 0, pickup **3/3**, lift/hold **3/3**, full task **0/3**
(not implemented). Same deterministic initial conditions, not randomized trials.
Each episode: 34.7833 s simulated including reset; 3 s hold; minimum cube centre
height 73.2735 mm (lift >53 mm above its initial centre); minimum hold pad forces
1.6050 / 1.5785 N; relative hold drift <0.0002 mm (simulation precision only).
Closure measured 21.7056 degrees with forces 3.3892 / 3.7540 N. Peak joint speed
1.084 rad/s and applied actuator torque 0.835 N m, below existing 10 rad/s / 10 N m
limits. Peak TCP tracking error during loaded motion was 4.113 mm; during approach
<=1.308 mm. Cube drift before intended contact was zero in all three episodes.
No forbidden pairwise force >=0.05 N was reported. Reset after each held-cube
trial restored the next starting cube pose and independently checked weight
support; no task-time object write or artificial attachment exists.

Five model tests plus three legacy geometry tests passed; all new source modules
compile. Reports include per-episode counts, FK errors, force/pose traces and
object orientation. Failed attempts exit nonzero and attempt an explicit episode
reset, recording whether that reset succeeds. This failure-reset path is
implemented but has not yet been exercised after its addition.

Coupled handover feasibility: nominal point (0,-0.27,0.18) m yielded 7 solutions
for one orthogonal-face grasp arrangement and 5 for its alternate. These passed
inter-arm conservative-envelope endpoint checks; other proposed points
(0,-0.24,0.18) and (0,-0.20,0.18) yielded no candidates in this bounded search.
Neither transport paths, loaded orientation stability nor physical transfer have
been verified. The original (0,0.015,0.18) remains unproven. Further work must use
the actual held-cube transform, not assume its orientation equals the TCP.

### Continuation: receiver and bin integration (2026-10-21, in progress)

The following supersedes the transport limitation above; it does not yet
establish complete manipulation success.

- `transport_pass.json`: earlier loaded transport passed once, maximum slip
  0.046165 mm / 0.18477 degrees, centre height 174.599 mm. Its receiver pose had
  insufficient approach margin, so a different coupled pose was selected.
- New nominal receiver: joint-linear pregrasp-to-grasp failed `cube:gripper`
  at sample 85/628. Replacing this with 16 Cartesian IK legs passed all
  intermediate geometry checks (`receiver_nominal.json`). Runtime replans at
  the actually measured cube pose; nominal clearance is not a physics pass.
- `pickup_regression.log`, report `pickup_20261021T201958882650Z.json`: updated
  controller passed pickup/lift 3/3 again, minimum height 73.2735 mm, sustained
  pad forces >=1.5785 N, zero precontact cube drift. Five model tests and three
  geometry tests passed. No dependency, asset, friction or driver changes.
- `handover.log`: the new transport orientation failed on measured cube slip
  3.94 mm / 10.10 degrees; pad forces fell to about 1.1 N. The 10-degree gate
  rejected the run. Failure reset then passed, including 0.4905 N table support.
  This is now evidence that the failure-reset path executes successfully.
- Bounded jaw-force feedback was added for loaded transport (3–4 N target
  band, at most 0.02 degree command change per physics tick, 15–30 degree command
  bounds). Contact, actuator and slip gates remain enabled. Its physical
  handover evaluation is in progress in `handover_force_control.log`.
- Bin feasibility: a right-hand target near (0.34,-0.04,0.11) m with a 15-degree
  tilted approach has >5 degrees joint margin. A nominal carried path passed
  2,379 samples after donor withdrawal. This assumes the measured cube transform
  remains held and is not evidence of physical placement. Withdrawal and full
  bin evaluation are being integrated; `--stage full` is experimental.

GUI is the default for every maintained simulator entry point. In the main
controller, successful GUI evaluation freezes the final scene and leaves the
window open; `--exit-after` opts into finite GUI execution. `--headless` remains
an explicit automated-test option. Each finished/failed run writes a timestamped
stage JSON with source hashes; `simulation.json` is also updated. Generated
reports/logs remain ignored and are not published to GitHub.

Publication: terminal HTTPS push failed because no credential helper supplied a
username; SSH had no existing trusted GitHub host key. The connected GitHub app
published the **exact tree** of verified local commit `f218911` on feature branch
`feat/fingertip-geometry-calibration`, remote commit `c907190`. Tree hash matched
`6b9f0eee5dfca8a005be0042dd6bc9395bb15919`. This is an API snapshot commit with
a different commit ID, not a claim that local history was pushed unchanged.
`main` was not changed. Further tested changes will update this feature branch.

Force feedback result: transport passed with maximum slip **0.10977 mm**, rotation
**0.17736 degrees**, hold pad forces >=**3.1138 N**, cube height **175.535 mm**.
The actual receiver path and all 16 Cartesian approach legs executed without
forbidden contact. Strong receiving closure then failed at **5.21 mm** relative
slip / **1.99 degrees**: both compliant arms deflected while squeezing the shared
cube. Right pad forces rose from 1.02/2.45 N to 2.54/4.67 N before failure.
The next evaluation uses 1 N receiving preload during dual support (>=0.6 N
after settling), then 3–4 N right-hand feedback during donor release/right-only
hold. This is a control change, not an actuator-gain or physics-stack change.
The existing slip gates remain unchanged. `full_first.log` records that trial.

`--stage full` now integrates donor withdrawal, loaded bin motion, release,
right-arm retreat and three seconds of measured cube containment, floor support
(0.3–0.7 N), linear speed <0.01 m/s and angular speed <0.1 rad/s. These are
implemented acceptance conditions, not a passed result yet. Active phase and
partial traces checkpoint every 120 physics ticks. Failure traces are copied
before reset so recovery motion cannot overwrite the evidence.

Light receiving preload subsequently cleared the physical handover gate: right
acquisition forces 1.0495 / 2.4685 N; the right-only hold trace showed zero left
pad forces, approximately 4.445 / 3.976 N right forces and 173.407 mm cube height.
The full trial then stopped in offline donor withdrawal planning: after a
40 mm retreat, direct HOME travel intersected the cube with the jaw at sample
443/3535. A longer 60 mm retreat instead reached shoulder/lower-arm self overlap.
No rejected path was executed. The correction inserts a pan turn away from the
cube before straightening the arm, and limits retreat candidates to 40–50 mm.

That failure also exposed a logging defect: NumPy `float32` hold height was not
JSON serializable, so the final `full_first` report could not be written; the
log and `placement_plan.json` survived. The scalar is now explicitly converted,
handover checkpoints precede placement planning, and the active trial is
`full_departure.log`. Do not count the interrupted final report as full success.
The new run additionally monitors carried-cube/table/bin forces at every tick;
>=0.05 N environment contact during loaded transport fails. Bin floor support
still supplies the independent final placement prerequisite.

### Complete basket placement and planning performance — 2026-10-22

First full-task pass: `full_20261022T084021118058Z.json`, exit 0, every stage
1/1, 209.1833 simulated seconds. Cube final centre
`(0.346718967,-0.032673143,0.029999895)` m; full containment and floor support
held for three seconds after both arms retreated. Minimum support 0.49049899 N;
maximum linear/angular speeds 0.00047460 m/s / 0.0109784 rad/s. Maximum carried
cube/environment force 0 N; peak measured joint speed 2.15124 rad/s and actuator
torque 1.51440 N m, within existing 10/10 limits. Right-only hold minimum forces
4.44493/3.97580 N; left forces exactly zero. This is a release above the basket
at about 106 mm centre height, followed by settling at 30 mm, not controlled
lowering into floor contact. Gravity and physical contacts perform the release.

Withdrawal fix: the old axial retreat brushed the fixed finger against the cube
at 0.050372 N in `left_retreat_4`. The donor now separates by a -2 degree pan
change before its 40 mm Cartesian retreat, turns away and returns HOME. The
first separation permits nominal contact-surface proximity, but runtime donor
contact remains forbidden throughout withdrawal; later legs require geometric
cube clearance too. The contact limit was not increased. The user's GUI
`pickup` report passed; its later withdrawal failure was recorded under `full`.
Startup now prints requested stage/episode count to make this distinction clear.

Performance: vectorized mesh bounds preserve exact collision decisions on a
1,080-check comparison corpus: scalar 76.550 s, vectorized 30.062 s (2.55x).
Approach planning independently decreased from ~205 s to 32.227 s with the same
3,794 samples. Separate closure/release/retreat caches bind geometry, endpoints,
cube and opposite-arm transforms plus margin/contact options. Fresh right
retreat validation measured 60.22 s / 2,661 samples. Cache-hit timing is printed;
fresh planner duration is printed by `[PLAN DONE]`. GUI rendering continues
without stepping physics during CPU planning.

The first repeated run passed episode one, then stopped before moving in episode
two because HiGHS returned unknown model status (primal infeasible) during a
near-tangent convex-envelope query. Reset succeeded. Added one dual-simplex
retry with presolve disabled; only definite statuses 0/2 are accepted, and a
second ambiguous result still raises. The saved failing measured state now
passes a complete offline placement plan (`lp_retry_reproduction.json`); the
transient unknown status did not recur in that standalone replay. Focused tests
force an ambiguous first result and exercise real definite overlap/separation
retries, plus rejection when both attempts remain ambiguous.

Measured reset variation was only 6.56e-7 rad in donor joints and 1.64e-6 in the
held transform, yet the old 1e-7 cache threshold forced expensive replanning.
Cache matching now uses 1e-5 per state element; the 0.6 m radius bound limits the
combined geometric variation below 0.1 mm, within the remaining allowance of
the >=1 mm planning margin. Actual contact/slip gates are unchanged. A direct
cache regression reuses 1 micrometre cube jitter but invalidates a 6 mm shift
into a finger and rejects it. Code/assets still invalidate all related plans.

Final headless validation: `full_final_repeated.log`, report
`full_20261022T085453054801Z.json`, exit 0. All seven stage success counts **3/3**
(approach, pickup, lift, transport, handover, placement, full task). Each episode
lasted 209.1833 simulated seconds. Minimum bin support across trials 0.49049899 N;
maximum linear/angular settling speeds 0.00047469 m/s / 0.01097843 rad/s. Maximum
carried environment contact 0 N; speed/torque maxima <=2.15124 rad/s / 1.51441 N m.
Final centres differed by less than 0.006 mm between trials. Episode two reused
all approach/transport/receiver/placement and three segment caches, recomputing
right retreat once (59.65 s) because the settled pose changed beyond tolerance.
Episode three reused **every** plan, including right retreat: no fresh collision
solve. First cold validation timings (seconds): approach 42.89, lift 1.06,
transport 43.64, receiver 150.01, donor release 23.11, placement 126.77, right
retreat 60.21. Cold computation is intentionally retained when inputs change;
warm repeat computation is skipped only with matching certified context.

Seven model/geometry tests plus three original finger tests passed (10 total).
GUI regression is running with `python scripts/run_manipulation.py --stage full
--episodes 1` (no `--headless`), log `full_gui_final.log`; its final scene will stay
open for human inspection. Physical visual review is not yet confirmed.
Stack source commits independently rechecked: Isaac Lab
`3c6e67bb5c7ada942a6d1884ab69338f57596f77`, LeIsaac `24d3bcd`; Python 3.11.16.
No dependency, driver, physics asset, friction, gravity or actuator-gain changes.
README now explains FK, five-DOF bounded TRF IK, coupled handover solving,
collision sampling/LP, quintic trajectory formulas, force control, stage commands
and cache behavior. Keep generated reports local/ignored; publish source/docs
only to the authorized feature branch.

## Learning notes / code map (current algorithms)

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
pairs first try separating axes of the actual half-space-inflated hulls. A
separating axis proves nonintersection; an inconclusive test retains the LP.
Acute-corner inflation is computed by half-space intersection rather than an
unsafe original-vertex radius approximation. Remaining pairs use convex envelopes: SciPy
[`linprog(method='highs')`](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.linprog-highs.html)
tests whether their inflated half-space inequalities have a common point.
Gripper/cube tests use authored triangles, separating-axis tests and containment
checks because convexifying the whole gripper would fill its empty grasp region.
The carried cube is transformed along the path and checked against obstacles.
These geometric checks predict clearance; measured forces and slip remain
independent execution gates.

For joint-linear segments, the number of samples is
`ceil(0.6 metres × sum(abs(q_end-q_start)) / (margin - 0.0001 metres)) + 1`, at least two.
The 0.6 m downstream radius bounds nominal point movement to at most half the
collision margin, minus a 0.1 mm cache-motion reserve, from the nearest sample. Typical margins are 1–3 mm depending
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


Exact peak nominal quintic derivatives are `1.875 |dq|/T`,
`(10/sqrt(3)) |dq|/T²`, and `60 |dq|/T³`. Each segment is rest-to-rest, but
jerk is discontinuous at joins and force feedback/phase target changes are not
covered by those nominal formulas. The USD/LeIsaac actuators specify position,
10 rad/s speed and 10 N m effort limits, **no acceleration or jerk limits**.
The evaluator records finite-difference acceleration/jerk without inventing
hardware limits. No real-robot dynamic certification is claimed.

`BenchmarkEnv` owns all direct reset state writes, actuator submission, 60 Hz
physics, pairwise force measurements and observations. `TaskEvaluator` consumes
measured `Frame`s with no imports of the expert or planner and no phase/success
inputs. Lift, right-only hold and released floor-supported placement each need
three continuous seconds; failure is sticky until a verified reset. The expert
retains early motion/contact guards, but cannot override this evaluator.
Default policy observations contain both arms' joint positions/velocities and
the preceding position action. `observe(privileged=True)` exposes simulator
state to the expert/evaluator explicitly. Cameras and learning are absent.
The policy adapter holds absolute `(2,6)` joint position targets (radians) for
three physics ticks (20 Hz); the force-regulated expert currently updates at
60 Hz. Future comparisons must declare this frequency/privilege advantage or
run a matched-frequency expert; this is not yet a fair learning leaderboard.


## Engineering audit — benchmark hardening (2026-10-22)

Preserved known-good local baseline `9af7dc75007a3ed85295ff5a19e45bfe8986a107`
on `feat/fingertip-geometry-calibration`, never modified main. The remote feature
head observed during this mission was `4c6a9d3c6d5751f7978ae07a43adf5fe9b726fb4`,
a different, older snapshot. New mission explicitly requires approval before
push/merge/deletion. No further publication attempts or deletions. Earlier
automatic approval review rejected publishing baseline source/environment
details because its destination/export authorization was insufficiently
specific; this was not bypassed. No dependencies/system settings changed.

### Correctness findings and decisions

| Subsystem | Finding / action | Remaining limit |
| --- | --- | --- |
| USD / FK | Actual composed collision triangles, authored joint frames; independent USD-rest and measured link checks retained | Authored envelopes differ from cooked PhysX hulls |
| TCP / aperture | Triangle-derived TCP and full-footprint 42 mm clearance at 45°, user-confirmed finger surfaces | Simulator calibration, no real hardware contact calibration |
| Five-DOF IK | Bounded deterministic multi-start TRF, achieved-pose checks; coupled 10-joint transfer solve | Local solver, full 6D requests often infeasible |
| Collision scene | Task allowed contacts separated from expert labels; actual every-tick forces, payload and full paths checked | No certified tracking tube; runtime guard detects contact rather than proving avoidance under arbitrary disturbance |
| Timing | Quintic analytic speed/acceleration/jerk peaks recorded; stricter sample spacing reserves cache tolerance | No authored acceleration/jerk limits; measured phase transients remain |
| Actuation | Original implicit PD + PhysX robot gravity feed-forward retained | Compliant arms deflect during load transfer |
| Grippers | Measured bilateral preload, bounded jaw feedback; staged receiver ramp measured | No tactile patch/pressure or real-motor model |
| Handover | Cube/TCP/wrist/relative pose/joint/torque/forces logged every tick | PhysX reported velocity and pose finite differences disagree during sustained grasp; unresolved |
| Placement | Entire cube, floor weight, low root speeds and both arms HOME for 3 s | Releases above basket rather than controlled floor-contact lowering |
| Reset / evaluation | Common adapter plus independent observation-driven evaluator; reset verifies pose/orientation/velocity/weight | Single environment, not registered Gym/Isaac Lab RL task |

Initial independent integration correctly rejected a release because it used a
35° jaw-angle proxy for release. Measured pad force was already zero before
that angle. Replaced the proxy with loss of **all** hand contact, without
weakening containment, floor support or speed criteria; added regression.
The older 4.1 mm TCP headline covered pickup only. New metrics cover both arms
over the entire episode (baseline peaks 16.0 / 27.85 mm).

A proposed nominal-target donor hold reduced dip but left 0.115 N donor force;
rejected. Receiver staging alone passed: maintain 1–2 N preload until donor
forces are <0.05 N for six ticks, then ramp the lower force bound to 3 N using
a one-second quintic. Jaw rate, gains, friction, mass and effort caps unchanged.
An attempted lift target based on post-contact cube seating (0.85 mm shifted)
changed the coupled solution and caused receiver-path interference. It was
rejected before receiver motion. Intended task lift target is now separate
from measured collision/payload geometry. This demonstrates planner sensitivity,
not a successful robustness trial.

### Performance and collision validity

`profile_before.json`: original LP approach, 3,794 samples, 32.5325 s profiled
wall / 32.6561 s CPU; 1,958 LP calls consumed 22.6101 s. FK 0.246 s, IK 0.239 s.
`profile_after.json`: same 3,794 samples, 17.8379 s profiled wall; convex queries
2.6495 s, no LP fallback on this approach. Revalidation with preprocessed
geometry 13.1102 s; **not** a persisted cache hit. Results are local timing
samples, not universal hardware benchmarks. The subsequent cache correctness
change increases sampling density to explicitly reserve 0.1 mm for <=1e-5
state differences; final profile/tests below record the denser path separately.

CPU tests compare every optimized approach query with the original definitive
LP, plus rotated boxes, acute inflated corners, near-touching boundaries,
ambiguous-LP retry, containment, FK/IK, cache invalidation, independent task
failures, timeout/reset and exact trajectory derivative bounds. Testing an
all-facet SAT prototype created unnecessarily large matrices; stopped its own
diagnostic processes, replaced with a small deterministic sufficient-axis
subset and preserved the LP fallback. No safety margin/resolution was reduced.

### Planning alternatives / official references

The current task-specific CPU planner remains the reference backend.
[MoveIt planning-scene concepts](https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html)
provide a useful model for robot/world state, padding and explicit allowed
contacts; adopting ROS 2 is unnecessary for this fixed task.
[MoveIt trajectory processing](https://moveit.picknik.ai/main/doc/concepts/trajectory_processing.html)
separates geometric paths from time parameterization using joint limits; TOTG
would need trustworthy acceleration limits and new swept-path validation.

[Isaac Lab 2.3 SkillGen](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/imitation-learning/skillgen.html)
pins cuRobo `ebb71702f3f70e767f40fd8e050674af0288abe8` with CUDA 12.8 source build.
The [generic cuRobo installation page](https://curobo.org/get_started/1_install_instructions.html)
still lists Python 3.8–3.10 and older Isaac Sim examples, so it does not establish
compatibility with Python 3.11 / RTX 5090. No cuRobo install or measurement was
performed. An approved experiment should use an isolated clone of the existing
environment and that exact commit, verify Blackwell compiler/kernel support,
build an SO-101 kinematic and collision-sphere model, then compare identical
queries against independent mesh checks. Measure cold/warm latency, false-clear
collisions, IK feasibility, trajectory derivatives, reproducibility and the
same full-task evaluator. Reject a switch if correctness or reliability drops.
This is a proposed experiment, not evidence that GPU planning will win.

### Repository audit and proposed cleanup (approval required)

All tracked files and the pre-existing untracked calibrator were inspected.
Active implementation: model, planning, manipulation, shared task scene, finger
audit, common environment/evaluator. Required tests: all three test modules.
Configuration: .gitignore, .vscode/settings.json (existing Conda integration).
Essential docs: README, this handoff, AGENTS operating contract. Historical
diagnostics: scene inspection, FK inspection, motion/gripper/reset regression,
two workspace probes. Generated outputs/logs/caches/datasets are ignored and
not staged. No simulator assets or environments belong in Git.

Proposed removals only: `scripts/probe_workspace.py` and
`scripts/probe_coordinated_workspace.py` (unsafe exploratory grids, superseded by
validated collision planning); `scripts/calibrate_so101_fingertips.py`
(pre-existing untracked failed instance-proxy discovery, superseded by the
maintained triangle audit). **No file removed.** Preserve motion/reset, FK and
finger tests because they protect independent failure modes.

### Hardened nominal baseline and CPU regression

Commands in fresh activated shells, BLAS/OMP threads=1:

```bash
python -m unittest discover -s tests -v
python scripts/run_manipulation.py --headless --stage full --handover-mode baseline --episodes 1
```

CPU run `review_cpu_final.log`: **23/23 passed**, 52.868 s, including
2,177 optimized/reference LP comparisons across 4,215 approach samples (the
0.1 mm sampling reserve adds checks to the earlier 3,794-sample corpus).
Full run `review_final_baseline_restored.log`, report
`full_20261022T113512685297Z.json`: **1/1 full pass**, independent evaluator all
four flags true, zero collisions, correct reset. Sources were dirty relative
to `9af7dc7`; report hashes identify the exact evaluated files.
Cold start 5.3466 s (includes Kit launch, geometry 0.5354 s and calibration).
Initial approach 13.8980 s; remaining cold planning 118.7613 s; episode wall
191.0824 s including planning / 72.3211 s excluding it. Episode 209.1833
simulated seconds includes reset; evaluator excludes reset (206.1833 s).
Environment subtimes: actuation 3.6647, physics 50.6135, observations 9.2119,
evaluation 2.0571, rendering 0 s. Reset 0.6150 s overlaps its physics/observation
subtimes; do not sum it twice. Checkpoint serialization 3.6891 s per run.
Measured per-episode timing deltas, not cumulative timings, are saved.

Baseline transfer reproduces the earlier paired result exactly: cube/TCP dip
17.3271/17.1831 mm, wrist 6.2327 mm, relative motion 1.0370 mm / 2.1416°, pose
convergence 1.85 s. This supports compliant-arm deflection/tracking as the main
dip component, with smaller contact settlement/slip. It is a measured
attribution, not identification of all contact-model parameters. Cube COM
local offset is exactly zero, so a COM-to-actor translation does not explain
reported vz=+0.0380 m/s while final pose-derived vz=0. Velocity disagreement
remains explicitly unresolved. Full-episode actual speed 2.15124 rad/s,
effort 1.51440 N m, TCP tracking left/right 16.0237/27.8468 mm, sole-holder slip
0.2101/1.2814 mm; acceleration 73.4027 rad/s² / jerk 7433.4863 rad/s³ are
measurements, not passes against absent limits.

### Staged preload repeated regression / policy adapter

`python scripts/run_manipulation.py --headless --stage full --handover-mode staged --episodes 3`
completed with exit 0, report `full_20261022T114046453102Z.json`, log
`review_final_staged3.log`. All seven expert stage counts **3/3**, all four
independent flags true in every episode, zero collision violations. Sources
match the control implementation committed in `7148636` (run began before the
commit; hashes rather than its dirty HEAD label identify it). Seed 101 each
reset. Three episodes each lasted 209.1667 simulated seconds including reset.
Cube dip 9.98315–9.98352 mm, TCP dip 10.06565–10.06608 mm, relative translation
<0.920 mm / rotation <1.984°, pose convergence 1.25 s. Compared with the exact
baseline: **42.4% less cube dip**, lower receiver tracking error (3.582° →
2.621°), transfer peak right effort 1.5144 → 1.2322 N m, peak receiving forces
5.956/5.293 → 3.771/3.357 N. Gravity compensation ~0.451 N m is unchanged in
scale. No stronger squeeze or altered gains/materials were used.

Minimum final bin floor support across three episodes 0.4904992 N, maximum
linear speed 0.00076356 m/s, angular speed 0.0129624 rad/s. Full-episode effort
<=1.35272 N m, speed <=2.15130 rad/s. Left/right full-episode TCP tracking
16.024 / 20.710 mm; sole-holder slip <0.211 / 0.790 mm. Full trajectory derivative
metrics remain in the report. Nominal peak commanded speed is 0.35 rad/s.
These are repeatability measurements at one condition; no IID success-rate
confidence interval is justified from identical deterministic repeats.

Cold run start 5.321 s, approach 14.250 s, remaining planning
120.046 s. Episode wall/planning seconds: 192.445/120.046,
92.167/9.041 (post-drop retreat revalidation), 94.172/0 (all plans reused).
Physics was ~50.7 s each; observation ~9.3 s, evaluation ~2.1 s. Whole-run
checkpoint serialization grew to **43.512 s**, so a subsequent code change
retains only the active phase and 12 recent samples in the two-second checkpoint.
Complete history/traces persist in timestamped reports at every episode
boundary and on handled failures. Later runs measure this improvement; the
above timings intentionally describe the original logging overhead.

`--stage model` now exercises the common policy interface in real physics:
three ticks per action (20 Hz), three `(2,6)` proprioceptive arrays with no
oracle fields, and rejection of wrong-shape/NaN/out-of-limit actions before
stepping. `review_policy_adapter.log` exited 0. Extended pure evaluator tests
passed 10/10, including nonfinite target/contact forces and invalid quaternion.
Raw contact matrices are checked for finite values before threshold filtering.
No change to valid finite-state physical control results from these guards.

The adapter follows [Isaac Lab 2.3 action/physics decimation](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/tutorials/03_envs/create_direct_rl_env.html)
concepts while remaining a small application adapter rather than a registered
RL task. [Isaac Lab 2.3 reproducibility guidance](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/features/reproducibility.html)
limits cross-hardware determinism claims. Here seeds initialize NumPy and
PyTorch; exact source hashes, fixed task poses and the pinned rigid-body/GPU
stack define the evaluated conditions. No runtime material randomization.

Final isolated CPU profile (`review_profile_final.log`, `profile_final.json`):
geometry 0.4842 s, calibration 0.9338 s; **13.0298 s wall / 13.0311 s CPU** cold
approach, **4,215 samples**, 2,177 convex queries in 2.1573 s, zero LP fallbacks.
FK 0.2392 s, IK 0.2406 s; geometry-warm revalidation 11.5534 s, no persisted
plan cache. This denser final profile is separate from the same-3,794-sample
32.53 → 17.84 s comparison above; host load and profiler overhead affect timing.
The single model USD composes from one 23,241,268-byte used layer, already
included in the asset fingerprint; resolved triangles/limits/TCP/scene source
are hashed as well. GPU utilization read during earlier CPU planning was 1%
(on the RTX 5090); this is not a measured GPU-planner comparison.

A final review found a potential stale-evaluator report when offline approach
planning rejected before reset. Removed that shortcut: every attempted episode
now resets and establishes fresh evaluator state before reporting a rejected
plan. Reset also clears evaluator state at entry, including prerequisite
failures. Failed cases retain a category, measured observation and plan-evidence
path. This prevents a preceding successful episode from contaminating later
failure evidence.

### Fixed perturbation suite — actual failures retained

`python scripts/run_manipulation.py --headless --stage full --suite perturb`
(default staged mode) produced `full_20261022T115237457681Z.json`,
`review_perturb.log`, **exit 1**. Full task 4/5 including nominal;
**perturbed poses 3/4**. All five reset/approach/pickup/lift/transport stages
passed. Seeded configurations were independently planned, not replayed from the
nominal cube trajectory. All six report source hashes match `6fd9fd2` exactly,
although the run began before that commit and recorded a dirty `7148636` HEAD.
The later CLI-help/timing-label edits do not change this controller/planner.

| Case / seed | Full task | Cube dip mm | Episode simulated s | Episode wall / planning s |
| --- | --- | ---: | ---: | ---: |
| nominal / 101 | pass | 9.984 | 209.167 | 78.159 / 9.137 |
| X +1 mm / 102 | pass | 9.886 | 209.200 | 186.281 / 117.520 |
| X −1 mm / 103 | pass | 9.986 | 209.367 | 185.958 / 117.015 |
| yaw +1° / 104 | pass | 11.074 | 210.250 | 178.236 / 108.863 |
| yaw −1° / 105 | planning rejection | not reached | 47.400 control s | 46.534 / 31.459 |

Successful cases: zero independent collision violations, bin support >=0.4904325 N,
settled linear speed <=0.00076332 m/s, angular speed <=0.01296422 rad/s.
Across successful perturbations peak speed <=2.31662 rad/s, effort <=1.45229 N m,
left/right stable-grasp slip <=0.288 / 0.771 mm. Maximum measured acceleration
92.688 rad/s² and jerk 9310.246 rad/s³ remain uncategorized against absent limits.

The −1° scene was rejected at sample 2,808/3,134 of open-HOME → pregrasp,
`opposite:gripper/gripper`; the receiving robot had not moved toward the cube.
Independent flags were pickup/lift true, handover/placement false, zero runtime
collision violations. The stored failure duration 53.4 s includes initial and
recovery reset (6 s); independent control elapsed was 47.4 s. Both resets passed.
This is a valid safe failure, and exposes task-specific path sensitivity.
Two bounded offline diagnostics tried pan-first and arm-fold-first staging;
both were rejected for `cube:gripper` along the alternative segment. Evidence:
`receiver_turn_first.json`, `receiver_fold_first.json` and corresponding review
logs. Neither route was executed or added to production. No broad speculative
search, reduced padding or allowed-contact exception was introduced.

Suite wall 736.514 s includes four new pose plans, startup and recovery. Compact
checkpoint logging 4.1505 s across five cases, versus 43.512 s in the preceding
three-repeat logger. The nominal warm episode excluding its retreat revalidation
was **69.0223 s**, versus 94.1721 s for the earlier all-cache third episode.
Physics remains ~50.7 s per nominal episode. These are observed run times,
not controlled statistical timing estimates. Final profile measurements were
isolated; CPU regression ran concurrently with part of this diagnostic suite.
Fixed cases are not IID samples, so 3/4 is a test-set fraction rather than a
confidence-bounded estimate of general robustness. Need a held-out pose
specification and more coverage before claiming a robust learning benchmark.

### Deliberate physical failures

`python scripts/run_manipulation.py --headless --stage full --suite negative`
produced `full_20261022T120705264350Z.json`, `review_negative.log`, **exit 0**,
`passed_negative_regression`. The donor-open omission yielded zero bilateral
forces and `failed_grasp`; receiver-open omission yielded `failed_transfer`
after the left acquisition/transport. **2/2 expected omissions detected, 0/2
full-task successes**, no handover/placement flags, independent success false
for both, verified recovery reset for both. Expert prerequisites abort early;
the independent evaluator refuses completion without observed support. Its
policy-independent timeout is separately unit-tested, not bypassed in these
reports. Logging 0.3140 s. This regression success must never be counted as a
manipulation success.

Final CPU run on the committed model/evaluator/planner (`review_tests_committed.log`):
**23/23 passed**, 41.149 s, including all 2,177 query comparisons / 4,215 approach
samples. `--help` now lists task-specific arguments before launching Kit;
`review_cli_help.log` and byte-compilation exited 0. Source/report provenance,
cache settings and discrete physics limits are explicit. Full cold planning
historically ~548 s at `9af7dc7` versus ~134 s in the hardened nominal run; this
historical comparison includes changed sampling density and host conditions,
not a controlled backend contest. No dependency/driver/material/gain changes.

### Strict reset-case audit

Final review found the inherited 2 mm / 2° reset tolerance could not distinguish
our smaller 1 mm / 1° test cases. Tightened the independent reset pose gate to
**0.1 mm three-dimensional position / 0.1° orientation**, retained existing
velocity, arm and weight prerequisites, and record actual reset pose/quaternion
and errors in every report. Added unit rejection of a misplaced 1 mm cube and a
wrong 1° yaw. No physical parameters or reset commands changed.

`python scripts/run_manipulation.py --headless --stage model --suite perturb`
completed exit 0, `model_20261022T121432061342Z.json`, `review_reset_cases.log`:
**5/5 physical reset/interface checks passed**. Maximum measured reset position
error 0.0017271 mm, orientation error 0.0019348°. Actual X coordinates were
−0.33900103 / −0.34100103 m in the shifted cases, and actual quaternions resolve
both signed yaw perturbations. This independently audits realization of the
case settings; it is not an additional full-task robustness run. The earlier
4/5 full suite used the same reset commands/physics with the looser gate.
Extended evaluator unit tests: 10/10 in 0.193 s (`review_reset_unit.log`).
No change to existing geometric cache validity; only reset verification/report
metadata changed. The policy audit also rejected invalid actions in all cases.

### GUI regression / rendering profile

`python scripts/run_manipulation.py --stage full` (default GUI + staged preload)
completed objective evaluation: `full_20261022T120935212643Z.json`,
`review_final_gui.log`, `passed_requested_stage`, all independent flags true,
zero collisions. Its numerical physical metrics match the headless nominal
trial (cube dip 9.98352 mm, peak speed 2.15129 rad/s, effort 1.35271 N m).
All plans reused; cached approach load plus fingerprint validation 0.0318 s.
Startup 8.958 s; 209.1667 simulated seconds took 406.387 s wall. Rendering alone
213.360 s, physics 56.931 s, actuation 10.222 s, observations 18.261 s,
evaluation 2.424 s; deliberate 8 ms/tick viewer pacing adds about 100 s.
Checkpoint logging 1.101 s. This separates GUI/rendering cost from collision
planning. A small portion overlapped the separate reset audit / final regression;
GUI wall timing is descriptive, not an isolated rendering benchmark.
The process is still open at `[VIEWER]`, physics frozen after completion.
No exit code is claimed while it awaits window closure. Human confirmation of
the original sequence was supplied in the mission brief; visual review of the
new staged preload remains pending. Watch receiver dip, clean donor withdrawal
and cube settling; all numerical gates already passed. This GUI run preceded
the stricter reset-only evaluator gate; the final headless repeats below test
that gate on the unchanged controller. Source hashes distinguish both revisions.

### Final strict-gate nominal regression / review assessment

`python scripts/run_manipulation.py --headless --stage full --episodes 3`
completed **exit 0**, `full_20261022T121526133312Z.json`,
`review_strict_final3.log`: every expert stage **3/3**, every independent flag
true, zero collision violations. All six report source hashes were compared
with the final working files and matched exactly before the final commit.
The report records dirty `6fd9fd2` because the strict reset/report edits were
not yet committed; the final audit commit contains those identical files.
All three measured reset errors: 0.0017241 mm / 0.0018969°.
Each episode lasted 209.1667 simulated seconds, with independent control
206.1667 s. Cube dip 9.98315–9.98352 mm; min bin support 0.4904992 N, max settled
linear/angular speed 0.00076357 m/s / 0.0129624 rad/s. Full-episode peak speed
<=2.15130 rad/s, effort <=1.35272 N m, no carried-object environment contacts.
Left/right stable slip <=0.211 / 0.790 mm; full TCP tracking <=16.024 / 20.710 mm.

Warm approach load + fingerprint check 0.03117 s. Episode wall/planning seconds:
102.107/0, 98.472/10.312, 88.025/0; checkpoint logging 1.935 s over three repeats.
The GUI viewer was active concurrently, so these final correctness-run times
are **not** replacements for the isolated ~69 s warm execution measurement.
Nominal physical metrics reproduced despite the extra rendering workload.

Review status: the deterministic nominal reference is ready for code review,
with a preserved baseline, independent evaluator, repeated actual physics,
quantified transfer improvement, profiled conservative collision acceleration,
real failure tests, strict reset-case realization, documented action/observation
contracts and no stack changes. It is **not a generally robust manipulation
benchmark**: the fixed perturbed test fraction remains 3/4, COM velocity versus
pose-derived velocity during grasp remains unresolved, acceleration/jerk limits
are absent, and execution has no certified tracking tube/continuous-contact
guarantee between physics ticks. The 20 Hz proprioceptive adapter and privileged
60 Hz expert are explicitly different experimental tracks. No learning, dataset
generation or real-robot success claimed.

Created local hardening commits `7148636` and `6fd9fd2`, followed by the final
strict-reset/audit commit (see `git log -3`). Main was not changed. Remote branch
rechecked via the connected GitHub read API: still `4c6a9d3c6d5751f7978ae07a43adf5fe9b726fb4`,
so it does not contain the latest successful local task or this hardening work.
No push/merge/deletion performed under the new brief. Twenty tracked text/source
files total about 301 KB; generated outputs are ignored. Credential-pattern
scan flagged no tracked files. The sole pre-existing untracked calibrator is
preserved and remains outside commits.

Remaining user checkpoint: visually review staged receiver loading (roughly
10 mm dip), clean donor withdrawal and settled basket placement. GUI defaults
and the open completed scene are preserved. After visual review and acceptance
of the documented coverage limits, decide whether to publish this feature branch
and approve the three proposed removals above. No merge into main is authorized.


## Imitation-learning mission — motion milestone (2026-10-22)

New mission explicitly authorizes normal non-force feature push, isolated
Python 3.12 LeRobot training, and removal of the three named obsolete probes.
Baseline `05a0c13` is preserved: all six source hashes matched the strict
`full_20261022T121526133312Z.json` 3/3 report before edits. Branch remains
`feat/fingertip-geometry-calibration`; main untouched.

`git push --dry-run origin feat/fingertip-geometry-calibration` failed exit 128:
`fatal: could not read Username for 'https://github.com': No such device or address`.
No push succeeded; remote last independently observed at `4c6a9d3...`. The
remote API snapshot may also require a history-preserving merge before a normal
push. Do not force, rewrite, or substitute an API snapshot for the requested push.

### Continuous paths: actual matched local physics

Commands (activated original environment, OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1):

```bash
python scripts/run_manipulation.py --headless --stage full --motion-mode legacy
python scripts/run_manipulation.py --headless --stage full --motion-mode continuous --episodes 3
```

Both exited 0. Reports `full_20261022T143607932087Z.json` (legacy 1/1) and
`full_20261022T143935357354Z.json` (continuous 3/3), logs `il_legacy_baseline.log`
and `il_continuous3.log`. All full-task and independent flags true; zero forbidden
contacts. Same physics/materials/actuators/evaluation, run sequentially.
Legacy simulated duration 209.1667 s; continuous 150.65 s including 3 s reset
(28.0% reduction). Explicit waypoint settling 70.5 -> 24 s. Warm execution
excluding CPU planning: legacy 68.987 s, continuous 47.907 / 47.927 / 48.131 s.
Physics wall 50.513 -> 34.837–34.959 s; observation 9.174 -> 6.577–6.636 s.
Cold episode planning legacy 118.721 s plus separately logged pickup; continuous
169.777 s plus pickup, since the spline itself gets additional geometric checks.
Warm second continuous episode rechecked final retreat in 9.049 s; third all
plans reused. Faster execution does not mean faster cold planning.

Continuous max joint tracking left/right 2.0572/2.7197 degrees; speed 2.27470
rad/s, effort 1.42672 N m (limits 10/10). Stable slip .210/.7524 mm. Receiver
dip 10.2952–10.2959 mm vs legacy 9.9835 mm; pose convergence 1.2167 vs 1.25 s.
Measured acceleration/jerk 84.852 rad/s² / 9240.31 rad/s³ vs 73.726 / 7412.07.
Those limits are absent in the asset; no acceleration/jerk certification.
PhysX residual COM velocity remains; no velocity-settled claim.

Implementation: `JointPath` uses clamped CubicSpline through cumulative joint
arc-length knots, composed with quintic normalized time. It is C2 throughout
and has zero velocity/acceleration at endpoints. Exact polynomial extrema
provide joint bounds and conservative composed derivative peaks. Curve checks
sample spline parameter with the same 0.6 m downstream-motion bound and
0.1 mm cache reserve, rather than checking only waypoint chords. The initial
donor pan separation, contact stabilization, release/closure and final large
segments retain their prior pauses. Core math is now in cache fingerprints.

GUI command `python scripts/run_manipulation.py --stage full --motion-mode continuous`
launched for one user visual checkpoint, log `il_continuous_gui.log`; no artificial
8 ms/tick pacing unless `--viewer-delay 0.008` is selected. Human reply pending.

CPU regression on these motion sources: `python -m unittest discover -s tests -v`,
24/24 passed in 37.647 s (`il_motion_tests.log`), exit 0. Includes independent
evaluator, geometry/LP equivalence and analytic C2/velocity/acceleration checks.


## Camera, dataset and official-policy pipeline (2026-10-22)

Milestone implementation extends BenchmarkEnv instead of duplicating scene or
evaluation. CLI `--record-dir` implies actual 20 Hz sample-and-hold expert targets
and opts into Isaac Lab 2.3 RGB sensors before AppLauncher. The expert still uses
privileged contact/geometry at 60 Hz internally; its commanded 12 joint targets
are held unchanged for all three physics ticks. Policies have only 24 joint
position/velocity scalars and RGB, with no object/contact/phase/completion input.
The shared 60 Hz implicit PD and robot gravity compensation remain unchanged.
No task-time physical state writes or attachments were introduced.

### Exact physical recording / replay

`--headless --stage full --motion-mode continuous --record-dir datasets/raw_smoke`
produced `full_20261022T145012629805Z.json`: full independent success, 150.6833 s
including reset, 147.65 s / 2,953 complete policy frames plus two terminal ticks.
Joint order is left then right: shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper. Absolute radians target labels are the exact
float32 actuator values, not next observed states. NPZ retains each actual
3-tick target sequence, image/physics indices, seconds and q/qd. PNGs are
pre-action. Privileged cube/contact measurements live in separate diagnostics.
Terminal partial transitions are excluded; final state and tick count retained.
Completed failures and interrupted attempts are distinct from successful data.
The final writer saves attempt metadata immediately, partial trajectories every
200 frames, and never overwrites an existing attempt directory.

Initial recording physically passed and durably saved, but **process exit 134**:
Replicator camera destruction hit a stale weak-reference callback in Sim 5.1.
Fixed by strongly owning scene/env through teardown and destroying cameras while
PhysX views still exist, before clearing callbacks. `il_camera_exit.log` model
check exited 0. Later replay/model audits also exited normally. No upstream
package/driver changes. Camera capture costs 19.408 s in this initial recording;
wall 282.278 s includes 175.803 s cold planning. Other GPU processes were active;
this is not an isolated rendering benchmark.

`--headless --stage full --replay datasets/raw_smoke/nominal_20261022T145014937559Z`
produced `full_20261022T150147142022Z.json`, **exit 0, full success**, no expert
feedback, no planner and no physical state copying. Independent completion at
144.85 simulated seconds; zero collisions, peak speed 2.15485 rad/s, effort
1.35452 N m, stable slip .214/.418 mm. It holds the last actual target for up to
three seconds if terminal dwell needs it; this run completed before the original
expert stopped logging. Replay confirms action semantics; it is not learned
policy performance or a disturbance/generalization claim.

The initial camera RGB was valid but pose metadata buffers were all zeros:
CameraCfg defaults to not updating poses after set_world_poses_from_view.
Enabled supported `update_latest_camera_pose=True`, verified actual eye and unit
quaternion before recording, and now reject invalid camera calibration in raw
validation. `model_20261022T150815933446Z.json`, `il_camera_metadata.log`, exit 0,
contains measured calibration. Thus the FIRST raw/converted smoke remains an
explicit diagnostic artifact and fails the new strict calibration validator;
do not quietly treat its stale extrinsics as production measurements.

Two 224-square views: overview (0,-.12,.92)->(0,-.12,.04) m; front
(0,.72,.56)->(0,-.14,.12) m; focal length 18 mm, aperture 20.955 mm,
clipping .01–10 m. RGB-only capture renders before advancing physics and reads
fresh buffers with update_period=0; scheduled capture is 20 Hz. Measured focal
pixels 192.4123, principal point (112,112). Both images visually inspected.
Using separately measured calibration, 15 visible-view samples over eight
recording indices projected measured cube centres within .521–4.194 px of red
pixel centroids (`il_camera_projection.json`), with visibility/occlusion bias.
This supports geometry/time association; it is not a sub-tick latency certificate.
Front resolves the cube when the overhead view is occluded by the hand.

### Distribution diagnostics (all failures counted)

Initial preselected four draws from X±10 mm, Y±7 mm, yaw±6 degrees:
`full_20261022T150333657825Z.json`, `il_coverage4.log`, exit 1: **2/4 full successes**.
train_000 and train_002 rejected pre-grasp; train_001 (yaw +4.56665 degrees,
y +6.329 mm) and train_003 (x -3.3775 mm, yaw +.96305 degrees) physically passed.
Repeated after IK continuation/camera repair:
`full_20261022T151441433258Z.json`, `il_coverage4_seeded.log`, **2/4**, exit 1.
Actual successful raw records in `datasets/raw_production_train`: 2,958 + 2,900
frames = **5,858 frames / 292.9 simulated seconds**. Both failed attempts have
metadata and zero action frames; never hidden or added to training. These are
interim uncommitted-source diagnostic runs, not final source-frozen coverage.
Early classification reported these as reset_prerequisite because the guard
ran after reset; final code explicitly labels the no-approach phase plan_pickup,
so these are correctly expert planning failures, not bad reset realizations.

Nominal IK continuation finds pre/grasp candidates for the previously missed
initialization, but the near-limit pregrasp intersects the cube. New regression
requires retaining that collision rejection. Trials of 30/25 mm pre-height had
clear endpoints and rejected HOME paths: no endpoint-only PASS was introduced.
Forty random poses in each original/inward range had IK candidates (not paths);
12 new inward draws all passed full approach geometry in `il_inward_geometry.log`.
Candidate v2 range X±10 mm, Y offsets -12 to -4 mm, yaw±6 degrees; distinct
32 train / 8 validation / 30 held-out configurations seeded 811/812/813. These
are fixed BEFORE outcomes. Full physical v2 coverage still pending at this entry.
Original outward failures remain reported; no generic robustness claim.

Receiver -1 degree still fails before receiver motion. Two new actual attempts:
`full_20261022T150440088313Z.json` (60 Hz) and
`full_20261022T151426995650Z.json` (20 Hz), both exit 1 / planning_rejection.
Offline free-roll 40 mm pregrasp cleared one saved 60 Hz geometry, but had only
1.647 degrees joint reserve; 35 mm retained reserve and hit the cube on travel.
A -5 degree pregrasp pitch cleared that old geometry, but failed the distinct
actual 20 Hz held pose. Cube-symmetry flip lacked IK; four existing alternative
handover candidates had safe carried travel but no receiver route in the bounded
offline check. No unsafe route was executed and these experimental receiver
rewrites were removed. Preserve the original receiver planner. Next diagnostic
would need a genuinely different checked transit path; do not keep retrying the
same local interpolation or relax contact margins to call this fixed.

### Isolated training stack and actual model smoke results

User explicitly authorized the separate environment. Created
`/home/rynem8/miniforge3/envs/so101-il-py312`, Python **3.12.15**. Exact installed
package versions are `configs/il_requirements.lock.txt`. Direct pins:
LeRobot 0.6.1, numpy 2.2.6, torch 2.7.1+cu128, torchvision .22.1+cu128,
torchcodec .5.0, av 15.1.0. Official release pyproject requires Python>=3.12 and
NumPy>=2, incompatible with the locked simulator environment. No changes to
leisaac-sim/Isaac/NVIDIA/system configuration. Initial PyPI torch chose cu126;
corrected ONLY the new env to official cu128 wheels for sm_120. `pip check`
passed; actual CUDA 512x512 matrix product and ACT/Diffusion imports passed on
RTX 5090 (12,0). Setup logs in /tmp/so101_il_{environment,install,cuda128}.log.

Local conversion uses official LeRobotDataset.create/add_frame/save_episode/
finalize with image features, not a guessed parquet format. Raw validation runs
first; v3 reader reopens and verifies shape. The first conversion initially
failed on the restricted default HF cache after durable data creation; moving
HF_HOME/TORCH_HOME to ignored workspace outputs fixed reader verification.
`datasets/lerobot_smoke`: one episode, 2,953 frames, codebase_version v3.0.
`datasets/lerobot_initial_train`: two calibrated episodes, 5,858 frames, v3.0,
`il_convert_initial_train.log` exit 0. No data/model upload. New converter refuses
mixed partitions and counts incomplete attempts explicitly. Validation/test
must be separate dataset roots so their normalization statistics cannot leak.

Official model classes, not custom transformer/U-Net code. `policy_pipeline.py`
uses official saved pre/post processors for normalization/unnormalization. ACT
width256, 8 heads, 4 encoder/1 decoder, 40-step predicted chunk /20 executed.
Diffusion two observations, horizon32 /16 executed, U-Net (128,256,512), DDIM10.
Local diagnostic training: AdamW lr1e-4, weight_decay1e-4, clip norm10, constant
LR (not a claim to reproduce a published optimized recipe); seed701, batch16.
Longer runs save model/processor checkpoints every500; final optimizer/Torch RNG
saved for diagnosis, CLI resume is not implemented.

Commands in separate training environment (outputs immutable on rerun):

```bash
python scripts/policy_pipeline.py train --dataset datasets/lerobot_smoke --output checkpoints/act_smoke --steps 20 --no-pretrained --log-every 5
python scripts/policy_pipeline.py train --dataset datasets/lerobot_smoke --output checkpoints/diffusion_smoke --method diffusion --steps 20 --no-pretrained --log-every 5
```

Both exited0, gradients finite, weights/processors saved, valid12-radian targets
produced. ACT loss81.9431->6.0602, 7.878s training; Diffusion1.1264->1.0073,
7.543s. NOT task successes. Servers load checkpoints in Python3.12 and exchange
only allowlisted RGB bytes and24 scalar joint state over private local Unix
sockets. Lists/bytes avoid NumPy1/2 pickle coupling. Original Python3.11 env
holds targets for3ticks and evaluates every physics tick, with no hidden oracle.

Actual ACT smoke evaluation `full_20261022T145946086701Z.json`, exit1:
**0/1, no stages, timeout at300s**, 6,000 inference requests. Mean/p95/max model
latency .834/3.750/162.492 ms; most requests reuse the action chunk. Round trip
.942/3.841/162.680ms. Zero contact collisions; speed/effort touched authored limits.
Actual Diffusion smoke `full_20261022T150741124656Z.json`, exit1: **0/1**, actuator
limit at .05s, two collision violations; first inference191.373ms. Too few/poorly
trained models to rank methods, and no 30-trial held-out comparison yet. Inference
pauses simulated time; 20 Hz interface does NOT imply all queries meet50ms wall.

A longer two-demonstration ACT diagnostic completed 1,000 updates:
`checkpoints/act_initial_2ep_1000/pretrained_model`, checkpoints also at500/1000,
`il_act_initial_1000.log`: loss79.6424->.216413, 389.845s training; ImageNet
ResNet18 weights44.7MB downloaded only to ignored outputs/torch from official
PyTorch. Actual policy evaluation pending at this entry; loss is not success.

Source-frozen runs now guard reset/planning against live edits so an old loaded
planner cannot tag a plan with a new on-disk source fingerprint. Reports and
recordings share a process-start source-hash snapshot. Initial raw diagnostic
runs preceded this guard; retain their process reports, do not overstate exact
intermediate-source reproducibility. New physical generation must start AFTER
committing and leave controller/model/recorder files frozen until completion.

CPU pipeline regression: **28/28**, 38.556s (`il_pipeline_tests_sourceguard.log`),
including complete 2,177 LP comparisons /4,215 path samples, source/cache guard,
IK+collision rejection, independent evaluator and dataset contracts. Follow-up
actual-file interruption/terminal tests 3/3 (`il_record_interruption_tests.log`).
Subsequent changes only add attempt checkpoints/provenance and correct failure
phase labeling; final source-frozen physical runs must audit these as well.

Removed exactly the three authorized, unused probes after inspection. Tracked
workspace probes remain recoverable at e887eb1 /baseline05a0c13; no active imports.
SHA256 before removal: probe_workspace632949f3063905cb4e27178d47cdbb66ed9b9d41c900b0d62230507f8737a52c;
probe_coordinated_workspace acfa94f274bd554486cd6983674819460a0375e1411e8ce19d3ed0c924e0b316;
untracked failed calibrator0e907cdc8b72a3ef43c678f43d3bdd556c1d1ea825c7edd4342d9f62aae7a3fc.
Failed collider discovery was superseded by the independent triangle audit and
contained no uniquely valuable uncommitted work. No other tracked files deleted.

Motion GUI numeric regression: `full_20261022T144603146927Z.json` passed,
227.281s total /218.024episode wall,150.65simseconds; rendering153.499s,
physics37.852s; old406.387wall/209.17sim run included artificial pacing and longer
motion. New recording overlapped part of the GUI run, so not an isolated matched
GUI speed ratio. Sample simultaneous workload GPU92% /18,691MiB /220.63W;
process average CPU at one sample GUI158%, ACTsim209%, server35%; this measures
concurrent aggregate activity, not planner GPU acceleration. Human continuous
motion visual checkpoint remains pending; completed reference GUI is left open.

Incomplete: substantial source-frozen data, broad expert coverage/receiver fix,
ACT/Diffusion manipulation success and held-out rates, optional cuRobo experiment,
30-attempt statistics/comparison and professional comparison demo. Do not publish
smoke failures or declining loss as a successful learned portfolio benchmark.


Final pre-collection source check: `il_pipeline_commit_tests.log`, 28/28 in
37.969s, exit0, plus py_compile and diff whitespace/credential-pattern review.
Remote feature SHA rechecked through connected GitHub read API:
`4c6a9d3c6d5751f7978ae07a43adf5fe9b726fb4`; terminal push still lacks credentials.

### Optional cuRobo compatibility audit (source research only)

[Isaac Lab **2.3.0** SkillGen instructions](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/imitation-learning/skillgen.html)
specify commit ebb71702f3f70e767f40fd8e050674af0288abe8 and CUDA12.8; the example
architecture flag is 8.0 and must match the device. Do not follow newer develop
instructions that target a different CUDA stack. The [pinned setup.py](https://raw.githubusercontent.com/NVlabs/curobo/ebb71702f3f70e767f40fd8e050674af0288abe8/setup.py)
compiles custom CUDA extensions and supplies no register-cap option by default.
An [open Blackwell report on cuRobo's own tracker](https://github.com/NVlabs/curobo/issues/716)
describes L-BFGS launch resource exhaustion on RTX5090/torch2.7+cu128, including
a register-limit workaround. This is a report for v0.7.6, not proof that our
exact pinned commit fails. No cuRobo install, kernel test, SO101 configuration,
FK agreement or timing measurement has been performed. Retain CPU planner;
separate-environment runtime experiment remains incomplete and must not block
recording/training or modify the working simulator. No unsupported-hardware or
GPU-speedup claim follows from this source audit.


## Verified expert coverage repairs and episode clock (2026-10-22, later session)

The first source-frozen v2 batch at 2fe2a77 exposed a real recorder error:
`full_20261022T154248994206Z.json`, `il_train_v2_4.log`, exit1. Results:
train_v2_000 full success (3,074 frames); train_v2_001 and 003 failed in open with
`Previous policy transition is incomplete` after one tick; train_v2_002 rejected
receiver planning. **1/4**, not silently filtered. The successful episode has
153.7333 control seconds, zero collisions, max2.56664rad/s /1.50240Nm.
The next episode began at a global tick not divisible by3: first held command
lasted one tick, then the recorder attempted a new transition. Fixed scheduling
relative to BenchmarkEnv.policy_origin_step after EACH reset. Tests explicitly
cover origins180,181,182,9041 and real-file recordings at each phase offset.
Interrupted-attempt metadata/partial files were already supported. No reset or
physical criterion was relaxed.

Old fixed-jaw expert on the first four v2 held-outs:
`expert_pilot/full_20261022T155106747997Z.json`, `il_expert_heldout4.log`, exit1:
**1/4 full successes**, heldout_v2_002 passed. The other three established initial
bilateral pressure, then lost the cube on lift (minheight .020m, pad forces0).
The clock bug affects recorded transitions, not the first successful pickup,
but subsequent legacy 20Hz first-command alignment was also imperfect. An
attempt to interrupt this completed batch returned `No such process`; it was
NOT interrupted, all four attempts remain in the report.

### Measured grip feedback repair

The expert previously enabled its existing force servo only after lift/hold.
Added `--grip-feedback-from-lift`, retaining the fixed-jaw reference by default.
It applies the SAME3–4N hysteresis, bounded15–30degree jaw targets and slow gains
through lift/hold; preserves the actual updated command when transport begins.
No material, mass, gravity, gain or actuator-limit changes. Learned policies get
no equivalent hidden servo. On the same four held-out pickups:
`lift_feedback_audit/pickup_20261022T160532115098Z.json`, `il_lift_feedback4.log`,
**4/4 pickup/lift/three-second hold passed**, exit0. Min heights .073376–.073932m;
min bilateral forces at least3.0057N. These shorter-stage successes were never
counted as full-task training demonstrations. This run preceded final worker
configuration; its requested limit1 was ignored and actual OPENBLAS env remained32.

### Why the shell's numerical-thread limit did not work

Local pinned SimulationApp source overwrites OPENBLAS_NUM_THREADS with32. Lab2.3
also filters the supported SimulationApp limit_cpu_threads keyword out of its
whitelist, so merely passing that keyword did NOTHING. Measured inclusive
process CPU/wall for the first v2 receiver plan was2858.85/118.56s, reflecting
many numerical/SDK workers, not pure planner CPU. No dependency was installed to
control this. CPU-only NumPy and scipy.linalg now initialize before Kit with the
requested environment, and supported public Kit args set Carbonite/TBB worker
counts. Restore the requested numerical environment after startup. Per-process
only: no upstream file, OS governor, driver or persistent setting modified.

`thread_fix_audit/approach_...json`, `il_threads_fixed_approach.log`, exit0:
actual carb_workers1, tbb_workers1, torch_intraop1, OPENBLAS env1. Full cold
nominal approach12.1647wall /24.6085inclusive processCPU, physical approach passed.
Do not compare its wall time as an identical-pose speed ratio against the
17.32s wider-case plan. [Isaac Sim5.1 thread controls](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html)
and [OpenBLAS startup variables](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/)
explain the settings. The CLI `--cpu-threads` controls SDK workers separately
from the pre-launch numerical limit;32 restores original SDK worker count.

### Bounded collision-checked global transit

Lift feedback changed the stable held pose and exposed a blocked HOME->receiver
pregrasp even for nominal: `full_feedback_audit/full_20261022T161200699400Z.json`,
exit1. Orientation-only alternatives did not clear the connecting path. They
were removed, leaving the original model/IK API unchanged. Added a deterministic
RRT-Connect fallback inside CollisionWorld for ONLY the blocked open-hand transit.
Existing direct routes remain first. Five arm joints sampled; jaw fixed open;
USD limits preserved; nearest-node steering max.25rad L2; seed101,300 iterations,
60s operational budget (checked between edge calls). Every edge uses the same
1mm padded, displacement-bounded segment checker with .1mm state reserve.
Only collinear subdivision nodes are compressed; no unchecked shortcuts or
spline smoothing. Transit turns use independent rest-to-rest quintics; the
17-node insertion retains its separately checked C2 spline.

The actual saved failing geometry yielded a four-waypoint route in two tree
iterations,14 checked edges /4,545 samples /7.3504s (`il_receiver_rrt_probe.json`).
Independent analytic slab/rectangle tests verify the returned detour and no-path
budget failure. **31 CPU tests passed**,38.480s (`il_rrt_full_tests.log`).

First actual RRT run reached the receiver with no collisions but correctly
FAILED right-only transfer: `full_feedback_rrt_audit/full_20261022T162818010552Z.json`.
Right pads ~3.08N, cubeheight .1769m, but open donor fixed finger retained .271N.
Opening angle45degrees is not proof of release. The independent evaluator's
handover flag remained false. No success was claimed from the safe geometric
plan or the right-hand forces alone.

### Contact-based donor release and final full regression

The calibrated fixed pad lies on negative TCP X. If donor contact remains
above .05N after jaw opening, and receiver bilateral support/height are already
adequate, plan a4/8/12mm outward-normal donor movement. Endpoint must clear Cube
and all other geometry. Only this small departing finger segment allows initial
Cube contact, as in the existing first donor-departure check; all other pairs
retain normal checks. At most two measured attempts. Then require the ORIGINAL
three-second right-only hold: both right forces>.05N, donor forces<.05N,
cubeheight>=.15m. No forced success flag, contact exception in evaluator or
physical attachment. Offline first candidate4mm:63 intermediate samples,
.093mm IK position error/.143degree full orientation error, safe.

```bash
python scripts/run_manipulation.py --headless --stage full --motion-mode continuous --expert-hz 20 --grip-feedback-from-lift --episodes 3 --output-dir outputs/controller/full_clearance_audit
```

`full_clearance_audit/full_20261022T163854656945Z.json`, `il_full_clearance3.log`,
**exit0, 3/3 all expert and independent stages**, zero collisions. Each156.1833s
including3s reset, versus209.1667s old reference (25.3% shorter); the previous
narrow continuous mode was150.65s. One checked4mm normal departure per episode;
right-only minimum pads3.0700/3.1918N; max donor pads0/0N. Peak speed2.01100rad/s,
effort1.34962Nm; bin floor support>=.4904994N. Same physical metrics reproduced
in all three runs. These verify the new complete deterministic control mode,
not broad robustness. All seven source SHA256 hashes were compared with current
files and matched exactly before commit. Baseline05a0c13 remains in history.

First wall184.722s included120.347s fresh episode planning (pickup separately
logged12.26s). Warm repeats64.309/64.541s, each~1.47s normal-clearance planning.
Worker limit1 can reduce execution throughput even while reducing numerical
thread oversubscription; an eight-worker matched-cache comparison is pending.
Do not advertise a universal wall-time speedup. Normal-departure frames were
not yet included in transfer_trace; its sampled5.67mm dip is incomplete during
that gap and must NOT be called a certified full transient maximum. Continuous
trace collection will be corrected separately. Independent safety evaluation
DID inspect every physical tick, including normal departure.

Final control CPU regression31/31 in38.526s (`il_control_commit_tests.log`),
including RRT exact-edge fixture, no-path bound, reset-relative clocks, actual
recording chronology/terminal guards, geometry/LP equivalence and evaluator.

### Actual 1,000-update learned pilots

Both official methods used the SAME two calibrated interim demonstrations
(5,858frames) and seed701/batch16/1,000updates, ImageNet ResNet18; constant
AdamW lr1e-4/clip10. ACT18,725,388parameters,389.972s total, loss79.6424->.216413.
Diffusion40,901,452parameters,439.091s total, noise loss1.15133->.0590476.
These different loss definitions are not quality rankings.

ACT `full_20261022T154252429358Z.json`, `il_act_initial_eval5.log`: **0/1 nominal
+0/4 held-out**, all300s timeouts, zero pickup stages. Diffusion
`diffusion_pilot/full_...json`, `il_diffusion_initial_eval5.log`: **0/1 +0/4**;
actuator limits at1.1833/.8833s; left gripper/table contact at.2167s; right
wrist/jaw contacts at.9/1.1s. All failures counted, exit1. The earlier Diffusion
server did not seed its denoising RNG per episode; these are stochastic pilot
observations, not reproducible final comparisons. ACT inference uses latent0
and eval mode. A versioned, seeded reset/model-identity handshake remains needed.
The v2 held-outs are a distribution shift from the two original successful
training poses, further limiting interpretation. These pilots verify working
model-to-physics interfaces and illustrate insufficient data/training; NOT a
serious method ranking, learned task success or the requested30-trial benchmark.

Data expansion must use the corrected clock and improved expert AFTER source
commit. Preserve failed initial batches and report their separate fractions.
Human visual confirmation is needed for the new RRT/normal-departure behavior;
original continuous GUI remains open until replaced with the verified revision.

### Frozen provenance, bounded actions and seeded inference (2026-10-22)

The user confirmed the continuous GUI motion was correct. This confirms the
original continuous replay, not yet the newer RRT transit / donor normal move.
The original visible process56345 was retained until this confirmation.

Control commit `ba0bb04` preserves the exact seven source hashes of the three
successful new-control regressions. An eight-worker matched-cache follow-up also
passed2/2 (`full_clearance_audit/full_20261022T165449889359Z.json`); wall93.909/
72.996s, physics59.623/50.411s versus one-worker warm64.309/64.541s and physics
45.50/45.40s. First eight-worker episode overlapped CPU tests. No evidence that
increasing workers improves this workload; retain process-local default1.

Raw schema1.1 now embeds authored six-joint radian bounds, verified for both
arms with only2e-7rad float32 rounding tolerance. A test corrupts both nominal
and actual-held targets to90 and requires rejection. Legacy1.0 is readable but
reports whether action bounds were independently verified. Recording metadata
uses the process-start revision/dirty state and captured source hashes rather
than consulting changing Git state at each reset. Transfer traces now include
every donor normal-clearance tick, and expose nonuniform sampling / maximum gap;
previous incomplete transient maxima remain explicitly uncertified.

The versioned policy handshake `so101-policy-1.0` resets Python/NumPy/Torch RNGs
using the case seed, resets policy queues, and reports checkpoint weight hash,
adapter hash and software identity. The simulator refuses an obsolete server.
Actual official Diffusion checkpoint reset813000 twice on real saved RGB/state
produced EXACTLY equal12-target actions (`il_seeded_protocol.json`). This tests
repeatability of the adapter, not manipulation success. Original unseeded pilot
failure results remain historical and must not become a reproducible ranking.

Final CPU regression:33/33 passed in37.919s, `il_schema_trace_final_tests.log`;
command is the README CPU unittest command. `git diff --check` passed. Includes
all2,177 LP-equivalence queries /4,215 checked approach samples. A prior32-test
run passed37.878s and the added transfer-trace unit suite13/13 passed.190s.

After the user reported Git authentication complete, an actual escalated
`git fetch origin feat/fingertip-geometry-calibration` still failed128 with
`fatal: could not read Username for 'https://github.com': No such device or address`.
No credential helper or gh executable is configured in this terminal context.
Asked for a successful `git ls-remote origin refs/heads/feat/fingertip-geometry-calibration`
in the same workspace. No push is claimed and no remote ref changed.

### Published history and corrected four-case cohort

Authentication subsequently worked. Fetch obtained feature tip4c6a9d3; its entire
Git tree85a24610df2bbaf36122744b1ca35c5ce2e12a65 EXACTLY matched local historical
c4620be, with no unique remote file changes. A feature-only `ours` merge joined
both histories while preserving the tested c1d8336 tree byte for byte. Normal
non-force push succeeded; `git ls-remote` independently confirmed
`0f6793d000ad0fef02169949e8aab8f00f29a76d`. Main unchanged; no rewrite/publication.

The user's visual confirmation actually corresponds to GUI report
full_20261022T165446538383Z.json: all SEVEN source hashes match ba0bb04 exactly,
although its recorded parent revision2fe2a77 was dirty at execution. Thus the
confirmation already covered RRT/normal donor clearance. A subsequent20Hz RGB
GUI run was also visually confirmed, but the user closed the window before the
final terminal recording finished. Capture stopped at3,043 images near bin_settle;
task-owned process59805 was interrupted with SIGINT (exit0 through Kit handling).
Preserved raw incomplete attempt and interrupted_gui_checkpoint.json; NO completed
numerical success claimed for that attempt. Pinned SimulationContext.step and
STOP callback can render while waiting for playing; this is consistent with the
stall, not proof of the exact callback responsible. Wait for RESULT/VIEWER before
closing a recording GUI; headless generation avoids this interaction.

Separate nominal automated record: `corrected_nominal/full_20261022T172237776401Z.json`,
exit0, all stages/safety gates passed;3,063frames,153.15control seconds, schema1.1
bounds verified, source0f6793d. Checked caches copied from the interrupted GUI;
normal clearance remeasured/planned,1.47s. Full60Hz transfer trace563 samples,
maximumgap.0166666667s, uniform; cube dip5.8784mm including normal clearance.
Right-TCP dip5.7597mm; peak joint speed2.0110rad/s, effort1.34962Nm. Reported COM
vertical velocity residual persists (.04299m/s vs pose-derived.00000268m/s);
no velocity-settled claim. Total135.977wall includes recording and overlap with
other jobs, not an isolated speed benchmark. Camera teardown exited0.

```bash
python scripts/run_manipulation.py --headless --stage full --motion-mode continuous --expert-hz 20 --grip-feedback-from-lift --cases-file outputs/controller/train_v2_4.json --record-dir datasets/raw_train_v2_corrected --output-dir outputs/controller/corrected_train
```

`corrected_train/full_20261022T171227680021Z.json`, exit0, **4/4 full successes**,
every independent stage true, zero collision violations. Schema1.1 validation
also passed all four: frames3,042/2,920/3,073/2,951 = **11,986**,599.3s at20Hz.
All sourcec1d8336 with tracked dirtyFalse; current Git changes during run did not
relabel already-loaded source. Actual episode starting ticks180/9,486/18,426/
27,827; the last is NOT divisible by3, demonstrating corrected relative clock
on real multi-episode physics. Peak speed2.47281rad/s, effort1.50913Nm across
cohort. This is four preselected draws, not validated32/8/30 coverage or a
statistical robustness claim. Initial failed batches remain separately counted.
Raw validation results in il_corrected_train4_validation.json.

Seeded official Diffusion1,000-update pilot:
`diffusion_seeded_pilot/full_20261022T172619027187Z.json`, exit1, **0/5**;
nominal/heldout000/001/002 actuator violations at.85/1.50/.90/1.0167s;
heldout003 right wrist/jaw forbidden contact at.5167s. No pickup stages.
Per-case seed101,813000..813003 and checkpoint weight hash recorded in report.
Maximum model action-request time across cases23.65ms, including queued actions;
this is not full closed-loop wall time or a manipulation success. These replace
the unseeded pilot observations for reproducible diagnostics, NOT the requested
credible trained-policy comparison. Both models still trained on the original
two-demo shifted distribution, not the corrected four-case dataset.

### Donor axial-retreat limitation and bounded fallback

Fixed -1degree retest `corrected_yaw_minus/full_20261022T172435530481Z.json`
failed safely at plan_placement. Pickup, lift and right-only handover succeeded;
no unsafe withdrawal issued. Receiver RRT resolved the original earlier planning
failure, but40mm fixed-orientation donor IK exhausted its feasible range after
about20mm. Saved measured pose used for an offline test: a clear -2degree pan
separation then bounded donor RRT found a3-node route to HOME,1iteration,
10checked edges/3,042samples/20.42s (`il_donor_rrt_probe.json`).

Reused that SAME collision world/RRT only after all existing Cartesian departures
fail. Exactly one bounded global attempt; open jaw fixed, all arm/Cube/right/bin
checks retained, no spline across RRT turns. Runtime executes separate quintics.
Full offline placement (including carried cube travel and release opening)
selected the fallback in35.257s (`il_donor_fallback_offline.json`). This is
geometric evidence only; physical -1degree recovery still pending below.

Fallback CPU regression: **34/34 tests passed in70.029s**,
il_donor_fallback_tests.log, including full saved-pose donor/loaded-bin geometry
regression. `git diff --check` passed. Four-case dataset conversion exited0,
reader verified4episodes/11,986frames/v3.0 in datasets/lerobot_corrected_train4,
il_corrected_train4_conversion.log. No validation/test statistics were used.

Physical fallback attempt ef2d476 stopped SAFELY before withdrawal:
`corrected_yaw_fallback/full_20261022T173702154236Z.json`, exit1,
`Transit jaw must stay fixed`, controller_guard during plan_placement. Pickup,
lift and independent handover true; zero collisions. Simulator q arrays are
float32; assigning pi/4 into a copied float32 vector rounded it relative to the
float64 HOME jaw. The original JSON/offline fixture was float64 and missed this.
The saved-pose unit changed to actual float32 inputs reproduced the SAME error
in8.797s (`il_donor_float32_before.log`). Fixed only the fallback caller's array
dtype before assigning the exact jaw target; strict fixed-jaw guard unchanged.
Full34 tests passed72.781s (`il_donor_float32_fixed_tests.log`), diff check clean.

To avoid changing frozen source during generation, task-owned batches were
interrupted ONLY immediately after EPISODE markers and saved results: first two
heldout_v2_000/001 independently full success (sourceef2d476), and additional
train_v2_004/005 full successes. Batch requested4/8 but completed2/2; remaining
cases were NOT attempted and must be resumed explicitly. Interruption sidecars
preserved; exit0 through Kit SIGINT handling is NOT proof all requested cases ran.
Six validated training files now total18,118frames; the frozen four-episode
LeRobot dataset used for current training is unchanged at11,986frames.

Current-schema recorded-target replay, with NO expert plans or contact servo:
`corrected_replay/full_20261022T173946110719Z.json`, exit0, all independent stages
true, full success150.35s, zero collisions, speed2.0110rad/s, effort1.34962Nm.
This verifies action semantics through real physics; no recorded states copied.

### Verified -1degree full recovery, sourcefb408f5

`corrected_yaw_float32/full_20261022T175104954670Z.json`: all independent stages
true, FULL success, sourcefb408f58d8737c928694b8aa5dbfb6ee176d73a3, dirtyFalse,
zero collisions. User confirmed "Clean withdrawal and placement" for this exact
visible run.138.0333s independent elapsed; peak2.34637rad/s,1.45375Nm, stable
right slip1.16477mm/.99672degree. Acceleration160.135rad/s2 and jerk18,109.0rad/s3
measured, no authored limits to certify.563 contiguous60Hz transfer samples,
max gap1/60s, dip4.4587mm. Donor RRT1iteration/10edges/3,042samples/21.826s,
three transit nodes plus the initial clear separation. Episode410.707wall,
planning100.141wall amid other jobs; not isolated playback throughput.
Original failed planning/float32 attempts remain counted in their own cohorts.
This is one fixed-case recovery, not a rerun of the full five-case suite.

Independent raw/v3 preservation check:12 samples (start/middle/end of each of
four episodes), exact float32 state and actions, ZERO RGB byte error in BOTH
views after official reader decoding, timestamps within1e-5s,11,986 totalframes.
`il_v3_raw_alignment.json` /log, exit0. No privileged features in the dataset.

### User-requested stop and inference handoff — 22 October 2026

The user requested that experiments stop, documentation be updated and the feature
branch be pushed so they can diagnose inference themselves. All five remaining
task-owned processes were stopped: two expert batches, the ACT evaluation and
both policy servers. Training and recording had already finished. No new training
or simulator run was launched after the request. Stop evidence is preserved in
`outputs/controller/user_requested_stop.json`; the audit of saved reports,
recordings and training metadata is `outputs/controller/user_stop_audit.json`.
Generated evidence, datasets and weights remain local and ignored by Git.

Latest corrected data and expert results (earlier failed cohorts remain above):

| Cohort | Completed evidence | Local report directory under `outputs/controller/` |
| --- | --- | --- |
| Training cases 000–003 | 4/4 full successes, 11,986 frames | `corrected_train` |
| Training cases 004–005 | 2/2 full successes; larger batch stopped at a saved boundary | `corrected_train_next8` |
| Training cases 006–011 | 6/6 full successes | `corrected_train_remaining6` |
| Validation cases 000–001 | 2/2 full successes, 6,050 frames | `corrected_validation2` |
| Held-out cases 000–001 | 2/2 full successes; larger batch stopped at a saved boundary | `corrected_heldout4` |
| Held-out cases 002–003 | 2/2 full successes | `corrected_heldout_remaining2` |
| Held-out cases 004–008 | 5/5 full successes; case 009 interrupted at user stop | `expert_heldout_004_016` |
| Held-out cases 017–021 | 5/5 full successes; case 022 interrupted at user stop | `expert_heldout_017_029` |

Twelve distinct training recordings passed strict raw schema 1.1 validation:
**36,268 frames / 1,813.4 policy seconds** in `datasets/raw_train_v2_corrected`.
Two separate validation recordings passed the same checks: **6,050 frames /
302.5 seconds** in `datasets/raw_validation_v2_corrected`. Neither raw root has
an incomplete episode directory. Only training cases 000–003 were converted to
`datasets/lerobot_corrected_train4` and used for the current models and their
normalization statistics. The remaining eight training recordings and both
validation recordings were not used for those weights.

The expert held-out total is **14 completed successes, zero completed failures,
two interrupted attempts and fourteen unstarted cases**, out of thirty planned.
This is not a completed thirty-case success rate. Cases 010–016 and 023–029 were
not started. Original partial reports retain `in_progress`; their completed
episode entries were not rewritten. New `user_stop.json` sidecars in both active
expert directories identify the interrupted case and original report. Older
boundary interruptions retain their existing `batch_interruption.json` sidecars.

Both latest official LeRobot 0.6.1 models finished **4,000 updates**, seed 701,
batch 16, two loader workers, AdamW learning rate 1e-4 and gradient clipping 10,
using the same four demonstrations / 11,986 frames:

| Model | Local checkpoint root | Parameters | Training wall seconds | First → last loss |
| --- | --- | ---: | ---: | --- |
| ACT | `checkpoints/act_corrected_train4_4000` | 18,725,388 | 1,507.8297 | 89.56226 → 0.0513568 |
| Diffusion | `checkpoints/diffusion_corrected_train4_4000` | 40,901,452 | 1,474.0479 | 1.147505 → 0.0120214 |

Inference loads each root's `pretrained_model` directory, including its saved
official processors. Intermediate 500-update checkpoints and optimizer/RNG state
remain available; the current CLI does not implement exact training resume.
Losses have different definitions and are not a manipulation comparison.
Training logs are `outputs/controller/il_act_corrected_train4_4000.log` and
`il_diffusion_corrected_train4_4000.log`; each root contains `training.json`.
`outputs/controller/user_stop_checkpoint_hashes.json` preserves file hashes.
Final weight SHA-256 values:

- ACT: `cb51e88ec754343ddeeee8c9564175fe33bd8941e5377932d5954c0feba8ef14`
- Diffusion: `53a4cb096a11a655f5c719eff8c13e066a4cf207b49445048a7f02e69172643f`

The latest learned-policy physics results are unsuccessful:

- ACT: `act_corrected4_pilot_workers8/full_20261022T180832930960Z.json` records
  **0/4 completed attempts**, all 300-s simulated timeouts without pickup
  (nominal and held-out 000–002). Held-out 003 was interrupted at the user's
  request; its `user_stop.json` sidecar preserves that distinction. The last
  periodic log reached 4,801 policy steps with no task stages achieved.
- Diffusion: `diffusion_corrected4_pilot/full_20261022T182249865879Z.json` records
  **0/5**, with no pickup: nominal actuator limit at 0.85 s; held-out 000 left
  jaw/table forbidden contact at 4.1667 s; held-out 001 actuator limit at
  3.2167 s; held-out 002 right gripper/bin contact at 2.5167 s; held-out 003
  right jaw/bin contact at 1.5333 s. These remain counted failures.

An earlier ACT launch with one Kit worker stalled during startup, before reset
or a policy action. It was terminated and recorded in `il_act_startup_failure.json`;
it is an infrastructure failure, not an attempted manipulation. The eight-worker
retry initialized and produced the four completed evaluations above. The README
uses process-local `--cpu-threads 8`; no driver, dependency or OS setting changed.
The first Diffusion nominal target differed from measured joints by up to
0.22441566 rad. Installed official Diffusion observation/action indices were
checked; no erroneous one-frame action-label shift was found. Recorded-target
replay and raw/v3 byte preservation passed separately, as documented above;
neither establishes learned-policy quality.

The already-tested optional policy tracing change is retained for diagnosis:
`serve --trace-dir ... --trace-images` writes reset seed, model/adapter hashes,
24-value pre-action proprioception, requested 12-joint targets and actual RGB
inputs. It never records evaluator labels or claims requested targets were
applied. Traces are not demonstration episodes and omit terminal images. Model
timing excludes trace I/O; client round-trip timing includes it. Camera names
must match the model configuration and proprioception must be finite and shape 24.

Actual-model trace integration passed before the stop:
`outputs/controller/il_policy_trace_check.json` records two resets with seed
812000, exactly equal actions, exact JSON states/targets and byte-identical PNG
inputs. `il_trace_cli_guard.log` confirms `--trace-images` without `--trace-dir`
exits 2. The traced Diffusion server also completed the five physics evaluations
above. After stopping, only CLI help, Python compilation and diff checks were
run; no additional experiment was started. The most recent geometry/evaluator/
dataset regression remains **34/34 passed in 72.781 s** in
`il_donor_float32_fixed_tests.log`.

README section **View the trained models in Isaac Sim** gives complete two-shell
activation, checkpoint/socket commands for ACT and Diffusion, GUI defaults,
expected failures, report/trace paths and shutdown instructions. Source push
does not upload ignored weights, recordings or logs; run inference in this
existing workspace. The pinned simulator environment and NVIDIA driver are
unchanged. Full 32/8/30 collection/evaluation, successful learned manipulation,
statistical comparisons, cuRobo runtime work and a publishable comparison demo
remain unfinished. Any future work should begin with the saved input/target
traces and separate validation cases; no further work is running at handoff.

### Documentation correction and camera review — 22 October 2026

At the user's request, further training and visual simulator work remain stopped
and under the user's control. The user reported poor dataset images and asked
for clear procedures, exact demonstration provenance, policy rationale, camera
choice, environment explanations and a GitHub-reviewable source record. They
explicitly requested that datasets and weights NOT be uploaded.

README now serves as the user procedure and complete source inventory.
`docs/POLICY_REVIEW.md` explains the implementation, exact four-demo and earlier
training cohorts, fixed cameras rather than wrist cameras, action/observation
semantics and limitations of the custom training loop. The former README is
preserved at commit d1cb3be; no extra copy of the old experiment log was created.
`docs/POLICY_REVIEW_METADATA.json` contains a review snapshot of actual saved
configs/processors, training arguments, dataset statistics, conversion provenance,
source and checkpoint hashes. It is documentation metadata, not a runnable
trainer configuration, dataset, weight file or raw experiment log.

Read-only inspection of `train_v2_000_20261022T171242694160Z` frames 000000 and
001500 in both overview/front found poor initial cube visibility/detail and
partial hand occlusion during transfer. The initial front-view cube is roughly
ten pixels across; it is not clearly visible in the initial overview sample.
These four samples support the user's concern but do not isolate the sole cause
of learned failure. Prior nonblank/shape/timestamp checks did not establish
visual suitability, and main-viewport approval did not validate the policy views.
No wrist-camera design or comparative camera study was implemented.

Documentation explicitly distinguishes the official LeRobot model classes from
the custom trainer. Static source inspection showed ACT's backbone retains its
1e-5 parameter-group rate while other parameters use CLI 1e-4; Diffusion's saved
optimizer/scheduler defaults are not used by the custom AdamW loop. Its saved
`drop_n_last_frames=15` is not applied by an episode-aware sampler, and its padding
loss mask is disabled. These are review findings, not fixes or proven failure
causes. There is no validation loop or best-checkpoint selection.

All seven four-demo recorded source hashes matched Git revision c1d8336, and
that revision's policy_pipeline.py hash matched both latest training reports.
No task simulation/training/inference process was found in this workspace during
the read-only audit. Only documentation and review metadata were changed; no
experiment, training, dependency installation or simulator run was started.

Documentation verification: 41 local links/anchors resolved; 12 Bash blocks
passed `bash -n` without executing their commands; 11 documented Python command
lines matched source-declared flags; all 18 inventoried source/config/test hashes
matched the review snapshot. Both saved model configs and training arguments
matched their local originals, the four-demo frame sum was 11,986, and every
tracked source/config/test file appeared in the README inventory. `git diff
--check` passed. No runtime regression or learned-policy test was rerun for this
documentation-only change.


### Learning reset inventory — 23 October 2026

Baseline f47c358 protected by local tag `reference/pre-il-camera-v2`; new branch
`feat/il-camera-dataset-v2`. Physical recovery source fb408f5; 34 prior CPU tests.
Historical configurations/statistics remain in POLICY_REVIEW_METADATA.json and
failure outcomes above. No regression test references dataset/checkpoint artifacts.
Verified ordinary local files/directories, no symlinks or nested mounts using
lstat and /proc/self/mountinfo. No benchmark process active at deletion.

| Path | Files | Logical bytes | Allocated bytes | Disposition |
| --- | ---: | ---: | ---: | --- |
| `datasets/lerobot_corrected_train4` | 9 | 446125083 | 446144512 | Authorized deletion |
| `datasets/lerobot_initial_train` | 7 | 217890719 | 217907200 | Authorized deletion |
| `datasets/lerobot_smoke` | 6 | 109930991 | 109936640 | Authorized deletion |
| `datasets/raw_corrected_nominal` | 12221 | 229676722 | 256827392 | Authorized deletion |
| `datasets/raw_production_train` | 11728 | 219946527 | 246358016 | Authorized deletion |
| `datasets/raw_reference` | 1977 | 37462341 | 42086400 | Authorized deletion |
| `datasets/raw_smoke` | 5911 | 110937040 | 124174336 | Authorized deletion |
| `datasets/raw_train` | 11708 | 219288717 | 243822592 | Authorized deletion |
| `datasets/raw_train_v2` | 8360 | 157353308 | 176783360 | Authorized deletion |
| `datasets/raw_train_v2_corrected` | 72600 | 1362083487 | 1524039680 | Authorized deletion |
| `datasets/raw_validation_v2_corrected` | 12112 | 226729922 | 253128704 | Authorized deletion |
| `checkpoints/act_corrected_train4_4000` | 57 | 825557036 | 825724928 | Authorized deletion |
| `checkpoints/act_initial_2ep_1000` | 21 | 375145592 | 375209984 | Authorized deletion |
| `checkpoints/act_smoke` | 8 | 225008798 | 225030144 | Authorized deletion |
| `checkpoints/diffusion_corrected_train4_4000` | 57 | 1801234120 | 1801420800 | Authorized deletion |
| `checkpoints/diffusion_initial_2ep_1000` | 21 | 818713193 | 818790400 | Authorized deletion |
| `checkpoints/diffusion_smoke` | 8 | 491206546 | 491233280 | Authorized deletion |
| `outputs/policy_traces/diffusion_corrected4` | 514 | 9484200 | 10518528 | Proposed deletion only; retained |
| `outputs/policy_traces/manual_act` | 2538 | 47671963 | 52043776 | Proposed deletion only; retained |

Deletion completed: datasets/ and checkpoints/ recreated empty; no placeholders
needed because these artifact directories remain ignored. Policy traces, all
other output evidence, caches and external environments were retained.

### Camera-v2 research and official training preparation — 23 October 2026

Official sources consulted before camera implementation:
- https://github.com/TheRobotStudio/SO-ARM100
- https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101/assets
- https://github.com/TheRobotStudio/SO-ARM100/tree/main/Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module
- https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/core-concepts/sensors/camera.html
- https://isaac-sim.github.io/IsaacLab/v2.3.0/source/tutorials/04_sensors/add_sensors_on_robot.html

Mount assembly identifies the wrist-roll housing's hex-nut recesses. Local model
mesh mapping places wrist_roll_follower under rigid gripper, not moving jaw.
The upstream binary STL was inspected: 3,476 triangles, bounds and SHA recorded
in assets/camera_mount/README.md, with Apache license. Assembly image URL returned
404; no claim is made to have measured camera extrinsics from that image or CAD.
Mesh is a reference only, not instantiated mass/collision geometry. Sensor offsets
are explicitly initial simulation estimates. Camera properties and transforms
are in policy_cameras.py and README, with no installed robot USD modifications.

Official training/data sources:
- https://huggingface.co/docs/lerobot/en/envhub_leisaac
- https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
- https://raw.githubusercontent.com/huggingface/lerobot/v0.6.1/src/lerobot/scripts/lerobot_train.py

Installed LeRobot 0.6.1 CLI help ran successfully. Source inspection confirms
policy optimizer/scheduler factories, EpisodeAwareSampler with policy-specific
tail exclusion, checkpoint/resume handling and an optional episode evaluation
split. This replaces the previous custom optimizer path with a command-preparation
function that NEVER trains. Steps/batch size/evaluation choices are mandatory user
inputs. Hub uploads and EnvHub environment evaluation are disabled in the printed
command. Our task is not registered as an EnvHub environment; the external LeIsaac
example is not a drop-in benchmark or a reason to alter the pinned installation.
ACT uses its official conditional VAE/action loss and saved processors; Diffusion
uses its official sequence indices, padding policy and optimizer/scheduler preset.
Inference retains official select_action/pre/postprocessors through the socket.

Disposable compatibility fixture: 48 synthetic constant-image frames, three
640x480 RGB cameras, state24/action12, saved as official v3 image data in a temporary
/tmp directory and removed when the check ended. This is a file-format unit
fixture, not an expert demonstration or training dataset. Official ACT sequence
reader returned actions[100,12], image[3,480,640]; Diffusion returned actions[64,12],
images[2,3,480,640]. Log: outputs/controller/camera_v2_lerobot_contract.log.
No model instantiated, no optimizer update, no checkpoint. Real-data training
and official-checkpoint simulator inference remain gated and unverified.

### Future synthetic demonstration assessment (not implemented)

Sources:
- https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/imitation-learning/teleop_imitation.html
- https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/imitation-learning/skillgen.html
- https://curobo.org/get_started/1_install_instructions.html
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/index.html

Mimic requires a ManagerBasedRLMimicEnv-compatible task adapter and helpers for
end-effector poses, target-pose/action conversion, gripper commands, object poses
and subtask termination signals. Our five-axis arm cannot realize arbitrary
6-DOF target poses, so reuse constrained IK with explicit infeasibility failures.
Define left acquire/lift, carry-to-transfer, coordinated receiver acquisition,
donor release/withdrawal, right transport/release and retreat. Cube-relative
frames apply to acquisition, a shared transfer frame to the bimanual interval,
and bin-relative frames to placement. Order constraints must prevent donor
release before verified receiver support. Physical success remains independent.
An initial future experiment would transform a few approved demonstrations to
nearby feasible poses, with synchronized bimanual segments and full collision/
contact checking. Do not assume automatic stitching preserves a grasp.

SkillGen adds motion-planned free-space bridges to the demonstrated skills.
Lab 2.3's instructions pin cuRobo ebb71702f3f70e767f40fd8e050674af0288abe8 and discuss
CUDA 12.8; generic cuRobo install docs refer to older stacks. Neither is an
installation authorization. A dual-arm robot configuration, collision spheres,
world conversion, payload transform/attachment semantics and planner-to-joint-
target adapter need implementation and testing. Payload attachment in a planner
must not become a physical constraint on the cube. RTX5090 sm120 extensions,
Torch/CUDA ABI compatibility and five-axis pose feasibility are unresolved risks.
Planning estimate, not measured work: Mimic adapter/annotation is several engineer
days to weeks; robust coordinated SkillGen/cuRobo integration is additional weeks,
not a configuration toggle. No cuRobo software was installed.

Replicator can later vary lighting, materials and backgrounds with fixed seeds
and documented ranges, while keeping geometry/contacts fixed. It augments pixels;
it does not supply valid manipulation trajectories or repair invisible grasps.
A small controlled appearance experiment after camera acceptance is lower effort
(roughly days), but must check RGB/physics synchronization and visibility again.

### Camera-v2 verification and human review checkpoint — 23 October 2026

Final production changes: policy_cameras.py configures the fixed overview and
both sensors under /World/envs/env_0/{Left,Right}_Robot/gripper/{left,right}_wrist.
Sensor position in the rigid gripper link: (0.0141,-0.045,0.005) m. ROS optical
quaternion wxyz: (0,0,0.9756376988650908,0.21938796811409378), aimed at the calibrated
grasp region (0.0141,0,-0.09) m. ROS optical +X is negative parent X; +Y down,
+Z forward. Projection: 640x480 RGB, focal 12 mm, horizontal aperture20.955 mm,
fx=fy366.4996338px, principal point(320,240), horizontal FOV82.25005degrees,
clip0.01–10m. These are initial simulation estimates, not measured real hardware.
The upstream mount supports a32mm UVC module; neither its lens nor bracket-to-link
registration is calibrated here. No physical mount mass/collision geometry added.

Default pinned Isaac camera world-pose reads were stale against the measured
articulation. Disabling Fabric did not fix the issue. A Fabric read also exposed
an upstream Warp dtype error with explicit int64 camera indices; reading all
view indices avoids that error. A narrow project-owned subclass reads live Fabric
poses. Before each captured observation, sensor-only Fabric world transforms are
synchronized from the measured PhysX link and constant mounting offset. Actual
camera poses are then checked independently: translation atol2e-5m and rotation
matrix atol2e-4, rtol0. Robot/object physical state writes are unchanged. This uses
pinned private camera _view/_update_poses interfaces and must be reviewed if the
simulator is ever upgraded; installed packages were not patched.

Failed camera prerequisites/diagnostics remain under outputs/controller prefixes
camera_v2_nominal, camera_v2_diagnostic, camera_v2_checked, camera_v2_usd,
camera_v2_fabric, camera_v2_fabric_checked, camera_v2_live_link and
camera_v2_local_transform_probe. They are not successful manipulation tests or
demonstration-generation runs. No failing attachment tolerance was relaxed.

Complete nominal physics test, exit0:

```bash
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/run_manipulation.py --headless --cpu-threads 8 --stage full \
  --motion-mode continuous --expert-hz 20 --grip-feedback-from-lift \
  --camera-preview outputs/camera_v2/synchronized \
  --output-dir outputs/controller/camera_v2_synchronized
```

Do not rerun into the same preview directory. Log camera_v2_synchronized.log;
report camera_v2_synchronized/full_20261023T085954936832Z.json. Independent pickup,
lift, handover and placement all true;153.1833333simulated seconds, zero forbidden
collisions, peak2.011003rad/s and1.349620Nm. Stable grasp slip0.2673/0.2866mm for
left/right. Authored acceleration/jerk limits still absent; measured quantities
are not certified limits. The report preserves dirty source hashes from the
working tree at execution, baseline revisionf47c358; it does not assert that the
uncommitted camera code was already part of that historical commit.

51 synchronized three-view snapshots cover26phases, with each saved sample
validated at (physics_step-reset_origin)%3==0. This is a sparse camera audit,
not a contiguous20Hz recording. Eight representative phases are combined in
outputs/camera_v2/synchronized/review_contact_sheet.jpg. Each original row is
1920x508 with overview,left_wrist,right_wrist. audit.json includes actual poses,
intrinsics and physical ticks. Tensor RGB validity, ordering path, dimensions,
missing cameras and schedule checks are enforced; no pixel-based visibility or
occlusion score is claimed. Cube visibility and housing/other-arm occlusion need
human judgment. Overview transfer nears its lower edge; initial occlusion remains.

After the full run, the optional fixed front diagnostic was added (excluded from
policy features) and passed a separate reset/model test, exit0:
`python scripts/run_manipulation.py --headless --cpu-threads 8 --stage model
--camera-preview outputs/camera_v2/front_check --diagnostic-front
--output-dir outputs/controller/camera_v2_front_check`.
Default camera configuration stayed unchanged; the front option was not part of
the full-task run. File camera_v2_front_check.log preserves the result.

CPU verification:

```bash
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
USD_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
export PYTHONPATH="scripts:$USD_LIB"
export LD_LIBRARY_PATH="$USD_LIB/bin:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Baseline34/34 in68.918s (camera_v2_baseline_tests.log). Final37/37 in69.868s
(camera_v2_commit_tests.log), including wrong-parent/rotation rejection,
missing/bad RGB and invalid clock tests. Camera attachment comparison was factored
without changing tolerances after the physical run; final CPU tests passed.
Compilation, launcher/prepare-training help and diff checks also passed.

Additional retained artifact inventory: outputs/huggingface40files775,889,478
logical bytes (cache); outputs/torch1file46,830,571bytes, pretrained ImageNet
resnet18-f37072fd.pth. These lie outside the two deletion-authorized trees and
were not deleted. Previously inventoried policy traces total57,156,163bytes;
proposed cleanup is only outputs/policy_traces/diffusion_corrected4 and manual_act,
not executed. Other output evidence is retained. datasets/ and checkpoints/
verified empty again after all tests; format fixtures used /tmp, not these trees.

No deterministic model/planner/evaluator physics criteria or installed USD asset
were changed. dataset_io.py and all prior tests are retained. Converter dimensions
now come from metadata; attempted failure reasons and a quality summary remain in
conversion.json. Manual recording additionally requires --authorize-recording.
Custom optimizer training code was replaced with official command preparation;
source history retains it. No learning source files or useful tests were deleted.

Camera approval is pending. Exact GUI command is in README. The next action is
human inspection of all three policy views; no new dataset collection or training
is authorized by merely accepting the cameras. No remote push is authorized in
this assignment. No trained-model success, hardware calibration, automatic
occlusion certification, Mimic/SkillGen integration or new distribution coverage
is claimed.

### Conditional camera approval and overview comparison — 23 October 2026

User conditionally approved the wrist cameras, requested overview alternatives,
matching-phase comparison, continuous synchronization/attachment evidence, a
recommendation and one consolidated visual review. Dataset strategy/collection
and training remain separately gated. No remote push authorized.

Preserved reviewed configuration adf03b4 with local annotated tag
`reference/camera-v2-reviewed`. Original overview is retained as the baseline
profile; default selection remains baseline pending final framing approval.
Wrist extrinsics, optics, resolution and parent links are unchanged.

Measured baseline transport/handover cube position is approximately
(0,-0.270,0.178/0.174)m, rather than the provisional scene-constant handover Y.
Used measured working area to design fixed overview alternatives in world metres:

| Profile | Eye | Look-at target |
| --- | --- | --- |
| baseline | (0,-0.10,0.72) | (0,-0.10,0.02) |
| centered | (0,-0.20,0.78) | (0,-0.20,0.02) |
| angled_front | (0,0.07,0.78) | (0,-0.18,0.02) |
| angled_rear | (0,-0.45,0.78) | (0,-0.18,0.02) |
| angled_left | (-0.25,-0.18,0.78) | (0,-0.18,0.02) |

All use the same640x480/12mm focal/20.955mm aperture/0.01–10m clip and fixed world
pose for the entire run. Front/rear/left inclinations are about18.2/19.6/18.2degrees
from vertical. Initial reset-only screening (overview_v3_startup.log) tested
steeper front/rear views; they left excess background, so reduced inclinations
were chosen before the full comparison. Screening did not certify task framing.
Diagnostic sensors review_* are separate from the three policy input keys.

All five alternatives are rendered in a single deterministic episode and read
at the same physical ticks as the three policy tensors; no scene replay, cloned
object state or controller action changes are used to compare views. The
comparison flag records continuous20Hz policy RGB video and a JSONL clock/frame/
pose/hash audit under outputs only, without demonstration action labels. Sparse
matching-phase images show all alternatives. Cube corner projections use ground
truth only in this diagnostic; frustum inclusion is not an occlusion test.
Red-pixel counts are an appearance-specific proxy, not certified visible area.

Camera continuity tests reject missing3-tick samples, reversed time, stale
sensor frame IDs, missing streams and invalid reset-relative timestamps. Initial
preview and first action at the same physical tick produce one video frame.
Attachment assertions execute at every captured20Hz boundary, not just phase
snapshots. RGB hashes expose repeated frames for review, without failing legitimate
stationary observations merely because they repeat.

CPU regression:39/39 passed70.738s, outputs/controller/overview_v3_tests.log.
Source AST comparison against adf03b4 verified Episode.step/run/move/move_right and
BenchmarkEnv.advance/step/reset/_read unchanged. Model, planner, independent
evaluator, scene/actuator configuration and dataset_io.py are byte-identical.
Only camera configuration, diagnostic capture and orchestration options changed.

Exact comparison command (after existing simulation environment activation):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_manipulation.py \
  --headless --cpu-threads 8 --stage full --motion-mode continuous \
  --expert-hz 20 --grip-feedback-from-lift --compare-overviews \
  --camera-preview outputs/camera_overview_v3/comparison \
  --output-dir outputs/controller/overview_v3_comparison
```

For human GUI reproduction omit --headless and choose a fresh preview/output
root. Camera parameters never change during an episode. Final approval will
identify a named profile and exact recorded calibration/source revision; no
adaptive reframing or silent parameter tuning is permitted after approval.

Overview comparison completed, exit0. Report:
outputs/controller/overview_v3_comparison/full_20261023T142830953036Z.json;
log overview_v3_comparison.log. Independent full success153.1833333simulated seconds,
zero forbidden collisions, peak2.011003rad/s/1.349620Nm. Every independent stage
passed. Elapsed time, peak speed/effort and the COMPLETE transfer trace exactly
match camera_v2_synchronized/full_20261023T085954936832Z.json. This supports that
extra rendering/diagnostics did not alter this nominal task's measured behaviour;
it does not establish general robustness or wall-time performance.

50 sparse snapshots and nine selected phases compare all five views at identical
physical ticks. All eight projected cube corners were inside every candidate's
frustum in all50samples. This tests projection bounds, not occlusion or robot
mesh clipping. Red-pixel proxy sums across selected phases were baseline4635,
centered3527, front3341,rear3149,left3358; these are not a ranking of policy quality.
The left angle showed605red pixels initially but only1during lift/hold, a poor
whole-task tradeoff. The centered view balances handover margin, stable global
orientation, cube/bin context and less perspective/background than the oblique
alternatives. Recommendation: centered, pending final human approval. Do not
silently switch the production default away from baseline before approval.

Recommended overview eye(0,-0.20,0.78)m, target(0,-0.20,0.02)m; ideal ROS wxyz
(0,1,0,0). Recorded float quaternion is approximately(7.45e-9,1,-1.44e-8,1.44e-8).
Optics unchanged:12mm focal,20.955mm aperture,640x480RGB,fx=fy366.4996338px,
principal point(320,240),82.25005degree horizontalFOV,0.01–10m clip,20Hz observations.
Wrist offsets/rotations remain exactly as previously documented and conditionally
approved. Calibration is an initial simulation estimate, not measured hardware.
Proposed calibration hash36b7a6f81433eb927ee69e0e3a42057fc74ec359da14fa00d7919f440fce0592,
with full parameters in outputs/camera_overview_v3/comparison/proposed_calibration.json.
Parameters are fixed per run. Final approval must pin this profile/calibration
and source revision; subsequent changes require a new calibration review.

Continuous diagnostic evidence:3064unique RGB frames per policy camera,20Hz,
first/last physical ticks180/9369, every interval3ticks. Sensor frame IDs strictly
increase; actual wrist attachment checked at each observation. Independent
imageio_ffmpeg count/decode confirmed3064encoded frames and153.2seconds, matching
continuity.jsonl. independent_verification.json records checks and colour-framing
proxies. Software physical-step stamps are not independently measured RTX or
real-camera exposure timestamps. A final continuity-failure flag improvement
records failed checks instead of hard-coding success; five targeted camera tests
passed again after that and offline review-rendering additions. No optics,
extrinsics or physics changed after the full run.

One consolidated human review: outputs/camera_overview_v3/comparison/recommended_review.jpg
(original overview / proposed centered / original wrist images, matching ticks).
All alternatives: overview_comparison.jpg. Continuous video policy_streams_20hz.mp4
uses original baseline overview plus unchanged wrists; do not mislabel that video
as proposed-centered-camera playback. Sparse comparison images contain the actual
centered sensor pixels, not a crop or reprojected original image. Rebuild figures:

```bash
python scripts/policy_cameras.py --build-review outputs/camera_overview_v3/comparison --recommend centered
```

The inspector and figure builder write only diagnostic outputs. No demonstrated
actions/learning records, model weights or optimizer steps were generated. Dataset
and checkpoint directories remain empty. Camera approval and demonstration-strategy
approval are separate gates. No dependencies, assets, driver or system settings
changed, and no remote push was performed. Further camera work pauses at the
consolidated human review.

## 2026-10-23 — Camera freeze and demonstration protocol pipeline

User approved centered overhead plus both unchanged wrist sensors. No further
camera optimization. User requested a substantial research-backed demonstration
protocol and complete generation/QA tooling before authorizing any dataset or
training. **No collection, training, or remote push in this session.** Datasets
and checkpoint roots remain empty. Temporary synthetic format fixtures live only
under `/tmp/so101_protocol_format_iy7klat1`, not in either artifact root.

Current branch `feat/il-camera-dataset-v2`, started clean at `2aa6961`. New local
milestone commit is discoverable with `git log -1`; no push. Preserve both existing
reference tags; main and remote unchanged. README is now the run procedure and
complete maintained-file inventory. Research/choices/counts/ranges/budgets/gates
are in `docs/DEMONSTRATION_PROTOCOL.md`; do not confuse proposed settings with
executed experiments.

### Frozen environment/camera record

Stack unchanged: leisaac-sim Python3.11.16/NumPy1.26/torch2.7.0+cu128,
Sim5.1/Lab2.3/LeIsaac0.4. Isolated so101-il-py312 has Python3.12.15,
LeRobot0.6.1/NumPy2.2.6/torch2.7.1+cu128. No install/upgrade/driver/OS changes.

Approved source is `configs/policy_cameras.json`, version so101-cameras-3.0,
SHA256 `f8f7e56842602fbe040fea43b1cf1a51f75f56b3ad28357aae04f717c8e9fcbc`.
640x480 RGB at20Hz; focal12mm, aperture20.955mm, clips0.01..10m,
K=[[366.4996338,0,320],[0,366.4996338,240],[0,0,1]]. Overview eye(0,-.20,.78)m,
target(0,-.20,.02)m, ROS wxyz(0,1,0,0). Both wrists attach to rigid gripper link,
position(.0141,-.045,.005)m, aim at(.0141,0,-.09)m, ROS wxyz
(0,0,.9756376988650908,.21938796811409378). Not measured physical calibration.
Original proposal hash36b7a6f... is a different JSON structure, preserved as
provenance, not the new file hash. Centered is now default; historical profiles
remain diagnostic only. User approval did not authorize demonstrations.

### Implementation and evidence boundary

- `demonstration_pipeline.py`: deterministic protocol manifests, offline pilot
  preflight, dry-run/guarded existing-expert launches, start/finish ledger, resume
  without retries, exhaustive QA, review sheets/video/template, hash-bound human
  acceptance. Pilot12/train100/validation20/heldout100/stress20 are attempted counts.
- `demonstration_protocol.json`: proposed distribution and QA/training conditions;
  SHA256 `9ec7c3539c287e11bdc988eb52597a16001404d81ef43f8aad64fd0c6a4a3b3a`.
  Protocol approval and per-session collection authorization still required.
- Recorder instrumentation saves phase, actual frame IDs, measured camera and
  parent transforms at each policy boundary. Dataset validation also checks
  float32 state/actions and absolute physics time. No policy feature gains oracle
  information. Conversion requires accepted QA and unchanged raw bytes/inventory;
  failed/missing process attempts stay in the declared denominator.
- `so101_model.py`, `so101_planning.py`, `benchmark_core.py`, scene module remain
  byte-identical to2aa6961. BenchmarkEnv.advance/step/reset/_read and expert
  move/move_right AST unchanged. Episode.run differs only in recording metadata;
  Episode.step adds capture evidence; Episode.reset adds camera-file hash guard.
  No motion, contact, actuator, gain or task evaluation change was made.
- Quaternion-to-matrix camera helper uses NumPy so QA/conversion work in both
  locked environments. Frozen wrist quaternion is checked against its optical
  target. Sim camera adapter remains project-owned and sensor-only.

### Executed commands and actual results

All simulation commands source conda.sh, activate leisaac-sim, set assets root,
OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Official-reader commands activate the
existing so101-il-py312 instead. No optimizer/model construction in these tests.

1. `python scripts/demonstration_pipeline.py plan --output outputs/demonstration_protocol_v1`:
   fixed manifest/hash plan written. `collect` without --execute printed12 proposed
   launches; no simulator or recording from that command. Earlier pre-final
   manifests in outputs/demonstration_proposal_v1 are stale after the final protocol
   observation-history setting was added; use demonstration_protocol_v1.
2. Bundled USD library setup as in protocol, then
   `python scripts/demonstration_pipeline.py preflight --manifest outputs/demonstration_protocol_v1/pilot.json --output outputs/demonstration_protocol_v1/pilot_preflight.json`:
   **12/12 HOME/open/pregrasp/grasp geometry plans safe**, summed137.770959s.
   No physical contacts, lift, handover, bin or RGB task observability proved.
3. With bundled USD PYTHONPATH/LD_LIBRARY_PATH:
   `python -m unittest discover -s tests -v`:
   **45/45 passed in71.810s**, log outputs/demonstration_proposal_v1/milestone_tests.log.
   Six new protocol tests include mocked collector failure/resume (no real child
   simulator), fixed/tampered manifests, corruption outside sparse image samples,
   stale sensor clocks, wrong wrist parent, physics failure, production length gate,
   human acceptance hashes, missing ledger and mixed sources. Existing collision
   equivalence still checked2177 LP queries/4215 path samples. Earlier full44-test
   passes (71.511s,72.708s) precede the last added collector test.
4. `python scripts/run_manipulation.py --cpu-threads 8 --stage model --headless --exit-after --camera-preview outputs/approved_camera_freeze_model_retry --output-dir outputs/controller/approved_camera_freeze_model_retry`:
   **passed_requested_stage**, default centered profile,9.046423s wall. Approved K,
   overview extrinsics and actual measured-parent poses rechecked from audit.json.
   One reset/interface sample only; no action-labelled demonstration or complete
   manipulation episode. GUI defaults remain unchanged for user commands.
5. Two synthetic8-frame episodes created with test fixtures and synthetic review
   flags solely under /tmp. In so101-il-py312:
   `python scripts/policy_pipeline.py convert --raw /tmp/so101_protocol_format_iy7klat1/raw --output /tmp/so101_protocol_format_iy7klat1/lerobot_final --quality-report /tmp/so101_protocol_format_iy7klat1/qa_accepted/quality.json`:
   **2 episodes/16 frames, official v3.0** reopened successfully. Official ACT and
   Diffusion config/sequence readers tested indices0/7/8/15, actions[64,12], all
   three CHW[3,480,640] images, finite values,63 padded future targets at episode
   end; no cross-episode windows. ACT state[24], Diffusion history state[1,24].
   Logs: format_conversion_final.log, official_sequence_reader_retry.log in
   outputs/demonstration_proposal_v1. Synthetic flags do not prove physics success.
6. Rejection tests: adding an unreviewed raw directory or changing accepted raw
   bytes each caused conversion to fail BEFORE destination creation. File was
   removed only because it was a known temporary synthetic fault injected by this
   test. Log conversion_gate_tests.log. `prepare-training` on the synthetic
   converted root printed an official100k command only; its named NEVER_EXECUTED
   output directory does not exist. No model/optimizer/checkpoint created.

Initial failures were diagnosed rather than hidden: converter import initially
pulled SciPy from camera code, unavailable in the training lock. Removed that
cross-environment dependency using a checked NumPy quaternion helper; no install.
Standalone reader initially tried read-only ~/.cache: reran with HF_HOME and
HF_DATASETS_CACHE under ignored outputs/huggingface. First simulator command in
sandbox had no CUDA access; identical reset diagnostic with GPU permission passed.
Logs of these failures are retained. No changed physics was used to obtain a pass.

### Next action and research limitations

Review README and DEMONSTRATION_PROTOCOL. Approve/revise the protocol, then
separately authorize a12-attempt pilot. Exact final/train session authorization
and accepted pilot needed before scaling; never fabricate a true approval JSON.
No final dataset or training yet. The new recorder's complete action-labelled
physical run remains untested until that authorization. Varied full-task success,
real PNG throughput/storage, human pilot visibility and learned performance remain
unproven. A future separate-root offline policy-loss CLI is explicitly not yet
implemented; proposed primary uses fixed100k final checkpoint and independent
physical validation. Internal LeRobot eval_split is disabled in that proposal to
avoid mixing root statistics with separate validation data. No VLA chosen/downloaded.

## 2026-10-23 — Authorized physical pilot completed; stop for review

User provisionally accepted the protocol for testing and explicitly authorized
only all12 pilot attempts plus action replay/quality measurement. Final collection,
augmentation and training were excluded. The authorization record binds the pilot
manifest, camera and protocol hashes at outputs/protocol_v1_pilot_run/approval.json.
It must not authorize another batch. All12 attempts were run once, in order, with
actual PhysX and the unchanged existing expert; no replacements/retries.

Collection and all three replay tests ran on clean committed source877f9b1,
branch feat/il-camera-dataset-v2, GUI visible, cpu-threads8, BLAS/OMP1. Each terminal
explicitly activated leisaac-sim and set LEISAAC_ASSETS_ROOT. No installs/driver/OS,
physics, model, planner, expert, gains or camera parameter changes. Cameras remain
SHA256 f8f7e56842602fbe040fea43b1cf1a51f75f56b3ad28357aae04f717c8e9fcbc;
protocol JSON remains9ec7c3539c287e11bdc988eb52597a16001404d81ef43f8aad64fd0c6a4a3b3a.

Exact collection command after activation:

```bash
python scripts/demonstration_pipeline.py collect \
  --manifest outputs/demonstration_protocol_v1/pilot.json \
  --raw datasets/protocol_v1_pilot --output outputs/protocol_v1_pilot_run \
  --approval outputs/protocol_v1_pilot_run/approval.json \
  --execute --authorize-collection
```

The launcher emitted one existing run_manipulation.py process per case with
--stage full --motion-mode continuous --expert-hz20 --grip-feedback-from-lift,
--overview-profile centered, its single-case manifest, --record-dir,
--authorize-recording and --exit-after. It did not use --headless. Actual argv,
seeds, starts, return codes and process wall time are in attempts.jsonl. Each run
has an immutable full_*.json report plus simulation.json and a log. CPU plan
feasibility from the prior milestone was never counted as physical completion.

**Results:**12/12 independent pickup/lift/handover/placement successes;12/12
process exits0; zero forbidden-contact violations (0.05N threshold), no reported
drop/grasp/controller/evaluator failure and no failure-recovery reset. All cases
have actual reset position/quaternion and source/config hashes. Peak speed2.4415
rad/s, effort1.5271Nm and stable-grasp slip1.1648mm. The new nominal metrics AND
complete transfer trace exactly equal the preserved camera-regression reference
outputs/controller/overview_v3_comparison/full_20261023T142830953036Z.json.

Actual task durations: mean149.1472s, range138.0333..160.1000s excluding reset.
Process wall sum7905.2184s (2.1959h), mean658.7682s, range582.4607..697.9781s.
35,792 complete 20Hz rows /107,376 paired RGB frames. Six final partial holds
(10 physical ticks total) are diagnostic, excluded from complete rows;107,394
PNGs exist including18 terminal diagnostic images. Raw logical bytes11,222,342,915
(11.222GB), mean935,195,243 bytes per demonstration. Actual wall throughput4.5276
policy rows/s,5.4647 demos/hour, distinct from the simulated temporal20Hz rate.
Image write timers2596.22s, viewport rendering1861.84s, camera869.86s, physics457.76s,
planning including approach1663.53s. Reset overlaps stepping timers; do not sum all
categories. One read-only37.75s first-case audit overlapped the second collection
case; later batch QA overlapped replay only. Timings describe the observed workflow,
not an isolated hardware microbenchmark. Planning is not the only throughput cost.

Automated QA command (no human approval supplied):

```bash
python scripts/demonstration_pipeline.py validate \
  --manifest outputs/demonstration_protocol_v1/pilot.json \
  --raw datasets/protocol_v1_pilot --output outputs/protocol_v1_pilot_qa \
  --ledger outputs/protocol_v1_pilot_run/attempts.jsonl --review-video
```

12/12 automatic QA passes, zero automatic rejections, complete ledger,0 human
accepted, ready_for_conversion=false. It decoded all complete paired RGB rows,
checked finite float32 state/actions, authored limits, absolute/relative clock,
exact repeated3-tick targets, all3 sensor IDs, frozen K/pose, measured wrist-parent
attachment and required task phases. Human review remains pending for all12;
never change those flags from code tests or an AI image inspection. The sampled
projected-cube ROI colour diagnostic is not certified segmentation, contact or
continuous visibility. All96 sampled phases have >=12px cube-red proxy width in
at least one view; initial wrists see no cube, overview has partial occlusion.

Physical replay was preselected and logged BEFORE execution: pilot_000 corner,
pilot_008 nominal, pilot_009 minus1degree. Original raw actions were passed through
BenchmarkEnv.step (the policy20Hz/3-tick interface), same reset/actuators/gravity
compensation/independent evaluator; no expert IK/RRT/contact servo and no new expert
recordings. For each case the exact command is in
outputs/protocol_v1_pilot_replay/attempts.jsonl. Pattern:

```text
python scripts/run_manipulation.py --cpu-threads 8 --stage full \
  --motion-mode continuous --expert-hz 20 \
  --cases-file outputs/protocol_v1_pilot_run/cases/<case>.json \
  --replay <that-case-raw-directory> \
  --output-dir outputs/protocol_v1_pilot_replay/<case> --exit-after
```

3/3 actual physical successes, exits0. Replay sim durations157.2667/150.3500/135.2s;
process walls228.5/218.5/197.5s. The intended interface ends at first independent
success, omitting56 final recorded settling-tail rows in each selected case.
Do not claim every stored row or pointwise state equivalence was replayed. Aggregate
metrics differ only in elapsed time and desired-target tracking (float32 recorded
targets versus original higher-precision desired-pose calculation); slip, speed,
effort, stage outcomes and contacts match. Nine other recordings were not physically
replayed. Do not present format validation as physical replay success.

A reusable pilot-report subcommand was added only AFTER all physics tests finished.
It joins the actual ledger/reports/QA/replay evidence, rechecks full raw byte digests,
samples camera visibility and emits JSON/CSV plus a consolidated HTML review with
videos, sheets and full-resolution PNG links. It cannot collect, train or grant
human acceptance. See exact rebuild command in docs/PILOT_RESULTS.md. Outputs:

- outputs/protocol_v1_pilot_report/index.html — consolidated user review.
- outputs/protocol_v1_pilot_report/pilot_report.json, pilot_cases.csv — full per-case
  poses/seeds, stage/failure/reset evidence, durations/bytes, QA and replay results.
- outputs/protocol_v1_pilot_qa/quality.json, human_review_template.json,12 phase
  sheets and12 synchronized20Hz videos. Original PNGs remain in raw folders.
- docs/PILOT_RESULTS.md — curated measured results, storage table, limitations and
  next-size recommendation; README provides review/rebuild/visible replay commands.

Raw size projections:100=93.52GB,500=467.60GB,1000=935.20GB; these exclude converted
copies/weights/media. Measured GUI generation projections18.30/91.50/182.99hours.
Recommend retaining tested primary x[-.350,-.330],y[-.052,-.044]m,yaw+-6degrees,
fixed HOME, and considering100train/20validation only AFTER human acceptance and
separate authorization. Expected120raw~112.22GB,~21.96h generation before QA/human
review. Do not widen/narrow ranges, use the pilot for training, infer broad success
probability from boundary selection, or jump to500/1000 before a useful baseline.

All45 CPU tests passed in72.465s after report support was added. Report counts,
local links and raw totals were additionally checked against actual evidence.
One auxiliary NumPy inspection command in a fresh system shell lacked Conda;
reran its inspection using stdlib/activated Python, with no data/state changes.
No simulator or dataset test was repaired by installing/upgrading anything.

**STOP for user review.** No further collection, augmentation, official conversion
into an accepted training corpus, optimizer update, model creation or remote push.
The12 authorized expert attempts are consumed. Checkpoints remain empty. Raw pilot
is intentionally retained locally and ignored by Git. No user file/dataset was
deleted. Keep the approved camera file and provisional protocol JSON unchanged.

Final report verification: outputs/protocol_v1_pilot_report/report_verification.json
and source_preservation.json. All12 run evaluations match hash-bound raw episode
evaluations; all18 terminal diagnostic PNGs decoded; all35798 camera captures have
matching three-camera frame-counter values; all326 local HTML links exist; no
checkpoint files. All control/scene/camera/protocol source files are byte-identical
to877f9b1. Six focused protocol tests passed again in2.423s after final reporting
cleanup; python compile and git diff --check passed. Only reporting code and docs
changed after the physics runs. A new local milestone commit follows; no push.

## 2026-10-23 — Source and documentation publication for review

The user subsequently requested that all completed documentation and work be
pushed to the feature branch for review. This authorizes a normal, non-force push
of `feat/il-camera-dataset-v2`, including pilot milestone `6638a1f` and its preceding
camera/protocol/pipeline commits. It does not authorize final collection,
augmentation, training, merging, changes to `main`, or artifact uploads.

Preflight: clean working tree at `6638a1f`; terminal HTTPS authentication works.
`git ls-remote --heads origin main feat/il-camera-dataset-v2 feat/fingertip-geometry-calibration`
returned `main` at `c01651938f6220a23cb19ce3734670376e3eead5`, the prior feature branch
at `f47c358ed3a7a75215633f0ee558150143b98a14`, and no existing camera/dataset branch.
The new history contains no tracked datasets, outputs, checkpoints, logs or secret
files; the largest added blob is the 173,884-byte licensed wrist-camera mount STL.
README, pilot results and current gate text now distinguish publication permission
from the still-pending learning approvals and explain that local artifact links
will not work in a GitHub-only checkout.

Publication command: `git push -u origin feat/il-camera-dataset-v2`.
Verify its remote tip against `git rev-parse HEAD` using
`git ls-remote origin refs/heads/feat/il-camera-dataset-v2`; also recheck `main`
and `git status --short`. This publication update changes documentation only;
the physical runs and tests recorded above remain the measured validation.
