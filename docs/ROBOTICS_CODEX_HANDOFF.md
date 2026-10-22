# Project handoff — Simulation-first bimanual SO-101
Updated 2026-10-22. Verify local files and current Git state before acting.

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
2,175 optimized/reference LP comparisons across 4,215 approach samples (the
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
