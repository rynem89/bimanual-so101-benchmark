# Project handoff — Simulation-first bimanual SO-101
Updated 2026-10-21. This is a snapshot from ChatGPT; **not automatically synchronized with Codex**. Verify all local files and current Git state before acting.

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
