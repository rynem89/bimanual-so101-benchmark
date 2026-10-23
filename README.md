# Bimanual SO-101: deterministic expert and camera-gated learning

The engineering objective is a reproducible benchmark: left SO-101 picks a 40 mm
cube, transfers it in mid-air to the right arm, and the right arm releases it into
a basket. The validated geometric expert is preserved. Learning is being reset;
there are currently **no demonstration datasets or model checkpoints**.

## Current boundaries

- Work branch: `feat/il-camera-dataset-v2`; preserved reference tag:
  `reference/pre-il-camera-v2` at `f47c358`. Main and remote branches are untouched.
- The historical expert completed the task, including the visually verified
  minus-one-degree recovery at `fb408f5`. Existing physical success criteria remain.
- Camera v2 passed automated attachment/timing checks and one full nominal
  physics regression; human approval of this initial simulation calibration is pending. Do not collect demonstrations or train before that approval.
- All old learning artifacts in `datasets/` and `checkpoints/` were deleted with
  user authorization. Exact inventory and sizes are in the engineering handoff.
  Output evidence and old policy traces were retained; deletion of traces is only
  proposed. Old commands/weights in POLICY_REVIEW.md are historical.

The previous four-demo ACT/Diffusion runs failed. Poor image visibility, limited
coverage and differences in the custom training recipe remain concerns; the
experiments do not isolate a single cause. Historical configuration/statistics
remain in [the review](docs/POLICY_REVIEW.md) and its metadata snapshot.

## Environments

In every new simulation shell:

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
export LEISAAC_ASSETS_ROOT=/home/rynem8/robotics/leisaac/assets
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

This preserves Python 3.11.16 / NumPy 1.26 / torch 2.7.0+cu128, Isaac Sim 5.1,
Isaac Lab 2.3 and LeIsaac 0.4. No installed packages or drivers are changed.
Conversion and future model serving use a separate existing environment:

```bash
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate so101-il-py312
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

LeRobot 0.6.1 requires Python ≥3.12 and NumPy ≥2; separation avoids changing the
simulation stack. Both ACT and Diffusion use this same training environment.
During future inference a local socket carries images/joints and targets between
processes. There is no trained model available to serve after the reset.

## Inspect the expert and actual policy cameras

Expert without learning or recordings, in the simulation environment:

```bash
python scripts/run_manipulation.py --cpu-threads 8 --stage full \
  --motion-mode continuous --expert-hz 20 --grip-feedback-from-lift
```

Camera inspection, also in the simulation environment (choose a fresh output
folder each time; the inspector refuses to overwrite an existing folder):

```bash
python scripts/run_manipulation.py --cpu-threads 8 --stage full \
  --motion-mode continuous --expert-hz 20 --grip-feedback-from-lift \
  --camera-preview outputs/camera_v2/manual_review \
  --output-dir outputs/controller/camera_v2_manual
```

GUI is the default. `--headless` is only an explicit optional test flag. The
ordinary viewport is not the policy observation. Open the side-by-side images in
the preview directory: overview, left wrist and right wrist are captured from
actual synchronized RGB tensors. `audit.json` records physical ticks, phase,
actual extrinsics and intrinsics. Snapshots cover initial, approach, closure,
lift, transfer, receiver closure, transport and release/settling phases. They
contain no action trajectory and are not a demonstration dataset.

Check whether the cube and finger contacts remain visible in the relevant wrist
view, including approach and receiver takeover. Inspect the basket framing during
right transport/release. Human camera acceptance is required before collection.
A successful GUI run stays at `[VIEWER]`; failed runs can close with exit 1.
Read the physical result JSON; image capture alone is not manipulation success.

## Camera architecture and calibration

Policy keys are `observation.images.overview`, `.left_wrist`, `.right_wrist`.
Each RGB stream is 640 × 480 at the 20 Hz observation boundary. Initial pinhole
parameters: focal length 12 mm, horizontal aperture 20.955 mm, square pixels,
approximately 82.25° horizontal FOV, fx=fy≈366.50 pixels, centre=(320,240),
clipping 0.01–10 m. These are a plausible downsampled UVC-camera projection,
not measured specifications of the eventual physical lens.

The overview eye is (0, −0.10, 0.72) m, looking at (0, −0.10, 0.02) m.
Wrist sensors are children of `Left_Robot/gripper` and `Right_Robot/gripper`.
The selected upstream hex-nut mount attaches to the wrist-roll housing, whose
mesh belongs to this rigid gripper link. It does not follow the moving jaw.
Local sensor position is (0.0141, −0.045, 0.005) m; optical +Z looks toward
(0.0141, 0, −0.09) m in the same link. ROS axes are +X right, +Y down, +Z forward;
The ROS camera-to-parent quaternion (w,x,y,z) is
`(0, 0, 0.97563770, 0.21938797)`; optical +X is negative parent X.
`policy_cameras.py` computes the orthonormal rotation and quaternion.
This offset is an initial simulation estimate and needs physical measurement
before sim-to-real. Attachment checks compare sensor poses with measured link
poses; the robot USD, mass and collision shapes are unchanged.

The optional [mount STL](assets/camera_mount/README.md) is provided with its
license for mechanical review, not spawned collision geometry. Its registration
to the robot and cable clearance are not validated. Add `--diagnostic-front` to camera preview to save the former fixed front view
as separate diagnostic PNGs. It is excluded from policy features. Presentation
also uses the independent viewer camera. No wrist hardware or lens has been physically calibrated here.

Camera transform compatibility: in the pinned stack, the default USD camera pose
reader did not track the PhysX parent accurately. A project-owned camera subclass
reads live Fabric poses, while capture synchronizes only the camera Fabric world
transform from the measured link and constant local offset before rendering.
No robot/object state is written by this adapter. Actual sensor poses are then
checked against the measured link at 20 micrometres / rotation-matrix 2e-4 tolerance.


## Observation/action contract and future data procedure

The state is 24 float32 values: left q[6], right q[6], left qd[6], right qd[6].
Joint order per arm: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper. Positions/targets are radians; velocities are rad/s.
The 12 actions are absolute position targets, not Cartesian poses or deltas.
Each target is held for three 60 Hz physics ticks. The recorder pairs pre-action
RGB/state with that exact held command. Cube poses/contact diagnostics remain
separate from network inputs. Independent physical evaluation must verify every
successful demonstration; all failed attempts and reasons remain counted.

After camera approval, the user chooses pilot poses and quantity. Supply a small
manifest with schema `so101-cases-1.0`, `split: "train"`, and a `cases` list of
`{name, cube_xy_m, yaw_deg, seed}`. Existing `configs/task_distribution.json`
contains the historical inward distribution; it is not silently modified or
endorsed as the new benchmark distribution. Separate train, validation and
held-out manifests and artifact roots are required. Do not tune against the final
held-out set. Human teleoperation is a future input adapter, not implemented.

A one-case example is [configs/camera_v2_pilot.json](configs/camera_v2_pilot.json),
explicitly marked unapproved. It is configuration only and has not been executed.

The future expert recording interface is the expert command above plus
`--cases-file <approved-manifest> --record-dir datasets/<new-raw-root>` and
`--authorize-recording`. This explicit flag is only for a user-authorized session
following camera approval; do not use it now. Review actual images and task
outcomes before conversion. Conversion in the Python 3.12 environment uses:

```text
python scripts/policy_pipeline.py convert --raw <approved-raw-root> --output <new-converted-root>
```

Conversion validates chronology, held targets, calibrated cameras and image
presence; rejects mixed split labels; keeps failed-generation counts in its
manifest; and creates official LeRobot v3 image/parquet data. Statistics are
computed anew. Automated nonblank checks do not replace visual data acceptance.

## Official training preparation, after data acceptance

The former custom optimizer loop has been removed. `prepare-training` only
validates metadata and prints an official command; it never executes updates:

```text
python scripts/policy_pipeline.py prepare-training --method act \
  --dataset <approved-converted-train-root> --output <new-run-root> \
  --steps <user-choice> --batch-size <user-choice> \
  --validation-fraction <user-choice> --eval-steps <user-choice>
```

Use `diffusion` for the other policy. Budget and hyperparameters remain user
choices. The emitted command uses installed `lerobot.scripts.lerobot_train`, its
policy presets, official optimizer/scheduler and episode-aware sampling. Uploads
and environment evaluation are disabled; this custom simulator is not yet an
EnvHub/Gym environment. Official offline validation loss uses an episode split;
it is not physical success evaluation. Do not confuse that split with a separately
collected validation distribution. Inference still loads official processors and
`select_action` through the local socket adapter, with camera dimensions taken
from the saved policy configuration.

Verified so far: official CLI help and a disposable 48-frame format fixture with
three cameras loaded through ACT/Diffusion sequence readers. No model was created,
no optimizer was run, and no checkpoint was saved. A real approved-dataset training
run and full official-checkpoint inference remain unverified until later gates.
ACT versus Diffusion comparison must use agreed data/budgets, independent physical
evaluation, full attempted-case denominators and explicit observation privileges.

## Evidence ready for camera review

Final CPU regression: **37/37 passed** in 69.868 seconds, including optical-frame,
wrong-parent/rotation, RGB and timestamp checks. Exact commands and logs are in
the [handoff](docs/ROBOTICS_CODEX_HANDOFF.md).

The complete nominal camera-enabled run passed all independent stages in
153.1833 simulated seconds, with zero forbidden collisions, peak speed
2.0110 rad/s and peak effort 1.3496 Nm. Its report is
`outputs/controller/camera_v2_synchronized/full_20261023T085954936832Z.json`.
The physical regression used the default three-camera configuration; the optional
front diagnostic was separately checked in a reset-only run. Full history and
source hashes remain in those local reports. This is one nominal regression,
not validation of a varied collection distribution or real hardware.

Open the existing review files without rerunning simulation:

- [Eight-phase contact sheet](outputs/camera_v2/synchronized/review_contact_sheet.jpg)
- [All 51 phase snapshots and audit.json](outputs/camera_v2/synchronized/)

The left/right wrist view follows its arm and shows the held cube at a larger
scale. The overview still has hand occlusion and the transfer region approaches
its lower edge. Camera acceptance must assess these limitations. An automatic
occlusion/visibility score has not been implemented. The next gate is the user's
visual decision; approval does not authorize training or collection by itself.

Future inference after a separately approved official training run uses two
activated shells. In the training shell, choose the matching method and official
saved checkpoint directory:

```text
python scripts/policy_pipeline.py serve --method <act-or-diffusion> \
  --checkpoint <run-root>/checkpoints/last/pretrained_model \
  --socket /tmp/so101_user_policy.sock
```

After `[READY]`, in the simulation shell:

```text
python scripts/run_manipulation.py --cpu-threads 8 --stage full \
  --policy-socket /tmp/so101_user_policy.sock --case nominal
```

These are future interfaces, not runnable inference for the deleted old weights.

## Later roadmap

Mimic/SkillGen are not integrated. The existing IK expert remains primary. Mimic
would require a task adapter with object-relative cube/bin frames, coordinated
left/right subtask constraints and pose/action conversion for the five-axis arms.
SkillGen additionally needs a compatible cuRobo robot/world model and carried-cube
collision handling. Replicator changes lighting/appearance; it does not generate
valid grasp trajectories. Details and pinned sources are in the handoff.

SmolVLA comes after accepted cameras, demonstrations and ACT/Diffusion baselines.
Sim-to-real additionally requires measured camera intrinsics/extrinsics, wrist
mount/cable clearance, actuator calibration and hardware safety validation.

## What each file does

Start with `policy_pipeline.py` for learning, `dataset_io.py` for data semantics,
and `benchmark_env.py` for the images and commands seen by the policy. This table
covers every maintained Python source file; imported modules have no standalone
training command.

| File | Responsibility and inputs → outputs | How it is used |
| --- | --- | --- |
| [scripts/policy_cameras.py](scripts/policy_cameras.py) | Project-owned camera configuration, optical transforms and sparse phase contact sheets | Used by `--camera-preview`; no training records |
| [tests/test_policy_cameras.py](tests/test_policy_cameras.py) | Optical axes, missing cameras, RGB shapes and 20 Hz clock guards | CPU unittest |
| [configs/camera_v2_pilot.json](configs/camera_v2_pilot.json) | One nominal, explicitly unapproved pilot manifest | Edit/approve before user-authorized collection |
| [assets/camera_mount/README.md](assets/camera_mount/README.md) | Upstream optional mesh provenance, units, license and limitations | Reference only; not spawned |
| [scripts/policy_pipeline.py](scripts/policy_pipeline.py) | Raw folders → LeRobot dataset; dataset metadata → official training command; RGB/joints → predicted targets | Training environment: `convert`, `prepare-training`, `serve` commands above |
| [scripts/dataset_io.py](scripts/dataset_io.py) | Writes and validates 20 Hz raw episodes, joint order, pre-action timing and held targets; keeps privileged diagnostics separate | Imported by recording, conversion and replay |
| [scripts/run_manipulation.py](scripts/run_manipulation.py) | Main CLI; chooses expert, recorded-target replay or learned control; orchestrates phases and writes reports | Simulation environment: `--stage full`; `--record-dir`, `--policy-socket` or `--replay` select modes |
| [scripts/benchmark_env.py](scripts/benchmark_env.py) | Attached wrist cameras and overview, measured observations, reset, PhysX stepping, joint targets and gravity compensation | Imported by the main simulator runner; camera changes start here |
| [scripts/benchmark_core.py](scripts/benchmark_core.py) | Task specification, independent evaluator, fixed diagnostic cases, trajectory metrics | Imported by runner/environment; pure CPU |
| [scripts/inspect_task_layout.py](scripts/inspect_task_layout.py) | Asset paths, Isaac startup, robot/table/cube/bin configuration and initial poses | Shared scene module; standalone `python scripts/inspect_task_layout.py` is a GUI diagnostic, not training |
| [scripts/so101_model.py](scripts/so101_model.py) | Robot USD → transforms, calibrated TCP, meshes, FK and constrained IK | Imported by scene/control; standalone offline model audit requires bundled USD libraries |
| [scripts/so101_planning.py](scripts/so101_planning.py) | Geometry + measured poses → checked expert paths, RRT fallback and plan cache | Expert motion planning; `--profile` is an offline diagnostic |
| [scripts/audit_so101_finger_geometry_v2.py](scripts/audit_so101_finger_geometry_v2.py) | Finger mesh sections and jaw opening → calibration CSV/JSON | Maintained diagnostic: `--offline` for geometry; otherwise GUI |
| [scripts/verify_motion_gripper_reset.py](scripts/verify_motion_gripper_reset.py) | Robot joint/gripper/reset behavior → simulator regression evidence | Standalone GUI diagnostic, not a policy trainer |
| [scripts/inspect_bimanual_scene.py](scripts/inspect_bimanual_scene.py) | Initial two-arm scene → visual inspection | Preserved historical GUI diagnostic |
| [scripts/inspect_fk_frames.py](scripts/inspect_fk_frames.py) | Robot frame poses → FK inspection | Preserved historical GUI diagnostic |
| [tests/test_dataset.py](tests/test_dataset.py) | Raw files → synchronization, image, action-bound and terminal-transition assertions | CPU `unittest`; does not assess visual usefulness |
| [tests/test_benchmark.py](tests/test_benchmark.py) | Synthetic measured frames → success/failure/reset/evaluation assertions | CPU `unittest`; not learned-policy validation |
| [tests/test_so101_model.py](tests/test_so101_model.py) | USD geometry and fixtures → FK/IK/collision/path/cache assertions | CPU `unittest` with bundled USD libraries |
| [tests/test_finger_geometry.py](tests/test_finger_geometry.py) | Analytic meshes → section and prerequisite assertions | CPU `unittest` |
| [configs/task_distribution.json](configs/task_distribution.json) | Named cube poses/seeds for 32 train / 8 validation / 30 held-out cases | `--cases-file`; contains poses, not recordings or model hyperparameters |
| [configs/il_requirements.lock.txt](configs/il_requirements.lock.txt) | Exact installed training packages | Review/recreation of isolated training environment only |
| [docs/POLICY_REVIEW.md](docs/POLICY_REVIEW.md) | Detailed decisions, data provenance, policy mechanics and unresolved issues | Independent technical review |
| [docs/POLICY_REVIEW_METADATA.json](docs/POLICY_REVIEW_METADATA.json) | Saved configurations, statistics, arguments and hashes | Machine-readable review evidence; not executable configuration |
| [docs/ROBOTICS_CODEX_HANDOFF.md](docs/ROBOTICS_CODEX_HANDOFF.md) | Historical commands, results, physical evidence and environment locks | Experiment history and agent continuity |
| [AGENTS.md](AGENTS.md) | Repository operating rules | Agent instructions |
| `.gitignore`, `.vscode/settings.json` | Artifact/secret exclusions; existing editor environment selection | Repository configuration |
