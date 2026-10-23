# Bimanual SO-101 benchmark

Two SO-101 arms perform one task: **left picks a 40 mm cube → transfers it in
mid-air → right places it in the bin → both retreat**. The deterministic IK expert
has completed this task under measured PhysX contact and independent evaluation.
There is currently **no accepted learning dataset and no trained model** after the
user-authorized reset of the earlier failed experiments.

## Current deliverable and next decision

The user approved the centered overhead camera and both wrist cameras on
2026-10-23. They are now frozen in [configs/policy_cameras.json](configs/policy_cameras.json),
and `centered` is the default. No further camera optimization is planned.

The **12-attempt pilot is complete**: **12/12 independent physical task successes,
12/12 automated recording-QA passes, zero automatic rejections**. All 12 are
pending your human visual review. Three preselected recordings (corner, nominal,
−1° yaw) also completed the task through physical action replay; the other nine
were not physically replayed. No policy was trained.

Start with [the consolidated visual review](outputs/protocol_v1_pilot_report/index.html),
then [the measured results and recommendation](docs/PILOT_RESULTS.md).
The [machine-readable report](outputs/protocol_v1_pilot_report/pilot_report.json)
and [case CSV](outputs/protocol_v1_pilot_report/pilot_cases.csv) contain all seeds,
poses, stage outcomes, failures/reset evidence, timing, sizes, QA and replay results.
Raw recordings are in `datasets/protocol_v1_pilot/`.

The protocol is **provisionally accepted for this pilot, not frozen for final
collection**. All 12 authorized attempts have been consumed. Final collection,
augmentation and training remain paused for your review. Proposed next scale is
100 train /20 validation attempts in the same tested ranges, only after separate
authorization. Research and procedure remain in
[the protocol](docs/DEMONSTRATION_PROTOCOL.md), with machine-readable settings in
[configs/demonstration_protocol.json](configs/demonstration_protocol.json).

Use the [GitHub review branch `feat/il-camera-dataset-v2`](https://github.com/rynem89/bimanual-so101-benchmark/tree/feat/il-camera-dataset-v2)
for source code, configurations, tests and documentation. The user separately
authorized publishing this feature branch for review on 2026-10-23; `main` remains
unchanged. Raw recordings, generated reports/videos and checkpoints stay local
and are excluded from Git. Links into `outputs/` therefore require this local
workspace; GitHub reviewers should start with [the tracked pilot results](docs/PILOT_RESULTS.md)
and [the demonstration protocol](docs/DEMONSTRATION_PROTOCOL.md).
Preserved expert reference:
`reference/pre-il-camera-v2` (`f47c358`); original reviewed camera baseline:
`reference/camera-v2-reviewed` (`adf03b4`).

## Activate the correct environment

Every simulation terminal needs its own activation:

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
export LEISAAC_ASSETS_ROOT=/home/rynem8/robotics/leisaac/assets
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

Pinned stack: Python 3.11.16, NumPy 1.26, torch 2.7.0+cu128, Isaac Sim 5.1,
Isaac Lab 2.3, LeIsaac 0.4. RTX 5090 / 32 GB; driver 580.178.04. No installs/upgrades.

Conversion, later official training and model serving use the **same separate**
existing environment for both ACT and Diffusion:

```bash
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate so101-il-py312
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

This has Python 3.12.15, LeRobot 0.6.1, NumPy 2.2.6, torch 2.7.1+cu128; exact packages
are in [the training lock](configs/il_requirements.lock.txt). LeRobot's Python/NumPy
requirements conflict with the simulation lock. Two environments preserve both;
they are not two separately invented implementations of the policies. Later
inference uses a local Unix socket between them. Activation never persists into
a different terminal invocation.

## Run the proven expert in the visible simulator

In `leisaac-sim`:

```bash
python scripts/run_manipulation.py --cpu-threads 8 --stage full \
  --motion-mode continuous --expert-hz 20 --grip-feedback-from-lift
```

GUI is the default. It remains open at `[VIEWER]` after success. Read the saved
independent evaluation under `outputs/controller/` as well as observing motion.
To inspect the actual three policy image tensors, use a fresh diagnostic folder:

```bash
python scripts/run_manipulation.py --cpu-threads 8 --stage full \
  --motion-mode continuous --expert-hz 20 --grip-feedback-from-lift \
  --camera-preview outputs/approved_cameras_manual \
  --output-dir outputs/controller/approved_cameras_manual
```

The viewport is a presentation camera, not a policy observation. These preview
snapshots are diagnostic outputs, not training demonstrations. The historical
comparison is [the approved review sheet](outputs/camera_overview_v3/comparison/recommended_review.jpg).
The old `baseline` and angled profiles remain diagnostic alternatives only;
recording QA requires the approved centered profile.

## What the controller and policies do

The expert uses bounded SciPy TRF least-squares IK on **five arm joints** to find
reachable TCP positions/orientations. `so101_planning.py` checks intermediate robot,
scene and carried-cube geometry, with existing RRT recovery where needed.
Continuous motion interpolates joint waypoints with a clamped cubic spline; the quintic blend
`10u³−15u⁴+6u⁵` provides smooth endpoint timing, and geometric/path checks remain
necessary between endpoints. Grasp/handover feedback uses measured finger forces
and actual cube motion, not an attachment or task-time teleport. Details and
source responsibilities are below; historical numerical evidence is in the handoff.

State is 24 float32 values: left q6, right q6, left qd6, right qd6. Joint order is
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper. Units are
radians and rad/s. Action is 12 **absolute joint-angle targets**, held for three
60 Hz physics ticks at each 20 Hz boundary. The gripper value is an angle, not aperture.
RGB/state are captured before applying the paired action. Cube pose, contacts,
controller phase and future expert paths are excluded from policy inputs.

ACT predicts a chunk of future joint targets with its official transformer/CVAE;
Diffusion predicts targets through conditional denoising. Learned inference does
not call the expert's IK/RRT/contact servo. Both use the shared physical actuator
interface and evaluator. Neither has demonstrated success in the reset pipeline.
The [protocol](docs/DEMONSTRATION_PROTOCOL.md#fair-training-and-evaluation-proposal--not-authorized)
explains the proposed matched observation/chunk/execution settings, native-default
ablation, seeds, training budgets and future VLA comparison without claiming one
recipe is universally best.

## Approved camera calibration

Authoritative file: [configs/policy_cameras.json](configs/policy_cameras.json),
version `so101-cameras-3.0`. File SHA-256:
`f8f7e56842602fbe040fea43b1cf1a51f75f56b3ad28357aae04f717c8e9fcbc`.
This hash binds the approved configuration, not the differently structured
historical proposal JSON.

| Parameter | Frozen value |
| --- | --- |
| Keys | `overview`, `left_wrist`, `right_wrist` |
| RGB /resolution /rate | uint8 RGB, 640×480, 20 Hz |
| Overview world eye → target, m | (0,−0.20,0.78) → (0,−0.20,0.02) |
| Overview ROS quaternion wxyz | (0,1,0,0), optical axis vertically down |
| Wrist parents | `Left_Robot/gripper`, `Right_Robot/gripper`, rigid wrist-roll housing |
| Wrist position → target in parent, m | (0.0141,−0.045,0.005) → (0.0141,0,−0.09) |
| Both wrist ROS quaternions wxyz | (0,0,0.9756376988650908,0.21938796811409378) |
| Focal /horizontal aperture | 12mm / 20.955mm, square pixels |
| K, pixels | [[366.4996338,0,320],[0,366.4996338,240],[0,0,1]] |
| Clipping /horizontal FOV | 0.01–10m /approximately 82.25° |

ROS optical axes are +X right,+Y down,+Z forward; wrist optical +X is negative
parent X. These are approved simulation estimates; physical lenses, gantry and
mount offsets still need measurement. The optional licensed mount mesh is for
mechanical review, not spawned mass/collision geometry. No robot USD was changed.

The pinned stack needs a project-owned camera pose adapter that reads live Fabric
transforms and synchronizes **camera transforms only** from measured PhysX gripper
poses. It writes no robot/cube state. Every recorded boundary saves measured
parent/sensor transforms; QA independently checks translation within 20 µm and
rotation-matrix elements within 2e-4. Camera changes require a new reviewed version.

## Plan, collect, review and convert demonstrations

The pilot used the existing expert, one process per declared case, with the GUI
visible. Full attempt commands, authorization, start/finish ledger and logs are
under `outputs/protocol_v1_pilot_run/`; QA, review templates, videos and phase
sheets are under `outputs/protocol_v1_pilot_qa/`. No failed case was regenerated.
The complete [generation and QA procedure](docs/DEMONSTRATION_PROTOCOL.md#executable-workflow)
remains available for a later authorized session; existing pilot paths are occupied.

To rebuild the consolidated report without collecting or training, activate
`leisaac-sim` and choose a fresh output folder:

```bash
python scripts/demonstration_pipeline.py pilot-report \
  --manifest outputs/demonstration_protocol_v1/pilot.json \
  --run outputs/protocol_v1_pilot_run \
  --quality-report outputs/protocol_v1_pilot_qa/quality.json \
  --replay-run outputs/protocol_v1_pilot_replay \
  --output outputs/protocol_v1_pilot_report_rebuilt
```

To see the recorded nominal actions run through real simulation, in
`leisaac-sim` use:

```bash
python scripts/run_manipulation.py --cpu-threads 8 --stage full \
  --motion-mode continuous --expert-hz 20 \
  --cases-file outputs/protocol_v1_pilot_run/cases/pilot_008.json \
  --replay datasets/protocol_v1_pilot/pilot_008_20261023T171324135324Z \
  --output-dir outputs/controller/manual_pilot_nominal_replay
```

The GUI remains open at `[VIEWER]` after success. This is recorded-action replay,
not learned-model inference. It submits joint targets through the same policy
interface and stops at independent success; it may omit the recording's final
settling tail. The pilot's three physical replay tests all passed, while pointwise
state-trajectory equivalence was not measured. Full results and limitations are in
[the pilot report](docs/PILOT_RESULTS.md#recording-format-versus-physical-replay).

After separately authorized collection and human data acceptance, convert in
`so101-il-py312`:

```text
python scripts/policy_pipeline.py convert --raw <approved-raw-partition> \
  --output <new-lerobot-root> --quality-report <accepted-qa>/quality.json
```

Train and validation roots stay separate. Conversion writes/reopens official
LeRobot v3 image/Parquet data with only the allowed RGB/state/action features.
It checks that the raw bytes still match human acceptance. It runs no training.

## Future training and inference procedure

There are no runnable trained weights today. The previous four-demo ACT/Diffusion
experiments failed and their datasets/weights were deleted with permission.
Poor visual coverage, insufficient diversity and the former custom training
recipe were confounded; no experiment isolated a single cause. See
[the historical review](docs/POLICY_REVIEW.md) for what was actually used.

After data approval and separate training authorization, `prepare-training`
prints an official command without executing it:

```text
python scripts/policy_pipeline.py prepare-training --method act \
  --dataset <approved-train-lerobot> --output <new-act-run> \
  --steps 100000 --batch-size 8 --validation-fraction 0 --eval-steps 0
```

Use `diffusion` for the other method. Before executing the printed command, apply
and review the proposed common/method-specific flags in
[the training protocol](docs/DEMONSTRATION_PROTOCOL.md#fair-training-and-evaluation-proposal--not-authorized).
Use official processors, optimizer, scheduler, episode-aware sampler and saved
configurations. No custom optimizer remains. Keep separate validation data out of
normalization statistics; offline loss is not physical success.

After a real approved official training run, use two activated terminals. In
`so101-il-py312`, select the actual official checkpoint directory:

```text
python scripts/policy_pipeline.py serve --method <act-or-diffusion> \
  --checkpoint <run>/checkpoints/last/pretrained_model \
  --socket /tmp/so101_user_policy.sock
```

After `[READY]`, in `leisaac-sim`:

```text
python scripts/run_manipulation.py --cpu-threads 8 --stage full \
  --policy-socket /tmp/so101_user_policy.sock --case nominal
```

Inference uses the saved policy and processors, three actual images and measured
joints. No model-download or inference-success claim is implied by these future
commands. Full real-checkpoint inference remains a later verification gate.

## Evidence and limitations

Previous nominal camera regression: full independent task success in 153.1833 s
simulated, zero forbidden contacts, peak 2.0110 rad/s and 1.3496 Nm. Complete transfer
trace matched the original reviewed baseline. Diagnostic video contained 3,064
synchronized frames with continuous sensor IDs and correct wrist attachments.
Report: `outputs/controller/overview_v3_comparison/full_20261023T142830953036Z.json`.
These are existing physical results, not new varied-demonstration evidence.

Current measured evidence is in [the pilot results](docs/PILOT_RESULTS.md).
All 12 actual physics episodes and all three selected physical replays succeeded;
all 45 CPU tests passed in 72.465 s. The new nominal episode exactly matches the
preserved reference's independent metrics and complete transfer trace. Expert
motion, planner, model, scene and shared physics files were not changed.

The 12 raw demonstrations contain 35,792 complete 20 Hz rows /107,376 paired RGB
frames and occupy 11.22 GB. Mean task duration is 149.15 s simulated; mean process
wall time is 658.77 s, with 4.53 recorded rows/s wall throughput. All complete rows
passed camera/state/action synchronization and attachment checks. Final partial
physics holds remain diagnostic and are excluded from complete rows. Original
lossless PNGs are preserved; review videos are separate derivatives.

Human image/motion acceptance is pending. Sampled cube-colour ROI checks are
visibility proxies, not certified segmentation or contact evidence. Cameras and
protocol settings stay unchanged; broader-distribution reliability, accepted final
training data and learned-policy success remain unproven. `checkpoints/` is empty.
No final collection, augmentation or training was performed. Publishing the
source and documentation for review does not authorize those next steps.

## What each file does

Start with `demonstration_pipeline.py` and the protocol for dataset planning/QA,
`policy_pipeline.py` for official learning interfaces, `dataset_io.py` for data semantics,
and `benchmark_env.py` for the images and commands seen by the policy. This table
covers every maintained Python source file; imported modules have no standalone
training command.

| File | Responsibility and inputs → outputs | How it is used |
| --- | --- | --- |
| [scripts/demonstration_pipeline.py](scripts/demonstration_pipeline.py) | Protocol → fixed manifests; guarded expert launches → ledger; raw files → QA/review; measured evidence → JSON/CSV/HTML pilot report and visibility samples | Simulation env: `plan`, `preflight`, `collect`, `validate`, `pilot-report`; no more collection authorized |
| [configs/demonstration_protocol.json](configs/demonstration_protocol.json) | Versioned counts, pose support, seeds, QA limits and proposed training conditions | Proposed; reviewed before collection; no optimizer execution |
| [configs/policy_cameras.json](configs/policy_cameras.json) | Approved camera extrinsics, optics, resolution, FPS and provenance | Frozen source for default cameras; changes require review |
| [tests/test_demonstration_pipeline.py](tests/test_demonstration_pipeline.py) | Synthetic on-disk episodes → approval, corruption, attachment, clock, review-hash and denominator assertions | CPU unittest; synthetic flags are not physical evidence |
| [docs/PILOT_RESULTS.md](docs/PILOT_RESULTS.md) | Actual 12-attempt physical/QA/replay results, timings, storage projections and next-size recommendation | User review; no final collection/training approval |
| [docs/DEMONSTRATION_PROTOCOL.md](docs/DEMONSTRATION_PROTOCOL.md) | Research and explicit dataset/training/evaluation proposal, commands, estimates and unresolved gates | Main document for protocol approval |
| [scripts/policy_cameras.py](scripts/policy_cameras.py) | Camera profiles, optical transforms, continuity checks, comparison sheets and diagnostic video | Used by `--camera-preview`; no training records |
| [tests/test_policy_cameras.py](tests/test_policy_cameras.py) | Optical axes, attachment, RGB, reset-relative clock and continuous-frame rejection tests | CPU unittest |
| [configs/camera_v2_pilot.json](configs/camera_v2_pilot.json) | Historical one-case camera pilot placeholder | Retained for provenance; new protocol planner supersedes it |
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
