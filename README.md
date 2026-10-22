# Bimanual SO-101: user guide

This repository contains a scripted controller and an imitation-learning pipeline
for one task: **left arm picks up a 40 mm cube → transfers it to the right arm →
right arm releases it into the basket → both arms return home**.

The scripted controller has completed this task in Isaac Sim. The ACT and
Diffusion checkpoints have **not** demonstrated successful manipulation. They are
small diagnostic training runs, not finished policies. Training, camera changes
and visual evaluation are now left to the user. This documentation update runs
none of them.

## Start here

- [What was implemented and what trained the models](#what-was-implemented-and-what-trained-the-models)
- [Why two Python environments are used](#why-two-python-environments-are-used)
- [Cameras and the data-quality problem](#cameras-and-the-data-quality-problem)
- [Collect, inspect and convert demonstrations](#collect-inspect-and-convert-demonstrations)
- [Train ACT or Diffusion yourself](#train-act-or-diffusion-yourself)
- [View the trained models in Isaac Sim](#view-the-trained-models-in-isaac-sim)
- [What each file does](#what-each-file-does)
- [Detailed policy design and independent review](docs/POLICY_REVIEW.md)

The detailed review explains the choices, limitations, actual optimizer behavior
and exact demonstration provenance. [Review metadata](docs/POLICY_REVIEW_METADATA.json)
contains the saved configurations, dataset statistics, file hashes and training
arguments for inspection on GitHub. It is a documentation snapshot, not an input
to the trainer. Historical experiments remain in the
[engineering handoff](docs/ROBOTICS_CODEX_HANDOFF.md) and the
[previous README](https://github.com/rynem89/bimanual-so101-benchmark/blob/d1cb3befa15533cf177c7178814b8e000cd64162/README.md).

## What was implemented and what trained the models

There are three distinct operations:

1. **Expert control:** geometry, inverse kinematics, collision checks and contact
   feedback produce joint targets. This is ordinary programmed control.
2. **Training:** a neural network learns to predict those targets from recorded
   images and joint states. This is supervised imitation learning; no rewards,
   reinforcement learning or human teleoperation were used.
3. **Inference:** the trained network produces joint targets in the simulator.
   It does not call the expert planner or its gripper-contact feedback.

```text
Scripted expert → Isaac Sim → RGB + measured joints + commanded joint targets
                                       ↓
                             raw demonstration folders
                                       ↓ convert
                              LeRobot image dataset
                                       ↓ train
                             ACT / Diffusion checkpoint
                                       ↓ serve over local socket
                          Isaac Sim + independent evaluator
```

I implemented the scene integration, calibrated geometry/TCP, constrained IK,
collision-aware task planning, reset/evaluation, synchronized recorder and the
LeRobot conversion/training/inference adapter. ACT's transformer and Diffusion's
U-Net are imported from **LeRobot 0.6.1**, not implemented in this repository.
The training loop around them is custom; it is not the official LeRobot trainer.

**Both latest checkpoints trained on exactly these four successful scripted
expert episodes**, recorded at 20 Hz using continuous expert motion and contact
feedback from lift onward:

| Demonstration | Initial cube X, Y (world metres) | Yaw (degrees) | Seed | Frames |
| --- | --- | ---: | ---: | ---: |
| `train_v2_000` | −0.3481787, −0.0442980 | 0.32610 | 811000 | 3,042 |
| `train_v2_001` | −0.3414638, −0.0443834 | 4.56665 | 811001 | 2,920 |
| `train_v2_002` | −0.3492046, −0.0453542 | −4.81371 | 811002 | 3,073 |
| `train_v2_003` | −0.3433775, −0.0481709 | 0.96305 | 811003 | 2,951 |

Total: **11,986 frames / 599.3 seconds**, converted into
`datasets/lerobot_corrected_train4`. These are four long trajectories, not 11,986
independent demonstrations. Exact timestamped raw directories and historical
source versions are in [the review](docs/POLICY_REVIEW.md#exact-demonstration-provenance).

There are now twelve successful raw training recordings and two validation
recordings locally. The extra eight training recordings and both validation
recordings were **not used** in the latest weights or their normalization.
Earlier one-episode/20-update and two-episode/1,000-update runs are separately
identified in the review. The twelve-episode collection is not a trained model.

| Existing model | Directory to pass to `serve --checkpoint` | Training |
| --- | --- | --- |
| ACT | `checkpoints/act_corrected_train4_4000/pretrained_model` | Four episodes; 4,000 updates |
| Diffusion | `checkpoints/diffusion_corrected_train4_4000/pretrained_model` | Same four episodes; 4,000 updates |

Both used ImageNet-pretrained ResNet18 visual encoders; their manipulation
policies were trained locally. Latest completed evaluations were ACT **0/4**
(timeouts) and Diffusion **0/5** (actuator/contact failures), with one further ACT
attempt interrupted. Those outcomes do not isolate the cause of failure.

## Why two Python environments are used

These are **two software environments, not two Python models**. Both ACT and
Diffusion run in the same training environment. Only inference needs both
processes running together; offline training does not require a running simulator.

| Environment | Python / key packages | Used for |
| --- | --- | --- |
| `leisaac-sim` | Python 3.11.16; NumPy 1.26.0; torch 2.7.0+cu128; Isaac Sim 5.1 / Isaac Lab 2.3 / LeIsaac 0.4 | Expert playback, recording, physics, cameras and policy evaluation |
| `so101-il-py312` | Python 3.12.15; NumPy 2.2.6; torch 2.7.1+cu128; LeRobot 0.6.1 | Dataset conversion, neural-network training and model server |

Installed LeRobot 0.6.1 declares **Python ≥3.12 and NumPy ≥2.0**. Installing it
into the pinned simulator environment would conflict with that stack. A separate
environment was used to preserve the existing simulator. During inference, a
local Unix socket carries RGB bytes, joint-state lists and predicted targets
between the processes; neither environment imports the other one's packages.

Use these activation blocks in **every new terminal**. They refer to the existing
local installation; no installation or upgrade is needed here. A GitHub clone
alone does not include Conda environments, robot assets, recordings or weights.
The training dependency lock is [configs/il_requirements.lock.txt](configs/il_requirements.lock.txt).
External simulator paths and versions are in the handoff.

**Simulation terminal:**

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
export LEISAAC_ASSETS_ROOT=/home/rynem8/robotics/leisaac/assets
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

**Training / model-server terminal:**

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate so101-il-py312
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

## Cameras and the data-quality problem

The dataset uses **two fixed world cameras**, `overview` and `front`, each
224 × 224 RGB pixels. They are created in
[scripts/benchmark_env.py](scripts/benchmark_env.py), under the scene root. Neither
is attached to a wrist. The normal simulator viewport is a separate camera and
is **not** what the policy sees.

The design intent was to give the network a stationary view of both arms, the
transfer region and the basket. That simplified synchronized recording, but it
was not established as the best observation design. A wrist camera was not
implemented or compared. No camera-placement study or policy-view visual
acceptance gate was completed before training.

The user's inspection identified poor training images. Inspection of saved
`train_v2_000` frames 0 and 1,500 confirms the concern: the initial front-view cube
is only roughly ten pixels across; it is not clearly visible in the initial
overview; fingers obscure parts of the cube during transfer. A nonblank 224-pixel
scene image can still be a poor observation for precise contact manipulation.
Existing automated checks test dimensions, timestamps, calibration and missing
or blank images. They do **not** test object visibility, occlusion, usable grasp
detail or whether the cameras contain enough information to infer the action.

That gap should have been exposed before the longer training runs. Bad visibility
is a concrete concern, but the recorded experiments do not prove it is the sole
cause of failure. Four demonstrations, action-chunk feedback intervals and the
custom training recipe also remain unresolved. The
[camera review](docs/POLICY_REVIEW.md#camera-choice-and-image-quality) records the
exact camera parameters, local image paths and what a wrist-camera change would
involve. No camera or training code was changed for this documentation task.

## Collect, inspect and convert demonstrations

### 1. Inspect the expert in the simulator

In the **simulation terminal**, after its activation block:

```bash
python scripts/run_manipulation.py --cpu-threads 8 --stage full \
  --motion-mode continuous --expert-hz 20 --grip-feedback-from-lift \
  --output-dir outputs/controller/user_expert_view
```

This runs the programmed expert, not a learned policy. GUI is the default.
Planning can pause motion while computing collision checks. A successful run
prints `[RESULT]` and then `[VIEWER]`; close the window after inspection.
`--cpu-threads 8` uses eight Kit workers, the setting used for the successful
retry after a one-worker startup stall. It does not change the policy rate.

### 2. Inspect actual recorded policy images before selecting a dataset

An existing raw episode is:

```text
datasets/raw_train_v2_corrected/train_v2_000_20261022T171242694160Z/
├── episode.json       # reset pose, cameras, source identity, evaluation
├── trajectory.npz     # measured state, requested/applied targets, times
├── overview/*.png     # the policy's fixed overhead view
├── front/*.png        # the policy's fixed front view
├── privileged.json    # object/contact diagnostics, excluded from training
└── attempt.json       # completion/interruption record
```

Open PNGs from both views at approach, grasp, lift, transfer and placement. Judge
cube visibility, contact detail and occlusion yourself. Frame 1,500 is 75 seconds
after the first recorded sample at 20 Hz; episode timing varies. The converted
images are copied from these PNGs. Conversion cannot recover details that were
not visible when recorded. Do not select these recordings just because the
expert completed the task.

### 3. Optional: record a new dataset

Only run this after deciding that the current camera setup and task distribution
are suitable. In the **simulation terminal**:

```bash
python scripts/run_manipulation.py --cpu-threads 8 --stage full \
  --motion-mode continuous --expert-hz 20 --grip-feedback-from-lift \
  --cases-file configs/task_distribution.json --split train \
  --record-dir datasets/raw_train_user_v1 \
  --output-dir outputs/controller/user_train_collection
```

This attempts all **32 predefined training cases**, once each. It is a new
collection, not a resume of the twelve existing recordings. Each attempt gets a
timestamped directory; rerunning into the same root can add duplicate cases.
Choose a fresh root for each intended dataset. For a separate validation
collection, change `--split train` to `--split validation` and use different raw
and report directories. The manifest contains 8 validation and 30 held-out cases.
Do not combine them with training. GUI remains the default for these commands.

`--record-dir` enables the two policy cameras automatically and records exact
20 Hz held targets. A complete row contains the state/images **before** its action,
then the same target applied for three 60 Hz physics ticks. Failed and interrupted
attempts remain recorded, but the converter admits only validated full successes.
Success filtering is suitable for behavior cloning; all attempted cases must
still count when reporting expert reliability.

### 4. Convert the dataset you actually choose

In the **training terminal**, this example converts the twelve currently available
successful raw training episodes into a **new** LeRobot dataset:

```bash
python scripts/policy_pipeline.py convert \
  --raw datasets/raw_train_v2_corrected \
  --output datasets/lerobot_user_train12
```

Expected for the present local root: `episodes: 12`, `frames: 36268`, schema
`v3.0`. This is a format conversion, not a quality endorsement. If you collected
new data in step 3, replace both paths with the new raw root and a fresh converted
root. Do not overwrite `datasets/lerobot_corrected_train4`.

Inspect `<converted-root>/conversion.json`: it lists every attempted raw folder,
accepted episodes and source hashes. The converter checks synchronization, action
bounds and image presence, writes official LeRobot image/parquet data and reopens
it using the official reader. Dataset statistics are computed from accepted
frames in that root. The converter rejects mixed split labels; the trainer itself
does not enforce that its input is a training split, so choose the path carefully.

## Train ACT or Diffusion yourself

In the **training terminal**, select the converted dataset first. The commands
below use the twelve-episode conversion from step 4. They have **not** been run as
part of this documentation task. Run the methods sequentially and use new output
directories. A simulator does not need to be running.

```bash
python scripts/policy_pipeline.py train --method act \
  --dataset datasets/lerobot_user_train12 --output checkpoints/act_user_v1 \
  --steps 4000 --batch-size 16 --workers 2 --seed 701 --lr 0.0001 --log-every 100
```

```bash
python scripts/policy_pipeline.py train --method diffusion \
  --dataset datasets/lerobot_user_train12 --output checkpoints/diffusion_user_v1 \
  --steps 4000 --batch-size 16 --workers 2 --seed 701 --lr 0.0001 --log-every 100
```

These reproduce the existing **diagnostic recipe** on your selected data. They
are not a proven sufficient training budget or an optimized published recipe.
To repeat the previous four-demo experiment, use
`--dataset datasets/lerobot_corrected_train4` instead, keeping a new output path.

`--steps` counts optimizer updates, not episodes or epochs. `--batch-size` is the
number of sampled sequence anchors per update; `--workers` controls data-loading
processes. `--seed` initializes the random generators. `--lr` sets the optimizer's
base learning rate; ACT's visual backbone has a separate 1e-5 group rate.
Omitting `--no-pretrained` uses ImageNet ResNet18 initialization. It does not load
an existing robot policy. Architectures, chunk lengths and normalization choices
are currently defined in `policy_pipeline.py:train`, not exposed as CLI options.
The [review](docs/POLICY_REVIEW.md#actual-training-recipe) explains every setting
and where the custom loop differs from LeRobot defaults.

Training prints loss/gradient rows and ends with `[CHECKPOINT]`. It saves:

| File / directory under your output root | Meaning |
| --- | --- |
| `pretrained_model/` | Final weights, architecture configuration and saved normalization processors; pass this directory to `serve` |
| `step_000500/`, `step_001000/`, … | Model and processor snapshots every 500 updates; usable for inference |
| `training.json` | Actual CLI arguments, dataset size, source hash, parameter count and logged training metrics |
| `progress.json` | Metrics updated during training |
| `training_state.pt` | Optimizer and Torch RNG state for diagnosis; the CLI has no resume option |

The script performs no validation loop, early stopping or best-checkpoint
selection. Final means last update, not best-performing policy. Low loss and the
finite-action check only establish that optimization produced numerical outputs.
Physical inference must be evaluated separately. Do not replace old weights with
new ones until you have compared them yourself.

## View the trained models in Isaac Sim

Use **two terminals**: the training environment serves one policy; the simulation
environment executes it. Choose either ACT or Diffusion. The following commands
use existing four-demo checkpoints so you can inspect the previous work without
training anything.

### Terminal 1: start one model server

After the **training activation block** above, choose ACT:

```bash
python scripts/policy_pipeline.py serve --method act \
  --checkpoint checkpoints/act_corrected_train4_4000/pretrained_model \
  --socket /tmp/so101_manual_act.sock \
  --trace-dir outputs/policy_traces/manual_act --trace-images
```

Or choose Diffusion instead:

```bash
python scripts/policy_pipeline.py serve --method diffusion \
  --checkpoint checkpoints/diffusion_corrected_train4_4000/pretrained_model \
  --socket /tmp/so101_manual_diffusion.sock \
  --trace-dir outputs/policy_traces/manual_diffusion --trace-images
```

Wait for `[READY]`. To inspect your new training run, replace the checkpoint with
`checkpoints/act_user_v1/pretrained_model` or
`checkpoints/diffusion_user_v1/pretrained_model`. Keep the matching `--method`.
Do not start two servers on the same socket. If a path already exists, use a fresh
matching socket path in both terminals instead of deleting an unknown socket.

### Terminal 2: run the matching simulator command

After the **simulation activation block**, for ACT:

```bash
python scripts/run_manipulation.py --cpu-threads 8 --stage full \
  --policy-socket /tmp/so101_manual_act.sock --case nominal \
  --output-dir outputs/controller/manual_act_inference
```

For Diffusion instead:

```bash
python scripts/run_manipulation.py --cpu-threads 8 --stage full \
  --policy-socket /tmp/so101_manual_diffusion.sock --case nominal \
  --output-dir outputs/controller/manual_diffusion_inference
```

These open a simulator window and enable the policy cameras. No `--headless` flag
is used. The model receives RGB and joint states; its 12 absolute joint targets
are held for three physics ticks. There is no expert recovery planner or hidden
gripper servo in this mode. Physics pauses while waiting for inference, so 20 Hz
simulated control does not guarantee 50 ms wall-clock latency.

With the existing checkpoints, ACT has timed out without pickup and Diffusion has
failed within seconds. A failed evaluation exits 1 and can close the window
immediately. Read the terminal and `full_<timestamp>.json` in your output directory.
A successful GUI evaluation remains open at `[VIEWER]`. Stop the simulator before
switching models, then stop its server with Ctrl+C. The server otherwise stays
available after a simulator client disconnects.

The optional trace flags save actual input PNGs, joint states and requested
outputs in a new `seed_*` folder per reset. They do not save proof of applied
motion or terminal images. Omit both flags to avoid that extra disk I/O.

For all eight predefined validation poses, replace `--case nominal` with
`--cases-file configs/task_distribution.json --split validation`. This runs
**all** validation cases; adding `--case` does not narrow a `--cases-file` run.
Use held-out cases only for your final assessment, not repeated tuning. Nominal
is outside the inward-Y training strip, so distinguish it from in-range testing.
Held-out cases already used during development are not a pristine final test set.

## What each file does

Start with `policy_pipeline.py` for learning, `dataset_io.py` for data semantics,
and `benchmark_env.py` for the images and commands seen by the policy. This table
covers every maintained Python source file; imported modules have no standalone
training command.

| File | Responsibility and inputs → outputs | How it is used |
| --- | --- | --- |
| [scripts/policy_pipeline.py](scripts/policy_pipeline.py) | Raw folders → LeRobot dataset; dataset → official ACT/Diffusion weights; RGB/joints → predicted targets | Training environment: `convert`, `train`, `serve` commands above |
| [scripts/dataset_io.py](scripts/dataset_io.py) | Writes and validates 20 Hz raw episodes, joint order, pre-action timing and held targets; keeps privileged diagnostics separate | Imported by recording, conversion and replay |
| [scripts/run_manipulation.py](scripts/run_manipulation.py) | Main CLI; chooses expert, recorded-target replay or learned control; orchestrates phases and writes reports | Simulation environment: `--stage full`; `--record-dir`, `--policy-socket` or `--replay` select modes |
| [scripts/benchmark_env.py](scripts/benchmark_env.py) | Fixed cameras, measured observations, reset, PhysX stepping, joint targets and gravity compensation | Imported by the main simulator runner; camera changes start here |
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

## What is on GitHub and what stays local

All repository source used for recording, conversion, training and inference,
the training package lock, task poses and the review documents are tracked on
`feat/fingertip-geometry-calibration`. Historical source is preserved in Git.
Official LeRobot/Isaac packages are external dependencies, not vendored code.

At the user's request, **datasets and model weights are not uploaded**. Local
`datasets/`, `checkpoints/`, `outputs/` and caches remain ignored. The review JSON
contains metadata and statistics only; it cannot reproduce training or verify
visual quality without the local images. A remote reviewer can audit the code
and choices, but cannot independently rerun the existing weights from Git alone.

No simulation, training or inference process was found running in this workspace
at the documentation audit. No new experiment was started. Read the review for
limitations before treating any existing checkpoint as a useful controller.
