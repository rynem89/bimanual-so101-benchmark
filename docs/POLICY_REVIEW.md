# Historical review of the earlier failed learning experiments

> Archive, not current training or inference instructions. Its saved models/data
> were removed during the previously authorized reset. Names such as
> `current_models` in the unchanged metadata snapshot mean current at that old
> audit, not checkpoints available now. Preserve this record to understand the
> earlier camera/data/method failures. Use [POLICIES](POLICIES.md) and
> [STATUS](STATUS.md) for the current official pipeline and approval gates.

---

# Camera-v2 reset: current review status

As of 23 October 2026 all previous datasets and checkpoints were deliberately
deleted. The material below is HISTORICAL: its paths and training commands no
longer describe runnable current artifacts. Metadata is retained for audit only.
Current camera and learning procedures are in README.md. No training or dataset
collection is authorized until the new camera and demonstration gates pass.

# ACT and Diffusion: implementation and training review

This document explains the existing implementation and completed training runs.
It does not prescribe them as an optimized method. Use the [README](../README.md)
for commands. The companion [metadata snapshot](POLICY_REVIEW_METADATA.json)
contains exact saved configurations, arguments, statistics and provenance so a
reviewer can inspect them on GitHub without access to this workstation.

The documentation audit reads code and saved artifacts from `d1cb3be`. It makes
no change to training, cameras, observations, motion or evaluation. The user has
reserved further training and visual assessment for themselves. No dataset or
weights are published, as explicitly requested.

## What approach was taken, and why

The approach was **behavior cloning of a privileged geometric expert**. The
expert knows geometry, cube pose and contacts. It generates a full task trajectory
in simulation. The student sees only camera images and measured joint positions
and velocities, and learns the expert's commanded joint positions. This makes
the controller a source of demonstrations without requiring teleoperation.

The implementation used official LeRobot policy classes inside a small custom
training loop. The rationale below explains the design; it is not evidence that
these choices are optimal, sufficient or matched to a published training recipe.

| Choice made | Engineering rationale | What was not established |
| --- | --- | --- |
| Scripted expert demonstrations | Repeatable reset, exact action labels and measurable task completion | A physically successful expert trajectory need not be visually observable or easy to imitate |
| Official ACT and Diffusion classes | Reuse maintained transformer/U-Net and processor implementations | Official classes do not validate the custom data, cameras or optimization loop |
| Fixed overhead/front RGB cameras | Keep both arms, handover region and basket in stationary scene views | Grasp visibility was not accepted visually before training; no wrist-camera comparison |
| 20 Hz absolute joint targets | Match a simple actuator interface and record exact targets held for three physics ticks | The choice was not compared with delta actions or other control rates |
| ImageNet ResNet18 initialization | Reuse general visual features instead of learning all image features from four episodes | Transfer to these synthetic images was not validated; it cannot recover occluded detail |
| Reduced model widths and short diagnostic runs | Limit compute while checking recording, conversion, training and inference | Four trajectories and 4,000 updates were not shown sufficient for learning |
| Same four episodes for both models | Keep demonstration input consistent between diagnostic runs | Equal updates do not equal optimized training or a fair final method ranking |
| Separate Python environments | Satisfy LeRobot requirements while preserving the pinned simulator stack | This adds process/socket complexity; it does not itself improve the policy |

**This was not demonstrated to be best practice as a complete training setup.**
Useful practices were exact action recording, separation of privileged labels,
saved normalization, source/seed tracking and evaluation through real physics.
The missing camera-quality gate, small demonstration coverage, absent validation
loop and unexamined differences from the official optimizer/sampler are material
shortcomings. These should have been stated before longer training was presented
as progress toward a working learned controller.

## Exact demonstration provenance

The latest models used `datasets/lerobot_corrected_train4`, an image-backed
LeRobot v3.0 dataset with 4 episodes and 11,986 frames at 20 Hz. It was converted
from these directories, in order:

| Episode | Raw directory below `datasets/raw_train_v2_corrected/` | Frames |
| --- | --- | ---: |
| 0 | `train_v2_000_20261022T171242694160Z` | 3,042 |
| 1 | `train_v2_001_20261022T171744811433Z` | 2,920 |
| 2 | `train_v2_002_20261022T172208013812Z` | 3,073 |
| 3 | `train_v2_003_20261022T172657667892Z` | 2,951 |

All four recorded full-task expert successes. They were generated from clean
source revision `c1d8336397babd50dcba0255f55c13bd154cecf3`, with continuous motion,
staged handover, 20 Hz expert commands and `--grip-feedback-from-lift`. The saved
recording report used one Kit worker; current GUI instructions use eight after a
later startup stall. These are process-worker settings, not action frequencies.

The reset configurations are the first four `train` cases in
[configs/task_distribution.json](../configs/task_distribution.json). The manifest
contains 32 training, 8 validation and 30 held-out configurations with separate
seeds. Its inward-v2 range is X ±10 mm, Y −12 to −4 mm and yaw ±6° relative to
nominal `(-0.34, -0.04)` m. An earlier symmetric Y distribution exposed expert
planning failures and was replaced for this collection. This narrowed the task
support; it did not demonstrate general workspace robustness. Nominal lies
outside the new Y strip.

The raw folder later grew to twelve successful episodes. **Re-running conversion
on that folder now would select twelve, not recreate the four-episode dataset.**
The frozen converted dataset already exists locally and is the input to use for
repeating the old training recipe. The review metadata includes its original
`conversion.json`, the exact four-case recording document and per-episode hashes.
If reconstructing it elsewhere, select those exact four raw directories first;
changing only the number of optimizer steps cannot reproduce a changed dataset.

During this audit, all seven recorded source-file hashes matched their files at
`c1d8336`. The historical `policy_pipeline.py` at that revision has SHA-256
`8fabd205105146820e3821af957fe97afb6d4b4aaa4c552e74121aae5cf0fad8`, matching both
latest training reports. The current adapter adds optional input tracing; Git
retains the historical implementation for inspection. The converted reader was
previously checked at start/middle/end of each of these four episodes: exact
state/actions and byte-identical RGB samples. That establishes sample preservation,
not useful camera framing or learned task success.

Earlier completed training also exists locally:

| Checkpoint roots under `checkpoints/` | Dataset | Actual demonstrations | Updates / initialization |
| --- | --- | --- | --- |
| `act_smoke`, `diffusion_smoke` | `datasets/lerobot_smoke` | One nominal expert episode, 2,953 frames | 20 each; no pretrained visual backbone |
| `act_initial_2ep_1000`, `diffusion_initial_2ep_1000` | `datasets/lerobot_initial_train` | `train_001` and `train_003` from the earlier distribution, 5,858 frames | 1,000 each; ImageNet visual backbone |
| `act_corrected_train4_4000`, `diffusion_corrected_train4_4000` | `datasets/lerobot_corrected_train4` | The four episodes above, 11,986 frames | 4,000 each; ImageNet visual backbone |

These are separate runs, not sequential fine-tuning. Each creates a new policy;
there is no resume/fine-tune argument in this CLI. Earlier raw paths and training
arguments are also in the review JSON. No twelve-demo model has been trained.
Validation recordings were not used for normalization or learning, and there was
no validation-based selection of the saved final checkpoints.

The following equivalent commands reconstruct the latest training arguments from
the saved `training.json` files. They describe **completed historical runs**;
their output directories already exist, so the CLI will refuse to overwrite them.
Use the README's fresh output paths for your own run.

```bash
python scripts/policy_pipeline.py train --method act \
  --dataset datasets/lerobot_corrected_train4 \
  --output checkpoints/act_corrected_train4_4000 \
  --steps 4000 --batch-size 16 --workers 2 --seed 701 --lr 0.0001 --log-every 100
python scripts/policy_pipeline.py train --method diffusion \
  --dataset datasets/lerobot_corrected_train4 \
  --output checkpoints/diffusion_corrected_train4_4000 \
  --steps 4000 --batch-size 16 --workers 2 --seed 701 --lr 0.0001 --log-every 100
```

## Camera choice and image quality

Camera definitions are `CAMERA_VIEWS` and `configure_cameras()` in
[benchmark_env.py](../scripts/benchmark_env.py). The cameras are fixed at
`{ENV_REGEX_NS}/overview` and `.../front`, not parented to any robot link.

| Camera | Eye position, world metres | Look-at target, world metres |
| --- | --- | --- |
| `overview` | `(0, −0.12, 0.92)` | `(0, −0.12, 0.04)` |
| `front` | `(0, 0.72, 0.56)` | `(0, −0.14, 0.12)` |

Both use 224 × 224 RGB, focal length 18 mm, horizontal aperture 20.955 mm and
clipping range 0.01–10 m. Recorded intrinsics are approximately `fx = fy =
192.4123`, `cx = cy = 112` pixels. Recorded world/ROS quaternions and intrinsics
are in the review JSON's `conversion.cameras`. The ordinary viewer camera is
set separately in `run_manipulation.py` and is never used as a policy input.

**Why no wrist camera:** the implementation selected stationary global coverage
and used Isaac Lab scene-root sensors. It did not implement a link-mounted camera,
choose a wrist extrinsic, or compare visibility against a wrist view. A global
view can locate both arms and the basket, while a wrist view could provide more
local contact detail. Neither advantage proves the current framing was adequate.
There is no experimental basis here for claiming fixed cameras were the better
choice for this task.

Representative images inspected during documentation were:

```text
datasets/raw_train_v2_corrected/train_v2_000_20261022T171242694160Z/
  overview/000000.png
  front/000000.png
  overview/001500.png
  front/001500.png
```

At the initial sample, the front-view cube occupies roughly ten pixels across;
it is not clearly visible in the overhead sample. At sample 1,500, the cube is
partly obscured by the hands during transfer. These are qualitative observations
from four images, not a quantified visibility audit of all episodes. They support
the user's concern about poor observations. They do not establish whether a
particular wrist pose would solve occlusion, nor isolate images as the only cause
of policy failure.

The old checks required an RGB array of the correct size/type with standard
deviation ≥2, checked timestamps and camera pose buffers, and verified file
presence. Offline validation decoded first/middle/last images. These checks
catch empty or malformed recordings, but an arm or table can make an image
nonblank even when the cube is hidden. There was no cube pixel-size requirement,
visibility measurement or human acceptance of the actual policy views before
training. A human check of the main simulator window is not such acceptance.

To assess the data yourself, inspect both actual PNG streams throughout pickup,
handover and release, not only the start frame. Decide whether the cube and
opposing fingers remain distinguishable and whether simultaneous arm motion
occludes essential information. If you choose wrist cameras, the implementation
would need a robot-link parent, specified camera-to-link transform, matching
metadata and model feature names, then newly recorded data and newly trained
weights. Adding a new view only at inference would mismatch the learned inputs.
No such change is included in this documentation revision.

## Observation, action and timing contract

Each network receives:

- `observation.state`: 24 float32 values: left joint positions[6], right joint
  positions[6], left velocities[6], right velocities[6]. Units are radians and
  radians/second. Each arm's order is shoulder_pan, shoulder_lift, elbow_flex,
  wrist_flex, wrist_roll, gripper. The gripper value is a joint angle, not a jaw
  aperture in metres.
- `observation.images.overview` and `.front`: RGB stored as uint8 H×W×3 PNGs,
  decoded to 3×224×224 floating-point image tensors and normalized by the saved
  official processors. There is no depth input or wrist input.

The output is 12 **absolute joint-position targets in radians**, left six then
right six. These are actuator targets, not measured next positions, Cartesian
poses, torques, deltas or IK goals. The environment holds each target for three
60 Hz physics steps, giving a 20 Hz policy interface. Both students share the
expert's implicit joint PD actuators and gravity compensation.

The recorder captures images/state at time t, records the expert target for the
following three physics ticks and verifies that it was actually held. It excludes
an incomplete terminal hold from the training rows. `privileged.json` preserves
cube/contact diagnostics separately; conversion does not read it. Success labels,
expert phases, cube pose and contact forces are not network inputs. Task text is
dataset metadata; these policies do not use a language encoder.

## How ACT and Diffusion turn observations into motion

**ACT** uses one current observation, ResNet18 image features and a conditional
VAE transformer. Training predicts 40 successive targets (2 seconds at 20 Hz).
The loss is masked L1 action error plus 10 times the VAE KL term. The VAE uses
training action sequences to learn a latent representation; inference uses the
official implementation's inference latent rather than future expert actions.
The current configuration executes the first 20 targets before predicting again.
Temporal ensembling is disabled. Although images are sent every 50 ms, the queued
ACT commands do not respond to new observations until that queue empties, up to
one simulated second later.

**Diffusion** conditions a 1-D U-Net on two observation steps. Training adds noise
to action sequences and minimizes noise-prediction mean squared error. It uses
100 diffusion training timesteps with a squared-cosine beta schedule. Inference
starts from noise and uses 10 DDIM denoising steps to generate a horizon of 32
targets. The official sequence indices are observations `[-1, 0]` and actions
`[-1, ..., 30]`; executable actions start at index 1, corresponding to the current
time. It executes 16 targets (0.8 seconds) before generating another chunk.
Observation history is updated while queued actions execute. Seeds are reset at
the episode boundary, and the policy's observation/action queues are reset too.

Neither student runs online IK, collision planning or the expert contact servo.
The action chunk itself is the predicted joint-target sequence. It is applied
without the expert's quintic interpolation or a trajectory optimizer. Range and
runtime physics checks can reject commands or terminate a run; they are not a
collision-avoidance planner. Thus learned motion can fail even if every expert
trajectory was collision checked. Long intervals between chunk replanning are a
specific concern for contact-sensitive transfer, not a proven failure diagnosis.

## Actual training recipe

The exact model configuration and processors are preserved in
`POLICY_REVIEW_METADATA.json` under `current_models`. The values below describe
the completed four-demo runs, not a suggested optimized recipe.

| Setting | ACT | Diffusion |
| --- | --- | --- |
| Model implementation | `lerobot.policies.act.modeling_act.ACTPolicy` | `lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy` |
| Visual initialization | ImageNet ResNet18, trainable | ImageNet ResNet18, separate RGB encoder per camera, trainable |
| Sequence configuration | 1 observation; predict 40 / execute 20 | 2 observations; horizon 32 / execute 16 |
| Main architecture | Width 256; 8 heads; feed-forward 1,024; 4 encoder / 1 decoder layers | U-Net down dimensions `(128,256,512)`; kernel 5; groups 8 |
| Additional settings | VAE latent 32; VAE encoder 4 layers; KL weight 10; dropout 0.1 | DDIM 10 inference steps; 100 noise-training steps; epsilon prediction |
| Normalization | Images/state/actions: mean and standard deviation | Images: mean/std; state/actions: min/max |
| Parameters | 18,725,388 | 40,901,452 |
| Base learning rate | 1e-4; backbone group stays 1e-5 | 1e-4 for all parameters |

Both used batch size 16, seed 701, two loader workers, shuffled sequence anchors,
`drop_last=True`, 4,000 updates, AdamW weight decay 1e-4, optimizer default betas
`(0.9, 0.999)`, gradient-norm clipping at 10, float32 without AMP and constant
learning rates. Statistics came only from the four-demo training dataset. There
was no explicit image augmentation in the adapter; Diffusion's saved crop ratio
is 1.0 and resize/crop shapes are unset. Four thousand batches of sixteen are
64,000 sampled anchors, about 5.34 times the dataset frame count, not 4,000 epochs.

### Differences between saved defaults and actual execution

A reviewer must inspect `policy_pipeline.py:train`, not assume every field in
`pretrained_model/config.json` was applied:

- The adapter constructs `torch.optim.AdamW` directly. It does not call the
  official optimizer/scheduler factories. ACT's non-backbone rate therefore
  comes from CLI `--lr=1e-4`, although saved `optimizer_lr` is 1e-5. ACT's explicit
  backbone parameter group still uses saved `optimizer_lr_backbone=1e-5`.
- Diffusion's saved optimizer defaults include betas `(0.95, 0.999)`, weight decay
  1e-6, a cosine scheduler and 500 warm-up steps. The actual loop uses AdamW's
  default `(0.9, 0.999)`, decay 1e-4 and **no scheduler or warm-up**.
- Diffusion saves `drop_n_last_frames=15`, but the custom DataLoader does not use
  an episode-aware sampler to apply it. Boundary sequence padding can therefore
  enter training; `do_mask_loss_for_padding=False` means its loss includes padded
  positions. ACT's official loss masks padded actions. No boundary ablation was
  performed. This is a concrete recipe discrepancy for review, not a verified
  explanation of all failures.
- `push_to_hub=True` is an inherited configuration default. This adapter only
  calls local `save_pretrained`; it never invokes a Hub upload.
- The model returns component loss details, but the custom log stores only total
  loss and gradient norm. It does not provide separate ACT KL/L1 histories.

Model/processor checkpoints are saved every 500 updates and at the final update.
There is no validation evaluation, checkpoint ranking, early stopping, automatic
hyperparameter search or exact-resume CLI. The final optimizer/Torch RNG file is
not a complete resume implementation: Python/NumPy RNG and sampler position are
not restored by any command. Identical seeds alone do not prove bitwise identical
retraining across hardware/software or guarantee simulation reproducibility.

The current `train` function verifies tensor shapes and finite outputs at the
end. The current `serve` function reloads weights plus saved normalization. Prior
actual inference runs verified that loading works, not that the task was learned.

## Why the models failed: established facts and open questions

Established observations are poor visual detail/occlusion in the inspected PNGs,
four-trajectory training coverage, the recipe differences above and unsuccessful
physics evaluations. ACT completed four attempts with no pickup before 300-second
timeouts. A fifth was stopped at the user's request. Diffusion completed five
attempts with no pickup, failing joint/actuator or forbidden-contact checks.
Detailed run identities and earlier experiments remain in the engineering handoff.

The available tests do not separate image quality, coverage, optimization,
action representation or chunk-feedback effects. There is no evidence that simply
training longer, changing to a wrist camera alone or reducing loss would solve
the task. Conversely, a successful expert replay does not show a vision policy
has enough information to imitate it. The recorded-target replay and byte-alignment
checks narrow some interface concerns but do not eliminate other implementation
or learning failures.

For an independent review, useful questions are: are both grasp regions visible
at the required scale; can the selected demonstrations cover the intended starts;
should both methods use their official optimizer/sampler recipes; how should
validation select checkpoints; and is open-loop execution of 0.8–1-second chunks
appropriate for this contact sequence? These are review questions, not changes
made in this task. The user decides future data, training and observation work.

## What produces the expert demonstrations

The expert is implemented in `run_manipulation.py`, `so101_model.py` and
`so101_planning.py`. FK composes the USD joint frames. The calibrated TCP is
`(0.0141000004, 0, -0.09)` metres in the gripper link. IK uses bounded SciPy
trust-region-reflective least squares over five arm joints: position and an
approach direction can be constrained, while arbitrary six-dimensional poses
are not generally feasible for this arm. Accepted IK needs <1 mm position and
<1° orientation/direction error for the applicable solve.

Collision checks cover connecting paths, not just endpoints. Broad-phase AABBs
and convex separating axes reject separated pairs; ambiguous convex cases use
linear-program feasibility. Gripper cavities use triangle/box and containment
checks. A bounded RRT-Connect fallback searches joint space for difficult
receiver transits or donor retreats. RRT turns execute as separately checked
segments. These geometric checks are distinct from actual runtime contact checks.

A joint segment uses `q(u) = q0 + (10u³ − 15u⁴ + 6u⁵)(q1 − q0)`, with `u=t/T`
and `T=max(1 s, 1.875 max|q1−q0| / 0.35 rad/s)`. Continuous mode joins selected
Cartesian-IK legs using a checked clamped cubic joint spline with quintic time
scaling. Contact boundaries retain settling and measured jaw-force feedback.
The students imitate the resulting actuator targets; these trajectory formulas
are not inserted into learned inference.

`benchmark_core.py` evaluates measured physical states independently of expert
phase flags. Pickup and right-only handover require sustained physical support;
completion requires the released cube inside the basket, floor support, low
motion and both arms home for three seconds. Collision, actuator, slip and
300-second timeout criteria can fail a run. The cube is released above the basket
and settles under gravity; no task-time attachment or teleport carries it.

## How to audit the published files

Start with these source locations:

1. `dataset_io.py`: `JOINT_ORDER`, `EpisodeWriter.begin/applied/finish` and
   `validate_episode` define the supervision and what validation actually checks.
2. `benchmark_env.py`: `CAMERA_VIEWS`, `configure_cameras`, `capture_images`,
   `advance` and `step` define the actual student observations and control.
3. `policy_pipeline.py`: `convert`, `train` and `serve` define dataset admission,
   model settings, optimization, normalization and local inference transport.
4. `run_manipulation.py`: `evaluate_policy` and the expert recorder setup separate
   expert generation from student evaluation; `--cases-file` selects all cases
   in a split, overriding single-case selection.
5. `benchmark_core.py`: `TaskSpec` and `TaskEvaluator` define physical success and
   failure. The tests verify parts of this infrastructure, not policy quality.

The JSON review snapshot includes current source hashes, all six completed
training-run arguments, original four-demo conversion provenance, source
revisions, dataset feature definitions/statistics, both saved full policy and
processor configurations and recorded checkpoint hashes. Historical source and
current source are available on the feature branch. No hidden custom model file
is needed: network classes live in the pinned external LeRobot package. The
installed package metadata requires Python ≥3.12 and NumPy ≥2.0,<2.3; the existing
simulator is Python 3.11/NumPy 1.26.0, which explains the isolated environment.

The review snapshot is not a dataset or checkpoint. Per the user's request,
images, trajectories, weights, runtime logs, caches and external simulator assets
remain local. Their paths are documented so a local reviewer can inspect them;
a GitHub-only reviewer cannot independently verify image contents or rerun those
weights from metadata. The previous README remains in Git history and measured
results remain in the handoff, rather than mixing the experiment log into the
user procedure again.
