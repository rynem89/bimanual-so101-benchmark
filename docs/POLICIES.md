# User-controlled official ACT training

[README starting point](../README.md#5-act-and-diffusion) · [Raw format](DATASET.md#raw-attempt-layout) · [Environment locks](SIMULATION.md#environment-locks)

## Contents

1. [What is prepared](#1-what-is-prepared)
2. [Activate the learning environment](#2-activate-the-learning-environment)
3. [Check the dataset](#3-check-the-dataset)
4. [Choose and preview the experiment](#4-choose-and-preview-the-experiment)
5. [Start training yourself](#5-start-training-yourself)
6. [Monitor, stop and resume](#6-monitor-stop-and-resume)
7. [Inspect a saved checkpoint](#7-inspect-a-saved-checkpoint)
8. [How the model works](#8-how-the-model-works)
9. [Future evaluation and comparison](#9-future-evaluation-and-comparison)

## 1. What is prepared

The owner explicitly accepted these **15 successful, visually reviewed recordings**
on 25 October 2026. The assistant recorded that acceptance; it did not perform a
new visual review or claim new physical replays.

| Preserved source | Accepted episodes |
| --- | ---: |
| `datasets/generated_demos/20261025T001402372570Z_seed43/successful/` | 5 |
| `datasets/generated_demos/20261025T142358523645Z_seed44/successful/` | 10 |

The local training dataset is **`datasets/lerobot/so101_act15/`**. It has 15
separate episodes and **38,038 paired rows**, about 31 min 42 s at 20 Hz. All 15
are assigned to training for this initial experiment. There is no validation or
held-out corpus in this preparation; training loss cannot establish generalization.
Failures, interrupted recordings and earlier rejected pilots are excluded.
The converted directory occupies approximately **8.9 GiB**; the two original raw
successful directories together occupy approximately 697 MiB. Image files plus
embedded Parquet image columns account for the larger official image-format copy.

Preparation checked every state/action row, finite values, radian joint limits,
joint ordering, the three held 60 Hz targets per action, camera frame counts,
20 Hz timestamps, capture/sensor continuity, frozen camera calibration/attachment
records, independent physical-success metadata, completion markers and matching
collector/controller results. Every original episode file is bound by SHA-256.
Conversion decodes all paired RGB frames. Successful recordings may have one
terminal camera capture with only one/two following physics ticks: it is retained
in the raw recording and excluded from training rows by the existing format.

The official reader loaded all 15 episodes and sampled **45 action chunks**
(start/middle/end of each episode), including episode-end padding. All **38,038
converted state/action rows match their originals exactly**. Original episode
digests were rechecked after conversion and remain unchanged. Six focused CLI/
approval/preflight tests passed, including checks that preflight cannot create an
optimizer or launch the trainer. RGB checks cover the official image reader's
float32 [0,1] output as well as uint8 video-reader output.

These checks establish the recorded-data contract. They are not a new physical
replay, an independent visual-quality score, or a policy-success result. Original
recordings and controller/camera configuration are unchanged.

| Local path | Purpose |
| --- | --- |
| `outputs/act15_preparation/acceptance.json` | Exact original paths/digests, owner acceptance, actual checks and their limits |
| `outputs/act15_preparation/validation.log` | Per-episode preparation results |
| `outputs/act15_preparation/conversion.log` | Official writer output |
| `datasets/lerobot/so101_act15/conversion.json` | Accepted-source provenance and conversion inventory |
| `datasets/lerobot/so101_act15/meta/` | Official LeRobot v3 feature schema, statistics, tasks and episode boundaries |
| `datasets/lerobot/so101_act15/data/` | Official Parquet data, including encoded RGB image columns |
| `datasets/lerobot/so101_act15/images/` | Image files emitted by the existing official image writer |
| `outputs/act15_preparation/preflight.json` | Resolved official config, reader samples, parameter counts and cache verification |
| `outputs/act15_preparation/converted_integrity.json` | Exact state/action round trip, original-digest preservation and dataset byte size |
| `checkpoints/act15_seed17_2k/` | Created only when **you** start training |

Conversion keeps the existing LeRobot image-storage path: RGB decoded from the raw
MP4 streams is written losslessly as PNG/Parquet image columns. This costs more
storage than raw MP4 but uses the existing tested converter and official reader.
There is no extra lossy video encoding. Data, weights, local reports and caches
are ignored by Git; GitHub receives source and instructions only.

For reproducibility, this is the conversion command already used. **Do not rerun
it into the existing output**; conversion requires a fresh directory:

```bash
python scripts/so101.py dataset convert --raw \
  datasets/generated_demos/20261025T001402372570Z_seed43/successful \
  datasets/generated_demos/20261025T142358523645Z_seed44/successful \
  --quality-report outputs/act15_preparation/acceptance.json \
  --output datasets/lerobot/so101_act15 --execute
```

## 2. Activate the learning environment

Use this in **each new Ubuntu terminal**. Isaac Sim is not needed for training.
Do not install packages or activate the simulation environment for these commands.

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate so101-il-py312
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TORCH_HOME="$PWD/outputs/torch"
export HF_HOME="$PWD/outputs/huggingface"
python --version
command -v lerobot-train
```

Verified environment: Python 3.12.15, LeRobot 0.6.1, PyTorch 2.7.1+cu128,
torchvision 0.22.1+cu128, CUDA runtime 12.8, RTX 5090 32 GB, driver 580.178.04.
CUDA allocation and a tiny tensor operation passed. Training throughput and peak
training memory have **not** been measured.

The requested ImageNet ResNet-18 weights are already cached at
`outputs/torch/hub/checkpoints/resnet18-f37072fd.pth`; their SHA-256 was checked.
No download is needed while that file and `TORCH_HOME` remain as above.
`--inspect-model` refuses a missing/mismatched cache instead of downloading or
silently changing to random backbone weights. An optional TorchCodec import
warning can appear in this locked installation; the command selects existing
PyAV and this converted dataset uses image columns. Do not change dependencies
in response to that warning.

## 3. Check the dataset

This loads the complete official dataset metadata/Parquet inventory and samples
start, middle and final action chunks from **every episode**. It checks shapes,
finite state/actions, RGB dtype and episode-end padding. It runs no model,
optimizer or training loop:

```bash
python scripts/policy_pipeline.py check-training -- \
  --policy.type=act --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=local/so101 --dataset.root=datasets/lerobot/so101_act15 \
  --dataset.video_backend=pyav --dataset.eval_split=0 \
  --policy.chunk_size=64 --policy.n_action_steps=8 \
  --output_dir=checkpoints/act15_seed17_2k
```

The dataset contains only these policy features, plus official indexing/task
columns. Cube poses, forces, contacts, phases, planning variables and success flags
are excluded from observations and targets:

| Feature | Per training sample |
| --- | --- |
| `observation.state` | float32 `[24]`: left q[6], right q[6], left qd[6], right qd[6] |
| `observation.images.overview` | float32 `[3,480,640]` in [0,1], decoded by the official image-column reader from 640×480 RGB |
| `observation.images.left_wrist` | same |
| `observation.images.right_wrist` | same |
| `action` | float32 `[64,12]`: future absolute joint targets, radians |
| `action_is_pad` | bool `[64]`: official mask at episode boundaries |

The 12-joint order is left then right, each in the recorded `JOINT_ORDER`:
shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper.
Official processors scale images and normalize state/actions from the dataset's
saved mean/std. At batch size 4, actions have shape `[4,64,12]`; each camera has
shape `[4,3,480,640]`. No resize, crop, augmentation or privileged inputs are added.

## 4. Choose and preview the experiment

The existing `configs/policy_training.json` **matched-v1 remains unchanged**:
100,000 steps, batch 8, 64 predicted actions and 8 executed actions. The experiment
below is a separate, editable official CLI invocation: **2,000 steps, batch 4,
chunk 64, execute 8, seed 17**. It is a starting run, not evidence that 2,000 steps
or 15 episodes are sufficient. It samples 8,000 anchors before repeats/padding
considerations, about 0.21 times the number of available rows.

Copy this Bash array into the activated terminal. Edit values directly for later
experiments. Choose a **new output directory** for every fresh run.

```bash
ACT_ARGS=(
  --policy.type=act
  --dataset.repo_id=local/so101
  --dataset.root=datasets/lerobot/so101_act15
  --output_dir=checkpoints/act15_seed17_2k
  --steps=2000
  --batch_size=4
  --policy.optimizer_lr=0.00001
  --policy.optimizer_lr_backbone=0.00001
  --policy.optimizer_weight_decay=0.0001
  --policy.kl_weight=10
  --policy.chunk_size=64
  --policy.n_action_steps=8
  --policy.n_obs_steps=1
  --save_freq=500
  --log_freq=20
  --seed=17
  --num_workers=4
  --policy.device=cuda
  --policy.use_amp=false
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
  --use_policy_training_preset=true
  --cudnn_deterministic=true
  --dataset.video_backend=pyav
  --dataset.image_transforms.enable=false
  --dataset.eval_split=0
  --eval_steps=0
  --env_eval_freq=0
  --policy.push_to_hub=false
  --save_checkpoint_to_hub=false
  --wandb.enable=false
)
```

| Editable flag | Starting value | Meaning |
| --- | ---: | --- |
| `--steps` | 2000 | Total optimizer updates, not epochs |
| `--batch_size` | 4 | Number of observation/action-chunk samples per update |
| `--policy.optimizer_lr` | 0.00001 | AdamW learning rate for the transformer/action model |
| `--policy.optimizer_lr_backbone` | 0.00001 | Separate learning rate for the shared ResNet backbone |
| `--policy.optimizer_weight_decay` | 0.0001 | Official AdamW weight decay |
| `--policy.kl_weight` | 10 | Multiplier on the VAE KL term |
| `--policy.chunk_size` | 64 | Future targets supervised per anchor; 3.2 s at 20 Hz |
| `--policy.n_action_steps` | 8 | Actions executed before replanning at inference; 0.4 s; must be ≤ chunk size |
| `--save_freq` | 500 | Save every 500 updates, plus the final update |
| `--log_freq` | 20 | Print training metrics every 20 updates |
| `--seed` | 17 | Initialization and sampling seed; editable integer |
| `--num_workers` | 4 | CPU loader processes; 0 is useful for diagnosing worker errors |
| `--output_dir` | `checkpoints/act15_seed17_2k` | Local run directory; fresh runs refuse an existing directory |

The official ACT preset selects **AdamW**, no scheduler, and the official
padding-masked L1 reconstruction loss plus `kl_weight × KL`. Learning-rate flags
belong to `policy`, because `use_policy_training_preset=true` resolves the optimizer
configuration from ACT. No custom optimizer or loss is implemented here.

Preview the **same arguments** with the actual installed config parser and reader.
The optional model inspection constructs the official ACT on CPU solely to count
parameters, loading the verified cached backbone. It does not call forward,
backward, an optimizer, the training entry point or Isaac Sim:

```bash
python scripts/policy_pipeline.py check-training --inspect-model \
  --report outputs/act15_preparation/preflight.json -- "${ACT_ARGS[@]}"
```

## 5. Start training yourself

**This is the action that starts optimizer updates. It has not been executed by
the assistant.** Run only when you choose to train, after the preview succeeds:

```bash
set -o pipefail
lerobot-train "${ACT_ARGS[@]}" 2>&1 | tee outputs/act15_preparation/train_act15_seed17.log
```

`lerobot-train` is the installed official console command. The project does not
wrap or change its update loop. Hub uploads, W&B, simulator rollouts and validation
loss are disabled. A decreasing training loss alone does not establish successful
pickup, handover or placement.

The older `so101.py train --config matched-v1` interface still provides its original
100k-step preflight and guarded execution path. Its hash-bound execution approval
has not been created for this task. The direct official command above is the
owner-controlled path requested here; no automatic execution permission is inferred.

## 6. Monitor, stop and resume

In another terminal:

```bash
tail -f outputs/act15_preparation/train_act15_seed17.log
nvidia-smi -l 1
```

Use separate tabs for these continuous monitors. The trainer prints loss, gradient
norm, learning rate and timing; ACT also reports its L1/KL components. GPU usage is
reported by `nvidia-smi`. Check saved files without opening Isaac:

```bash
find checkpoints/act15_seed17_2k/checkpoints -maxdepth 3 -type f | sort
```

Expected checkpoints: `000500/`, `001000/`, `001500/`, `002000/`, plus a `last`
symlink. Each checkpoint contains `pretrained_model/` (weights, config, processors)
and `training_state/` (optimizer, RNG and step state). Keep the **whole checkpoint**
for a true resume. Ctrl+C stops the run; updates after the most recent saved
checkpoint are not recoverable.

After interruption, activate the same learning environment and resume from the
latest saved checkpoint:

```bash
lerobot-train \
  --config_path=checkpoints/act15_seed17_2k/checkpoints/last/pretrained_model/train_config.json \
  --resume=true 2>&1 | tee -a outputs/act15_preparation/train_act15_seed17.log
```

The saved configuration supplies data/output, optimizer settings and total steps.
If extending a completed run, append e.g. `--steps=4000`: that means 4,000 total,
not 4,000 additional updates. Keep batch size, worker/process setup and data
unchanged for reproducible continuation. No checkpoint or resume exists yet;
this command is documented for your later use.

## 7. Inspect a saved checkpoint

After your 2,000-step run completes:

```bash
python - <<'PY'
import json
from pathlib import Path
from safetensors import safe_open
p = Path('checkpoints/act15_seed17_2k/checkpoints/002000/pretrained_model')
c = json.loads((p / 'config.json').read_text())
t = json.loads((p / 'train_config.json').read_text())
with safe_open(p / 'model.safetensors', framework='pt', device='cpu') as weights:
    print('Readable weight tensors:', len(list(weights.keys())))
print('Policy:', c['type'], 'chunk:', c['chunk_size'], 'execute:', c['n_action_steps'])
print('Inputs:', c['input_features'], 'outputs:', c['output_features'])
print('Training steps:', t['steps'], 'batch:', t['batch_size'], 'seed:', t['seed'])
print('Checkpoint files:', sorted(f.name for f in p.iterdir()))
PY
```

A readable checkpoint establishes saved weights/configuration, not physical task
success. Do not launch evaluation or inference automatically after training.

## 8. How the model works

This is official LeRobot ACT, not a new model implementation. A **shared pretrained
ResNet-18** processes all three images. Its spatial features, current 24-dimensional
joint state and a latent token enter a 4-layer transformer encoder (width 512,
8 heads). A 1-layer transformer decoder uses 64 learned action queries and predicts
64 vectors of 12 absolute joint targets. Feed-forward width is 3200.

During training, a separate 4-layer VAE encoder reads the observed state and target
action chunk and predicts a 32-dimensional latent distribution. The KL term
regularizes it toward a unit Gaussian. Reconstruction uses the actual future
recorded targets, with padded positions masked. At inference the latent is zero;
ACT executes eight queued actions and requests a fresh chunk from the next current
observation. Temporal ensembling is off. Eight executed actions changes inference
replanning frequency; it does not shorten the 64-action training target.

This configuration has **51,603,340 trainable parameters and 51,603,340 total
parameters**. Major components contain 11,166,912 backbone parameters,
17,332,736 transformer-encoder parameters, 5,385,856 decoder parameters and
17,332,736 VAE-encoder parameters; projections, embeddings and the action head
make up the remainder. Exact component counts and inferred policy features are
saved in `outputs/act15_preparation/preflight.json`. Frozen batch-normalization
statistics are buffers rather than parameters. Full-resolution training memory
and throughput remain for the owner's first run to measure.

## 9. Future evaluation and comparison

No Diffusion run, trained checkpoint, new collection or simulator evaluation is
part of this preparation. Preserve `matched-v1` for a later controlled comparison.
Its ACT/Diffusion prediction and execution windows remain 64/8, with official
method-specific models, losses and optimizer presets. The historical 100/20/100
protocol is a proposal, not the split of this 15-episode experiment.

Future evaluation must use separately declared, physically reachable cases and
the same frozen `BenchmarkEnv`, cameras, 20 Hz action interface and independent
success evaluator. Do not report training-case success as held-out generalization.
The existing two-environment inference interface remains available after a real
checkpoint and separate owner authorization: `so101.py evaluate serve` in
`so101-il-py312`, followed by `so101.py evaluate run` in `leisaac-sim`. See the
[README evaluation step](../README.md#6-evaluation). No expert correction supplies
actions during learned-policy evaluation.

Implementation was checked against the **installed pinned source**:
`lerobot.configs.train`, `lerobot.policies.act.configuration_act`,
`lerobot.policies.act.modeling_act`, `lerobot.datasets.factory`,
`lerobot.datasets.lerobot_dataset`, `lerobot.scripts.lerobot_train` and
`lerobot.common.train_utils`. These installed modules are the authority for flag,
loss, padding and checkpoint behavior; no upstream upgrade was made.
