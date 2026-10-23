# Official ACT, Diffusion and policy evaluation

[Dataset prerequisites](DATASET.md) · [Approval gates](STATUS.md) · [Code map](ARCHITECTURE.md)

## Current state and approach

There is **no trained checkpoint or accepted training dataset in the reset
pipeline**. The earlier failed ACT/Diffusion experiments are documented in the
[historical review](POLICY_REVIEW.md); their images, small data coverage and
unvalidated choices are not a current recipe. No optimizer update is authorized
or executed in repository consolidation.

Both methods use the installed **official Hugging Face LeRobot 0.6.1** model,
processors, training loop, optimizer presets and checkpoint format.
`policy_pipeline.py` is a conversion/command/inference adapter, not an architecture
or optimizer implementation. The shared simulator remains in `leisaac-sim`; both
policies use the same `so101-il-py312` environment because of Python/NumPy
compatibility, not because they are unrelated custom Python models.

The controlled initial recipe matches observations and prediction/execution windows
so a comparison is understandable. Matching every internal optimizer/architecture
would instead redefine the methods. This is a proposed methodological choice,
not proof of an optimal training setup. Data acceptance and a measured learning
baseline must precede rankings.

## Inputs, targets and horizons

| Item | Both policies |
| --- | --- |
| Images | `observation.images.overview`, `.left_wrist`, `.right_wrist`; source RGB 480×640×3, tensor CHW, normalized by saved official processors |
| Proprioception | `observation.state`, float32 24 values: q then qd for both arms |
| Supervised target | `action`, 12 absolute position targets in radians, recorded pre-action and held for three physics ticks |
| Excluded oracle inputs | Cube pose, contacts, phases, expert plans and evaluator flags |
| Observation / prediction / execution | 1 current observation /64 predicted targets (3.2 s) /8 executed targets (0.4 s) at 20 Hz |

Official `select_action` owns each policy's observation/action queues. The simulator
still calls every 50 ms; a new model chunk is requested when its queue needs it.
There is no project-side interpolation or expert correction of a learned action.
The server resets queues and random seeds per episode and applies saved pre/post
processors, including action denormalization.

| Setting | ACT `matched-v1` | Diffusion `matched-v1` |
| --- | --- | --- |
| Official family | Conditional VAE / transformer action chunks | Conditional denoising U-Net, DDPM default |
| Window fields | `chunk_size=64`, `n_action_steps=8`, `n_obs_steps=1` | `horizon=64`, `n_action_steps=8`, `n_obs_steps=1` |
| Tail padding | Official padded-target mask | `drop_n_last_frames=0`, `do_mask_loss_for_padding=true` to include matching anchors |
| State/action normalization | Official mean/std | Official min/max |
| Image encoder | Official ImageNet ResNet18 initialization | Official ImageNet ResNet18 initialization; separate RGB encoders by default |
| Optimization | Official AdamW preset, no scheduler preset | Official Adam preset and cosine scheduler preset |
| Model-specific inference | Temporal ensembling off | Official 100-step DDPM schedule; inference count defaults to train count |

All omitted fields retain the installed implementation's defaults; inspect the
saved `train_config.json` before executing. We do not silently resize/crop the
source images or reduce the official model. The recipe uses FP32, batch 8, 100k
updates, checkpoints every 10k, workers 4, seeds 17/29/43, deterministic cuDNN,
no image augmentation, no Hub/W&B upload. RTX 5090 memory/throughput for this recipe
is **not measured**. ImageNet initialization may require its existing cache or
an explicitly permitted download when real training is later authorized.

## Verified command preparation — no training

Activate `so101-il-py312` as in [getting started](GETTING_STARTED.md#environment-locks).
The recipe file is [policy_training.json](../configs/policy_training.json).

```bash
python scripts/so101.py train --policy act --config matched-v1 \
  --dataset datasets/lerobot_train --output checkpoints/act_seed17 --seed 17
python scripts/so101.py train --policy diffusion --config matched-v1 \
  --dataset datasets/lerobot_train --output checkpoints/diffusion_seed17 --seed 17
```

Each command defaults to **preflight only**: it prints the exact resolved
`python -m lerobot.scripts.lerobot_train ...` command, parses the actual installed
LeRobot config through its official parser without constructing a model/optimizer,
and checks the converted train metadata/provenance. Missing/unaccepted data returns
a nonzero prerequisite error, which is expected now. Output is not created by
preflight. Parsing config options does not establish a valid forward pass, memory
budget, convergence or physical policy success.

Checks reject pilot/validation/held-out/legacy data, unaccepted QA, wrong cameras,
state/action dimensions/rate, mismatched converted episode inventory or reused
output directories for a new run. Resolved CUDA availability is reported; actual
execution refuses silent CPU fallback. The installed optional TorchCodec warning
is avoided for dataset decoding by selecting the existing PyAV backend.

The underlying `prepare-training` subcommand remains a legacy print-only utility;
use the named high-level recipe above instead of assembling undocumented flags.

## Start training later, only after acceptance and authorization

A successful preflight prints a `required_approval` object binding policy, named
recipe and file hash, dataset root, conversion/info hashes, output root, seed and
optional resume-config hash. A genuine approval record adds reviewer/date and
`approved_training: true`; do not manufacture approval on the user's behalf.
The data review/collection permissions and training session permission are separate.

Once those gates have actually been satisfied, **one explicit command** can launch
the official training implementation:

```bash
python scripts/so101.py train --policy act --config matched-v1 \
  --dataset datasets/lerobot_train --output checkpoints/act_seed17 --seed 17 \
  --execute --authorize-training --approval outputs/approvals/act_seed17.json
```

Use the same approved data/recipe/seed for Diffusion with its own output/approval.
The adapter delegates to the official process; it owns no optimizer/scheduler and
uploads nothing. This command is documented for future authorized use, not run now.

## Checkpoints and resume

Official training saves under the selected output's `checkpoints/<step>/`, with
`pretrained_model/` containing weights/config/processors and `training_state/`
containing optimizer/scheduler/RNG state. Preserve the whole checkpoint for a
true resume; loading weights alone is not optimizer-state continuation.

Future example, after that checkpoint exists:

```bash
python scripts/so101.py train --policy act --config matched-v1 \
  --dataset datasets/lerobot_train --output checkpoints/act_seed17 --seed 17 \
  --resume checkpoints/act_seed17/checkpoints/010000/pretrained_model/train_config.json
```

It prints/parses `--config_path=<saved train_config.json> --resume=true` with the
named primary settings. Execution requires a new approval bound to the resume
file and the same explicit flags. Original output may exist for resume; it must
be fresh for a new run. Resume config parsing can be checked without updates;
actual optimizer-state restoration has not been tested in this reset pipeline.

## Validation and fair evaluation

Keep separately accepted train/validation roots. Official `dataset.eval_split`
uses a single root's episode partitioning/statistics, so it is **0** for the
primary protocol to keep validation statistics out of training. A separate-root
offline validation-loss CLI is **not implemented**; do not invent one or imply
this milestone supplies it. Proposed primary selection is the final 100k checkpoint,
with the same 20-case physical validation screen for both methods and all declared
failures counted. Freeze choices before the 100-case held-out evaluation.

Report policy and expert success under identical reset, 20 Hz action interface,
physics/contact/evaluator, task timeout and cases. Model losses are not directly
comparable. Report stage/full success, collisions/drops/reset/policy errors,
slip, episode/compute time, model and round-trip latency, GPU memory and samples/s.
Keep train case IDs, image preprocessing and seed budgets fixed across methods.

## See a real trained checkpoint in the simulator later

Two terminals/processes are required because of the existing environment locks.
The CLI uses the current shared simulator/evaluator; no fallback expert supplies
actions during policy evaluation. There is no working model to run today.

Terminal A, **so101-il-py312**, preview the server command, then add `--execute`
only for a real official `pretrained_model` directory:

```bash
python scripts/so101.py evaluate serve --policy act \
  --checkpoint checkpoints/act_seed17/checkpoints/100000/pretrained_model \
  --socket /tmp/so101-act.sock
```

The real server checks checkpoint files/type and prints `[READY]` when listening.
Terminal B, **leisaac-sim**, after `[READY]`:

```bash
python scripts/so101.py evaluate run --socket /tmp/so101-act.sock \
  --manifest outputs/demonstration_protocol_v1/validation.json \
  --split validation --output outputs/evaluation/act_seed17_validation
```

GUI is default; `--dry-run` prints this simulation command. The client refuses an
absent socket, sends only three RGB byte buffers and q/qd lists, receives absolute
targets, and evaluates through `BenchmarkEnv.step`. The independent report, not
server readiness or a low loss, establishes physical outcome. Change policy,
checkpoint, socket and output together for Diffusion. Close the simulator after
its saved final state; stop the server with Ctrl+C when finished (its socket is
removed during cleanup).

## Official references and verification boundary

Checked against installed 0.6.1: `lerobot_train --help`, `lerobot_dataset_viz
--help`, `TrainPipelineConfig`, `DatasetConfig`, both policy config dataclasses,
parser wrapper, model serving/processors and checkpoint layout. See
[verification](VERIFICATION.md) for actual checks and failures found while adapting.
Primary references: [official training workflow](https://huggingface.co/docs/lerobot/v0.6.1/il_robots),
[ACT config](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/configuration_act.py),
[Diffusion config](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/diffusion/configuration_diffusion.py),
[trainer](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/lerobot_train.py)
and [dataset viewer](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/lerobot_dataset_viz.py).
