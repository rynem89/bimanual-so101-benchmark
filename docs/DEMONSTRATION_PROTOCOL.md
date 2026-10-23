# Proposed demonstration and evaluation protocol

**Decision requested: approve/revise this protocol, then separately authorize the
12-attempt pilot. Neither approval nor collection has happened. No training is
authorized.** The camera arrangement alone was approved on 2026-10-23.

## Research basis and scope

Use the existing PhysX-validated geometric expert as the first demonstrator. It
already solves this task nominally and exposes independent contact/reset/task
evaluation. Replanning each declared cube pose is more defensible than moving
old joint targets or relabeling images. This is a controlled, narrow-distribution
benchmark first; success here would not establish broad manipulation competence.

The [original ACT project](https://tonyzhaozh.github.io/aloha/) reports 50
demonstrations per task and uses fixed plus wrist RGB cameras. That motivates a
25/50/100 demonstration learning curve, not a guarantee that 50 is sufficient for
our much longer episodes. [Official ACT guidance](https://huggingface.co/docs/lerobot/en/act)
recommends starting from its implementation/presets. [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
predicts action sequences using conditional denoising and receding-horizon
execution; its published ranking is not a result for our SO-101 task.

[Isaac Lab 2.3 Mimic](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/imitation-learning/teleop_imitation.html)
requires task-specific object references, subtask signals, pose/action adapters
and multi-arm ordering constraints. It also validates generated candidates with
task success criteria. Installing Mimic/SkillGen or replacing the proven expert
would add an unvalidated layer now. No new stack or synthetic trajectory framework
is proposed for the primary dataset. These are methodological choices, not a
claim of universally best practice. Sources checked 2026-10-23 against installed
LeRobot 0.6.1 and pinned Isaac Lab 2.3.

## Counts, distribution and stage gates

All counts below are **fixed attempted cases**, not a promise of that many accepted
demonstrations. Failures, interrupted processes, reset faults and rejected images
remain in the denominator. No hidden retries or replacement draws.

| Partition | Attempts | Use and gate |
| --- | ---: | --- |
| Pilot | 12 | Eight corners of the proposed support; nominal; nominal yaw ±1°; inward centre. Excluded from training. Require 12/12 automatic AND human acceptances before final collection. |
| Train | 100 | Fresh stratified poses; only accepted episodes used for optimization. Require ≥95 accepted; report actual N. Nested first 25/50 accepted episodes can support later learning-curve experiments. |
| Validation | 20 | Separate fresh poses; never optimize on these. Require ≥19 accepted expert demonstrations; use all 20 declared poses for policy validation, counting failures. |
| Held-out | 100 | Fresh evaluation-only poses. No expert training recordings; keep unopened until selection/recipe is frozen. Same poses for every method/seed. |
| Stress | 20 | Separate exploratory evaluation, wider unverified support. Never merge with primary success rate or use for training. |

Primary support, in world metres: cube centre **x ∈ [−0.350,−0.330],
y ∈ [−0.052,−0.044], z reset to 0.020**, yaw **[−6°, +6°]**.
Relative to nominal (−0.34,−0.04), this is x ±10 mm and inward y −12…−4 mm.
This support is inherited as a *candidate* from the historical inward-v2 geometry
investigation; complete varied pickup/handover/bin success is **not established**.
The old 32/8/30 manifests are retained as history, not reused as the fresh test set.

Pilot nominal points at y=−0.04 deliberately check the previously demonstrated
reference outside the inward support. Stress support is x [−0.355,−0.325],
y [−0.055,−0.040], yaw ±8°; it is an explicit generalization test, not a promised
expert capability. Bin, cube dimensions/mass/friction, robot bases, lighting,
camera poses and controller gains remain fixed for this first comparison.

`plan` uses independent seeded Latin-hypercube strata for each non-pilot split,
with exact poses/seeds serialized before execution. Partitions have distinct case
IDs and seeds. Split by complete trajectory/source case, never random frames.
Any future augmented versions stay with their parent episode. If a pilot corner
fails, stop, diagnose the failed layer and propose protocol v2. Preserve this
version and its attempted outcomes; do not silently shrink the support.

Both robots start at the existing **HOME**, with zero requested velocity and an
independently measured settled reset. Seeds are controlled. Arbitrary joint
jitter could create new inter-arm collisions and invalidate the initial path;
therefore its amplitude is **0 rad** in this version. Later initial-state
robustness requires separately validated HOME perturbations and collision paths.
Randomizing only a seed without perturbing state is not trajectory diversity.

## How the expert supplies labels

`run_manipulation.py` resets, solves bounded five-arm-joint IK, checks paths and
executes the complete existing expert. `so101_model.py` uses SciPy bounded TRF
least-squares with reachable orientation constraints; these arms cannot satisfy
every arbitrary six-dimensional pose. `so101_planning.py` validates intermediate
geometry, includes the payload and uses existing deterministic RRT fallbacks.
The continuous controller uses its existing joint-path timing/interpolation,
bounded grip-force feedback and shared actuator/gravity compensation. Desired
geometry alone is never accepted as grasp evidence; measured force, cube motion,
hold, transfer and floor support drive the independent evaluator.

There is one expert strategy per sampled task pose. Its varying IK paths provide
initial diversity; copying nominal trajectories 100 times would not. No human
teleoperation adapter is implemented. Extra trajectory styles/recovery examples
may be valuable after a baseline exposes a coverage failure, but are not necessary
to establish the first controlled comparison. Naive joint noise, time warping,
mirror flips, object relocation in already recorded images and contact-segment
stitching can invalidate labels or bimanual timing. They are excluded. Image
augmentation is disabled in the primary experiment. A later shared photometric
augmentation ablation can use identical draws for both methods, with no camera
reframing or image crop that hides the interaction.

## Frozen cameras, synchronization and format

`configs/policy_cameras.json` is the authoritative approved camera file. It binds
the reviewed centered overhead and the unchanged wrist mounts, optics and 20 Hz
sampling. SHA-256 is recorded with every attempt. These are simulation estimates,
not physical camera calibration; see README for exact transforms and K.

At boundary k: compute the expert's held target; render all three actual sensor
tensors at the current measured state **without a physics advance**; record
`RGB(k), q(k), qd(k), action(k), physics_tick(k)`. Then apply that exact float32
absolute joint target for **three 60 Hz physics ticks**. Boundary k+1 is 50 ms
later. Record camera frame IDs, actual camera/parent poses, intrinsics and phase
at every boundary. Compare each wrist transform to its measured PhysX parent,
not to commanded joints. A repeated ID, missing boundary, misplaced camera or
nonheld target fails QA. A terminal partial hold is diagnostic only and is
excluded from training rows (at most one extra captured image row).

Raw layout per attempt:

```text
episode.json          # pose/seed, source/config hashes, schema, independent evaluation
attempt.json          # completed/in-progress marker; failures retained
trajectory.npz        # state[N,24], action[N,12], action_60hz[N,3,12], clocks
captures.jsonl        # per-boundary phase, sensor IDs, K and measured camera/parent poses
privileged.json       # cube/contact diagnostics, excluded from policy inputs
overview/000000.png   # lossless 640x480 RGB, same numbering for both wrists
left_wrist/...
right_wrist/...
```

State: left q6, right q6, left qd6, right qd6; radians and rad/s. Action:
left/right absolute q6 targets in radians, including the gripper angle. The
policy receives no cube pose, phase labels, contacts, expert path or success flag.
Those fields are diagnostic evidence only. A future VLA receives the same fixed
task sentence, not per-frame oracle instructions.

[Official LeRobot v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
supports timestamped state/action tables, image/video features and episode
metadata. Our converter uses installed `LeRobotDataset.create/add_frame/save_episode/finalize`,
with **image features and Parquet**, not a hand-written imitation of its schema.
Keys are `observation.state`, `action` and
`observation.images.{overview,left_wrist,right_wrist}`. Convert train and validation
into distinct roots. Reopen with the official reader and policy delta-timestamps;
episode padding must not cross boundaries. Recompute statistics from train only;
use the checkpoint's train-derived processors at validation/inference. No Hub upload.

## Acceptance and rejection

`validate` checks all PNGs, not just first/middle/last. Missing prerequisites fail.
Require all four independent stage flags, full success, no fatal failure, zero
forbidden collisions, speed ≤10 rad/s, effort ≤10 Nm, stable-grasp slip ≤5 mm /10°,
and reset error ≤2 mm /2°. These preserve existing evaluator limits, not hardware
safety certification. Record acceleration/jerk; this USD has no authored limits
for them, so do not invent a certification threshold.

Require 1,200–6,000 complete policy rows (60–300 s), float32 finite state/action,
authored action bounds, exact 3-tick holds, absolute/relative clock agreement,
three calibrated 640×480 RGB streams, monotonically increasing sensor IDs and
measured parent attachment at 20 µm /rotation-matrix 2e-4. Nonblank RGB std ≥2
is a corruption check, not object-visibility proof. Identical pixels for >60
successive frames raise a conservative stale-stream rejection; legitimate long
static occlusions could trigger it and must be investigated in the pilot, not
silently waived. No automatic camera optimization is performed.

Human review remains mandatory for **every candidate**: review synchronized video
and full-resolution interaction frames; confirm all streams, cube/gripper detail
at grasp, receiver acquisition, donor release, right-only hold and bin release,
complete motion, continuity and absence of clipping. Cube/contact detail must be
discernible in at least the relevant wrist view, with global context from overhead;
record the failed view/phase if not. For the pilot, annotate approximate cube
pixel width at these phases; <12 pixels or complete occlusion in all three views
requires review of observability before scaling. The 12-pixel guide is a proposed
human QA target, not a validated automatic detector or a contact proof.

QA emits contact sheets and optionally three-stream videos, plus a review JSON
template with acceptance false. A reviewer supplies name/date, explicit checkboxes,
decision and notes. A SHA-256 digest covers **all raw bytes**; altering an image,
state, label or metadata invalidates review. Conversion requires the final accepted
QA index, complete attempt ledger and partition acceptance gate. The raw directory
must contain only that partition. Duplicate/unknown cases, incomplete attempts,
source dirtiness or wrong configuration fail; final-quality summaries keep all
declared cases, including process failures before a raw folder existed.

## Executable workflow

Use the README's simulation activation in each fresh shell. The following planning
command is safe now and writes configuration under ignored `outputs/` only:

```bash
python scripts/demonstration_pipeline.py plan --output outputs/demonstration_protocol_v1
python scripts/demonstration_pipeline.py collect \
  --manifest outputs/demonstration_protocol_v1/pilot.json \
  --raw datasets/protocol_v1_pilot --output outputs/protocol_v1_pilot_run
```

The second command **prints** proposed launches; no simulator starts. New output
directories are required for `plan`/QA. `plan.json` provides exact manifest hashes.
An optional offline geometry check in `leisaac-sim` uses its existing bundled USD
libraries, with no simulator, images or action recording:

```bash
USD_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
export PYTHONPATH="scripts:$USD_LIB"
export LD_LIBRARY_PATH="$USD_LIB/bin:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python scripts/demonstration_pipeline.py preflight \
  --manifest outputs/demonstration_protocol_v1/pilot.json \
  --output outputs/demonstration_protocol_v1/pilot_preflight.json
```

This checks HOME→open→pregrasp→grasp intermediate geometry only. It cannot establish
physical contact, lift, handover or placement, and does not waive the pilot gate.
Executed on this proposal: **12/12 approach plans safe, 137.771 s summed planning
wall time**; `outputs/demonstration_protocol_v1/pilot_preflight.json`. These are
offline plans, not recorded demonstrations or physical trials.

After protocol approval and explicit session collection authorization, create an
approval JSON with `approved: true`, `reviewer`, `date`, `protocol_sha256`,
`camera_config_sha256`, `manifest_sha256` copied from the reviewed plan. This is
an audit record of real human authorization, not a substitute for asking the user.
Do not create a true approval record before approval. Then append to `collect`:

```text
--execute --authorize-collection --approval <reviewed-approval.json>
```

The launcher runs one existing expert process per declared case, on the current
clean committed branch. GUI is default, closing after each completed evaluation;
`--headless` is an explicit optional throughput choice after the pilot. Each case
gets a start/finish ledger record and log even if Isaac/reset fails. `--resume`
skips **all** started cases, including failed/interrupted ones; it cannot turn a
failure into an uncounted retry. A justified retry needs a new declared protocol.

First QA, after an authorized pilot:

```bash
python scripts/demonstration_pipeline.py validate \
  --manifest outputs/demonstration_protocol_v1/pilot.json \
  --raw datasets/protocol_v1_pilot --output outputs/protocol_v1_pilot_qa \
  --ledger outputs/protocol_v1_pilot_run/attempts.jsonl --review-video
```

Review `*.jpg`, `*.mp4`, original PNGs and episode evaluations. Copy/fill
`human_review_template.json` into your own review file. Repeat validation into a
fresh QA directory, adding `--human-review <completed-review.json>`. Only that
report may say `ready_for_conversion: true`; automated success alone cannot.

Final train/validation collection requires separate session authorization and
separate hashed approval files. Use their respective manifests, raw/run directories
and `--pilot-report <accepted-pilot-qa>/quality.json`. Repeat the same QA process.
In the existing **so101-il-py312** environment, convert each accepted partition:

```text
python scripts/policy_pipeline.py convert --raw <train-raw> --output <train-lerobot> \
  --quality-report <accepted-train-qa>/quality.json
python scripts/policy_pipeline.py convert --raw <validation-raw> --output <validation-lerobot> \
  --quality-report <accepted-validation-qa>/quality.json
```

Neither conversion nor command preparation trains a network. Real generation,
PNG throughput, varied physical feasibility and human pilot acceptance remain
unverified until authorized execution. The synthetic format tests exercise the
file interfaces and failure gates; their dummy success flags are not task evidence.

## Fair training and evaluation proposal — not authorized

Primary: identical accepted train episode IDs, camera resolution/order, state/action
contract, fixed task text, no image augmentation, common seeds **17/29/43**, batch
**8**, **100,000 updates**, checkpoints every **10,000**. Match one observation
frame, 64-target prediction window and **8 executed targets (0.4 s)** per query.
Both use their official optimizers/schedulers/architectures and ImageNet ResNet18
initialization. Report their different normalization rules and parameter counts.
No claim that equal updates equal compute; measure wall/GPU time, memory and samples/s.

Installed defaults differ: [ACT 0.6.1](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/configuration_act.py)
uses 1 observation, chunk100, execution100;
[Diffusion 0.6.1](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/diffusion/configuration_diffusion.py)
uses 2 observations, horizon64, execution32. Our matched setting is deliberately
a controlled comparison, not the untouched presets. Native-settings runs are a
separate future ablation. To sample the same anchors including terminal windows,
Diffusion's padding loss mask is enabled and its tail-drop is set to zero; ACT
already masks padded actions. Verify these settings in saved configs before training.

After independent data acceptance and a new training authorization, prepare the
base official command (prints only):

```text
python scripts/policy_pipeline.py prepare-training --method act \
  --dataset <train-lerobot> --output <new-act-run> \
  --steps 100000 --batch-size 8 --validation-fraction 0 --eval-steps 0
```

Review the printed command; add common official flags
`--seed=17 --save_freq=10000 --num_workers=4 --policy.device=cuda --policy.use_amp=false
--dataset.image_transforms.enable=false`, then method-specific flags:

```text
ACT:       --policy.chunk_size=64 --policy.n_action_steps=8
Diffusion: --policy.n_obs_steps=1 --policy.horizon=64 --policy.n_action_steps=8
           --policy.drop_n_last_frames=0 --policy.do_mask_loss_for_padding=true
```

Change method/output/seed deliberately for each run. These commands are proposals,
not instructions to execute this session. No training command or optimizer was run.
The installed trainer's internal `dataset.eval_split` uses the last episodes and
shared root metadata/statistics; **disable it here** to keep validation statistics
out of training. Separate-root offline loss evaluation using checkpoint processors
is not yet an implemented CLI; do not invent a validation command. For this first
protocol, use the fixed 100k final checkpoint as primary and 20 independent physical
validation cases for diagnosis. Any recipe revision is declared before opening
held-out results. Losses between ACT and Diffusion are not directly comparable.

Evaluate the expert and each selected policy on the same 100 declared held-out
poses, using the existing independent evaluator, 300 s timeout and identical
reset/actuation. Count startup, reset, policy and physical failures. Include stage
success, full success, failure reason, contacts, slips, task duration, speed/effort,
inference p50/p95, query interval and real-time factor. Report success/100 with
Wilson 95% intervals per seed, seed mean/spread, and paired case comparisons.
100 trials still have roughly ±10 percentage-point uncertainty near 50% success;
do not overstate small differences. Stress and known nominal regression cases are
separate tables. Expert failures remain visible in the benchmark denominator.

Future VLA: same accepted episodes and held-out poses, same three images/state,
same task sentence and shared physics/evaluation. Disclose external pretraining,
resolution/history/action adaptation, fine-tuning parameters and compute; a
pretrained VLA is a transfer-learning track, not a from-scratch architecture-only
comparison. No VLA architecture or download is selected now.

## RTX 5090 time/storage budget

Measured nominal reference: **153.1833 s simulated**, 3,064 policy boundaries.
The eight-camera comparison with continuous diagnostic video took **281.36 s wall**
including cold start (260.22 s episode). That is not a three-camera PNG collection
benchmark. Budget **5–10 min/attempt** initially for planning, rendering and lossless
PNG writes; measure per-stage time, bytes and failure rates during the pilot.
12 pilot attempts: **1–2 h**; 120 final attempts: **10–20 h**, plus human QA.
No collision checks are disabled to achieve these estimates. One fresh process
per case trades startup overhead for isolation and countable failures.

At nominal duration: 100 train episodes ≈306,400 rows /919,200 RGB images;
20 validation ≈61,280 rows /183,840 images; all 132 recorded attempts ≈404,448
rows /1,213,344 images. Raw uncompressed RGB is **8.47 GB/episode**. Assuming
30–160 kB per lossless PNG, budget **0.28–1.47 GB/episode**, about **40–200 GB**
for pilot+train+validation. These compressed sizes are planning assumptions, not
measurements. Allow a second dataset copy, review media and checkpoints; reserve
**400 GB** or revise storage after measured pilot. Never delete raw data without
user approval. Video-backed v3 is an optional later storage experiment requiring
decoded frame/timestamp and visual checks; do not change encoding mid-comparison.

Generate with one simulator and exactly three policy cameras. Do not run GPU
training concurrently. Keep collision/planning CPU workers8 and BLAS/OMP1; profile
existing plan caching before changing algorithms. Do not vectorize the current
single-scene environment until resets/contact filtering are validated independently.
NVIDIA's [Lab 2.3 performance guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/how-to/simulation_performance.html)
identifies rendering and collision complexity as costs. For this reference, keep
collision geometry, contacts and physics fidelity fixed; remove only the extra
diagnostic comparison cameras from collection. Do not apply its optional OS or
geometry tuning suggestions to this locked benchmark.
Use local SSD and four training data-loader workers initially. Keep 640×480 source
frames; both methods may share an explicitly approved resize ablation later.

Training time is **unmeasured**. At 1/2/5 updates/s, 100k updates take
27.8/13.9/5.6 h per method/seed; six primary runs take 6.9/3.5/1.4 GPU-days.
These are scenarios, not RTX5090 throughput claims. A separately authorized
500-update timing/memory check should precede long runs. Start FP32 for numerical
verification; consider official AMP for both only after equivalent finite-loss
checks and document the change. Do not change locked torch/CUDA/driver packages.
600 held-out policy episodes across two methods×three seeds could add **50–100 h**
at the same 5–10 min budget; early physical failures may shorten this. A one-seed
20-case validation screen precedes this expense. Evaluation images need not be
retained as training data; retain metrics and selected diagnostic evidence.

## Deliverable boundary

Implemented: camera freeze/default, deterministic manifests, guarded expert
orchestration with attempt ledger/resume, per-boundary capture evidence, exhaustive
image/physics/timing/attachment QA, human review sheets/video/template, hash-bound
acceptance and official conversion gate. **45/45 CPU tests passed in 71.810 s**.
Two synthetic episodes /16 frames converted and loaded through official ACT and
Diffusion sequence readers with 64-target windows and correct terminal padding.
Actual GPU reset/interface camera diagnostic passed in9.046 s, with approved
intrinsics and measured parent poses rechecked offline. No physical demonstration
was recorded and no model/optimizer was created. Not yet proven: varied expert success, real collection runtime/storage,
pilot visual/data acceptance, optimizer behavior or learned physical task success.
Next action is protocol review, followed by separately authorized pilot collection.
