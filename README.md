# Bimanual SO-101 benchmark

The left SO-101 picks up a 40 mm cube, hands it to the right SO-101 in mid-air,
and the right arm places it in a basket. Both arms retreat. Isaac Sim provides
physics and contact; an independent evaluator checks the result.

**Start here and follow steps 1–6 in order.** You can run the scene and expert
now. Demonstration collection and policy training require separate approval.

## Table of contents

1. [Setup](#1-setup)
2. [Scene and cameras](#2-scene-and-cameras)
3. [Expert IK and task execution](#3-expert-ik-and-task-execution)
4. [Demonstrations](#4-demonstrations)
5. [ACT and Diffusion](#5-act-and-diffusion)
6. [Evaluation](#6-evaluation)
7. [Project status](#project-status)
8. [Code, detailed guides and historical records](#code-detailed-guides-and-historical-records)
9. [Outputs and repository integration](#outputs-and-repository-integration)

## 1. Setup

These commands use the existing workstation. They do not install anything.
Activate the environment in **each new terminal**:

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
export LEISAAC_ASSETS_ROOT=/home/rynem8/robotics/leisaac/assets
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python --version
python scripts/so101.py --help
```

Expect Python **3.11.16**. The locked simulation stack is Isaac Sim 5.1, Isaac
Lab 2.3 and LeIsaac 0.4. The learning environment uses Python 3.12 and LeRobot
0.6.1. Keep both environments: their Python/NumPy requirements differ. Both ACT
and Diffusion use the same official LeRobot environment, not two custom models.
[Exact versions and asset paths](docs/SIMULATION.md#environment-locks).

## 2. Scene and cameras

```bash
python scripts/so101.py scene --frames
```

This opens two robots, the cube, basket and a panel with the actual overhead,
left wrist and right wrist RGB images. Use the selector to choose a camera.
Yellow guides show camera directions; axes show coordinate frames. You can orbit
the viewport and expand `/World/InspectionGuides` in the Stage panel.

The scene settles for three simulated seconds, then freezes physics. The camera
panel shows that settled state; the free viewport continues rendering. No expert
motion, recording or collection starts. **Close the whole Isaac window before
running the next command.** [Camera calibration and frames](docs/SIMULATION.md#cameras-and-coordinate-frames).

## 3. Expert IK and task execution

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_manipulation.py --stage full
```

The expert executes pickup → lift → handover → basket placement → retreat.
GUI is the default; the final state stays open for inspection. Close Isaac after
reviewing it. It runs once by default. An episode means one complete task attempt,
not a recorded demonstration. `python scripts/so101.py expert` is the equivalent
high-level entry point; the direct command above uses the same coordinated recipe.

The right starts moving **while the left opens and approaches the cube**. It now
waits lower, about 4 cm back along the final insertion direction. The donor brings
the cube in front of that waiting gripper; the receiver uses a direct insertion
instead of going around to a different handover pose. After verified transfer
and finger separation, the receiver makes one continuous move to the basket
while the donor returns HOME. The collision checker validates both moving arms
and the carried cube together. Existing checked fallback routes remain available.
The owner accepted 15 varied demonstrations under this controller. Historical
pilot successes describe the earlier sequential controller; broader reliability
remains to be measured.

The improved nominal task passed **two non-recording visible Isaac runs**: all four
physical stages succeeded with zero forbidden contacts in **114.88 simulated
seconds each**, including the 3 s reset. The previous coordinated version took
**134.12 s** and the sequential reference **153.18 s**. That is **14.3% shorter
than the previous version**. Excluding reset, the shorter motion would mean about
14.7% fewer 20 Hz observation/action rows if later recorded. The owner’s current diagnostic PNG batch remains preserved locally; it is not
an accepted training corpus. These nominal timings describe the pre-repair
controller. The owner subsequently recorded and visually accepted the 15
successful varied demonstrations listed in step 5; broader reliability is unmeasured.

The cold run took **232.24 s wall time**, including **64.81 s of stage planning**.
The aligned receiver insertion needed **6.61 s planning**; the new return/carry
needed **28.43 s**. First-run calculation is still noticeable. Planning pauses
physics, so those pauses add no dataset frames. Cached plans are reused only
when source, geometry and measured joint/cube inputs match.
The verified cached run took **168.69 s wall time**, with only **0.78 s
of stage planning**. Changing source, cube pose or measured grasp can require
fresh computation.
[Timing evidence and limits](docs/archive/VERIFICATION.md#aligned-handover-and-continuous-bin-carry--2026-10-24).

**What calculates the movement?** Our code uses SciPy bounded inverse kinematics
(IK) to find five arm-joint angles for a requested fingertip pose. It checks the
whole path against both robots, table, cube and basket. Joint transits use quintic
timing; Cartesian waypoint approaches use a cubic spline with quintic timing.
Those position targets are held
at 20 Hz while Isaac executes 60 Hz physics. Contact feedback controls gripping;
the cube is moved by simulated contact, not teleported.
[Equations and source references](docs/SIMULATION.md#expert-ik-and-motion).

Planning can take time: `[PLAN]` messages mean CPU collision checks are running
while physics pauses. Expect `[EPISODE] nominal/1: passed` and
`[RESULT] passed_requested_stage`. Check `full_task_success`, independent stage
flags and failures in `outputs/controller/simulation.json`; exit code alone
is insufficient. The timestamped `full_*.json` preserves the run. `plans/` stores
validated planning caches.

Preview the command without opening Isaac, or run a named regression case:

```bash
python scripts/so101.py expert --dry-run
python scripts/so101.py expert --case yaw_minus_1deg --output outputs/expert_yaw
```

## 4. Demonstrations

A demonstration is a **recorded physical expert episode**, not an IK solution or
a video alone. Each 20 Hz row pairs three RGB images and measured joint state
with the absolute joint target applied over the next three physics ticks.

### Generate your own demonstrations

Use **`scripts/collect_demos.py`**. It calls the existing coordinated expert,
recorder and `so101.py dataset review` in one multi-case simulator process.
You choose the ranges and run collection yourself. Activate `leisaac-sim` in each
new terminal using [step 1](#1-setup), including the asset path and thread settings.
No packages need installing.

- `--count N` asks for **N successful physical demonstrations**, not N attempts.
- `--max-attempts M` caps simulation attempts, including failures. Default is
  three times `--count`. Collection stops at the success target or the cap/available
  distinct poses. An unmet target returns a nonzero exit code and the actual count.
- `--x-offset-cm MIN MAX` and `--y-offset-cm MIN MAX` are **centimetres relative
  to nominal cube centre X = −0.34 m, Y = −0.04 m**. For example, X offset +2 cm
  means world X = −0.32 m. Negative/positive offsets follow the world axes.
- `--yaw-deg MIN MAX` controls cube rotation about the vertical axis in degrees.
- `--seed 42` reproduces the candidate ordering with the same ranges, count,
  attempt cap and unchanged model. Keep all these arguments identical between
  preview and recording.

The examples below request a **6 cm × 4 cm** region and yaw ±6°. These are
user-selected sampling bounds, not a claim that every pose will work. The sampler
stratifies the requested region, filters candidates, then chooses spread-out
positions at least **5 mm apart in XY**. It never substitutes the old small range.
Preview prints requested bounds, proposed world X/Y/yaw and actual feasible
coverage, plus rejection counts. If insufficient distinct poses are found, it
fails before simulation; change the ranges, seed or count explicitly.

Before recording, the geometry filter checks the **whole rotated 40 mm cube** is
supported by the table with a 10 mm edge margin, clear of the bin and both HOME
robots, and has valid open-gripper pregrasp/grasp IK under the existing joint
limits and collision checks. It also checks pregrasp descent and a predicted
closed-grasp payload lift using the same lift planner. Actual seating and the
prepositioned receiver still require runtime checks. This establishes geometric feasibility;
**it does not prove a full collision-free task or physical success**. The expert
still checks complete motion and applies all existing contact/drop/grasp guards.

**1. Preview only — no simulator, recording or output files:**

```bash
python scripts/collect_demos.py --count 1 --max-attempts 2 \
  --x-offset-cm -3 3 --y-offset-cm -2 2 --yaw-deg -6 6 --seed 42 \
  --preview-only
```

**2. Record visibly in Isaac Sim:**

```bash
python scripts/collect_demos.py --count 1 --max-attempts 2 \
  --x-offset-cm -3 3 --y-offset-cm -2 2 --yaw-deg -6 6 --seed 42 \
  --gui
```

**3. Record without the GUI, with the same three cameras and HTML reviews:**

```bash
python scripts/collect_demos.py --count 1 --max-attempts 2 \
  --x-offset-cm -3 3 --y-offset-cm -2 2 --yaw-deg -6 6 --seed 42 \
  --headless
```

GUI is the default. One Isaac instance runs the whole batch through the expert’s
existing multi-case loop. Each episode uses the existing physical reset and its
independent pose/velocity/weight checks. An unverified reset stops the batch.
The process stops at the success target or attempt cap; it does not restart
Isaac for each case. The examples above are deliberately limited to one success
and at most two attempts for owner testing.
Use **Ctrl+C in the terminal** to stop the whole collection. Closing the simulator window ends the shared batch.

The script prints a new run directory under
`datasets/generated_demos/<UTC timestamp>_seed42/`:

| Location inside that run | What it contains |
| --- | --- |
| `successful/<episode>/` | Finalized recordings with independent pickup/lift/handover/placement success; three RGB streams, measured states and held actions |
| `failed/<episode>/` | Completed physical/controller/recording failures; never counted toward `--count` |
| `interrupted/<episode>/` | User-interrupted or terminated partial recordings; reported separately from physical failures |
| `reviews/attempt_0001.html`, etc. | One interactive three-camera/state/action review per recording, where enough data exists; images stay in the corresponding episode |
| `cases.json`, `controller.log`, `controller/` | Exact batch poses, one simulator log and normal per-case plans/results; startup failures may have no recording |
| `summary.json` | Cube XY/yaw/seed, physical and collection result, duration, failure reason, episode and review paths, actual successful/failed/interrupted counts |

A failed grasp, drop, contact guard or other failure remains recorded; the script
tries another feasible position up to the cap. It does not relax safety checks.
HTML generation uses the existing review command and its built-in format checks;
no separate dataset QA, plots, physical replay or training runs are added.
Physical success is **not human acceptance or permission to train**. Review the
images and actions yourself; conversion/training still require their existing
separate approvals. The old pilot directories and their review links were
superseded and deleted at the owner's request.

### Recording storage and compatibility

New collection defaults to **three MP4/H.264 streams**, one per camera, still
640×480 RGB at 20 Hz. Encoding uses installed FFmpeg/libx264, CRF 18, ultrafast,
YUV420p and one CPU encoding thread per camera. It is **lossy**, not byte-identical
to PNG. Original recordings are never transcoded or overwritten.

Raw schema `so101-demonstration-1.2` stores `overview.mp4`, `left_wrist.mp4` and
`right_wrist.mp4` inside each episode. Metadata names the files, codec and exact
encoded frame counts. `trajectory.npz`, capture clocks, measured states, actual
held actions and privileged diagnostics retain their existing semantics. Video
frame i maps to paired row i, independent of wall time. A terminal partial
transition can have one extra video frame; it is excluded from training rows.
Streams block rather than drop frames, flush on normal failure/interruption,
and receive a seek index at finalization. Abrupt kills retain partial streams
where possible and never establish a successful demonstration.

The HTML slider decodes all three videos to the same requested frame before
updating their displayed canvases and joint/action table. Legacy PNG reviews
still work. The shared RGB reader also feeds decoded uint8 arrays into the
existing official LeRobot 0.6.1 conversion. Acceptance/training gates are unchanged.
Use `--record-format png` only when you explicitly want the old lossless storage.

Offline benchmark on **240 contiguous saved frames per camera, three cameras**,
including writer finalization and format/frame-count validation, excluding source
PNG loading:

| Saved source | PNG | H.264 | Recording I/O speedup |
| --- | ---: | ---: | ---: |
| Diagnostic attempt 1 | 17.84 s | 0.35 s | 51.5× |
| Diagnostic attempt 9 | 17.51 s | 0.37 s | 46.7× |

Separate codec tests decoded exactly 240 frames per camera at 640×480/20 Hz;
mean sampled PSNR was about 42 dB. These are offline serialization results,
**not measured new episode times or physical success rates**. Focused tests
cover old/new storage, terminal partial frames, seeking, interruption, HTML and
25 synthetic frames passed into the pinned official LeRobot dataset writer.
No new physics run, demonstration collection or bulk dataset QA was performed
for this repair. The owner must test the simulation and browser review before
further collection. [Legacy raw field definitions](docs/DATASET.md#raw-attempt-layout).

## 5. ACT and Diffusion

The owner approved **15 successful, visually reviewed demonstrations** from the
seed43 batch (5) and seed44 batch (10). They are prepared locally at
**`datasets/lerobot/so101_act15/`**: 15 episodes, 38,038 paired rows, three
640×480 cameras at 20 Hz, 24 state values and 12 absolute joint targets.
Original recordings are preserved. Failures and rejected pilots are excluded.
There is **no trained model**; no optimizer updates have been executed.

The first proposed ACT run is **2,000 steps, batch 4, chunk 64, execute 8,
seed 17**, using official LeRobot 0.6.1 ACT, AdamW and its masked L1 + KL loss.
This is an editable starting experiment. The existing `matched-v1` recipe
(100k steps, batch 8) remains unchanged. All 15 episodes are training data;
this preparation supplies no validation/held-out policy result.

**1. Activate the learning environment in each terminal:**

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate so101-il-py312
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TORCH_HOME="$PWD/outputs/torch"
export HF_HOME="$PWD/outputs/huggingface"
```

The ImageNet ResNet-18 weights are already cached under `TORCH_HOME`; no download
is needed. CUDA was checked on the RTX 5090. Training memory/throughput has not
been measured.

**2. Check the prepared dataset without training:**

```bash
python scripts/policy_pipeline.py check-training -- \
  --policy.type=act --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=local/so101 --dataset.root=datasets/lerobot/so101_act15 \
  --dataset.video_backend=pyav --dataset.eval_split=0 \
  --policy.chunk_size=64 --policy.n_action_steps=8 \
  --output_dir=checkpoints/act15_seed17_2k
```

This uses the official dataset reader and samples action chunks from every
episode. For the full experiment configuration preview and exact parameter
counts, follow [the editable arguments and preview procedure](docs/POLICIES.md#4-choose-and-preview-the-experiment).

**3. Start training yourself only when ready. This command performs updates:**

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=local/so101 \
  --dataset.root=datasets/lerobot/so101_act15 \
  --output_dir=checkpoints/act15_seed17_2k \
  --steps=2000 --batch_size=4 \
  --policy.optimizer_lr=0.00001 \
  --policy.optimizer_lr_backbone=0.00001 \
  --policy.optimizer_weight_decay=0.0001 \
  --policy.kl_weight=10 \
  --policy.chunk_size=64 --policy.n_action_steps=8 --policy.n_obs_steps=1 \
  --save_freq=500 --log_freq=20 --seed=17 --num_workers=4 \
  --policy.device=cuda --policy.use_amp=false \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --use_policy_training_preset=true --cudnn_deterministic=true \
  --dataset.video_backend=pyav --dataset.image_transforms.enable=false \
  --dataset.eval_split=0 --eval_steps=0 --env_eval_freq=0 \
  --policy.push_to_hub=false --save_checkpoint_to_hub=false --wandb.enable=false
```

Change these CLI values for later experiments and choose a fresh `--output_dir`.
Checkpoints will appear under `checkpoints/act15_seed17_2k/checkpoints/`, every
500 updates and at the final step. The assistant has **not run this command**.
[Step-by-step procedure, data provenance, every parameter, model explanation,
loss monitoring, GPU memory, resume and final-checkpoint inspection](docs/POLICIES.md).
No Diffusion run or simulator evaluation starts as part of ACT preparation.

## 6. Evaluation

After you generate a recording, you can replay its actual actions in a fresh
`leisaac-sim` terminal activated as in step 1. Replace the capitalized path parts
with a successful episode from your run’s `summary.json`:

```bash
python scripts/so101.py replay --episode datasets/generated_demos/RUN_ID/successful/EPISODE_DIRECTORY --output outputs/replay_manual
```

Replay sends recorded targets to real simulated actuators and the independent
evaluator. It does not write recorded robot/cube states into the scene. It stops
at independent success, so the final recorded settling tail may be omitted.

**For a future trained policy**, use two terminals: the learning environment
serves the real checkpoint; the simulation environment runs the shared task.
There is no trained checkpoint to run today.

Terminal A, `so101-il-py312`; add `--execute` only when this checkpoint exists:

```bash
python scripts/so101.py evaluate serve --policy act --checkpoint checkpoints/act_seed17/checkpoints/100000/pretrained_model --socket /tmp/so101-act.sock
```

Terminal B, `leisaac-sim`, after the real server prints `[READY]`:

```bash
python scripts/so101.py evaluate run --socket /tmp/so101-act.sock --manifest outputs/demonstration_protocol_v1/validation.json --split validation --output outputs/evaluation/act_seed17_validation
```

Use the same declared cases, reset, physics, 20 Hz action interface and evaluator
for expert, ACT and Diffusion. Report all failures and stage/full-task success.
[Inference details and fair comparison](docs/POLICIES.md#9-future-evaluation-and-comparison).

## Project status

This section is the current status and approval authority. Historical logs do
not override it.

| Established by evidence | Still unverified / not approved |
| --- | --- |
| Sequential reference pickup, lift, handover, placement and automatic reset | Broad-distribution reliability and hardware transfer |
| Coordinated expert: nominal reference and 15 owner-recorded, visually accepted varied successes | Reliability outside the accepted starting cases |
| Approved frozen overhead + two wrist cameras; owner accepted scene/HTML workflow and guide fix | Physical camera-mount registration and hardware calibration |
| Owner accepted 15 seed43/seed44 demonstrations; converted ACT data and recorded-data checks prepared | Training usefulness, generalization and hardware transfer |
| Official ACT parser/reader/model preflight and manual training procedure; matched-v1 preserved | Optimizer updates, trained inference and learned task success |

The owner now controls dataset generation through `scripts/collect_demos.py`.
The former 12-case pilot and the rejected `coordinated_pilot_10` batch and its
outputs were deleted with explicit approval. Reference benchmark evidence,
source/configurations, approved cameras and recovery tags are preserved.

For this milestone the assistant is authorized to validate and convert only the
15 owner-approved recordings, prepare editable ACT commands and run focused
configuration/reader checks. Training is **user-controlled**: the assistant must
not start updates, new collection, Diffusion or evaluation. Acceptance is recorded
at `outputs/act15_preparation/acceptance.json`; conversion provenance is inside
the prepared dataset. No new physical replay or independent visual acceptance is
claimed. The earlier 100/20/100 protocol remains historical context. Cameras,
scene, controller, physics, evaluator, environment locks and action semantics are
unchanged by this preparation.

## Code, detailed guides and historical records

After the six README steps, open only the guide needed for your question:

| Guide | Purpose |
| --- | --- |
| [Simulation and expert](docs/SIMULATION.md) | Environment locks, USD assets, camera transforms, FK/IK, collision checks, trajectories and physical evaluation |
| [Demonstrations](docs/DATASET.md) | Dataset design, exact record fields, acceptance, conversion and measured pilot results |
| [Policies and evaluation](docs/POLICIES.md) | Official ACT/Diffusion recipe, training/resume, inference and comparison limits |
| [Code map](docs/ARCHITECTURE.md) | Every maintained file/API, output retention and completed consolidation |

Core flow: `so101.py` → `run_manipulation.py` → shared `benchmark_env.py` and
`benchmark_core.py`. Model/planner compute expert motion; dataset and policy
modules record, validate, convert and serve actions. Imported modules are not
additional commands you need to run.

Historical engineering, failed-policy reviews, pilot report and old verification
logs live separately under [docs/archive](docs/archive). They preserve evidence;
they are not a second operating procedure.

## Outputs and repository integration

Source/configs/tests are versioned. Raw data, weights and generated outputs stay
local. Superseded pilot recordings and their generated reviews/reports were
deleted with the owner's explicit approval. Preserve the reference benchmark,
camera-approval evidence, source/configurations, Git history and recovery tags.
New user-generated runs live under `datasets/generated_demos/` with successful
and failed recordings separated. The 15 episodes in step 5 have owner acceptance;
other runs require their own acceptance.

The current tree has **42 tracked files, four active detailed guides, 14 Python
scripts, four configurations and seven tests** after adding this one collector.
The sole branch is `main`; no new branch was created. The historical
[consolidation record](docs/ARCHITECTURE.md#consolidation-record) describes the
previous eight-file retirement and its then-current counts. Existing recovery
tags are preserved.
