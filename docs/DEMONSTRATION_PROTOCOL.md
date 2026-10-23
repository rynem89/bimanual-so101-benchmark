# Provisional demonstration and evaluation protocol

This is the **design and rationale**, not the current execution log or permission
record. [STATUS](STATUS.md) is authoritative for approvals;
[PILOT_RESULTS](PILOT_RESULTS.md) contains the completed 12-attempt physical pilot.
The pilot is consumed; its human review is pending. Final collection and training
are not authorized. The original protocol JSON bytes remain unchanged for provenance.

## Purpose and research basis

The initial comparison asks whether official ACT and Diffusion Policy can learn
the same narrow bimanual task from the same physically verified deterministic
expert. The existing IK/contact controller is the demonstrator. Replanning each
initial object pose preserves task geometry and permits independent success
checks; shifting old action/image labels would not.

The [ACT project](https://tonyzhaozh.github.io/aloha/) uses chunked actions and
multiple fixed/wrist views. [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
uses conditional denoising of action sequences with receding-horizon execution.
These motivate a controlled comparison; neither publication establishes an
adequate dataset size or a winning method for this much longer SO-101 task.
[LeRobot's official learning workflow](https://huggingface.co/docs/lerobot/v0.6.1/il_robots)
provides the maintained policies, dataset reader and trainer used here.

[Isaac Lab 2.3 Mimic](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/imitation-learning/teleop_imitation.html)
requires task-specific pose/action adapters, object references and subtask/order
signals. Installing/replacing our proven demonstrator would add a new unvalidated
layer. No Mimic/SkillGen installation or trajectory augmentation is proposed in
the primary comparison. This is a defensible initial protocol, not a claim of
universally best practice. Sources and installed APIs were checked 2026-10-23.

## Attempt counts and splits

The active machine-readable design is
[demonstration_protocol.json](../configs/demonstration_protocol.json).
Counts refer to **attempted cases**, never a success quota. Failed starts, reset
faults, physical failures and image rejections stay in the denominator.

| Partition | Attempts | Purpose / acceptance requirement |
| --- | ---: | --- |
| Pilot | 12, completed | Eight support corners; nominal and nominal yaw ±1°; inward centre. Excluded from training. All 12 require automatic and human acceptance before final collection. |
| Train | 100 proposed | Fresh stratified poses, at least 95 accepted. Only accepted episodes enter optimization; report actual N. |
| Validation | 20 proposed | Separate poses, at least 19 accepted expert recordings. Never optimize on them; all declared poses count in policy validation. |
| Held-out | 100 proposed | Evaluation-only poses, identical across methods/seeds. No training recordings or tuning on outcomes. |
| Stress | 20 proposed | Separate wider, unverified evaluation support; report separately from primary results. |

Nested 25/50/100 accepted train subsets could later form learning curves, but
subsets/augmentation/new experiments need their own authorization. The historical
32/8/30 `task_distribution.json` is not this protocol or its evaluation set.

## Randomization and controlled state

Primary cube centre, world metres: x `[−0.350,−0.330]`, y `[−0.052,−0.044]`;
yaw `[−6°,6°]`. Reset spawns z=0.025 m and lets the cube settle near z=0.020 m.
Non-pilot partitions use independently seeded Latin-hypercube strata, distinct
case IDs/seeds and serialized exact poses. The pilot's nominal y=−0.040 m points
are deliberate regression checks outside the inward support.

Stress x `[−0.355,−0.325]`, y `[−0.055,−0.040]`, yaw ±8° is unverified. Bin, cube
size/mass/friction, bases, lights, cameras, gains and initial HOME remain fixed.
Initial joint jitter is **zero**: arbitrary perturbations could introduce new
inter-arm collisions or invalidate the initial path. A randomized seed without
a changed physical state is not added trajectory diversity.

All 12 selected pilot points physically succeeded; this does not prove every
interior/stress pose or establish an iid success probability. Do not silently
narrow distributions, replace failed cases or retry to achieve a target N.

## Expert strategy and augmentation

For each declared pose the existing expert resets, solves bounded five-joint IK,
checks intermediate motion including payload/opposite arm, and executes the full
task under actual physics. Independent contact/object evaluation, not completion
of commanded phases, establishes success. [Controller details](CONTROLLER.md)
explain the exact algorithms.

There is one validated strategy per pose. Varied IK paths provide initial
trajectory diversity. Identical nominal repeats do not cover new states. No
teleoperation adapter is implemented. Joint noise, time warps, mirror flips,
contact-segment stitching and relocating objects in saved images can invalidate
actions or bimanual timing; none is used. Image augmentation is disabled in the
primary recipe. A later common augmentation/recovery ablation must preserve source
case splits, be physically/data validated and receive separate approval.

## Data contract and quality

The three frozen RGB streams and 24-value q/qd state precede each 12-value absolute
position action. That exact float32 target is held for three 60 Hz ticks: 20 Hz
complete transitions. [DATASET](DATASET.md) specifies every stored array, capture
proof, partial terminal row, human review and conversion procedure.

Automatic QA requires independent full-task success, zero forbidden contacts,
verified reset, authored joint/action limits, speed ≤10 rad/s, effort ≤10 Nm,
stable slip ≤5 mm /10°, correct camera matrices/parents, matching physics clocks,
monotonic frame IDs and complete images. It decodes every RGB, checks contrast,
flags >60 identical frames, and requires relevant grasp/transfer/release phases.
Expected length is 1,200–6,000 complete rows. Preserve exact rejection reasons.
Human acceptance additionally checks all three streams, cube/gripper visibility,
full motion and synchronization, bound to the complete raw byte digest.
A 12-pixel sampled cube-ROI visibility proxy is a review guide, not segmentation
or human acceptance. See [actual pilot checks and limitations](PILOT_RESULTS.md).

## Split isolation and fair comparison

Partition by complete source case/trajectory, never random frames. Augmented
siblings, if approved later, remain with their parent. Held-out/stress cases never
become demonstrations. Conversion accepts one reviewed partition per root and
retains attempted/rejected counts and provenance. Training statistics come only
from the converted **train** root.

Primary proposal: same accepted train IDs, images, q/qd/actions, seeds 17/29/43,
batch 8, 100k updates and 10k checkpoint cadence. Match 1 observation, 64 predicted
targets and 8 executed targets (0.4 s per query). Keep official architectures,
normalization, ImageNet ResNet18 initialization and optimizer/scheduler presets;
equal updates do not equal equal compute. Diffusion masks padded target loss and
drops zero tail anchors to match ACT's sampled anchors. No resize/crop/photometric
augmentation is silently added. [POLICIES](POLICIES.md) documents verified config
fields, inference and the still-unimplemented separate-root offline loss loop.

Use fixed final checkpoints and separate 20-case physical validation for diagnosis;
freeze the recipe before exposing 100 held-out cases. Expert and policies share
reset, 20 Hz action interface, physics and independent evaluator, with all declared
failures counted. Report stage/full-task success, collisions, resets, slip, timing,
inference latency and GPU use, not just training loss. A future VLA uses these same
splits/observations/actions, with any pretrained-data/compute advantages disclosed.
It is not implemented or authorized here.

## Measured resource budget and next decision

The pilot measured mean 149.15 s task time, 658.77 s process wall, 4.53 complete
recording rows per wall second and 0.9352 GB raw per attempt. Current lossless PNG
collection cost, projected conditionally on the same workflow/distribution:

| Attempts | Raw storage (decimal GB) | Generation wall hours |
| --- | ---: | ---: |
| 100 | 93.52 | 18.30 |
| 500 | 467.60 | 91.50 |
| 1,000 | 935.20 | 182.99 |

Copies for conversion, review media and checkpoints require additional space.
120 proposed train/validation attempts imply ≈112.22 GB /21.96 h before QA/review.
These measurements supersede earlier pre-pilot estimates of 5–10 minutes/attempt.
Use one simulator, three cameras, CPU workers 8 and BLAS/OMP 1; preserve collision
checking and physics fidelity. Lossless writes/rendering are profiling candidates,
not permission to change encoding, geometry, drivers or OS.

Training throughput/memory remain unmeasured. At 1/2/5 updates/s, 100k steps would
take 27.8/13.9/5.6 h per run; these are arithmetic scenarios, not RTX 5090 results.
Do not run training concurrently with collection. Any timing experiment needs
explicit optimizer authorization. GPU availability/config parsing is not a memory
or convergence test.

Next: owner reviews the pilot and consolidation, then separately decides final
protocol/collection. [DATASET](DATASET.md) provides executable future steps.
