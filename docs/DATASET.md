# Demonstrations: design, recording, review and conversion

[Start with README step 4](../README.md#4-demonstrations) · [Policies](POLICIES.md)

## Contents

- [What data exists](#what-data-exists)
- [Inspect and view the preserved pilot](#inspect-and-view-the-preserved-pilot)
- [Raw attempt layout and fields](#raw-attempt-layout)
- [Demonstration design](#demonstration-design)
- [Planning and collection gates](#plan-and-collect-later--separate-authorization-required)
- [QA and human review](#automatic-qa-human-review-and-rejection)
- [Physical replay](#physical-action-replay)
- [LeRobot conversion](#convert-accepted-data-and-use-official-visualization)
- [Measured pilot results](#measured-pilot-results)

## What data exists

The preserved raw pilot is `datasets/protocol_v1_pilot/`: 12 attempted episodes,
all automatic QA passes and physical successes, **not yet human accepted**. Pilot
metadata uses `split: validation` because it is excluded from training; its
manifest's `partition: pilot` distinguishes it from the future 20-case validation
partition. Do not rename or relabel it into training data.

Future train/validation roots would hold fresh, separately authorized attempts.
Held-out/stress manifests define evaluation cases, not expert training recordings.
Converted LeRobot train/validation roots do not yet exist in this reset pipeline.
Ignored local datasets/reports/media are not available in a GitHub-only checkout;
tracked [pilot results](#measured-pilot-results) contain the remote-review summary.

## Inspect and view the preserved pilot

Use `leisaac-sim`; these operations do not collect or accept data:

```bash
python scripts/so101.py dataset inspect \
  --raw datasets/protocol_v1_pilot \
  --quality-report outputs/protocol_v1_pilot_qa/quality.json \
  --ledger outputs/protocol_v1_pilot_run/attempts.jsonl
python scripts/so101.py dataset review \
  --episode datasets/protocol_v1_pilot/pilot_008_20261023T171324135324Z \
  --output outputs/reviews/nominal.html
```

`inspect` reports completed/incomplete raw directories, per-case split, independent
physical result, supplied QA/human status and process-start counts. It reads
existing metadata, not a fresh byte validation. The ledger can contain attempts
that never produced an episode directory; do not equate directory count with
all attempted cases. Without a supplied QA report, QA is **unknown**, not passed.

Open the printed HTML address locally. One slider drives all three original RGB
frames and the aligned measured q, qd and **next** action target. Play uses a
20 Hz timeline; browser/I/O speed can limit display speed. It copies no images,
starts no simulation and grants no human acceptance. Choose a new HTML output
outside the raw episode, so review does not change its digest.

Existing consolidated pilot review: open
`outputs/protocol_v1_pilot_report/index.html` locally. It links the 12 synchronized
videos, phase sheets, full PNGs, JSON report and CSV. The three-camera videos are
review derivatives; original PNGs remain the data source.

## Raw attempt layout

```text
<raw-root>/<case>_<UTC timestamp>/
  attempt.json          # in_progress/completed marker and metadata
  episode.json          # finalized attempt, success/failure and provenance
  trajectory.npz        # complete, paired policy transitions
  captures.jsonl        # per-capture timing, sensor IDs and attachment evidence
  privileged.json       # cube/contact diagnostics; excluded from policy features
  overview/000000.png
  left_wrist/000000.png
  right_wrist/000000.png
  ...
```

Interrupted attempts may have `trajectory.partial.npz` and no `episode.json`.
Preserve them as failures/incomplete attempts. `EpisodeWriter` checkpoints every
200 complete rows; completed attempts remove only their own temporary partial
trajectory after writing final data. Consolidation does not delete raw files.

| `trajectory.npz` array | Shape / meaning |
| --- | --- |
| `state` | float32 `[N,24]`: left q[6], right q[6], left qd[6], right qd[6]; radians /rad·s⁻¹ |
| `action` | float32 `[N,12]`: absolute position targets, left then right, radians |
| `action_60hz` | `[N,3,12]`: actual target submitted on each of the next three physics ticks; exactly equals the repeated row action |
| `time_s` | `[N]` absolute physics time, seconds; 0.05 s row spacing |
| `physics_step` | `[N]` integer absolute physics ticks, spacing 3 |
| `image_step` | `[N,3]` capture tick for each camera; equals the row physics step |

Each arm's order is `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll`, `gripper`. State is **not** TCP/object pose. Action is **not** joint
velocity, torque, delta pose or a future measured q.

At boundary k: render all three cameras at measured state k without advancing
physics, save `RGB(k), state(k), action(k)`, then apply exactly that target for
three ticks. These are pre-action observations with a forward action label.
A terminal hold with only one/two ticks is excluded from N. Its extra PNG/capture
and terminal state remain diagnostic evidence; not every saved PNG is a paired
training row.

`episode.json` binds schema `so101-demonstration-1.1`, task, case/seed/initial pose,
source revision/hashes/dirty flag, split, protocol/camera/one-case-manifest hashes,
authored joint bounds, initial camera calibration, fps/image shape, controller
settings, complete frame count, partial terminal ticks, terminal state, write time,
physical success and independent evaluation. `captures.jsonl` contains row index,
physics step, expert phase, sensor frame IDs and measured per-camera/parent poses
and intrinsics. `privileged.json` records cube state/pad/environment forces only
for diagnosis/QA. Policies must never receive these oracle labels.

## Demonstration design

### Purpose and research basis

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

### Attempt counts and splits

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
subsets/augmentation/new experiments need their own authorization. The former 32/8/30 experiment is historical; it is not this protocol or its evaluation set.

### Randomization and controlled state

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

### Expert strategy and augmentation

For each declared pose the existing expert resets, solves bounded five-joint IK,
checks intermediate motion including payload/opposite arm, and executes the full
task under actual physics. Independent contact/object evaluation, not completion
of commanded phases, establishes success. [Expert details](SIMULATION.md#expert-ik-and-motion)
explain the exact algorithms.

There is one validated strategy per pose. Varied IK paths provide initial
trajectory diversity. Identical nominal repeats do not cover new states. No
teleoperation adapter is implemented. Joint noise, time warps, mirror flips,
contact-segment stitching and relocating objects in saved images can invalidate
actions or bimanual timing; none is used. Image augmentation is disabled in the
primary recipe. A later common augmentation/recovery ablation must preserve source
case splits, be physically/data validated and receive separate approval.

### Split isolation and fair comparison

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


## Plan and collect later — separate authorization required

The existing planner is deterministic and does not start Isaac:

```bash
python scripts/so101.py collect plan --output outputs/protocol_review_plan
python scripts/so101.py collect run --split train \
  --manifest outputs/protocol_review_plan/train.json \
  --raw datasets/protocol_train --output outputs/protocol_train_run
```

The second command prints launches only. `plan` requires a fresh directory and
writes pilot/train/validation/heldout/stress manifests plus `plan.json` hashes.
Planning does not authorize using any manifest. Do not execute the consumed pilot
again merely because a new plan exists.

Actual execution additionally requires separate user authorization for this
session, `--execute --authorize-collection --approval <reviewed-file>`, and, for
train/validation, `--pilot-report <human-accepted-pilot-QA>/quality.json`.
The approval record names `approved: true`, reviewer/date and the exact
`protocol_sha256`, `camera_config_sha256`, `manifest_sha256`. It records genuine
approval, not a way to bypass it. No accepted pilot report exists yet.

The maintained collector revalidates the reproducible manifest, rejects held-out
recording, requires clean committed source, and writes append-only started/finished
ledger rows and process logs. Each case is attempted once. `--resume` skips every
previously **started** case, including failed/interrupted starts; it is not retry.
The high-level `--split` must match the manifest's actual partition. No wrapper
weakens the collector's existing gates.

## Automatic QA, human review and rejection

The existing QA command can be rerun on existing data into a fresh output:

```bash
python scripts/so101.py dataset validate \
  --manifest outputs/demonstration_protocol_v1/pilot.json \
  --raw datasets/protocol_v1_pilot --output outputs/pilot_qa_review \
  --ledger outputs/protocol_v1_pilot_run/attempts.jsonl --review-video
```

It checks all images and complete transitions, camera attachment/intrinsics/frame
continuity, source/config hashes, reset, physical task/contact/slip limits and
ledger completion. It writes `quality.json`, phase JPGs, optional synchronized
MP4s and `human_review_template.json`. Failed/incomplete/missing cases retain
precise reasons. Geometry/IK feasibility is not physical completion.

Review all streams/motion, relevant cube/gripper interactions and timing. Fill a
copy of the human template only after actual review: reviewer/date/protocol hash,
each raw digest, accept/reject, notes, and six checks (`three_streams`,
`cube_at_grasp`, `cube_at_handover`, `cube_at_release`, `full_motion`,
`synchronization`). Repeat QA into another fresh directory with
`--human-review <completed-review.json>`. Acceptance is bound to every raw byte;
changed data requires review again. Pilot requires 12/12 accepted; proposed final
partitions require ≥95% accepted, with failures still reported. Automatic success
alone never sets `ready_for_conversion` true.


### Exact automatic rejection limits

Require independent full-task success, zero forbidden contacts, a verified reset,
finite authored joint/action limits, speed ≤10 rad/s, effort ≤10 Nm and stable
slip ≤5 mm /10°. Require 1,200–6,000 complete rows, all required manipulation
phases, correct image dimensions/calibration, monotonic camera IDs and matching
physics boundaries. Decode every RGB, check contrast and flag >60 identical frames.
Store the precise reason for each rejected attempt. Human image/motion acceptance
remains separate; a nonblank image or sampled colour ROI is not sufficient.

## Physical action replay

```bash
python scripts/so101.py replay \
  --episode datasets/protocol_v1_pilot/pilot_008_20261023T171324135324Z \
  --output outputs/replay_nominal_manual
```

GUI is default. The interface derives a one-case reset manifest from the original
recording, refuses to write inside raw data or overwrite a replay output, then
passes the original targets through `BenchmarkEnv.step`: actual actuators,
contacts, 20 Hz holds and independent evaluation. It does not write recorded
physical states into the simulator. Format QA and physical replay are different
tests. Replay terminates at first independent success, so the terminal settling
tail may be omitted; pointwise trajectory equivalence is not asserted.

## Convert accepted data and use official visualization

Switch to **so101-il-py312**. Future accepted train conversion, preview by default:

```bash
python scripts/so101.py dataset convert --raw datasets/protocol_train \
  --quality-report outputs/train_qa_accepted/quality.json \
  --output datasets/lerobot_train
```

Add `--execute` only after actual data acceptance/authorization. The converter
checks QA/root/camera identity, exact attempt inventory and unchanged full raw
digests. It converts accepted episodes of one split only, using official
`LeRobotDataset.create/add_frame/save_episode/finalize`, then reloads the reader
and checks shapes. Schema is LeRobot v3 with image-backed storage and a
`conversion.json` preserving accepted provenance **and rejected attempt counts**.
No videos, frame resize or privileged labels enter training. Convert a separately
accepted validation partition to its own root; never merge it into train stats.

When a converted dataset exists, the installed official viewer can display
images/state/actions on its native timeline:

```bash
python scripts/so101.py dataset viz --dataset datasets/lerobot_train --episode-index 0
```

Add `--execute` to open `lerobot.scripts.lerobot_dataset_viz` (Rerun). Its CLI was
checked against the installed 0.6.1 implementation; no accepted real dataset was
converted/opened in this milestone. Do not describe command preparation as proof
of end-to-end real-data conversion, visualization or learning.

## Measured pilot results

All 12 declared expert attempts ran once in actual visible Isaac physics on
committed source `877f9b1`: **12 successes, zero physical failures, 12 automatic
QA passes, zero automatic rejections, 12 human acceptances pending**. No failed
attempt was regenerated. All four independent task stages passed in every case.
No forbidden-contact violation, failed grasp, unintentional drop, unexpected reset
or incomplete attempt was reported. This is evidence for these purpose-selected
cases, not an iid estimate of general reliability.

### Every declared attempt

Positions are requested settled world XY in metres; seeds are controlled per case. Actual settled position/quaternion, initial pose errors, independent stage flags, contacts, failure/reset information, timing, source hashes and data-QA/replay evidence are in the JSON. These are designed corners/reference poses, not 12 uniform random draws.

| Case | Seed | Cube XY, m | Yaw | Physics stages | Sim s | Process wall s | Raw GB | Auto QA |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |
| pilot_000 | 2026102300 | [-0.35, -0.052] | -6° | pickup/lift/handover/place PASS | 160.100 | 697.98 | 1.007 | PASS |
| pilot_001 | 2026102301 | [-0.35, -0.052] | +6° | pickup/lift/handover/place PASS | 152.767 | 685.48 | 0.961 | PASS |
| pilot_002 | 2026102302 | [-0.35, -0.044] | -6° | pickup/lift/handover/place PASS | 155.783 | 691.50 | 0.979 | PASS |
| pilot_003 | 2026102303 | [-0.35, -0.044] | +6° | pickup/lift/handover/place PASS | 151.850 | 678.49 | 0.955 | PASS |
| pilot_004 | 2026102304 | [-0.33, -0.052] | -6° | pickup/lift/handover/place PASS | 148.500 | 657.45 | 0.936 | PASS |
| pilot_005 | 2026102305 | [-0.33, -0.052] | +6° | pickup/lift/handover/place PASS | 145.333 | 650.48 | 0.912 | PASS |
| pilot_006 | 2026102306 | [-0.33, -0.044] | -6° | pickup/lift/handover/place PASS | 146.650 | 652.47 | 0.914 | PASS |
| pilot_007 | 2026102307 | [-0.33, -0.044] | +6° | pickup/lift/handover/place PASS | 145.267 | 648.47 | 0.910 | PASS |
| pilot_008 | 2026102308 | [-0.34, -0.04] | +0° | pickup/lift/handover/place PASS | 153.183 | 661.46 | 0.959 | PASS |
| pilot_009 | 2026102309 | [-0.34, -0.04] | -1° | pickup/lift/handover/place PASS | 138.033 | 582.46 | 0.859 | PASS |
| pilot_010 | 2026102310 | [-0.34, -0.04] | +1° | pickup/lift/handover/place PASS | 146.200 | 648.50 | 0.911 | PASS |
| pilot_011 | 2026102311 | [-0.34, -0.048] | +0° | pickup/lift/handover/place PASS | 146.100 | 650.47 | 0.920 | PASS |

Failure breakdown: startup/reset 0; physical grasp/lift/transfer/placement 0; forbidden-contact violations 0; incomplete attempts 0; automated recording-quality rejections 0. Human acceptance is pending for all 12; pending review is separate from a failed demonstration. There is no accepted training dataset and the pilot is excluded from future training partitions.

### Timing, throughput and storage

- Task duration: mean **149.147 s**, range **138.033–160.100 s**, excluding initial reset. Runner duration including reset is retained separately.
- Collection process wall time: **7905.22 s /2.196 h**, mean **658.77 s**, range **582.46–697.98 s**. This includes startup, planning pauses, simulation/rendering, PNG writes and shutdown.
- **35,792 complete 20 Hz observation/action rows**, **107,376 paired RGB frames**. Actual wall throughput **4.528 rows/s**, **5.465 demonstrations/hour**; the temporal dataset rate remains 20 Hz.
- 107,394 PNGs total: 107,376 paired frames plus 18 terminal diagnostic images in six attempts. Their 1–2 final physics ticks are excluded from complete action rows. No partial hold is mislabeled as a full 50 ms action.
- Raw data: **11,222,342,915 bytes /11.222 GB**, mean **0.935 GB** per completed demonstration. Sizes include lossless PNGs, arrays and diagnostic metadata.
- Measured aggregate timers: image writes **2,596.22 s**, viewport rendering **1,861.84 s**, camera capture/readback **869.86 s**, physics stepping **457.76 s**, planning including initial approach **1,663.53 s**. Reset time overlaps some physics/render timers; these are not independent additive totals. Collision planning is significant, but is not the sole wall-time cost.

| Completed demonstrations | Estimated raw GB | Observed-size range, GB | GUI generation hours at measured rate |
| ---: | ---: | ---: | ---: |
| 100 | 93.52 | 85.90–100.67 | 18.30 |
| 500 | 467.60 | 429.50–503.36 | 91.50 |
| 1,000 | 935.20 | 859.00–1006.72 | 182.99 |

GB means 10⁹ bytes. Projections are conditional on this narrow distribution/duration and current GUI/PNG workflow, not guarantees. They exclude converted copies, checkpoints and review media. A second full copy approximately doubles raw storage. The next proposed 100 train +20 validation attempts imply about **112.22 GB raw /21.96 h generation**, before QA/human review; the earlier 10–20 h estimate was optimistic.

### Synchronization and image usefulness

All 12 passed finite float32 state/action dimensions and authored joint bounds, absolute and relative clocks, exact held targets for all three 60 Hz ticks, advancing sensor frame IDs, complete RGB files, frozen intrinsics/extrinsics, required task phases and measured wrist attachment. Sensor attachment tolerance is 20 µm /rotation-matrix 2e-4. Each complete pre-action state/action row has the three corresponding images at the same physics tick. This is software simulation synchronization, not independent real-camera exposure measurement.

The report includes eight matched phase samples per case. A red-pixel proxy restricted to the projected 40 mm cube ROI found a cube-coloured region at least 12 px wide in at least one camera in all 96 samples. At initial HOME, only overhead sees the cube; its visible red ROI width is 15–28 px and some robot occlusion remains. Relevant grasp/transfer wrist samples have roughly 168–184 px red ROI width; the receiver wrist shows release at 103–105 px. These are colour/ROI measurements, not certified segmentation, full-object visibility or a quality score for every frame. Human review of complete video and original PNGs is still required. No camera parameters were changed.

Peak measured joint speed **2.4415 rad/s**, effort **1.5271 Nm**, and stable-grasp slip **1.1648 mm** remain within the existing benchmark criteria. These are simulation measurements, not physical-hardware safety certification.

### Recording format versus physical replay

All 12 recording formats passed QA. Separately, the preselected **corner pilot_000, nominal pilot_008 and −1° pilot_009** were executed through `BenchmarkEnv.step` using their recorded 12-target actions, actual reset/physics/actuators/gravity compensation and independent evaluation. No expert IK, planner or grip-feedback controller supplied replay actions. These diagnostic tests did not create new expert demonstrations.

| Replay | Physical stages | Replay sim s | Original sim s | Unused final recorded rows |
| --- | --- | ---: | ---: | ---: |
| pilot_000 | all four PASS | 157.267 | 160.100 | 56 |
| pilot_008 | all four PASS | 150.350 | 153.183 | 56 |
| pilot_009 | all four PASS | 135.200 | 138.033 | 56 |

**3/3 physical replay successes; the other nine are not physically replayed.** The intended policy interface terminates at the first independent success. It therefore omits 56 recorded settling-tail rows in each tested case; this is not playback of every stored row. Pointwise joint/cube-state equivalence was not measured. Only elapsed time and desired-target tracking metrics differ from the original aggregate evaluations; the target representation is float32, while original desired-pose calculations used higher precision. No physical failures were hidden or repaired. Replay logs/commands/results: `outputs/protocol_v1_pilot_replay/`.

The preserved nominal expert episode exactly matches the prior camera-regression reference independent metrics and complete transfer trace (`outputs/controller/overview_v3_comparison/full_20261023T142830953036Z.json`). Expert/model/planner/scene/physics files remain unchanged.


Retain the tested primary position/yaw support and fixed HOME. The measured
projections include startup, planning, GUI rendering and lossless PNG writing;
training throughput and memory are not measured. 100k updates at 1/2/5 updates/s
would take 27.8/13.9/5.6 h: arithmetic scenarios, not RTX 5090 benchmarks.
Do not run training concurrently with simulation collection. No collection,
augmentation, profiling experiment or optimization is authorized by this guide.

The full dated pilot report is preserved in
[the archive](archive/PILOT_RESULTS.md). The local report, raw data and QA remain
at their original paths, so existing review links and provenance stay valid.
