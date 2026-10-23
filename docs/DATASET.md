# Demonstration dataset procedure and field reference

[Protocol design](DEMONSTRATION_PROTOCOL.md) · [Actual pilot results](PILOT_RESULTS.md) ·
[Approval gates](STATUS.md) · [Official policies](POLICIES.md)

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
tracked [pilot results](PILOT_RESULTS.md) contain the remote-review summary.

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
