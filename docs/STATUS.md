# Current project status and approval gates

Date: 2026-10-23. This file is the current-status authority; historical documents
do not override it. Read [pilot results](PILOT_RESULTS.md) for measured task/data
results and [consolidation verification](VERIFICATION.md) for interface regression.

## What is established

- The deterministic full-task expert has physically completed pickup, lift,
  mid-air handover, basket placement and retreat with independent contact/task
  evaluation. Repeated-episode reset is implemented and was verified earlier.
- The centered overhead and unchanged two wrist cameras received user visual
  approval on 2026-10-23. Their parameters are frozen.
- Exactly 12 authorized pilot attempts completed in actual Isaac physics, with
  12 full-task successes and 12 automatic QA passes; there were no hidden retries.
- Three preselected recordings completed the task through the shared 20 Hz
  physical replay interface. Nine were not physically replayed. Replay stopped
  at independent success and omitted the recorded final settling tail.

## Repository consolidation

The unified CLI, stationary camera inspection, named official-policy preflight,
connected guides and complete file/cleanup inventory are implemented locally.
CPU checks and limited expert/replay physics regressions passed; exact conditions
and remaining UI checks are in [VERIFICATION](VERIFICATION.md). The owner confirmed
the stationary camera viewer and nominal raw-review HTML are good, but reported
`InspectionGuides` flickering/collapsing in the Stage inspector. Repeated USD
visibility edits have been removed; a focused human recheck of that fix remains.
The owner also requested cleanup: 73 verified generated cache/preview files were
removed (776,310,957 bytes); pilot data, benchmark evidence, model-backbone cache
and tracked files remain intact. No tracked-file/branch deletion or publication
is included.

## What is pending

**Human acceptance of all 12 pilot videos/data is pending.** Automatic QA is not
human approval. There is no accepted learning corpus and no trained checkpoint in
the reset pipeline. Broad-distribution reliability, model optimization, learned
physical success and hardware transfer are unverified.

The active protocol is provisionally tested, not frozen for final collection.
Its proposed next scale is 100 train /20 validation attempts after separate
approval. The purpose-selected pilot does not estimate general success probability.

## Permissions for this milestone

Authorized: source/configuration audit, safe reversible CLI/documentation
consolidation, meaningful local commits on `feat/repository-consolidation`, CPU
tests, a stationary scene/camera diagnostic, and limited expert/replay physical
regression without demonstration recording. Ask for one consolidated human visual
checkpoint when the new viewing workflow is ready.

Additional permission now covers inspected, disposable generated cache/preview
cleanup only. See [the exact cleanup record and remaining candidates](REPOSITORY_AUDIT.md#generated-output-cleanup-and-retention).

Not authorized: more demonstration collection, dataset augmentation, acceptance on
behalf of the user, final protocol changes, optimizer updates, training, VLA
experiments, dependency/driver/OS changes, tracked-file deletion, branch/tag
deletion, merging or pushing this cleanup branch. The previous push permission
was consumed by publishing `a5cdc18` on `feat/il-camera-dataset-v2`; the new brief
explicitly requires separate permission for publication.

Do not modify approved camera calibration, expert motion, physical contacts, task
evaluation or state/action semantics as part of consolidation. Preserve raw pilot
data and previous evidence. Historical collection approval files are not fresh
session authorization and must not be reused to collect more data.

## Authoritative files

| Fact | Authority |
| --- | --- |
| Current state / human permission gates | This file |
| Frozen camera parameters | [configs/policy_cameras.json](../configs/policy_cameras.json) |
| Active provisional sampling/quality design | [configs/demonstration_protocol.json](../configs/demonstration_protocol.json), explained in [protocol](DEMONSTRATION_PROTOCOL.md) |
| Environment versions and activation | [GETTING_STARTED.md](GETTING_STARTED.md#environment-locks), learning package lock and installed external checkouts |
| Actual 12-attempt results | [PILOT_RESULTS.md](PILOT_RESULTS.md); hash-bound raw/evaluator evidence remains local |
| Consolidation regression evidence | [VERIFICATION.md](VERIFICATION.md) |
| Future named policy recipe | [configs/policy_training.json](../configs/policy_training.json); proposed, not training authorization |

Frozen SHA-256 values:

```text
policy_cameras.json:
f8f7e56842602fbe040fea43b1cf1a51f75f56b3ad28357aae04f717c8e9fcbc
demonstration_protocol.json:
9ec7c3539c287e11bdc988eb52597a16001404d81ef43f8aad64fd0c6a4a3b3a
```

The protocol JSON's original `status` is a conservative proposal label, not an
execution log. Its bytes stay unchanged so existing manifest/approval/data hashes
remain valid. Current completion and permission state belong here.

## Recovery and next decision

Baseline: `a5cdc18`, already published on `feat/il-camera-dataset-v2`; local
recovery tag `reference/pre-repository-consolidation`. Existing expert/camera
recovery tags remain intact. Main is protected. See [repository audit](REPOSITORY_AUDIT.md)
for branch ancestry and proposed cleanup; no removal is authorized.

After consolidation regression and visual review, the owner reviews the code/docs.
Pilot acceptance and any final collection/training protocol are separate decisions.
