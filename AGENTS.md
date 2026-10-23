# Operating contract — bimanual SO-101 benchmark

## Start and scope

Read README.md first. It is the single starting point and current status/permission
authority. Follow its six-step sequence; use only the relevant linked guide.
`docs/archive/` holds historical evidence, never current commands or permission.

The deterministic pickup → mid-air handover → basket placement benchmark is
established. Preserve the 12 original pilot attempts, QA, three selected physical
replays, reference benchmark and approved cameras. Human acceptance of all pilot
data is still pending. No accepted training corpus or trained policy exists in
the reset pipeline. Do not restart completed geometry/control/pilot milestones.

Repository consolidation is complete. `main` is the sole remaining local/remote
branch and is protected from unrequested changes. The approved eight-file and
four-feature-branch cleanup is recorded in `docs/ARCHITECTURE.md#consolidation-record`;
that authorization does not cover future removals or unrelated work. No new branch,
framework, wrapper, documentation file or research feature is requested. Never
reset, clean, force-push, delete other files/refs or overwrite uncommitted work.

## Preserve semantics and environment

Keep approved camera/protocol JSON bytes, controller, physics, evaluator, recorded
state/action semantics and raw evidence unchanged. Runtime reads the existing
LeIsaac USD. No dependency, environment, driver, CUDA or OS changes. Activate the
correct existing Conda environment separately in each terminal; versions and
paths are in `docs/SIMULATION.md#environment-locks`.

No collection, augmentation or optimizer updates without explicit authorization
for that session and the existing data/approval gates. Historical approvals cannot
be reused. Diagnostic snapshots are not demonstrations. Human scene/HTML approval
is not acceptance of the pilot data. Never claim physical success from IK,
geometry, commanded stages or compilation; use independent measured evaluation.

## Keep the repository understandable

README owns ordered operating instructions and current status.
`docs/SIMULATION.md` owns environment/assets/cameras and expert mathematics;
`docs/DATASET.md` owns demonstration design, format, review and measured results;
`docs/POLICIES.md` owns official learning/inference; `docs/ARCHITECTURE.md` owns the
complete maintained file/API map, outputs and integration scope. Update existing
guides instead of adding documents or chronological logs. Append test evidence
only to the existing `docs/archive/VERIFICATION.md` when useful.

Keep `so101.py` as the supported interface and GUI as the simulation default.
Preserve validated implementations; no stylistic rewrites of working algorithms.
Check imports, subprocess paths, tests and active configurations before retirement.
Keep retired experiments recoverable from Git history rather than copying scripts
into an archive. Never commit secrets, data, weights, environments or outputs.

Run focused checks for the changed layer. Docs need links/anchors, command syntax
and preservation checks; unused-file retirement needs dependency and CLI checks.
Do not rerun broad CPU/physics suites when runtime behavior is unchanged. Use a
physical regression and one narrow human checkpoint only when relevant behavior
changed. Report actual versus prepared counts, exact operations, test limits,
commit/branch state and any required approval. Do not mark pending work complete.
