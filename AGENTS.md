# Codex operating contract — bimanual SO-101 benchmark

## Start here

Read `README.md` and `docs/STATUS.md` at the start of each task, then the relevant
linked guide. `docs/ROBOTICS_CODEX_HANDOFF.md` and `docs/POLICY_REVIEW.md` are
historical evidence, not current commands or work orders. Trust repository code
and measured results; report conflicts and update the current authority.

The deterministic left pickup → mid-air right handover → basket placement task
is established. Do not restart geometry/TCP/IK milestones or the consumed pilot.
Next work is the repository-consolidation brief and its verification, followed
by owner review. Human pilot acceptance remains pending; no accepted training
corpus or trained model exists in the reset pipeline.

## Preserve the proven benchmark

- Keep pinned `leisaac-sim` and `so101-il-py312` environments. Each shell needs its
  own activation; use `docs/GETTING_STARTED.md`. No installs/upgrades, NVIDIA,
  CUDA, OS, Isaac Sim/Lab or LeIsaac changes without explicit consent.
- Preserve `configs/policy_cameras.json` and `configs/demonstration_protocol.json`
  byte-for-byte. Their hashes are in `docs/STATUS.md`. Do not alter controller
  algorithms, task physics/evaluator or action/observation semantics for cleanup.
- No physical success claims from compilation, transforms, commanded phases or
  offline planning. Use actual independent simulator evaluation and contact data.
- Never collect demonstrations or execute optimizer updates without explicit user
  authorization for that session and required approval records. The 12-attempt
  pilot authorization is consumed. Preserve failures and declared denominators.
- Cameras require attachment/timing checks and human image approval. Demonstrations
  require automatic QA, independent physics evaluation and human quality review
  before conversion/training. Diagnostic snapshots are not training data.

## Work autonomously within scope

Inspect, implement, test and diagnose coherent milestones. Use the simple
`scripts/so101.py` interface and existing modules; do not add a wrapper per action
or invent a new framework. Use descriptive names, frame/unit docstrings and
comments for non-obvious mechanics. Keep help lightweight and GUI the default.

Pause only for required human visual inspection, missing physical facts, genuinely
blocked work, destructive/privileged changes or significant architecture changes.
Batch simulator observation into one consolidated checkpoint where practical.
State the exact command, observation and narrowest missing evidence. Continue
independent safe work while awaiting optional clarification.

## Git and evidence

Inspect status, branches, commits, tags, ignores and file inventory first. Work
on `feat/repository-consolidation` from preserved `a5cdc18` for this milestone.
Main is protected. Never reset/clean/force-push or overwrite uncommitted work.
Commit coherent verified changes locally; no merge/push without new authorization.
Present exact deletion candidates and obtain approval before removing tracked or
ambiguous files, branches, tags, datasets, checkpoints or evidence. See
`docs/REPOSITORY_AUDIT.md`; none of its proposals are authorized deletions.

Never commit credentials, environments, binaries, generated outputs/logs, raw
images/datasets or weights. Preserve `.gitignore` and local evidence. Source/docs
publication never authorizes artifact upload or collection/training.

## Documentation and verification

README is the concise entrance; `docs/ARCHITECTURE.md` is the complete maintained
file inventory and public-function map. Keep both accurate. Current gates belong
only in `docs/STATUS.md`, environment locks in `docs/GETTING_STARTED.md`, measured
pilot results in `docs/PILOT_RESULTS.md`, regression commands/results in
`docs/VERIFICATION.md`. Do not append long chronological logs to the README or
reactivate historical instructions.

Run applicable CPU tests before expensive physical checks. Fail when prerequisites
fail. Verify CLI help/defaults, dry runs, collection/training gates, official
LeRobot config parsing, doc links and preserved hashes. Run limited expert and
recorded-action physics regressions; do not regenerate the pilot or start training.
Report proven behaviour, unverified limits, changed files, exact test evidence,
Git state and the next human decision. Link the architecture/controller guides
for learning rather than repeating a long tutorial in the final response.
