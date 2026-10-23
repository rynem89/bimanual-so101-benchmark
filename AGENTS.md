# Codex operating contract — bimanual SO-101 benchmark

Read `README.md` (if present) and `docs/ROBOTICS_CODEX_HANDOFF.md` at the start of each new task. Treat repository code and measured results as source of truth; report conflicts. Main mission: **left SO-101 picks a cube -> transfers it in mid-air to right SO-101 -> right places cube in bin** under deterministic control, auto-reset and objective evaluation **before AI training**. Preserve the pinned Isaac Lab / LeIsaac stack.

## Work autonomously, deliver milestones
- Take ownership of a requested milestone end-to-end: inspect, plan, implement, test, diagnose and iterate. Do not stop after an exploratory script or ask for approval after every small edit. Batch dependent low-risk code changes into coherent milestones.
- **Pause** for user action when: human simulator/visual inspection is required; missing physical facts prevent progress; a destructive/privileged operation or dependency/driver/OS change is contemplated; a proposed significant architecture change; or a nonrecoverable blocker. State exactly what the user should observe/do and the narrowest evidence needed. Do not claim physical grasp success from code compilation or proxy geometry.
- Execute tests autonomously whenever local environment and permissions allow. When GUI cannot be operated, produce a headless automated test plus an exact reproducible GUI command and ask for one human verification per milestone, not per small diagnostic.
- Avoid busywork, infinite retries and speculative broad rewrites. If blocked after a small number of evidence-based attempts, report failing layer and propose smallest next diagnostic.

## Git safety and hygiene
- First inspect `git status`, branch, commits, ignore patterns and file inventory. Treat `main` as protected. Work on a purpose-specific feature branch or Codex worktree from a verified baseline; account for existing uncommitted files. Do not reset, clean, force-push, delete branches or overwrite uncommitted work.
- Maintain a concise inventory distinguishing production code, tests, archived diagnostics, generated outputs and unknown files. **Do not blindly delete**: propose candidates with reasons; obtain user approval before removing anything tracked or ambiguous; remove untracked temporary artifacts only after inspection and with clear provenance. Prefer Git history for superseded diagnostics; archive only when genuinely needed, never to inflate files.
- Preserve working scripts until replacement passes an equivalent objective test. Inspect diffs and tests. Commit small, meaningful verified milestones on the feature branch, never push to `main` or merge without explicit user approval. Publishing/pushing remote changes needs user authorization.
- Never commit credentials, `.env` secrets, caches, venvs, Isaac binaries, USD giant assets, datasets, logs or generated temp outputs. Maintain `.gitignore`.

## Engineering and learning contract
- Minimal cohesive architecture; reusable scene, robot model, configuration, transforms, controllers, evaluation and tests only when needed. No unnecessary wrappers, frameworks, dependencies, notebooks, dozens of one-off probes or speculative files.
- Document each maintained source file in **`README.md`** in a succinct file inventory (purpose, entry point, inputs/outputs, run command, status). Update it when adding/removing files and at milestones. Record exact commands, results, expected output, failures and environment locks in `docs/PROGRESS.md` / handoff docs; create only if absent and useful.
- Code style: readable Python, typed APIs when helpful, concise meaningful comments/docstrings explaining frames, units, non-obvious mechanics, failure modes and external APIs. Avoid noise comments restating code. Explicitly distinguish meters/radians, world/base/TCP frames, actual measured vs desired state. Keep performance reasonable.
- Fail tests when prerequisites fail. No `PASS` based only on command issued, endpoints without path validation, stationary "disturbance", or visual claims from mere transforms. Record quantitative tolerances and independent checks.
- At milestone end provide: what works (proven), what doesn't (not proven), measured results, files changed and why, test evidence, current Git status/commit, user action required, and next milestone. **Don't turn every answer into a long tutorial**: provide compact explanation and a "Learning notes / code map" section for deeper self-study in another chat.
- Research official documentation for relevant APIs and compatibility. Do not install or upgrade Isaac Sim/Lab/LeIsaac/CUDA/torch/NVIDIA/ROS/MoveIt or modify system settings without explicit consent. Preserve `leisaac-sim`.

## Human-controlled learning gates
- Never collect demonstrations or execute optimizer updates without explicit user authorization for that session.
- Camera tensors must pass attachment/timing checks and human visual approval before collection.
- Demonstrations must pass visual/data-quality review and independent task evaluation before training.
- Camera inspection snapshots under outputs/ are diagnostics, not training demonstrations.
- Preserve failed-generation counts; never narrow distributions silently.
- Camera arrangement approved 2026-10-23: freeze `configs/policy_cameras.json`. The authorized 12-attempt pilot is complete (12 physical successes, 12 automatic QA passes; 3 selected physical replay passes). Stop for human review; protocol remains provisional. No further collection, augmentation, training, or remote push until separately authorized. Read `docs/PILOT_RESULTS.md` and the handoff.

## Current order of work
1. Geometry/TCP: independently verify actual opposing finger surfaces, real aperture for 40-mm cube, jaw angles and TCP transform (the CSV collision bounds currently missing). Treat mesh AABBs as proxies only; don't certify physical contact from AABB.
2. IK feasibility for pre-grasp target, with pose orientation feasibility (SO-101 arm has 5 controlled arm joints); offline check first.
3. Collision-aware HOME -> PREGRASP, verify intermediate poses, cube untouched; deterministic reset.
4. Physical cube acquisition, lift and hold with measurable contact/grasp evidence.
5. Bimanual handover with both robot arms coordinated and verified transfer.
6. Bin placement, retreat, complete repeated episodes, auto reset, success/failure metrics.
7. Only then dataset generation, ACT, diffusion policy, VLA, benchmarking, sim-to-real.

Follow `docs/ROBOTICS_CODEX_HANDOFF.md` for actual paths, states, documented limitations and exact environment activation. If that document conflicts with live results, trust live evidence and update the record.
