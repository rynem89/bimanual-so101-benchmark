# Repository audit and proposed cleanup

Date: 2026-10-23. [Full file inventory](ARCHITECTURE.md) · [Current gates](STATUS.md)

## Preserved baseline and branch state

Preflight found a clean working tree at local/remote
`a5cdc18f38342a91ab95cb98bfa0b9afee68973f`, branch `feat/il-camera-dataset-v2`.
Remote refs were inspected directly with `git ls-remote --heads --tags origin`;
there were four remote branches and no remote tags. Existing local tags were
`reference/pre-il-camera-v2` and `reference/camera-v2-reviewed`.

New local recovery tag: `reference/pre-repository-consolidation` → `a5cdc18`.
Dedicated local branch: `feat/repository-consolidation`, created from that exact
validated baseline. No branch/tag was deleted; no merge or push was performed by
this milestone. Source and evidence inventory was saved locally under
`outputs/repository_consolidation/baseline.json` before editing.

| Branch at audit | Tip | Relationship / recommendation |
| --- | --- | --- |
| `main` | `c016519` | Protected; 30 commits behind the reviewed feature tip. No automatic merge. |
| `feat/gripper-characterization` | `c121012` | Ancestor of reviewed tip; its work is already incorporated. Candidate for deletion only after final main includes reviewed consolidation and owner approves. |
| `feat/fingertip-geometry-calibration` | `f47c358` | Ancestor of reviewed tip; same conditional cleanup. |
| `feat/il-camera-dataset-v2` | `a5cdc18` | Reviewed/published baseline and ancestor of cleanup branch. Keep through review; candidate only after final integration/recovery approval. |
| `feat/repository-consolidation` | New local work | Owner review first; not published or merged under this brief. |

Local and remote versions of the first four branches matched. Ancestry was
checked with `git merge-base --is-ancestor`, not assumed from names. Do not blindly
merge all feature branches; a later reviewed integration should preserve the
single cumulative history. No destructive cleanup is necessary to use the project.

## Source review and consolidation decisions

All 14 pre-existing tracked Python source modules, six test modules, configurations,
assets and documentation were reviewed by role, imports, public functions, tests
and recorded results. The architecture inventory includes new interface/test files.

Preserved algorithm/semantics files: `run_manipulation.py`, `so101_model.py`,
`so101_planning.py`, `benchmark_core.py`, `benchmark_env.py`, `dataset_io.py` and
`inspect_task_layout.py`. They remain byte-identical to `a5cdc18`; frozen camera
and protocol JSON hashes also remain unchanged. Added camera inspection is a
separate function in the existing camera module; its established sensor/capture
functions remain unchanged. The learning adapter adds named official command
preparation and execution gates, without a new model, optimizer or dataset format.

The 980-line controller combines CLI parsing, stage execution, instrumentation
and report orchestration. Safe consolidation here is the supported top-level
CLI/defaults plus documented function map, not a broad control rewrite. Experimental
low-level flags remain available for historical reproduction but are absent from
new quick starts. Old foundation/FK scripts do duplicate parts of scene/model
inspection; their motion experiments are not equivalent to stationary camera
inspection. They are clearly classified instead of silently replaced/deleted.

## Exact proposed file deletions — NOT performed or approved

| Candidate | Why it could be removed from the active tree | Dependency / approval condition |
| --- | --- | --- |
| `scripts/inspect_fk_frames.py` | Old preliminary FK scene; no active imports, largely superseded by task/model inspection | Preserve recovery commit, review its unique motion/pose outputs and confirm replacement coverage first |
| `scripts/inspect_bimanual_scene.py` | Old no-basket foundation scene, separate defaults and joint-stability/gravity experiments | Its gravity/stability experiments are not covered by the new viewer; owner must decide to retire them rather than claim equivalent replacement |
| `configs/camera_v2_pilot.json` | Superseded single-pose placeholder, not the completed pilot | Update provenance references only after approved retirement; preserve Git history |
| `configs/task_distribution.json` | Historical 32/8/30 proposal; misleading beside the active protocol | Historical policy review/metadata reference its cases/hash; preserve recoverability and update references after approval |

No current module/test/accepted calibration should be deleted. In particular,
`inspect_task_layout.py` is an active imported runtime dependency despite its
name. The finger audit still provides triangle/instance evidence; the reset
diagnostic still tests an actual disturbance. Keep both. Do not delete
`POLICY_REVIEW_METADATA.json`: it is small machine-readable evidence explaining
the earlier failed experiments. Keep the full handoff/history; it is now explicitly
marked archival and removed from the main learning path.

## Local/generated evidence and unknown files

The 12 raw pilot directories and existing QA/report/replay outputs remain local
and ignored. Their episode metadata, preserved summary hashes and file inventory
are checked against the baseline during verification. No raw pixels, trajectories,
checkpoints or old benchmark evidence are removed or published. The current reset
pipeline has no checkpoint files; no model is implied by historical names.

Generated consolidation tests/HTML/PNG/JSON/logs also stay under ignored `outputs/`.
No new dependencies, environment, package skeleton or family of wrapper scripts
was introduced. There were no untracked source files at baseline; new maintained
files are explicitly inventoried. Existing generated/cached directories are not
automatically deletion candidates. Obtain exact scope approval before any cleanup.

## Documentation contradictions resolved

- README is now an operating entrance, with valid tracked guide links instead of
  hyperlinks into ignored outputs. Local artifacts are labelled code paths.
- The protocol acknowledges all 12 physical attempts and measured collection
  cost, distinguishing them from earlier offline approach checks.
- AGENTS reads current status, not a completed geometry-first work order.
- Camera/protocol hashes stay intact; status/permission history is not stored by
  mutating configuration bytes used by recorded data.
- Historical configs, failed learning metadata and the 2,300-line handoff are
  labelled historical, with a single current-status authority.
- No trained-model, real-hardware-calibration or untested GUI/training claim is
  promoted into a verified accomplishment. Pending checks stay explicit.
