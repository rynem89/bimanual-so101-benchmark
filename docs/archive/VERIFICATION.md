> Historical record, moved out of the operating guides during final consolidation.
> Statements below describe their recorded dates, not current permissions.
> Start with [README](../../README.md). Historical source/configuration paths
> are recoverable at commit `bbb15f5` and earlier history.

# Repository consolidation verification

Date: 2026-10-23. Baseline `a5cdc18`; branch `feat/repository-consolidation`.
[Current gates](../../README.md#project-status) · [Original pilot results](PILOT_RESULTS.md)

This is interface/documentation regression, not a new dataset or training run.
Full evidence remains local in `outputs/repository_consolidation/`. No existing
pilot attempts are repeated or replaced. The final verification table below is
updated from actual commands; pending items are not completion claims.

| Check | Measured result |
| --- | --- |
| Baseline CPU suite | 45 tests passed in 71.842 s before edits |
| Consolidated CPU suite | 55 tests passed in 71.623 s, including 10 CLI/authorization tests |
| Stationary headless scene | Three actual calibrated RGB tensors, 41 render captures, 180 reset ticks, **0 viewer task ticks**, no expert/recording |
| Visible stationary GUI smoke | Three sensor panels created, all selector choices checked, 32 captures, **0 viewer task ticks**; subsequent owner usability review passed after the guide fix |
| Official LeRobot CLIs | Installed training and dataset-viz `--help` succeeded; no updates |
| ACT/Diffusion configs | Both named 1-observation/64-prediction/8-execution recipes and saved-config resume flags parsed by the installed official config wrapper; CUDA resolved on GPU-enabled check; no model/optimizer constructed |
| Full physical expert | 1/1 full-task success, zero forbidden contacts; independent evaluation **and complete transfer trace exactly match** preserved reference |
| Recorded-action physical replay | 1/1 preserved nominal recording passed through shared physics; independent evaluation exactly matches its previous physical replay |
| Camera motion / unchanged calibration | 51 phase samples /153 camera-pose checks passed frozen intrinsics, actual attachment and common 20 Hz boundary checks; original exhaustive pilot continuity evidence preserved |
| Documentation/examples | 139 internal links resolve; 42 example commands parse; all 49 maintained files inventoried |
| Raw pilot preservation | All 12 full raw byte digests, episode inventory and original QA/report hashes unchanged |
| Actual gate rejection checks | Unauthorized collection, unaccepted pilot conversion, missing train data and missing checkpoint all rejected; no data/model/socket created |
| Human visual checkpoint | Owner accepted camera viewer and raw-review HTML; after the targeted guide fix, confirmed "Guides expand and select normally" |

## Reproduce the cheap checks

Activate `leisaac-sim`; run the CPU command in
[getting started](../SIMULATION.md#environment-locks). CLI help has no heavy imports:

```bash
python scripts/so101.py --help
python scripts/so101.py scene --help
python scripts/so101.py expert --dry-run
python scripts/so101.py collect run --help
python scripts/so101.py dataset --help
python scripts/so101.py train --help
python scripts/so101.py evaluate --help
python scripts/so101.py replay --help
```

Tests check that automatic QA does not grant human acceptance, pilot/held-out data
cannot become training data, absent/incomplete training authorization cannot launch
the official process, collection cannot relabel a manifest, and replay preview
never changes raw files. Existing manifest tamper/ledger/human tests remain active.
Synthetic tests are not physical evidence or accepted demonstrations.

## Simulator checks

Actual commands already run:

```bash
python scripts/so101.py scene --headless --duration 3 --frames \
  --snapshot-dir outputs/repository_consolidation/scene_headless
python scripts/so101.py scene --duration 4 --frames \
  --snapshot-dir outputs/repository_consolidation/scene_gui_smoke
```

The GUI smoke exercises actual `omni.ui` providers and selection callbacks; it is
not human layout acceptance. Snapshot PNGs exclude optical guides. Scene reports
check frozen intrinsics and measured static camera-to-gripper attachment. Task
physics is frozen after the standard reset.

The full-task expert/replay regressions below used the new interface without
recording demonstrations. Their actual independent reports and source hashes
establish the results, not only process exits.

## Official policy verification boundary

The installed CLI/config sources were inspected in `so101-il-py312`; actual
`python -m lerobot.scripts.lerobot_train --help` and
`python -m lerobot.scripts.lerobot_dataset_viz --help` succeeded.
A config-only callback through LeRobot's parser checked both named recipes.
An initial adapter attempt supplied a postponed string annotation; LeRobot 0.6.1
requires a concrete dataclass annotation. The adapter was fixed and both configs
then parsed successfully. No training function, dataset loader or model was called.
The restricted-shell config check observed CUDA falling back to CPU; execution
now explicitly refuses silent CPU fallback. A later GPU-enabled check resolved CUDA for both recipes and parsed their saved-config resume flags, still without any model or optimizer. Actual optimizer-state restoration is untested. Optional TorchCodec could not load; existing PyAV is selected,
without installing or changing dependencies.

Current train preflight is expected to reject absent/unaccepted converted data.
There is no checkpoint for real model serving, trained physical evaluation or
optimizer-state resume tests. Their documented commands are future interfaces,
not successful experiments. Official viewer help is verified; opening an accepted
converted real dataset remains pending that dataset's existence.

The raw-review HTML was generated from the preserved nominal episode. No browser
surface is enabled for computer use in this session; the owner subsequently
accepted its layout/playback in the consolidated human checkpoint. An attempted
repository-wide loopback HTTP server was rejected by automatic approval for excessive local data
exposure; it was not started. No workaround server or dataset upload was used.

Environment metadata was read again: simulation Python 3.11.16, NumPy 1.26.0,
SciPy 1.15.3, torch 2.7.0+cu128, LeIsaac 0.4.0; learning Python 3.12.15,
NumPy 2.2.6, torch 2.7.1+cu128, LeRobot 0.6.1. Isaac Lab Git remains exactly
the pinned `v2.3.0` commit; its internal package version is 0.47.2, now explained
in the environment guide. No dependency or external checkout was changed.

## Full expert regression result

```bash
python scripts/so101.py expert --exit-after \
  --camera-preview outputs/repository_consolidation/expert_cameras \
  --output outputs/repository_consolidation/expert
```

Visible Isaac, committed source `ae93781`, clean at launch. Process exit 0;
`passed_requested_stage`, all seven controller success counts 1. Independent
pickup/lift/handover/placement all true, failure null, zero forbidden contacts.
Evaluator elapsed 153.183333 s; runner includes reset at 156.183333 s; process
wall 359.532727 s. Peak joint speed 2.011003 rad/s, effort 1.349620 Nm, stable
grip slip left/right 0.267299/0.286576 mm. Evaluation JSON and the entire transfer
trace exactly equal the preserved `overview_v3_comparison` reference. This is a
regression without lossless recording, not a collection-throughput improvement.

The 51 actual three-camera phase captures passed independent frozen K/pose/parent
checks and shared software policy-boundary checks. Both wrists moved through the
trajectory while overhead stayed fixed. This is sparse sampling, not a new
exhaustive sensor-frame continuity test; those original pilot proofs remain
unchanged. Raw actions/observations/camera semantics and expert physics are untouched.

The new `collect plan` and dry `collect run` were exercised: five fixed manifests
were generated, 12 pilot launch commands printed, `executed: false`; no raw
directory or run ledger was created. The consumed pilot was not recollected.

## Recorded-action replay regression result

```bash
python scripts/so101.py replay \
  --episode datasets/protocol_v1_pilot/pilot_008_20261023T171324135324Z \
  --output outputs/repository_consolidation/replay --exit-after
```

Visible Isaac, process exit 0; all four independent stages true, failure null,
zero forbidden contacts. Evaluator elapsed **150.350000 s**, process wall
**217.673920 s**, episode wall **208.604351 s**. Every independent metric exactly
matches the previous nominal physical replay. Targets were the original float32
recorded actions through `BenchmarkEnv.step`; no expert plans/servo or physical
state playback supplied the task. The final recorded settling tail is still
omitted at first success; no pointwise-state equivalence is claimed.

The report names `ae93781` with `tracked_source_dirty: true` because documentation
was being updated during this run. Runtime source hashes match the committed
interface version and all preserved control/physics/data files match `a5cdc18`.
No attempt to relabel this diagnostic as clean training data was made.

## Stationary-viewer final guard and human checkpoint

After the physics regressions, only the new standalone viewer received an
additional verification guard: read both robots' joints and the cube transform
**directly from PhysX**, plus the independent simulator clock, before/after the
render-only loop. A bounded headless recheck produced 41 captures with **0 s
physical-time change, 0 rad joint change, 0 m cube translation and 0 quaternion
change**. It rejects nonfinite readings as well as changes. This strengthens the
stationary claim beyond an application step counter; established sensor functions
and expert/replay algorithms remain unchanged.

Five camera tests passed again in **0.010 s** and ten CLI/gate tests in **0.219 s**
after this viewer addition. All scripts/tests compile. Full raw byte digests and
baseline QA/report hashes remain unchanged for all 12 pilot episodes. GPU identity
was read again: RTX 5090, driver 580.178.04, 32,607 MiB; no stack change.

The consolidated **human usability** check is complete (separate from accepting
the 12 pilot videos). In `leisaac-sim`, the initial stationary review used:

```bash
python scripts/so101.py scene --frames \
  --snapshot-dir outputs/repository_consolidation/scene_human_review
```

The owner accepted the actual camera-feed viewer and
`outputs/repository_consolidation/nominal_review.html`, which synchronizes three
images with the joint/action table. The initial Stage-tree issue and its successful
focused recheck are documented below. This does not grant pilot-data acceptance,
further collection, training, deletion, merge or push permission.

## Owner feedback, InspectionGuides fix and output cleanup

The owner confirmed that the new stationary viewer and `nominal_review.html`
are good, and reported that `InspectionGuides` flickers/cannot stay expanded in
the Stage inspector. This is UI acceptance of the image/review workflow, not
acceptance of the 12 pilot demonstrations.

The original stationary loop called `MakeInvisible` and `MakeVisible` on the
guide USD prim every refresh, causing repeated authored stage changes. The fix
captures real RGB at the already frozen state **before** creating guides, updates
the image providers once, and subsequently only renders the viewport. It does not
modify camera calibration, task physics or the expert/recording capture path.
A USD `ObjectsChanged` observer independently counts changes under the guide root;
deliberate user guide edits remain allowed.

Actual bounded Isaac check: `scene --headless --duration 3 --frames` with fresh
snapshot directory `outputs/repository_consolidation/guides_fix_check` produced
**53 viewport renders, one frozen-state three-camera capture and zero authored
guide-change notices**. Physical clock, joint position and cube pose changes
were all zero. Five camera tests passed in 0.010 s and ten CLI/gate tests in
0.224 s; existing sensor functions remain AST-identical to the preserved baseline.
The owner tested the corrected visible viewer at committed source `a23bfe7` and
confirmed: **"Guides expand and select normally."** This completes the focused
Stage expansion/selection recheck and the consolidated usability checkpoint.

Cleanup removed 73 verified stale cache/preview files /776,310,957 bytes;
[the exact scope and retained outputs](REPOSITORY_AUDIT.md#generated-output-cleanup-and-retention)
are documented. The per-file hashes/reasons are local in
`outputs/repository_consolidation/cleanup_manifest.json`. All final verification,
benchmark results, approved-camera evidence, source/configs, the ResNet backbone
cache and raw pilot data are retained. No dataset collection, optimizer update,
tracked deletion, branch deletion, merge or push was performed.

After cleanup, all 12 complete raw-episode digests were recomputed and matched
the original QA record. The original QA/report bytes and both frozen config
hashes also remain unchanged; the local summary is
`outputs/repository_consolidation/post_cleanup_preservation.json`.

The final audit of the 12 current guides passed: 132 local links/anchors resolve,
29 Bash blocks pass syntax checks, all 42 CLI examples parse without execution,
and all 49 tracked files are named in the architecture inventory. Runtime/config
preservation and the absence of all 73 removed files were rechecked in
`outputs/repository_consolidation/guides_cleanup_audit.json`.

## Final documentation/source retirement preparation — 2026-10-23

Continuation baseline: `bbb15f5`, same `feat/repository-consolidation` branch.
The README now gives the ordered Setup → Scene → Expert IK → Demonstrations →
ACT/Diffusion → Evaluation procedure. Four existing detailed guides contain the
active explanations; six existing records were relocated into this archive.
The four replaced active documents are temporary redirects pending the same
approval as two unused probes and two superseded configs. No new guide was added.

A temporary 41-file copy excluded all eight proposed removals. Its existing
**10 CLI/authorization tests passed in 0.222 s**. This checks the actual proposed
source/config tree without removing approval-gated originals. Static imports and
literal command/config paths in maintained source/tests contain **zero references**
to the four retired code/config paths. The full CPU suite and physical tasks were
not rerun: every remaining source/test, active config, controller and physics file
is byte-identical to the baseline. Prior physical and human-viewer evidence remains
valid under that scope; no new physical success is claimed.

The prepared final tree has 41 tracked files versus 49 before: 13 scripts versus
15, four configs versus six, seven tests unchanged, four active detailed guides
versus 11. Historical records are five Markdown files plus the byte-identical
policy-review JSON in this directory. The eight removals and main/branch operations
remain behind the single approval specified in the code map.

All current/archive links and command examples were checked, including links that
would otherwise point at removed guides. Pilot inventory remains 107,454 files /
11,222,342,915 bytes with identical size/mtime fingerprint; original QA/report,
attempt ledger and reference benchmark hashes match. Full raw byte digests were
already verified in the prior cleanup and were not needlessly recomputed here.
No collection, training, data/evidence deletion, dependency change or new branch.
The previous 73-file cache cleanup is not repeated or counted again. No identical
stationary PNG preview groups were found for further deletion.

Final link/syntax pass: **140 local links/anchors**, including
temporary redirects, **24 Bash blocks** and **27 CLI examples** passed.
All 41 final files are named in the code map. The existing focused CLI result
was reused after prose-only corrections; it was not rerun.
