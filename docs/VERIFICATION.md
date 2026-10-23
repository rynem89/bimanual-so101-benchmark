# Repository consolidation verification

Date: 2026-10-23. Baseline `a5cdc18`; branch `feat/repository-consolidation`.
[Current gates](STATUS.md) · [Original pilot results](PILOT_RESULTS.md)

This is interface/documentation regression, not a new dataset or training run.
Full evidence remains local in `outputs/repository_consolidation/`. No existing
pilot attempts are repeated or replaced. The final verification table below is
updated from actual commands; pending items are not completion claims.

| Check | Measured result |
| --- | --- |
| Baseline CPU suite | 45 tests passed in 71.842 s before edits |
| Consolidated CPU suite | 55 tests passed in 71.623 s, including 10 CLI/authorization tests |
| Stationary headless scene | Three actual calibrated RGB tensors, 41 render captures, 180 reset ticks, **0 viewer task ticks**, no expert/recording |
| Visible stationary GUI smoke | Three sensor panels created, all selector choices checked, 32 captures, **0 viewer task ticks**; visual usability pending human review |
| Official LeRobot CLIs | Installed training and dataset-viz `--help` succeeded; no updates |
| ACT/Diffusion configs | Both named 1-observation/64-prediction/8-execution recipes and saved-config resume flags parsed by the installed official config wrapper; CUDA resolved on GPU-enabled check; no model/optimizer constructed |
| Full physical expert | Pending limited regression through new CLI |
| Recorded-action physical replay | Pending one preserved nominal recording regression |
| Camera motion / unchanged calibration | Frozen file hashes checked; new expert phase capture check pending; original exhaustive pilot continuity evidence preserved |
| Documentation/examples | 138 internal links resolve; 39 example commands parse; all 49 maintained files inventoried |
| Human visual checkpoint | Pending, consolidated scene/feed and expert inspection |

## Reproduce the cheap checks

Activate `leisaac-sim`; run the CPU command in
[getting started](GETTING_STARTED.md#cpu-tests). CLI help has no heavy imports:

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

Limited full-task expert/replay regressions will use the new interface, without
recording demonstrations. Source hashes and actual independent reports, not only
process exits, will be recorded here before completion.

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
surface is enabled for computer use in this session, so browser layout/playback
remain part of the consolidated human checkpoint. An attempted repository-wide
loopback HTTP server was rejected by automatic approval for excessive local data
exposure; it was not started. No workaround server or dataset upload was used.
