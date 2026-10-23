# Getting started

[README](../README.md) · [Current status](STATUS.md) · [Camera details](SIMULATION.md)

## 1. Open the existing workspace

These are workstation instructions for the already installed environments. A new
machine needs separately planned asset/software provisioning; this milestone does
not provide or execute an installer.

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
export LEISAAC_ASSETS_ROOT=/home/rynem8/robotics/leisaac/assets
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python --version
python scripts/so101.py --help
```

Expect Python **3.11.16**. The explicit interpreter is
`/home/rynem8/miniforge3/envs/leisaac-sim/bin/python`. Activation applies only to
this shell. CLI help and `expert --dry-run` also work with plain Python, without
starting Isaac, importing torch or loading a USD. Runtime commands check the
active environment name and report an error if it is wrong.

## 2. Inspect the task and cameras

```bash
python scripts/so101.py scene --frames
```

Expect two robots mounted beside the table, the red cube near the left arm and a
blue basket near the right. A separate **SO-101 actual policy camera feeds** panel
shows overhead, left wrist and right wrist images. Use its selector to display
one stream or all three. The viewport can be orbited independently; it is not a
policy observation. Yellow lines show optical +Z. Optional axes use red X, green
Y, blue Z. Guides are hidden during sensor capture and have no physics.

The cube settles during the established 180-tick reset. The subsequent inspection
freezes task physics and refreshes RGB only. HOME wrist views need not see the
cube: their useful contact views occur later in the expert trajectory. This
viewer never picks up the cube. Close the Isaac application window to exit;
closing only the camera panel does not close Isaac.

For saved diagnostic PNGs and measured calibration, use a **new** directory:

```bash
python scripts/so101.py scene --frames \
  --snapshot-dir outputs/scene_manual
```

A bounded automated check is `scene --headless --duration 3 --snapshot-dir
outputs/scene_automated`. It can check tensors/transforms and zero task steps;
it does not prove the GUI layout is useful. [Simulation details](SIMULATION.md)
explain the USD, dynamic camera attachments and hardware-calibration limitations.

## 3. Run the expert

Close scene inspection first; use one simulator at a time.

```bash
python scripts/so101.py expert --output outputs/expert_manual
```

The full-task recipe is fixed by the interface: continuous motion, 20 Hz held
position targets, grip feedback from lift, staged handover and the approved
centered camera profile. Geometry planning can take time; `[PLAN]` messages mean
physics is paused while CPU collision checks run. Do not mistake paused planning
for a successful grasp or frozen simulator.

Expect left pickup and lift, right approach, dual contact, left release/withdrawal,
right placement/release, and both arms retreating. `[EPISODE] nominal/1: passed`
and `[RESULT] passed_requested_stage` accompany a saved independent evaluation.
The final pose stays open at `[VIEWER]`; close Isaac after inspection.

Outputs: `simulation.json` is the current interruption/completion summary;
`full_<UTC timestamp>.json` preserves the run, including each attempt, independent
stage/contact evidence, failures, timings and source hashes. `plans/` contains
geometry-validated caches. An exit code of zero is insufficient alone: inspect
`full_task_success`, `independent_evaluation.success`, stage flags and failures.

Useful variants:

```bash
python scripts/so101.py expert --dry-run
python scripts/so101.py expert --case yaw_minus_1deg --output outputs/expert_yaw
python scripts/so101.py expert --episodes 3 --output outputs/expert_reset
python scripts/so101.py expert --headless --exit-after --output outputs/expert_batch
```

`--episodes` resets between repetitions. It does not replace failed cases. For
actual policy-camera phase PNGs, add `--camera-preview outputs/expert_images` to
a single expert run; these are diagnostics, not demonstrations.

## 4. Review data, then decide the next milestone

Use [dataset instructions](DATASET.md) for the existing 12-attempt pilot. Review
human visual quality separately from automatic QA and physical evaluation.
[STATUS](STATUS.md) holds the approval gates. No final collection or training is
authorized by this consolidation assignment.

## Environment locks

This is the authoritative human-readable environment table. Installed packages
and pinned external checkouts remain the source of truth if they disagree.

| Component | Simulation | Learning/conversion/serving |
| --- | --- | --- |
| Conda environment | `leisaac-sim` | `so101-il-py312` |
| Python | 3.11.16 | 3.12.15 |
| NumPy | 1.26.x | 2.2.6 |
| PyTorch | 2.7.0+cu128 | 2.7.1+cu128 |
| Framework | Isaac Sim 5.1.0, Isaac Lab 2.3.0, LeIsaac 0.4.0 | LeRobot 0.6.1 |
| GPU/driver | RTX 5090 32 GB / NVIDIA 580.178.04 | Same workstation GPU |

Isaac Lab checkout: `/home/rynem8/robotics/IsaacLab`, commit
`3c6e67bb5c7ada942a6d1884ab69338f57596f77`. LeIsaac checkout:
`/home/rynem8/robotics/leisaac`, commit prefix `24d3bcd`. Existing compatibility
patch: Isaac Lab's `flatdict` requirement is 4.1.0. Preserve it.
The complete installed learning packages are in
[il_requirements.lock.txt](../configs/il_requirements.lock.txt).

For conversion, training preflight and policy serving:

```bash
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate so101-il-py312
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

LeRobot and the locked simulator require incompatible Python/NumPy versions.
Separate environments preserve both; both policies use the **same official
LeRobot implementation/environment**, not separate custom implementations.
Inference exchanges RGB bytes, lists and targets through a local Unix socket.
The installed optional TorchCodec decoder may warn about shared FFmpeg libraries;
use the existing PyAV backend. No dependency repair is part of this assignment.

## CPU tests

In `leisaac-sim`, expose the existing bundled USD libraries, then run:

```bash
USD_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
PYTHONPATH="scripts:$USD_LIB" \
LD_LIBRARY_PATH="$USD_LIB/bin:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python -m unittest discover -s tests -v
```

These check code contracts and geometry. Physical success still requires actual
simulator evaluation. See [verification](VERIFICATION.md) for this milestone.
