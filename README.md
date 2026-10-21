# Bimanual SO-101 Benchmark

Simulation-first bimanual manipulation benchmark using NVIDIA Isaac Sim,
Isaac Lab, and LeIsaac.

## Task

Left SO-101 grasps a cube -> moves it to a handover region ->
right SO-101 grasps the cube -> left releases ->
right SO-101 places the cube into a target bin.

## Current milestone

Build the deterministic simulation foundation before any imitation learning.

Initial development target:

- two SO-101 follower arms
- simple table/workspace
- one cube
- articulation inspection
- independent left/right control

## Validated simulation stack

- Ubuntu 24.04.5 LTS
- NVIDIA driver 580.178.04 open
- CUDA 12.8.1
- Python 3.11.16
- PyTorch 2.7.0+cu128
- Isaac Sim 5.1.0
- Isaac Lab v2.3.0
- LeIsaac 0.4.0

The simulator environment is treated as locked unless a demonstrated
compatibility problem requires a change.

## Current evidence (2026-10-21)

Geometry audit now reads all three collision meshes, including instance proxies
with USD purpose `guide`. Each mesh's bounds are checked independently against
all transformed vertices (1 micrometre tolerance). Authored surfaces are inputs
to convex decomposition; they do not certify cooked PhysX contact geometry.
Both jaws tracked 0/45/90/0 degrees within 0.012 degrees, with zero measured cube
drift. The finite headless audit exits 0 after clearing the Isaac Lab STOP callback. M1 remains incomplete: contact surfaces, 40 mm cube fit and TCP require
validation. No IK, collision-free approach, pickup or handover is claimed.
See [handoff](docs/ROBOTICS_CODEX_HANDOFF.md) for measurements and limitations.

## Run in the existing environment

Every new shell needs activation; do not install or upgrade dependencies:

```bash
cd /home/rynem8/robotics/bimanual-so101-benchmark
source /home/rynem8/miniforge3/etc/profile.d/conda.sh
conda activate leisaac-sim
export LEISAAC_ASSETS_ROOT=/home/rynem8/robotics/leisaac/assets
python --version
```

Headless physics audit (120-second timeout is a failure guard; verified exit 0):

```bash
mkdir -p outputs/finger_audit
timeout --signal=TERM --kill-after=10s 120s python scripts/audit_so101_finger_geometry_v2.py --headless > outputs/finger_audit/headless.log 2>&1
```

Visual checkpoint:

```bash
python scripts/audit_so101_finger_geometry_v2.py
```

Watch the left jaw at 0, 45, 90, 0 degrees. A yellow line marks the triangle
surface section at gripper-frame y=0, z=-80 mm; it disappears when the moving
jaw no longer intersects that section. Check whether its ends lie on the
intended inner finger surfaces and whether the cube stays still. This line is
not a TCP marker. Close the simulator after inspection.

CPU-only asset audit and focused tests use the USD libraries already bundled
with this exact Isaac Sim installation (no Kit/GPU launch):

```bash
USD_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
export PYTHONPATH="$USD_LIB${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$USD_LIB/bin:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python scripts/audit_so101_finger_geometry_v2.py --offline --output-dir outputs/finger_audit_offline
python tests/test_finger_geometry.py
```

Use a fresh activated shell for the simulation commands after this CPU-only
setup. Offline output never substitutes for live physics measurements.

## File inventory / learning notes

All script entry points below run with `python scripts/<name>.py` after the
activation above. Simulator scripts read the existing LeIsaac USD; older
interactive diagnostics keep their viewer open until closed.

| File | Purpose, inputs / outputs, status |
| --- | --- |
| `scripts/inspect_task_layout.py` | Shared task scene and live FK inspection; USD + fixed layout → console poses. Active scene foundation; importing launches Kit. |
| `scripts/audit_so101_finger_geometry_v2.py` | Maintained geometry/jaw diagnostic; USD and optionally measured body transforms → hierarchy, geometry CSV, surface sections and JSON in `outputs/finger_audit`. `--headless`, `--offline`, `--output-dir` supported. Not a calibrated TCP. |
| `tests/test_finger_geometry.py` | `python tests/test_finger_geometry.py` with USD paths above; analytic triangle inputs → three unittest checks for face-interior intersections, degenerate/parallel triangles and failing prerequisites. |
| `scripts/verify_motion_gripper_reset.py` | Existing regression; scene → console motion/gripper/reset checks. Preserved unchanged; deliberately displaces cube only in reset test. |
| `scripts/inspect_bimanual_scene.py` | Earlier scene, independent motion and stability diagnostic; USD → console. Preserved baseline. |
| `scripts/inspect_fk_frames.py` | Earlier world/base/link transform experiment; USD → console. Preserved baseline. |
| `scripts/probe_workspace.py` | Historical joint-grid exploration; scene → console distances. Known collision risk; not an active planner. |
| `scripts/probe_coordinated_workspace.py` | Historical coordinated joint-grid exploration; scene → console distances. Known collision risk; not an active planner. |
| `scripts/calibrate_so101_fingertips.py` | Pre-existing untracked calibration attempt, preserved unchanged. Misses instance-proxy colliders; its proxy distance is not a TCP. Not an active validation entry point. |
| `docs/ROBOTICS_CODEX_HANDOFF.md` | Environment locks, mission gates, measured results and next checkpoint. |
| `AGENTS.md` | User-provided operating contract, preserved. |
| `.gitignore`, `.vscode/settings.json` | Repository hygiene and existing editor configuration. |

There is no production manipulation controller yet. Historical diagnostics are
retained in place; nothing has been deleted. `outputs/` and Python caches are
ignored generated evidence, not source. No remaining unknown source files were
found in the initial inventory.

For self-study: `mesh_evidence` shows USD row-vector transforms and triangle
planes; `x_surface_intersections` uses barycentric coordinates to intersect
face interiors; `finger_sections` applies measured jaw rotation/translation.
Keep authored geometry, actual body pose and effective physical contact distinct.

API references: [OpenUSD bounds cache](https://openusd.org/release/api/class_usd_geom_b_box_cache.html)
and [Isaac Sim 5.1 application lifecycle](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html).
