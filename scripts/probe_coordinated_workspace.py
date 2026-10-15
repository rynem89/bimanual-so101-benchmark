"""Bimanual SO-101 coordinated-joint workspace sampling (not IK).

Run from project root: python scripts/probe_coordinated_workspace.py

Reuses inspect_task_layout.py, which owns the canonical scene geometry and
launches Isaac Sim before simulation-dependent imports. Normal gravity is kept.

We sample moderate three-joint combinations, take *measured* gripper-link poses,
rank proximity to pickup/handover/bin approach targets, and read both gripper
and moving-jaw link origins. Link-origin separation is NOT finger clearance.

This experiment does NOT establish collision-free motion, fingertip TCP,
orientation suitability, grasp success, or full task reachability.
"""
from __future__ import annotations

import itertools
import math

import inspect_task_layout as layout  # This initializes AppLauncher.
import torch

# Angles in degrees. Each tuple is (pan, lift, elbow).
# Moderate commands are used so we can learn the combined influence of joints
# before switching to a proper IK algorithm and collision-aware motion planner.
PAN_DEG = (-25.0, 0.0, 25.0)
LIFT_DEG = (0.0, 15.0, 30.0)
ELBOW_DEG = (0.0, 15.0, 30.0)
MOVE_STEPS = 120
HOLD_STEPS = 180
TOP_N = 5


def set_pose(robot, pan: float = 0, lift: float = 0, elbow: float = 0, gripper: float = 0) -> None:
    """Set a full target vector using joint *names*, leaving other joints at defaults."""
    target = robot.data.default_joint_pos.clone()
    for name, deg in (("shoulder_pan", pan), ("shoulder_lift", lift), ("elbow_flex", elbow), ("gripper", gripper)):
        j = layout.joint_index(robot, name)
        target[:, j] = math.radians(deg)
        low, high = robot.data.soft_joint_pos_limits[0, j].tolist()
        if not low <= target[0, j].item() <= high:
            raise ValueError(f"{name} target {deg}° outside soft joint range")
    robot.set_joint_position_target(target)


def link_xyz(robot, name: str) -> torch.Tensor:
    """Get a link frame's measured world position (not the surface of its mesh)."""
    j = layout.body_index(robot, name)
    return robot.data.body_pos_w[0, j].detach().clone()


def fmt_xyz(p: torch.Tensor) -> str:
    return "(" + ", ".join(f"{v:+.3f}" for v in p.cpu().tolist()) + ")"


def sample_one_arm(sim, scene, label: str, robot, inactive, goals: dict[str, torch.Tensor]) -> None:
    """Try three-joint commands one at a time and sort *observed* distances."""
    set_pose(inactive)
    set_pose(robot)
    layout.step_physics(sim, scene, HOLD_STEPS)

    print(f"\n=== {label} COORDINATED WORKSPACE SAMPLE ===")
    print(f"Sampling {len(PAN_DEG) * len(LIFT_DEG) * len(ELBOW_DEG)} commanded combinations")

    ranking: dict[str, list[tuple[float, tuple[float, float, float], tuple[float, float, float], float]]] = {
        key: [] for key in goals
    }
    jaw_frame_distances = []

    for pan, lift, elbow in itertools.product(PAN_DEG, LIFT_DEG, ELBOW_DEG):
        set_pose(robot, pan, lift, elbow)
        layout.step_physics(sim, scene, MOVE_STEPS)
        tip_link = link_xyz(robot, "gripper")
        jaw_link = link_xyz(robot, "jaw")
        frame_sep = (tip_link - jaw_link).norm().item() * 1000.0
        jaw_frame_distances.append(frame_sep)

        q = robot.data.joint_pos[0]
        measured = tuple(round(math.degrees(q[layout.joint_index(robot, name)].item()), 2)
                         for name in ("shoulder_pan", "shoulder_lift", "elbow_flex"))
        pos_tuple = tuple(round(v, 4) for v in tip_link.cpu().tolist())
        # Keep track of distance for each goal independently. This is *not* a
        # task score, because orientation and collision checks are still absent.
        for key, target_xyz in goals.items():
            dist_mm = (tip_link - target_xyz).norm().item() * 1000.0
            ranking[key].append((dist_mm, (pan, lift, elbow), pos_tuple, max(abs(measured[0]-pan),abs(measured[1]-lift),abs(measured[2]-elbow))))

    for name, rows in ranking.items():
        rows.sort(key=lambda record: record[0])
        print(f"\n  Target: {name} at {fmt_xyz(goals[name])} m")
        for dist, angles, xyz, tracking in rows[:TOP_N]:
            print(f"    {dist:6.1f} mm | target pan/lift/elbow={angles!s:<23} | link xyz={xyz} | max angle error={tracking:.2f}°")
    print(f"\n  GRIPPER versus JAW link-frame-origin distance across samples: "
          f"{min(jaw_frame_distances):.1f}–{max(jaw_frame_distances):.1f} mm")
    print("  NOTE: link-frame origin distance is NOT measured inner-fingertip clearance.")

    set_pose(robot)
    layout.step_physics(sim, scene, HOLD_STEPS)
    print(f"  [RESET] {label} commanded back to the default pose")


def main() -> None:
    sim = layout.SimulationContext(layout.sim_utils.SimulationCfg(device=layout.args.device))
    sim.set_camera_view(eye=(1.25, -1.25, 0.85), target=(0.0, 0.08, 0.10))
    cfg = layout.BimanualSceneCfg(num_envs=1, env_spacing=2.0)
    cfg.left_arm.init_state.pos = layout.LEFT_BASE_M
    cfg.right_arm.init_state.pos = layout.RIGHT_BASE_M
    scene = layout.InteractiveScene(cfg)
    sim.reset()
    left, right = scene["left_arm"], scene["right_arm"]
    set_pose(left)
    set_pose(right)
    layout.step_physics(sim, scene, HOLD_STEPS)
    cube_xyz = scene["cube"].data.root_pos_w[0].detach().clone()
    print("\n=== PROVISIONAL TARGETS (METRES, WORLD FRAME) ===")
    print("Cube centre:", fmt_xyz(cube_xyz))
    print("Bin centre XY:", layout.BIN_CENTER_XY_M)
    print("Handover:", layout.HANDOVER_CENTER_M)
    print("Gravity: ON; gripper command: 0°; target orientation: NOT CONSTRAINED")

    def point(vals, device):
        return torch.tensor(vals, device=device, dtype=cube_xyz.dtype)

    left_goals = {
        "cube approach (link-origin proxy)": point((cube_xyz[0].item(), cube_xyz[1].item(), 0.12), left.device),
        "handover (link-origin proxy)": point(layout.HANDOVER_CENTER_M, left.device),
    }
    right_goals = {
        "bin approach (link-origin proxy)": point((layout.BIN_CENTER_XY_M[0], layout.BIN_CENTER_XY_M[1], 0.165), right.device),
        "handover (link-origin proxy)": point(layout.HANDOVER_CENTER_M, right.device),
    }
    sample_one_arm(sim, scene, "LEFT", left, right, left_goals)
    sample_one_arm(sim, scene, "RIGHT", right, left, right_goals)
    print("\n[COMPLETE] Both arms returned to default commands. No IK or grasp attempted.")
    print("[NEXT] Define physical TCP/finger-clearance and solve Cartesian targets with IK.")
    print("[INFO] Close the Isaac Sim window or press Ctrl+C.")
    while layout.app.is_running():
        layout.step_physics(sim, scene, 1)


if __name__ == "__main__":
    try:
        main()
    finally:
        layout.app.close()
