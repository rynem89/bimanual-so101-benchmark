"""Measured SO-101 workspace probe for the bimanual task.

Run: python scripts/probe_workspace.py

Reuses the verified task scene; no IK, grasp, geometry change, or actuator tuning.
For each robot, tests conservative joint configurations and measures the GRIPPER
LINK origin against the relevant task goals. It does not prove TCP reachability.
Stack: Isaac Sim 5.1 / Isaac Lab 2.3.0 / LeIsaac 0.4.0.
"""

from __future__ import annotations

# Importing the existing scene module starts Isaac Sim (AppLauncher must precede
# Isaac-dependent imports). Its main() does not run because of its __main__ guard.
import inspect_task_layout as layout
import torch

# Degrees. Values are modest probes, not planned trajectories or grasp poses.
# One nonzero arm joint per probe simplifies cause-and-effect interpretation.
PROBES = (
    ("default", {}),
    ("pan outward -20", {"shoulder_pan": -20.0}),
    ("pan inward +20", {"shoulder_pan": +20.0}),
    ("lift -20", {"shoulder_lift": -20.0}),
    ("lift +20", {"shoulder_lift": +20.0}),
    ("elbow -20", {"elbow_flex": -20.0}),
    ("elbow +20", {"elbow_flex": +20.0}),
)


def command_joint_pose(robot, angles_deg: dict[str, float]) -> torch.Tensor:
    """Build one full 6-joint target from defaults, overwriting named joints."""
    target = robot.data.default_joint_pos.clone()
    for name, angle_deg in angles_deg.items():
        index = layout.joint_index(robot, name)
        target[:, index] = torch.deg2rad(target.new_tensor(angle_deg))
        low, high = robot.data.soft_joint_pos_limits[0, index].tolist()
        if not low <= target[0, index].item() <= high:
            raise ValueError(f"{name} probe {angle_deg}° is outside model limits")
    robot.set_joint_position_target(target)
    return target


def measured_distance_mm(point_a: torch.Tensor, point_b: torch.Tensor) -> float:
    return torch.linalg.vector_norm(point_a - point_b).item() * 1000.0


def run_probe(sim, scene) -> None:
    left, right = scene["left_arm"], scene["right_arm"]
    cube = scene["cube"]
    # Fixed world goals; bin target is the centre above its open rim.
    # The cube goal is elevated above the cube to avoid tabletop collisions.
    goals = {
        "LEFT": {
            "cube approach": torch.tensor(
                [*layout.CUBE_START_M[:2], 0.12], device=left.device
            ),
            "handover": torch.tensor(layout.HANDOVER_CENTER_M, device=left.device),
        },
        "RIGHT": {
            "bin approach": torch.tensor(
                [*layout.BIN_CENTER_XY_M, layout.BIN_FLOOR_M + layout.BIN_WALL_HEIGHT_M + 0.08],
                device=right.device,
            ),
            "handover": torch.tensor(layout.HANDOVER_CENTER_M, device=right.device),
        },
    }

    print("\n=== TASK TARGETS (WORLD FRAME; PROVISIONAL) ===")
    print("Cube measured centre:", layout.xyz_m(cube.data.root_pos_w[0]))
    for label in ("LEFT", "RIGHT"):
        for name, position in goals[label].items():
            print(f"{label} {name}: {layout.xyz_m(position)}")
    print("NOTE: distances are from the gripper *link origin*, not between fingertips.")

    # Hold the other robot at default during each set of experiments.
    for label, active, inactive in (("LEFT", left, right), ("RIGHT", right, left)):
        print(f"\n=== {label} SINGLE-JOINT WORKSPACE PROBES ===")
        command_joint_pose(inactive, {})
        best = {goal_name: (float("inf"), "") for goal_name in goals[label]}
        for title, angles in PROBES:
            target = command_joint_pose(active, angles)
            layout.step_physics(sim, scene, 180)
            state = layout.snapshot(active)
            errors_deg = torch.rad2deg(
                torch.abs(state["joints"] - target[0])
            )
            errors_deg[layout.joint_index(active, "gripper")] = 0.0
            xyz = layout.xyz_m(state["world_p"])
            distances = {
                goal_name: measured_distance_mm(state["world_p"], position)
                for goal_name, position in goals[label].items()
            }
            result = " | ".join(f"{name}: {d:.1f} mm" for name, d in distances.items())
            print(f"{title:>19}: link={xyz} | {result} | max arm tracking error={errors_deg.max().item():.2f}°")
            for name, d in distances.items():
                if d < best[name][0]:
                    best[name] = (d, title)
        command_joint_pose(active, {})
        layout.step_physics(sim, scene, 180)
        print("Best *sampled* distances (not IK/reachability):")
        for name, (distance, pose_name) in best.items():
            print(f"  {name}: {distance:.1f} mm at {pose_name}")

    print("\n[COMPLETE] Both robots returned to default commands.")
    print("[NEXT] Use measured fingertip geometry and IK; this probe alone cannot certify grasp poses.")


def main() -> None:
    sim = layout.SimulationContext(layout.sim_utils.SimulationCfg(device=layout.args.device))
    sim.set_camera_view(eye=(1.25, -1.25, 0.85), target=(0.0, 0.08, 0.10))
    cfg = layout.BimanualSceneCfg(num_envs=1, env_spacing=2.0)
    cfg.left_arm.init_state.pos = layout.LEFT_BASE_M
    cfg.right_arm.init_state.pos = layout.RIGHT_BASE_M
    scene = layout.InteractiveScene(cfg)
    sim.reset()
    command_joint_pose(scene["left_arm"], {})
    command_joint_pose(scene["right_arm"], {})
    layout.step_physics(sim, scene, 180)
    run_probe(sim, scene)
    print("[INFO] Close the Isaac Sim window or Ctrl+C to exit.")
    while layout.app.is_running():
        layout.step_physics(sim, scene, 1)


if __name__ == "__main__":
    try:
        main()
    finally:
        layout.app.close()
