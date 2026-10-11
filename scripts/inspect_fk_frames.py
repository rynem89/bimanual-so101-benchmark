"""Inspect SO-101 gripper poses in world and robot-base frames.

This is a measured forward-kinematics experiment, not a grasp or IK controller.
Run with: python scripts/inspect_fk_frames.py

System flow: preflight -> launch -> scene -> physics -> settled baseline ->
            independent joint motions -> end-effector pose and cube measurements.

Stack lock: Isaac Sim 5.1, Isaac Lab v2.3.0, LeIsaac 0.4.0.
Units: metres, radians, seconds unless a printout explicitly says degrees.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# PRE-LAUNCH: LeIsaac resolves its asset root when imported. Verify/set the
# location BEFORE starting Isaac Sim, not after it has taken several seconds.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = PROJECT_ROOT.parent / "leisaac" / "assets"
ASSETS_ROOT = Path(os.environ.get("LEISAAC_ASSETS_ROOT", DEFAULT_ASSETS)).expanduser().resolve()
ROBOT_USD = ASSETS_ROOT / "robots" / "so101_follower.usd"
if not ROBOT_USD.is_file():
    raise FileNotFoundError(
        f"Missing SO-101 asset: {ROBOT_USD}\n"
        "Provide LEISAAC_ASSETS_ROOT pointing to your existing LeIsaac assets directory."
    )
os.environ["LEISAAC_ASSETS_ROOT"] = str(ASSETS_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

# POST-LAUNCH: simulation-dependent imports must come after AppLauncher.
import torch
from isaaclab.utils.math import subtract_frame_transforms
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG

# OUR benchmark geometry (not changes to upstream LeIsaac's robot asset).
TABLE_SIZE_M = (1.20, 0.80, 0.05)
TABLE_TOP_Z_M = 0.0
# USD geometry measurement: bottom of the visible base is 30.081 mm above root.
ROBOT_ROOT_Z_M = -0.030081
LEFT_BASE_M = (-0.25, -0.30, ROBOT_ROOT_Z_M)
RIGHT_BASE_M = (+0.25, -0.30, ROBOT_ROOT_Z_M)
CUBE_EDGE_M = 0.04
# Start 5 mm above table so the dynamic cube can settle under gravity.
CUBE_START_M = (0.0, 0.10, TABLE_TOP_Z_M + CUBE_EDGE_M / 2 + 0.005)

@configclass
class BimanualSceneCfg(InteractiveSceneCfg):
    """Declarative blueprint; InteractiveScene creates the runtime objects."""

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE_M,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -TABLE_SIZE_M[2] / 2)),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_EDGE_M,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_START_M),
    )

    left_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Left_Robot"
    )
    right_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Right_Robot"
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500, color=(0.75, 0.75, 0.75)),
    )




HOLD_STEPS = 180
MOVE_STEPS = 240


def joint_index(robot, name: str) -> int:
    """Resolve joint names against the loaded model instead of assuming an index."""
    ids, names = robot.find_joints(name, preserve_order=True)
    if len(ids) != 1 or names != [name]:
        raise RuntimeError(f"Expected one {name!r} joint; got {names!r}")
    return ids[0]


def body_index(robot, name: str) -> int:
    """The link named 'gripper' is our provisional FK reference frame."""
    ids, names = robot.find_bodies(name, preserve_order=True)
    if len(ids) != 1 or names != [name]:
        raise RuntimeError(f"Expected one {name!r} body; got {names!r}")
    return ids[0]


def step_physics(sim, scene, steps: int) -> None:
    """Send actuator targets, advance PhysX, refresh the measured link/joint state."""
    dt = sim.get_physics_dt()
    for _ in range(steps):
        if not app.is_running():
            raise RuntimeError("Simulator closed during FK experiment")
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)


def xyz_m(v: torch.Tensor) -> str:
    """Format XYZ positions as metres, then also show human-friendly millimetres."""
    values = v.detach().cpu().tolist()
    return f"({values[0]:+.4f}, {values[1]:+.4f}, {values[2]:+.4f}) m"


def snapshot(robot) -> dict:
    """Read world gripper pose, base pose, and base-relative gripper transform.

    T_base_gripper = inverse(T_world_base) * T_world_gripper.
    This is measured FK from PhysX, not an analytical FK implementation.
    Quaternions use (w, x, y, z), as Isaac Lab does.
    """
    idx = body_index(robot, "gripper")
    base_p = robot.data.root_link_pos_w
    base_q = robot.data.root_link_quat_w
    grip_p = robot.data.body_pos_w[:, idx, :]
    grip_q = robot.data.body_quat_w[:, idx, :]
    rel_p, rel_q = subtract_frame_transforms(base_p, base_q, grip_p, grip_q)
    return {
        "world_p": grip_p[0].detach().clone(),
        "world_q": grip_q[0].detach().clone(),
        "base_p": base_p[0].detach().clone(),
        "relative_p": rel_p[0].detach().clone(),
        "relative_q": rel_q[0].detach().clone(),
        "joints": robot.data.joint_pos[0].detach().clone(),
    }


def print_pose(label: str, state: dict, cube_world: torch.Tensor) -> None:
    """Compare the gripper-link origin with cube centre (not yet the grasp point)."""
    distance = torch.linalg.vector_norm(state["world_p"] - cube_world).item()
    print(f"\n[{label}] gripper link-frame origin")
    print(f"  World XYZ:   {xyz_m(state['world_p'])}")
    print(f"  Base XYZ:    {xyz_m(state['relative_p'])}")
    print(f"  Base origin: {xyz_m(state['base_p'])}")
    q = state['world_q'].cpu().tolist()
    print(f"  World quaternion [w,x,y,z]: ({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f})")
    print(f"  Gripper-origin to cube-centre distance: {distance * 1000:.1f} mm")


def command_pose(left, right, left_pan_deg: float, right_pan_deg: float) -> None:
    """Maintain remaining joints at their defaults while independently commanding pan."""
    lt = left.data.default_joint_pos.clone()
    rt = right.data.default_joint_pos.clone()
    lt[:, joint_index(left, 'shoulder_pan')] = torch.deg2rad(
        torch.tensor(left_pan_deg, device=left.device)
    )
    rt[:, joint_index(right, 'shoulder_pan')] = torch.deg2rad(
        torch.tensor(right_pan_deg, device=right.device)
    )
    left.set_joint_position_target(lt)
    right.set_joint_position_target(rt)


def run_fk_experiment(sim, scene) -> None:
    """Measured FK snapshots at rest and after two known-safe shoulder-pan commands."""
    left = scene['left_arm']
    right = scene['right_arm']
    cube = scene['cube']
    for label, robot in [('LEFT', left), ('RIGHT', right)]:
        if robot.num_joints != 6 or robot.num_bodies != 7:
            raise RuntimeError(f"Unexpected {label} SO-101 articulation")
        print(f"[MODEL] {label}: joints={robot.joint_names}, gripper body index={body_index(robot, 'gripper')}")

    command_pose(left, right, 0.0, 0.0)
    step_physics(sim, scene, HOLD_STEPS)
    cube_pos = cube.data.root_pos_w[0].detach().clone()
    print('\n=== BASELINE / NORMAL GRAVITY ===')
    print('Cube centre in WORLD:', xyz_m(cube_pos))
    baseline = {}
    for label, robot in [('LEFT', left), ('RIGHT', right)]:
        baseline[label] = snapshot(robot)
        print_pose(label, baseline[label], cube_pos)

    for phase, left_angle, right_angle in [
        ('LEFT PAN +30 deg', 30.0, 0.0),
        ('RIGHT PAN -30 deg', 0.0, -30.0),
        ('RETURN BOTH TO 0 deg', 0.0, 0.0),
    ]:
        print(f'\n=== {phase} ===')
        command_pose(left, right, left_angle, right_angle)
        step_physics(sim, scene, MOVE_STEPS)
        cube_pos = cube.data.root_pos_w[0].detach().clone()
        print('Cube centre in WORLD:', xyz_m(cube_pos))
        for label, robot in [('LEFT', left), ('RIGHT', right)]:
            state = snapshot(robot)
            print_pose(label, state, cube_pos)
            movement_mm = 1000 * torch.linalg.vector_norm(
                state['world_p'] - baseline[label]['world_p']
            ).item()
            pan = torch.rad2deg(state['joints'][joint_index(robot, 'shoulder_pan')]).item()
            print(f'  Measured shoulder_pan: {pan:+.3f} deg')
            print(f'  Gripper-origin displacement from baseline: {movement_mm:.1f} mm')

    print('\n[NOTE] The `gripper` link origin is not yet a verified TCP (tool-centre point).')
    print('[NOTE] These are measured transforms; no IK solve or cube grasp is attempted.')


def main() -> None:
    print('[PREFLIGHT] robot asset:', ROBOT_USD)
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device))
    sim.set_camera_view(eye=(1.25, -1.25, 0.85), target=(0.0, 0.05, 0.15))
    scene_cfg = BimanualSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.left_arm.init_state.pos = LEFT_BASE_M
    scene_cfg.right_arm.init_state.pos = RIGHT_BASE_M
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print('[INFO] Physics initialized; measuring live simulation poses.')
    run_fk_experiment(sim, scene)
    print('\n[INFO] FK experiment complete. Close the window or press Ctrl+C.')
    while app.is_running():
        step_physics(sim, scene, 1)


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
