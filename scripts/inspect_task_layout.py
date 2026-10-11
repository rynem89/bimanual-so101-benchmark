"""Inspect bimanual SO-101 frame poses in a first complete task layout.

This is a measured forward-kinematics experiment, not a grasp or IK controller.
Run with: python scripts/inspect_task_layout.py

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
# Provisional: cube is forward and outside the left robot. Confirm reachability with IK.
# Start 5 mm above table so the dynamic cube can settle under gravity.
CUBE_START_M = (-0.34, -0.04, TABLE_TOP_Z_M + CUBE_EDGE_M / 2 + 0.005)
# Provisional: receiving bin is forward and outside the right robot.
# Static bin remains unchanged in size; check approach clearance after IK.
BIN_CENTER_XY_M = (0.34, -0.04)
BIN_OUTER_XY_M = (0.16, 0.16)
BIN_WALL_M = 0.01
BIN_FLOOR_M = 0.01
BIN_WALL_HEIGHT_M = 0.075
HANDOVER_CENTER_M = (0.0, 0.015, 0.18)


def static_box_cfg(name: str, size_m: tuple, centre_m: tuple, colour: tuple):
    """One rigidly mounted collidable part, not an actuated/movable object."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size_m,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=centre_m),
    )


BX, BY = BIN_CENTER_XY_M
SX, SY = BIN_OUTER_XY_M
WT = BIN_WALL_M
FZ = BIN_FLOOR_M / 2
WZ = BIN_FLOOR_M + BIN_WALL_HEIGHT_M / 2
BIN_INNER_X_M = SX - 2 * WT
BIN_INNER_Y_M = SY - 2 * WT


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

    # Open-top receiving bin: floor + four fixed collidable walls. Every
    # component has a separate USD prim so geometry is easy to debug and edit.
    bin_floor = static_box_cfg(
        "BinFloor", (SX, SY, BIN_FLOOR_M), (BX, BY, FZ), (0.12, 0.45, 0.65)
    )
    bin_left_wall = static_box_cfg(
        "BinLeftWall", (WT, SY, BIN_WALL_HEIGHT_M),
        (BX - SX / 2 + WT / 2, BY, WZ), (0.12, 0.45, 0.65)
    )
    bin_right_wall = static_box_cfg(
        "BinRightWall", (WT, SY, BIN_WALL_HEIGHT_M),
        (BX + SX / 2 - WT / 2, BY, WZ), (0.12, 0.45, 0.65)
    )
    bin_front_wall = static_box_cfg(
        "BinFrontWall", (SX - 2 * WT, WT, BIN_WALL_HEIGHT_M),
        (BX, BY - SY / 2 + WT / 2, WZ), (0.12, 0.45, 0.65)
    )
    bin_back_wall = static_box_cfg(
        "BinBackWall", (SX - 2 * WT, WT, BIN_WALL_HEIGHT_M),
        (BX, BY + SY / 2 - WT / 2, WZ), (0.12, 0.45, 0.65)
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
    print('\n=== TASK LAYOUT (PROVISIONAL) ===')
    print('Cube requested start:', CUBE_START_M, 'm')
    print('Cube settled centre:', xyz_m(cube_pos))
    print('Bin centre XY:', BIN_CENTER_XY_M, 'm')
    print('Cube XY relative to left robot root:', (round(CUBE_START_M[0] - LEFT_BASE_M[0], 3), round(CUBE_START_M[1] - LEFT_BASE_M[1], 3)), 'm')
    print('Bin XY relative to right robot root:', (round(BX - RIGHT_BASE_M[0], 3), round(BY - RIGHT_BASE_M[1], 3)), 'm')
    print('Bin clear opening XY:', (BIN_INNER_X_M, BIN_INNER_Y_M), 'm')
    print('Bin interior floor Z:', BIN_FLOOR_M, 'm')
    print('Bin wall top Z:', BIN_FLOOR_M + BIN_WALL_HEIGHT_M, 'm')
    print('Candidate mid-air handover XYZ:', HANDOVER_CENTER_M, 'm')
    fits = BIN_INNER_X_M > CUBE_EDGE_M and BIN_INNER_Y_M > CUBE_EDGE_M
    print('[GEOMETRY] Cube nominally fits bin XY:', fits)
    cube_xy = cube_pos[:2].detach().cpu().tolist()
    inside_bin = (abs(cube_xy[0] - BX) < SX/2 + CUBE_EDGE_M/2 and
                  abs(cube_xy[1] - BY) < SY/2 + CUBE_EDGE_M/2)
    print('[GEOMETRY] Cube starts clear of bin footprint:', not inside_bin)
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
    print('[NOTE] Bin is a static five-part open receptacle; no placement success is claimed.')


def main() -> None:
    print('[PREFLIGHT] robot asset:', ROBOT_USD)
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device))
    sim.set_camera_view(eye=(1.25, -1.25, 0.85), target=(0.0, 0.08, 0.10))
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
