"""Inspect the minimal bimanual SO-101 simulation scene.

Milestone:
    - spawn two SO-101 follower arms
    - spawn a simple static workspace/table
    - spawn one dynamic cube
    - inspect the runtime articulation structure

No robot control is implemented yet.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# Command-line interface and Isaac Sim launch
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Spawn and inspect a minimal bimanual SO-101 scene."
)

# AppLauncher adds Isaac Lab arguments such as --device and --headless.
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# Isaac Sim must be launched before importing most simulation-dependent modules.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Simulation imports
#
# These imports intentionally come AFTER AppLauncher has started Isaac Sim.
# -----------------------------------------------------------------------------

import torch
import omni.usd
from pxr import Usd, UsdGeom
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG


# -----------------------------------------------------------------------------
# Scene constants
#
# Units:
#   position / size -> metres
#   joint angles    -> radians
# -----------------------------------------------------------------------------

TABLE_SIZE = (1.20, 0.80, 0.05)
TABLE_TOP_Z = 0.0

# Measured from the SO-101 USD:
# visible base bottom is 30.081 mm above the articulation root frame.
ROBOT_BASE_Z_OFFSET = -0.030081

LEFT_ARM_POS = (-0.25, -0.30, ROBOT_BASE_Z_OFFSET)
RIGHT_ARM_POS = (0.25, -0.30, ROBOT_BASE_Z_OFFSET)

CUBE_SIZE = 0.04
CUBE_POS = (0.0, 0.10, TABLE_TOP_Z + 0.025)


# -----------------------------------------------------------------------------
# Scene configuration
# -----------------------------------------------------------------------------

@configclass
class BimanualInspectionSceneCfg(InteractiveSceneCfg):
    """Minimal scene containing two SO-101 arms, a table, and a cube."""

    # Static workspace.
    #
    # AssetBaseCfg means that we want Isaac Lab to spawn this asset,
    # but we do not need a runtime Python object for controlling it.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.35, 0.35, 0.35)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # The table is 5 cm thick.
            # Putting its centre at -2.5 cm makes its top exactly z = 0.
            pos=(0.0, 0.0, -TABLE_SIZE[2] / 2.0),
        ),
    )

    # Dynamic cube.
    #
    # RigidObjectCfg gives the cube a runtime rigid-body representation,
    # so later we can read and reset its pose and velocity.
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CUBE_POS,
        ),
    )

    # Two independent SO-101 articulations.
    #
    # We reuse LeIsaac's validated SO-101 configuration but give each robot
    # its own USD prim path in our scene.
    left_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Left_Robot"
    )

    right_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Right_Robot"
    )

    # Lighting for GUI inspection.
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def print_vertical_bounds(prim_path: str, label: str) -> tuple[float, float]:
    """Print the world-space vertical bounds of a USD prim."""

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise RuntimeError(f"Invalid USD prim path: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
    )

    world_bound = bbox_cache.ComputeWorldBound(prim)
    aligned_box = world_bound.ComputeAlignedBox()

    min_z = aligned_box.GetMin()[2]
    max_z = aligned_box.GetMax()[2]

    print(
        f"{label:<18} min_z = {min_z:.6f} m, "
        f"max_z = {max_z:.6f} m"
    )

    return float(min_z), float(max_z)

def print_articulation_info(name: str, robot) -> None:
    """Print the runtime structure and initial state of one articulation."""

    print("\n" + "=" * 80)
    print(f"{name} SO-101")
    print("=" * 80)

    print(f"Prim path:      {robot.cfg.prim_path}")
    print(f"Fixed base:     {robot.is_fixed_base}")
    print(f"Joint count:    {robot.num_joints}")
    print(f"Body count:     {robot.num_bodies}")

    print("\nJoint structure:")
    print(
        f"{'idx':>3}  {'joint':<20} "
        f"{'lower [rad]':>12} {'upper [rad]':>12} "
        f"{'lower [deg]':>12} {'upper [deg]':>12} "
        f"{'initial [rad]':>14}"
    )

    joint_limits = robot.data.joint_pos_limits[0].detach().cpu()
    initial_joint_pos = robot.data.default_joint_pos[0].detach().cpu()

    for index, joint_name in enumerate(robot.joint_names):
        lower_rad = joint_limits[index, 0].item()
        upper_rad = joint_limits[index, 1].item()
        initial_rad = initial_joint_pos[index].item()

        lower_deg = torch.rad2deg(joint_limits[index, 0]).item()
        upper_deg = torch.rad2deg(joint_limits[index, 1]).item()

        print(
            f"{index:>3}  "
            f"{joint_name:<20} "
            f"{lower_rad:>12.4f} "
            f"{upper_rad:>12.4f} "
            f"{lower_deg:>12.2f} "
            f"{upper_deg:>12.2f} "
            f"{initial_rad:>14.4f}"
        )

    print("\nBody / link structure:")

    for index, body_name in enumerate(robot.body_names):
        print(f"  [{index:>2}] {body_name}")

    # LeIsaac's SO-101 IK configurations use the body named "gripper"
    # as the end-effector body. Verify that it actually exists at runtime.
    gripper_indices, gripper_names = robot.find_bodies(
        "gripper",
        preserve_order=True,
    )

    print("\nEnd-effector / gripper lookup:")
    print(f"  matching indices: {gripper_indices}")
    print(f"  matching names:   {gripper_names}")

    print("\nInitial articulation state:")

    print(
        "  root position [m]:",
        robot.data.default_root_state[0, :3].detach().cpu().tolist(),
    )

    print(
        "  root quaternion [w, x, y, z]:",
        robot.data.default_root_state[0, 3:7].detach().cpu().tolist(),
    )

    print(
        "  joint positions [rad]:",
        robot.data.default_joint_pos[0].detach().cpu().tolist(),
    )


def print_control_summary(left_arm, right_arm) -> None:
    """Describe the control dimensionality available at articulation level."""

    left_dofs = left_arm.num_joints
    right_dofs = right_arm.num_joints

    print("\n" + "=" * 80)
    print("CONTROL INTERFACE SUMMARY")
    print("=" * 80)

    print(f"Left articulation DOFs:   {left_dofs}")
    print(f"Right articulation DOFs:  {right_dofs}")
    print(f"Combined articulation DOFs: {left_dofs + right_dofs}")

    print(
        "\nNo ManagerBasedRLEnv / ActionManager exists yet, "
        "so this scene does not yet have a formal environment action dimension."
    )

def get_single_joint_index(robot, joint_name: str) -> int:
    """Resolve one joint name to exactly one runtime joint index."""

    joint_indices, joint_names = robot.find_joints(
        joint_name,
        preserve_order=True,
    )

    if len(joint_indices) != 1:
        raise RuntimeError(
            f"Expected exactly one joint named '{joint_name}', "
            f"but found indices={joint_indices}, names={joint_names}"
        )

    return joint_indices[0]

def run_independent_joint_control_test(
    sim: SimulationContext,
    scene: InteractiveScene,
    left_arm,
    right_arm,
) -> None:
    """Verify that each SO-101 articulation can be commanded independently."""

    sim_dt = sim.get_physics_dt()

    left_shoulder_pan = get_single_joint_index(left_arm, "shoulder_pan")
    right_shoulder_pan = get_single_joint_index(right_arm, "shoulder_pan")

    print("\n" + "=" * 80)
    print("INDEPENDENT JOINT CONTROL TEST")
    print("=" * 80)

    print(f"Left shoulder_pan index:  {left_shoulder_pan}")
    print(f"Right shoulder_pan index: {right_shoulder_pan}")

    # Start from each robot's default joint configuration.
    left_target = left_arm.data.default_joint_pos.clone()
    right_target = right_arm.data.default_joint_pos.clone()

    # -------------------------------------------------------------------------
    # Phase 1: move only the LEFT shoulder_pan to +30 degrees.
    # -------------------------------------------------------------------------

    left_target[:, left_shoulder_pan] = torch.deg2rad(
        torch.tensor(
            30.0,
            device=left_arm.device,
        )
    )

    print("\n[TEST] Phase 1: LEFT shoulder_pan -> +30 deg")
    print("[TEST] Right arm should remain at its default pose.")

    left_arm.set_joint_position_target(left_target)
    right_arm.set_joint_position_target(right_target)

    for _ in range(240):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    left_pos = left_arm.data.joint_pos[0, left_shoulder_pan].item()
    right_pos = right_arm.data.joint_pos[0, right_shoulder_pan].item()

    print(
        f"[RESULT] Left shoulder_pan:  "
        f"{torch.rad2deg(torch.tensor(left_pos)).item():.2f} deg"
    )
    print(
        f"[RESULT] Right shoulder_pan: "
        f"{torch.rad2deg(torch.tensor(right_pos)).item():.2f} deg"
    )

    # -------------------------------------------------------------------------
    # Phase 2: return left to default and move only the RIGHT shoulder_pan.
    # -------------------------------------------------------------------------

    left_target = left_arm.data.default_joint_pos.clone()
    right_target = right_arm.data.default_joint_pos.clone()

    right_target[:, right_shoulder_pan] = torch.deg2rad(
        torch.tensor(
            -30.0,
            device=right_arm.device,
        )
    )

    print("\n[TEST] Phase 2: RIGHT shoulder_pan -> -30 deg")
    print("[TEST] Left arm should return to its default pose.")

    left_arm.set_joint_position_target(left_target)
    right_arm.set_joint_position_target(right_target)

    for _ in range(240):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    left_pos = left_arm.data.joint_pos[0, left_shoulder_pan].item()
    right_pos = right_arm.data.joint_pos[0, right_shoulder_pan].item()

    print(
        f"[RESULT] Left shoulder_pan:  "
        f"{torch.rad2deg(torch.tensor(left_pos)).item():.2f} deg"
    )
    print(
        f"[RESULT] Right shoulder_pan: "
        f"{torch.rad2deg(torch.tensor(right_pos)).item():.2f} deg"
    )

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    """Create the scene, initialize physics, inspect it, then keep GUI alive."""

    # Verify that our external LeIsaac asset location was explicitly provided.
    assets_root_env = os.environ.get("LEISAAC_ASSETS_ROOT")

    if assets_root_env is None:
        raise RuntimeError(
            "LEISAAC_ASSETS_ROOT is not set. "
            "Set it to ~/robotics/leisaac/assets before launching this script."
        )

    assets_root = Path(assets_root_env).expanduser().resolve()
    so101_asset = assets_root / "robots" / "so101_follower.usd"

    if not so101_asset.is_file():
        raise FileNotFoundError(
            f"SO-101 USD was not found at: {so101_asset}"
        )

    print(f"[INFO] LeIsaac assets: {assets_root}")
    print(f"[INFO] SO-101 USD:      {so101_asset}")

    # SimulationContext owns the physics stepping configuration.
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Camera is only for the interactive GUI view.
    sim.set_camera_view(
        eye=(1.25, -1.25, 0.85),
        target=(0.0, 0.05, 0.15),
    )

    # Build one environment containing our four physical entities.
    scene_cfg = BimanualInspectionSceneCfg(
        num_envs=1,
        env_spacing=2.0,
    )

    # Initial base locations are part of OUR task geometry.
    # We intentionally keep LeIsaac's validated robot model itself unchanged.
    scene_cfg.left_arm.init_state.pos = LEFT_ARM_POS
    scene_cfg.right_arm.init_state.pos = RIGHT_ARM_POS

    scene = InteractiveScene(scene_cfg)

    # sim.reset() starts/initializes physics.
    #
    # Runtime joint/body metadata becomes valid only after the PhysX
    # articulations have been created.
    sim.reset()

    print("\n[INFO] Simulation initialized successfully.")

    left_arm = scene["left_arm"]
    right_arm = scene["right_arm"]

    print("\n" + "=" * 80)
    print("VERTICAL GEOMETRY BOUNDS")
    print("=" * 80)

    _, table_top = print_vertical_bounds(
        "/World/envs/env_0/Table",
        "Table",
    )

    left_base_bottom, _ = print_vertical_bounds(
        "/World/envs/env_0/Left_Robot/base",
        "Left base",
    )

    right_base_bottom, _ = print_vertical_bounds(
        "/World/envs/env_0/Right_Robot/base",
        "Right base",
    )

    print(f"\nLeft visual gap:  {left_base_bottom - table_top:.6f} m")
    print(f"Right visual gap: {right_base_bottom - table_top:.6f} m")

    print_articulation_info("LEFT", left_arm)
    print_articulation_info("RIGHT", right_arm)
    print_control_summary(left_arm, right_arm)

    run_independent_joint_control_test(
    sim,
    scene,
    left_arm,
    right_arm,
)

    # Keep the simulation alive for visual inspection.
    sim_dt = sim.get_physics_dt()

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()