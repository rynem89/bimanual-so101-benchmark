"""Bimanual SO-101 foundation tests: scene, independent arms, and grippers.

This is a *diagnostic* program, not the pick/handover controller.
Run with: python scripts/inspect_bimanual_scene.py --test all

System flow:
  preflight assets -> launch Isaac Sim -> build scene -> initialize physics
  -> inspect metadata -> command joint targets -> step physics -> report state

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
parser.add_argument(
    "--test", choices=("all", "arms", "grippers"), default="all",
    help="Which diagnostics to run after building the scene.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

# POST-LAUNCH: simulation-dependent imports must come after AppLauncher.
import torch
import omni.usd
from pxr import Usd, UsdGeom

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

# This is an exploratory sweep, NOT calibrated OPEN/CLOSED settings.
GRIPPER_SWEEP_DEG = (0.0, 45.0, 90.0, 0.0)
STEPS_PER_TARGET = 180


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


def world_z_bounds(prim_path: str) -> tuple[float, float]:
    """Inspect USD *visual* extents, not an articulation-root pose."""
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"USD prim missing: {prim_path}")
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    ).ComputeWorldBound(prim).ComputeAlignedBox()
    return float(bounds.GetMin()[2]), float(bounds.GetMax()[2])


def joint_index(robot, joint_name: str) -> int:
    """Name lookup prevents accidental dependence on joint ordering."""
    indices, names = robot.find_joints(joint_name, preserve_order=True)
    if len(indices) != 1 or names != [joint_name]:
        raise RuntimeError(f"Expected one '{joint_name}' joint, got: {names}")
    return indices[0]


def degrees(rad: float) -> float:
    return rad * 180.0 / 3.141592653589793


def inspect_scene(scene: InteractiveScene) -> None:
    """Fail early if models, joint structure, or mounting geometry are wrong."""
    print("\n=== SCENE INSPECTION ===")
    table_top = world_z_bounds("/World/envs/env_0/Table")[1]
    for label, robot_key, prim_name in (
        ("LEFT", "left_arm", "Left_Robot"),
        ("RIGHT", "right_arm", "Right_Robot"),
    ):
        robot = scene[robot_key]
        base_bottom = world_z_bounds(f"/World/envs/env_0/{prim_name}/base")[0]
        print(f"{label}: fixed={robot.is_fixed_base}, joints={robot.num_joints}, "
              f"bodies={robot.num_bodies}, base gap={1000 * (base_bottom-table_top):.3f} mm")
        print(f"  joint names: {robot.joint_names}")
        print(f"  body names:  {robot.body_names}")
        gripper_id = joint_index(robot, "gripper")
        limits = robot.data.joint_pos_limits[0, gripper_id]
        print(f"  gripper limits: {degrees(limits[0].item()):.1f}° "
              f"to {degrees(limits[1].item()):.1f}°")
        assert robot.num_joints == 6 and robot.num_bodies == 7
        assert abs(base_bottom - table_top) < 0.001, "Robot base is not flush with table"


def advance(sim: SimulationContext, scene: InteractiveScene, steps: int) -> None:
    """Actuator targets -> Isaac Lab buffers -> PhysX -> refreshed state."""
    dt = sim.get_physics_dt()
    for _ in range(steps):
        if not app.is_running():
            raise RuntimeError("Isaac Sim closed before diagnostics finished")
        scene.write_data_to_sim()  # submit buffered targets
        sim.step()                  # advance physics by dt
        scene.update(dt)            # refresh runtime state (joint_pos etc.)


def command_defaults(left, right) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent (1, 6) target matrices; clones protect default joint poses."""
    return left.data.default_joint_pos.clone(), right.data.default_joint_pos.clone()


def test_independent_arms(sim, scene, left, right) -> None:
    """Move one shoulder while holding the other at zero, then exchange roles."""
    print("\n=== INDEPENDENT ARM CONTROL ===")
    li, ri = joint_index(left, "shoulder_pan"), joint_index(right, "shoulder_pan")
    for label, left_deg, right_deg in (("LEFT", 30.0, 0.0), ("RIGHT", 0.0, -30.0)):
        lt, rt = command_defaults(left, right)
        lt[:, li] = torch.deg2rad(torch.tensor(left_deg, device=left.device))
        rt[:, ri] = torch.deg2rad(torch.tensor(right_deg, device=right.device))
        left.set_joint_position_target(lt)
        right.set_joint_position_target(rt)
        advance(sim, scene, STEPS_PER_TARGET)
        measured_l = degrees(left.data.joint_pos[0, li].item())
        measured_r = degrees(right.data.joint_pos[0, ri].item())
        print(f"{label} active: commanded L={left_deg:+.1f}°, R={right_deg:+.1f}° "
              f"| measured L={measured_l:+.2f}°, R={measured_r:+.2f}°")
        if max(abs(measured_l-left_deg), abs(measured_r-right_deg)) > 2.0:
            raise AssertionError("Shoulder-pan tracking error exceeds 2°")


def test_grippers(sim, scene, left, right) -> None:
    """Explore the gripper's physical direction without assuming OPEN/CLOSED."""
    print("\n=== GRIPPER CHARACTERIZATION (NOT YET CALIBRATED) ===")
    lg, rg = joint_index(left, "gripper"), joint_index(right, "gripper")
    for side, active, other, active_idx, other_idx in (
        ("LEFT", left, right, lg, rg),
        ("RIGHT", right, left, rg, lg),
    ):
        print(f"\n{side} gripper sweep; opposite robot held at default pose")
        for angle in GRIPPER_SWEEP_DEG:
            active_targets = active.data.default_joint_pos.clone()
            other_targets = other.data.default_joint_pos.clone()
            active_targets[:, active_idx] = torch.deg2rad(
                torch.tensor(angle, device=active.device)
            )
            active.set_joint_position_target(active_targets)
            other.set_joint_position_target(other_targets)
            advance(sim, scene, STEPS_PER_TARGET)
            measured = degrees(active.data.joint_pos[0, active_idx].item())
            opposite = degrees(other.data.joint_pos[0, other_idx].item())
            # q[0:5] measures the five arm joints, excluding the gripper.
            arm_drift = (active.data.joint_pos[0, :5] -
                         active.data.default_joint_pos[0, :5]).abs().max().item()
            print(f"  target={angle:5.1f}° | measured={measured:6.2f}° "
                  f"| other gripper={opposite:6.2f}° "
                  f"| max arm-joint deviation={degrees(arm_drift):.3f}°")
            if abs(measured - angle) > 3.0:
                print("  [NOTE] Target not reached; inspect actuator/contact behaviour.")
    print("\nObserve the GUI to establish which angle opens/closes the jaw. "
          "Do not label values OPEN/CLOSED from joint limits alone.")


def main() -> None:
    print(f"[PREFLIGHT] robot asset: {ROBOT_USD}")
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device))
    sim.set_camera_view(eye=(1.25, -1.25, 0.85), target=(0.0, 0.05, 0.15))

    config = BimanualSceneCfg(num_envs=1, env_spacing=2.0)
    config.left_arm.init_state.pos = LEFT_BASE_M
    config.right_arm.init_state.pos = RIGHT_BASE_M
    scene = InteractiveScene(config)
    sim.reset()  # creates/initializes PhysX articulation handles

    inspect_scene(scene)
    left, right = scene["left_arm"], scene["right_arm"]
    if args.test in ("all", "arms"):
        test_independent_arms(sim, scene, left, right)
    if args.test in ("all", "grippers"):
        test_grippers(sim, scene, left, right)

    print("\n[INFO] Tests complete. Close the Isaac Sim window to exit.")
    while app.is_running():
        advance(sim, scene, 1)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
