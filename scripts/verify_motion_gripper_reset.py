"""Measured bimanual motion, gripper motion and nontrivial cube reset.

Run from the benchmark repository root:
    python scripts/verify_motion_gripper_reset.py

This is a regression/diagnostic test, NOT an IK or grasp controller.
No new dependencies or changes to the pinned SO-101 asset are required.

The test deliberately performs THREE independently validated actions:
  1. Move both shoulder_pan joints by safe, previously tested angles; return.
  2. Open/close both gripper joints while reading the moving jaw-link pose.
  3. Teleport the cube 120 mm sideways; verify it IS displaced BEFORE reset;
     then restore its settled pose and zero velocity and check the result.

It reports PASS only after actual measured motion/disturbance meets thresholds.
Lengths in metres internally, joint angles in radians internally.
"""

from __future__ import annotations

import math
import time

# This import performs SO-101 asset preflight and starts Isaac Sim BEFORE
# importing simulation-dependent modules. Reuses our tested task scene.
import inspect_task_layout as layout

import torch
from isaaclab.utils.math import subtract_frame_transforms

PAN_TEST_DEG = 15.0
PAN_MAX_ERROR_DEG = 1.0
MIN_GRIPPER_ORIGIN_MOTION_MM = 35.0
PAN_STEP_COUNT = 150

GRIPPER_ANGLES_DEG = (0.0, 45.0, 90.0, 0.0)
GRIPPER_TARGET_ERROR_DEG = 1.0
GRIPPER_STEP_COUNT = 90

CUBE_SHIFT_X_M = 0.120
RESET_TRIALS = 3
MIN_VERIFIED_CUBE_SHIFT_MM = 100.0
RESET_POSITION_TOLERANCE_MM = 2.0
MAX_RESET_LINEAR_SPEED_M_S = 0.025
MAX_RESET_ANGULAR_SPEED_RAD_S = 0.10
SETTLE_STEPS = 180

# A short wall-clock delay AFTER rendered physics steps lets humans watch
# the motion; it does not affect simulated physics or controller targets.
DISPLAY_DELAY_S = 0.016


def run_steps(sim, scene, count: int, display: bool = False) -> None:
    for _ in range(count):
        layout.step_physics(sim, scene, 1)
        if display:
            time.sleep(DISPLAY_DELAY_S)


def pos3(tensor: torch.Tensor) -> str:
    x, y, z = tensor.detach().cpu().tolist()
    return f"({x:+.4f}, {y:+.4f}, {z:+.4f}) m"


def distance_mm(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b).item() * 1000.0)


def angle_deg(robot, joint_name: str) -> float:
    idx = layout.joint_index(robot, joint_name)
    return math.degrees(float(robot.data.joint_pos[0, idx].item()))


def gripper_origin(robot) -> torch.Tensor:
    idx = layout.body_index(robot, "gripper")
    return robot.data.body_pos_w[0, idx].detach().clone()


def jaw_relative_pose(robot):
    gi = layout.body_index(robot, "gripper")
    ji = layout.body_index(robot, "jaw")
    p, q = subtract_frame_transforms(
        robot.data.body_pos_w[:, gi, :],
        robot.data.body_quat_w[:, gi, :],
        robot.data.body_pos_w[:, ji, :],
        robot.data.body_quat_w[:, ji, :],
    )
    return p[0].detach().clone(), q[0].detach().clone()


def quat_difference_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    # The absolute dot product handles q and -q representing the same rotation.
    a = a / torch.linalg.vector_norm(a)
    b = b / torch.linalg.vector_norm(b)
    dot = float(torch.abs(torch.dot(a, b)).clamp(0.0, 1.0).item())
    return math.degrees(2.0 * math.acos(dot))


def command_grippers(left, right, degrees: float) -> None:
    for arm in (left, right):
        targets = arm.data.default_joint_pos.clone()
        targets[:, layout.joint_index(arm, "gripper")] = math.radians(degrees)
        arm.set_joint_position_target(targets)


def inspect_arm_motion(sim, scene, left, right) -> None:
    print("\n=== 1. VERIFIED ARM MOVEMENT (WATCH BOTH SHOULDERS) ===", flush=True)
    layout.command_pose(left, right, 0.0, 0.0)
    run_steps(sim, scene, SETTLE_STEPS)
    initial_l, initial_r = gripper_origin(left), gripper_origin(right)
    initial_cube = scene["cube"].data.root_pos_w[0].detach().clone()

    print("Commanding LEFT +15 deg, RIGHT -15 deg ...", flush=True)
    layout.command_pose(left, right, PAN_TEST_DEG, -PAN_TEST_DEG)
    run_steps(sim, scene, PAN_STEP_COUNT, display=True)

    observed_l = angle_deg(left, "shoulder_pan")
    observed_r = angle_deg(right, "shoulder_pan")
    travel_l = distance_mm(gripper_origin(left), initial_l)
    travel_r = distance_mm(gripper_origin(right), initial_r)
    print(f"LEFT:  target +{PAN_TEST_DEG:.1f} deg, measured {observed_l:+.2f} deg; gripper-link moved {travel_l:.1f} mm")
    print(f"RIGHT: target -{PAN_TEST_DEG:.1f} deg, measured {observed_r:+.2f} deg; gripper-link moved {travel_r:.1f} mm")

    moved = (
        abs(observed_l - PAN_TEST_DEG) <= PAN_MAX_ERROR_DEG
        and abs(observed_r + PAN_TEST_DEG) <= PAN_MAX_ERROR_DEG
        and travel_l >= MIN_GRIPPER_ORIGIN_MOTION_MM
        and travel_r >= MIN_GRIPPER_ORIGIN_MOTION_MM
    )
    if not moved:
        raise RuntimeError("FAIL ARM MOTION: commanded pan or gripper-link displacement did not occur")

    print("Returning both arms to default ...", flush=True)
    layout.command_pose(left, right, 0.0, 0.0)
    run_steps(sim, scene, PAN_STEP_COUNT, display=True)
    return_l = distance_mm(gripper_origin(left), initial_l)
    return_r = distance_mm(gripper_origin(right), initial_r)
    print(f"Return error: LEFT {return_l:.2f} mm; RIGHT {return_r:.2f} mm")
    if max(return_l, return_r) > 3.0:
        raise RuntimeError("FAIL ARM RETURN: gripper-link did not return near starting position")
    cube_drift = distance_mm(scene["cube"].data.root_pos_w[0], initial_cube)
    print(f"Cube drift during arm test: {cube_drift:.2f} mm")
    if cube_drift > 2.0:
        raise RuntimeError("FAIL NON-CONTACT: cube was disturbed by arm motion")
    print("[ARM MOVEMENT RESULT] PASS — both arm movements AND return measured", flush=True)


def inspect_grippers(sim, scene, left, right) -> None:
    print("\n=== 2. VERIFIED JAW MOVEMENT (WATCH GRIPPERS) ===", flush=True)
    reference_q = {}
    initial_cube = scene["cube"].data.root_pos_w[0].detach().clone()
    max_jaw_rotation = {"LEFT": 0.0, "RIGHT": 0.0}
    for target in GRIPPER_ANGLES_DEG:
        command_grippers(left, right, target)
        run_steps(sim, scene, GRIPPER_STEP_COUNT, display=True)
        print(f"  Commanded {target:.0f} deg:")
        for label, arm in (("LEFT", left), ("RIGHT", right)):
            measured = angle_deg(arm, "gripper")
            rel_p, rel_q = jaw_relative_pose(arm)
            if label not in reference_q:
                reference_q[label] = rel_q
            rotation = quat_difference_deg(reference_q[label], rel_q)
            max_jaw_rotation[label] = max(max_jaw_rotation[label], rotation)
            print(
                f"    {label} measured {measured:+.2f} deg, "
                f"jaw-link rotation since closed {rotation:.2f} deg, "
                f"jaw-frame offset {pos3(rel_p)}"
            )
            if abs(measured - target) > GRIPPER_TARGET_ERROR_DEG:
                raise RuntimeError(f"FAIL GRIPPER MOTION: {label} did not track {target} deg")
    if any(rotation < 20.0 for rotation in max_jaw_rotation.values()):
        raise RuntimeError("FAIL JAW ROTATION: moving jaw orientation did not measurably change")
    cube_drift = distance_mm(scene["cube"].data.root_pos_w[0], initial_cube)
    print(f"Cube drift during jaw test: {cube_drift:.2f} mm")
    if cube_drift > 2.0:
        raise RuntimeError("FAIL NON-CONTACT: cube was disturbed by gripper motion")
    print("[GRIPPER MOVEMENT RESULT] PASS — joint angles and jaw-link rotation verified", flush=True)
    print("[NOTE] Jaw LINK orientation is NOT measured inner-finger clearance or a calibrated TCP.")


def verify_cube_reset(sim, scene) -> None:
    print("\n=== 3. VERIFIED CUBE DISTURBANCE + RESET ===", flush=True)
    cube = scene["cube"]
    run_steps(sim, scene, 60)
    # Capture the ACTUAL settled pose (nominal spawn is 5 mm above the table).
    baseline = cube.data.root_state_w.detach().clone()
    baseline[:, 7:13] = 0.0  # zero both linear and angular momentum
    original = baseline[0, :3].clone()
    original_quat = baseline[0, 3:7].clone()
    print("Baseline cube centre:", pos3(original), flush=True)

    for trial in range(1, RESET_TRIALS + 1):
        print(f"\nTrial {trial}/{RESET_TRIALS}: displace 120 mm, verify, reset, verify", flush=True)
        shifted = baseline.clone()
        shifted[:, 0] += CUBE_SHIFT_X_M  # move horizontally, not vertically
        cube.write_root_state_to_sim(shifted)
        run_steps(sim, scene, 15)
        disturbed_pos = cube.data.root_pos_w[0].detach().clone()
        shifted_mm = abs(float((disturbed_pos[0] - original[0]).item())) * 1000.0
        print("Cube centre AFTER deliberate sideways displacement:", pos3(disturbed_pos), flush=True)
        print(f"Verified sideways displacement from baseline: {shifted_mm:.1f} mm", flush=True)

        if shifted_mm < MIN_VERIFIED_CUBE_SHIFT_MM:
            raise RuntimeError(
                f"FAIL DISTURBANCE on trial {trial}: cube did not move far enough; "
                "reset cannot be awarded PASS"
            )
        # Keep it visibly displaced for >1 sec of real time.
        print("Holding cube at displaced position (look at the viewport) ...", flush=True)
        run_steps(sim, scene, 70, display=True)
        still_shifted = abs(float((cube.data.root_pos_w[0, 0] - original[0]).item())) * 1000.0
        print(f"Displacement BEFORE reset: {still_shifted:.1f} mm")
        if still_shifted < MIN_VERIFIED_CUBE_SHIFT_MM:
            raise RuntimeError(f"FAIL DISTURBANCE PERSISTENCE on trial {trial}: cube already returned")

        print("Restoring cube's settled pose and zeroing velocity ...", flush=True)
        cube.write_root_state_to_sim(baseline)
        cube.reset()  # clears externally applied wrench buffers; pose written above
        run_steps(sim, scene, 70, display=True)
        current = cube.data.root_state_w[0].detach().clone()
        err_mm = distance_mm(current[:3], original)
        orientation_err_deg = quat_difference_deg(current[3:7], original_quat)
        lin_speed = float(torch.linalg.vector_norm(current[7:10]).item())
        ang_speed = float(torch.linalg.vector_norm(current[10:13]).item())
        print("Restored cube centre:", pos3(current[:3]))
        print(f"Position error: {err_mm:.2f} mm (maximum {RESET_POSITION_TOLERANCE_MM:.1f} mm)")
        print(f"Orientation error: {orientation_err_deg:.2f} deg (maximum 2.0 deg)")
        print(f"Linear speed: {lin_speed:.4f} m/s; angular speed: {ang_speed:.4f} rad/s")
        passed = (
            err_mm <= RESET_POSITION_TOLERANCE_MM
            and orientation_err_deg <= 2.0
            and lin_speed <= MAX_RESET_LINEAR_SPEED_M_S
            and ang_speed <= MAX_RESET_ANGULAR_SPEED_RAD_S
        )
        if not passed:
            raise RuntimeError(f"FAIL RESET on trial {trial}: pose or velocity outside tolerances")
        print(f"[CUBE RESET TRIAL {trial}] PASS — displaced >100 mm, THEN restored", flush=True)

    print("[CUBE RESET RESULT] PASS — 3 verified disturbances AND resets", flush=True)


def main() -> None:
    print("[PREFLIGHT] Existing SO-101 asset:", layout.ROBOT_USD, flush=True)
    print("[PREFLIGHT] Using existing task scene, real gravity, no new dependencies.", flush=True)
    print("[NOTE] This test WILL move the arm shoulders and gripper jaws. It does NOT approach the cube.", flush=True)
    sim = layout.SimulationContext(layout.sim_utils.SimulationCfg(device=layout.args.device))
    sim.set_camera_view(eye=(1.00, -1.00, 0.78), target=(0.0, -0.04, 0.10))
    cfg = layout.BimanualSceneCfg(num_envs=1, env_spacing=2.0)
    cfg.left_arm.init_state.pos = layout.LEFT_BASE_M
    cfg.right_arm.init_state.pos = layout.RIGHT_BASE_M
    scene = layout.InteractiveScene(cfg)
    sim.reset()

    left, right = scene["left_arm"], scene["right_arm"]
    inspect_arm_motion(sim, scene, left, right)
    inspect_grippers(sim, scene, left, right)
    verify_cube_reset(sim, scene)

    print("\n[OVERALL RESULT] PASS — arm motion, jaw motion, and meaningful cube reset validated", flush=True)
    print("[NOT YET VERIFIED] Actual finger contact/TCP, IK, collision-free approach, grasp, handover.")
    print("[INFO] Close Isaac Sim or Ctrl+C to exit.", flush=True)
    while layout.app.is_running():
        layout.step_physics(sim, scene, 1)


if __name__ == "__main__":
    try:
        main()
    finally:
        layout.app.close()
