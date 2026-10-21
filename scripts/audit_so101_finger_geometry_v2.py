"""SO-101 finger geometry audit, using the verified Isaac Lab scene and jaw control.

Unlike the previous experiment, this is NOT a TCP calibration claim:
- Inspects the full composed USD hierarchy including instance proxies.
- Detects collision API inherited through geometry parent prims (common for URDF assets).
- Records mesh/shape bounds relative to their rigid link when accessible.
- Produces useful diagnostic reports even when finger geometry cannot be resolved.
- Commands visible, wall-clock-paced jaw movements and measures actual joint/link motion.
- Explicitly reports GEOMETRY INCOMPLETE instead of a misleading PASS on zero shapes.

From ~/robotics/bimanual-so101-benchmark:
    python scripts/audit_so101_finger_geometry_v2.py

Outputs: outputs/finger_audit/{hierarchy.txt, geometry.csv, measurements.json}
Use --headless for a finite physics test; --output-dir overrides the report directory.
Use --offline with the installed USD library paths to inspect the asset without Kit.
No added dependencies, no USD changes, no robot arm sweeps, no IK, no grasp.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument("--offline", action="store_true", help="Read USD only; no simulator or measured motion")
parser.add_argument("--output-dir", type=Path, default=Path("outputs/finger_audit"))
audit_args, remaining_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining_args]
if not audit_args.offline:
    # AppLauncher must run before importing simulation-dependent USD modules.
    import inspect_task_layout as layout

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics
if not audit_args.offline:
    from isaaclab.utils.math import subtract_frame_transforms

OUT_DIR = audit_args.output_dir
ROBOT_ROOT = "/World/envs/env_0/Left_Robot"
MOVING_ANGLES_DEG = (0.0, 45.0, 90.0, 0.0)
STEPS_PER_POSE = 120
VIEWER_PACING_S = 0.020  # Real-time pause *between* simulation ticks, not blocking frame updates.
MAX_CUBE_DRIFT_MM = 2.0
MAX_GRIPPER_TRACK_ERROR_DEG = 2.0
COLLISION_TRIANGLES = {}


def steps(sim, scene, count: int, visible: bool = False) -> None:
    for _ in range(count):
        layout.step_physics(sim, scene, 1)
        if visible:
            time.sleep(VIEWER_PACING_S)


def angle_deg(arm) -> float:
    idx = layout.joint_index(arm, "gripper")
    return math.degrees(float(arm.data.joint_pos[0, idx].item()))


def relative_jaw(arm):
    fixed = layout.body_index(arm, "gripper")
    jaw = layout.body_index(arm, "jaw")
    p, q = subtract_frame_transforms(
        arm.data.body_pos_w[:, fixed, :], arm.data.body_quat_w[:, fixed, :],
        arm.data.body_pos_w[:, jaw, :], arm.data.body_quat_w[:, jaw, :],
    )
    return p[0].detach().clone(), q[0].detach().clone()


def rotation_diff_deg(a, b):
    dot = float(torch.abs(torch.dot(a / a.norm(), b / b.norm())).clamp(0, 1).item())
    return math.degrees(2.0 * math.acos(dot))


def command_grippers(left, right, angle: float):
    for arm in (left, right):
        command = arm.data.default_joint_pos.clone()
        command[:, layout.joint_index(arm, "gripper")] = math.radians(angle)
        arm.set_joint_position_target(command)


def full_prims(root):
    """OpenUSD traversal that includes descendants of instanceable prims."""
    return list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))


def rigid_owner_path(prim, robot_path):
    """Find nearest ancestor carrying RigidBodyAPI (including the geometry itself)."""
    cursor = prim
    robot_len = len(robot_path)
    while cursor and cursor.IsValid() and len(str(cursor.GetPath())) >= robot_len:
        if cursor.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(cursor.GetPath())
        cursor = cursor.GetParent()
    return None


def ancestor_collision_state(prim, link_root):
    """Find an authored collision schema on any ancestor inside this link.

    A CollisionAPI may be applied to an Xform containing a Mesh, not to Mesh itself.
    Returning 'disabled' as well as 'enabled' avoids treating explicitly disabled collision as active.
    """
    cursor = prim
    saw = False
    disabled = False
    while cursor and cursor.IsValid():
        if cursor.HasAPI(UsdPhysics.CollisionAPI):
            saw = True
            enabled_attr = UsdPhysics.CollisionAPI(cursor).GetCollisionEnabledAttr().Get()
            if enabled_attr is False:
                disabled = True
        if cursor.GetPath() == link_root.GetPath():
            break
        cursor = cursor.GetParent()
    return ("disabled" if disabled else "enabled") if saw else "not-authored"


def relative_bounds(prim, link_root, bbox_cache):
    """Conservative USD AABB expressed in the link frame, not a contact-surface model."""
    bbox = bbox_cache.ComputeRelativeBound(prim, link_root).ComputeAlignedRange()
    if bbox.IsEmpty():
        return None
    minimum, maximum = bbox.GetMin(), bbox.GetMax()
    lo = [float(minimum[i]) for i in range(3)]
    hi = [float(maximum[i]) for i in range(3)]
    if any(not math.isfinite(v) for v in lo + hi):
        return None
    return lo, hi


def mesh_evidence(prim, link_root, xforms):
    """Read ALL authored vertices/triangles in the rigid-link frame, in metres.

    USD uses row-vector matrices. These surfaces are inputs to convex
    decomposition, not the cooked PhysX hulls or measured contact surfaces.
    """
    mesh = UsdGeom.Mesh(prim)
    raw = mesh.GetPointsAttr().Get()
    if raw is None or not len(raw):
        raise RuntimeError(f"Missing mesh points: {prim.GetPath()}")
    transform = xforms.GetLocalToWorldTransform(prim) * xforms.GetLocalToWorldTransform(link_root).GetInverse()
    points = np.array([transform.Transform(Gf.Vec3d(*p)) for p in raw], dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    if not np.isfinite(points).all() or not np.all(counts == 3) or indices.size != 3 * counts.size:
        raise RuntimeError(f"Expected finite triangle mesh: {prim.GetPath()}")
    if indices.size == 0 or indices.min() < 0 or indices.max() >= len(points):
        raise RuntimeError(f"Invalid mesh topology: {prim.GetPath()}")
    triangles = points[indices.reshape(-1, 3)]
    if ancestor_collision_state(prim, link_root) == "enabled":
        COLLISION_TRIANGLES[str(prim.GetPath())] = triangles
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(cross, axis=1)
    valid = length > 1e-14
    normals = cross[valid] / length[valid, None]
    centres = triangles[valid].mean(axis=1)
    # Group coplanar triangles to identify candidate pads without selecting
    # housing/hinge vertices by nearest distance. No automatic pad certification.
    offsets = (normals * centres).sum(axis=1)
    keys = np.column_stack((np.round(normals, 4), np.round(offsets, 5)))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    areas = np.bincount(inverse, weights=length[valid] / 2)
    patches = []
    for i in np.argsort(areas)[-12:][::-1]:
        selected = inverse == i
        weights = length[valid][selected]
        patches.append({"normal_link": unique[i, :3].tolist(), "plane_offset_m": float(unique[i, 3]),
                        "area_mm2": float(areas[i] * 1e6),
                        "centroid_link_m": np.average(centres[selected], axis=0, weights=weights).tolist()})
    return points.min(axis=0).tolist(), points.max(axis=0).tolist(), len(points), len(counts), patches


def x_surface_intersections(triangles, y_m, z_m):
    """Intersect an X-directed line with triangles using YZ barycentrics.

    This measures surfaces at a stated section, not AABB or vertex proximity.
    Parallel triangles are excluded; repeated edge hits are deduplicated.
    """
    a = triangles[:, 0]
    u, v = triangles[:, 1] - a, triangles[:, 2] - a
    determinant = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    valid = np.abs(determinant) > 1e-15
    a, u, v, determinant = a[valid], u[valid], v[valid], determinant[valid]
    dy, dz = y_m - a[:, 1], z_m - a[:, 2]
    b = (dy * v[:, 2] - dz * v[:, 1]) / determinant
    c = (u[:, 1] * dz - u[:, 2] * dy) / determinant
    inside = (b >= -1e-8) & (c >= -1e-8) & (b + c <= 1 + 1e-8)
    return np.unique(np.round((a[:, 0] + b * u[:, 0] + c * v[:, 0])[inside], 9))


def finger_sections(relative_p, relative_q):
    """Diagnostic gaps across distal fingers, using the measured jaw transform.

    A 1-D section cannot establish 3-D cube fit or effective PhysX clearance.
    At wide jaw angles the moving pad can leave a section entirely.
    """
    fixed = [t for path, t in COLLISION_TRIANGLES.items() if "/gripper/" in path and "wrist_roll_follower" in path]
    moving = [t for path, t in COLLISION_TRIANGLES.items() if "/jaw/" in path]
    if len(fixed) != 1 or len(moving) != 1:
        raise RuntimeError("Expected exactly one fixed-finger and one moving-jaw mesh")
    q = relative_q.detach().cpu().numpy().astype(float)
    q /= np.linalg.norm(q)
    points = moving[0]
    cross = 2 * np.cross(q[1:], points)
    moved = points + q[0] * cross + np.cross(q[1:], cross) + relative_p.detach().cpu().numpy()
    sections = []
    for z in (-0.06, -0.07, -0.08, -0.09, -0.10):
        f = x_surface_intersections(fixed[0], 0.0, z)
        m = x_surface_intersections(moved, 0.0, z)
        sections.append({"y_gripper_m": 0.0, "z_gripper_m": z,
                         "fixed_surface_x_m": f.tolist(), "moving_surface_x_m": m.tolist(),
                         "section_gap_mm": float((m.min() - f.max()) * 1000) if len(f) and len(m) else None})
    return sections


def show_section(stage, arm, sections):
    """Draw the z=-80 mm section in yellow, without adding a physics collider."""
    from isaaclab.utils.math import quat_apply

    curve = UsdGeom.BasisCurves.Define(stage, "/World/FingerSurfaceSection")
    section = next(s for s in sections if s["z_gripper_m"] == -0.08)
    if section["section_gap_mm"] is None:
        UsdGeom.Imageable(curve).MakeInvisible()
        return
    local = torch.tensor([[max(section["fixed_surface_x_m"]), 0, -0.08],
                          [min(section["moving_surface_x_m"]), 0, -0.08]], device=arm.device)
    idx = layout.body_index(arm, "gripper")
    world = quat_apply(arm.data.body_quat_w[0, idx].expand(2, -1), local) + arm.data.body_pos_w[0, idx]
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr([Gf.Vec3f(*p) for p in world.cpu().tolist()])
    curve.CreateWidthsAttr([0.001])
    curve.CreateDisplayColorAttr([Gf.Vec3f(1, 0.8, 0)])
    UsdGeom.Imageable(curve).MakeVisible()


def usd_audit(stage):
    root = stage.GetPrimAtPath(ROBOT_ROOT)
    if not root.IsValid():
        raise RuntimeError(f"Cannot find robot at {ROBOT_ROOT}; do not guess a replacement path")
    prims = full_prims(root)
    hierarchy_lines = [
        "SO-101 COMPLETE COMPOSED USD HIERARCHY (LEFT ROBOT)",
        "Instance proxies are included. This describes authored geometry, not effective PhysX contact distances.",
        f"Robot root: {ROBOT_ROOT}", f"Prims enumerated: {len(prims)}", "",
    ]
    body_roots = {name: stage.GetPrimAtPath(f"{ROBOT_ROOT}/{name}") for name in ("gripper", "jaw")}
    for name, link_root in body_roots.items():
        hierarchy_lines.append(f"EXPECTED LINK /{name}: exists={link_root.IsValid()} instance={link_root.IsInstance() if link_root.IsValid() else '-'}")
    hierarchy_lines.append("")

    for prim in prims:
        if str(prim.GetPath()) == ROBOT_ROOT:
            continue
        flags = []
        if prim.IsInstance(): flags.append("INSTANCE")
        if prim.IsInstanceProxy(): flags.append("PROXY")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI): flags.append("RIGID")
        if prim.HasAPI(UsdPhysics.CollisionAPI): flags.append("COLLISION_API")
        if prim.IsA(UsdGeom.Gprim): flags.append("GEOMETRY")
        owner = rigid_owner_path(prim, ROBOT_ROOT)
        hierarchy_lines.append(
            f"{str(prim.GetPath())} | type={prim.GetTypeName() or 'untyped'} | "
            f"flags={','.join(flags) or '-'} | nearest_rigid={owner or '-'}"
        )

    bboxes = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        useExtentsHint=False,
        ignoreVisibility=True,
    )
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    records = []
    for name, link_root in body_roots.items():
        if not link_root.IsValid():
            continue
        children = full_prims(link_root)
        for prim in children:
            if not prim.IsA(UsdGeom.Gprim) or not prim.IsA(UsdGeom.Boundable):
                continue
            # If there are nested rigid bodies, they belong to a different link.
            owner = rigid_owner_path(prim, ROBOT_ROOT)
            if owner and owner != str(link_root.GetPath()):
                continue
            collision = ancestor_collision_state(prim, link_root)
            evidence = None
            try:
                bound = relative_bounds(prim, link_root, bboxes)
                if prim.IsA(UsdGeom.Mesh):
                    evidence = mesh_evidence(prim, link_root, xforms)
                    if bound is None or not np.allclose(np.array(bound), np.array(evidence[:2]), atol=1e-6, rtol=0):
                        raise RuntimeError("USD bounds and independent mesh-vertex bounds disagree >1 micrometre")
                problem = ""
            except Exception as exc:
                bound = None
                problem = f"{type(exc).__name__}: {exc}"
            records.append({
                "link": name,
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "instance_proxy": prim.IsInstanceProxy(),
                "collision_schema": collision,
                "purpose": str(UsdGeom.Imageable(prim).ComputePurpose()),
                "vertex_count": evidence[2] if evidence else 0,
                "triangle_count": evidence[3] if evidence else 0,
                "surface_patches": evidence[4] if evidence else [],
                "min_x_m": bound[0][0] if bound else "",
                "min_y_m": bound[0][1] if bound else "",
                "min_z_m": bound[0][2] if bound else "",
                "max_x_m": bound[1][0] if bound else "",
                "max_y_m": bound[1][1] if bound else "",
                "max_z_m": bound[1][2] if bound else "",
                "problem": problem or ("no bound" if bound is None else ""),
            })

    # If expected links are empty, keep full robot geometry inventory in the report
    # rather than silently asserting 'no colliders'. It's vital for diagnosis.
    for name in ("gripper", "jaw"):
        items = [r for r in records if r["link"] == name]
        colliders = sum(r["collision_schema"] == "enabled" for r in items)
        print(f"[USD] {name}: {len(items)} boundable shapes; {colliders} collision-tagged shapes", flush=True)
        if items:
            for record in items[:15]:
                print(f"      {record['type']} {record['path']} ({record['collision_schema']})")
        else:
            print(f"[USD] WARNING: no shape found under {name}; check hierarchy report", flush=True)

    print(f"[USD] Total composed prims under left robot: {len(prims)}", flush=True)
    return hierarchy_lines, records


def write_reports(hierarchy, geom, measurements, status):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hierarchy.txt").write_text("\n".join(hierarchy) + "\n", encoding="utf-8")
    with (OUT_DIR / "geometry.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["link", "path", "type", "instance_proxy", "collision_schema", "purpose", "vertex_count", "triangle_count", "min_x_m", "min_y_m", "min_z_m", "max_x_m", "max_y_m", "max_z_m", "problem"]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(geom)
    payload = {
        "status": status,
        "robot_usd": str(robot_usd()),
        "task_scene": "scripts/inspect_task_layout.py",
        "note": "Geometry bounds are not fingertip contact surfaces. TCP is not calibrated.",
        "geometry_record_count": len(geom),
        "geometry": geom,
        "mode": "offline authored geometry" if audit_args.offline else "live simulation",
        "jaw_joint_and_frame_measurements": measurements,
    }
    (OUT_DIR / "measurements.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for filename in ("hierarchy.txt", "geometry.csv", "measurements.json"):
        file = OUT_DIR / filename
        print(f"[REPORT] {file} ({file.stat().st_size} bytes)", flush=True)


def robot_usd():
    assets = Path(os.environ.get("LEISAAC_ASSETS_ROOT", Path(__file__).resolve().parents[2] / "leisaac/assets"))
    return assets / "robots/so101_follower.usd"


def validate_geometry(geom):
    colliders = [r for r in geom if r["collision_schema"] == "enabled"]
    if {r["link"] for r in colliders} != {"gripper", "jaw"} or any(r["problem"] or not r["triangle_count"] for r in colliders):
        raise RuntimeError("GEOMETRY INCOMPLETE: missing collider triangles or invalid bounds; see reports")


def offline_main():
    global ROBOT_ROOT
    stage = Usd.Stage.Open(str(robot_usd()))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise RuntimeError("Expected asset in metres")
    ROBOT_ROOT = str(stage.GetDefaultPrim().GetPath())
    hierarchy, geom = usd_audit(stage)
    write_reports(hierarchy, geom, [], "AUTHORED GEOMETRY INSPECTED; TCP/contact NOT VERIFIED")
    validate_geometry(geom)
    print("[PASS] Authored collision triangles and bounds cross-check only; no physics run")


def main():
    sim = layout.SimulationContext(layout.sim_utils.SimulationCfg(device=layout.args.device))
    cfg = layout.BimanualSceneCfg(num_envs=1, env_spacing=2.0)
    cfg.left_arm.init_state.pos = layout.LEFT_BASE_M
    cfg.right_arm.init_state.pos = layout.RIGHT_BASE_M
    scene = layout.InteractiveScene(cfg)
    sim.reset()
    left, right = scene["left_arm"], scene["right_arm"]
    layout.command_pose(left, right, 0.0, 0.0)
    steps(sim, scene, 160)
    baseline_cube = scene["cube"].data.root_pos_w[0].detach().clone()
    # Explicit close camera, so a 45/90-degree jaw movement is easier to SEE.
    idx = layout.body_index(left, "gripper")
    p = left.data.body_pos_w[0, idx].detach().cpu().tolist()
    sim.set_camera_view(eye=(p[0] - 0.23, p[1] - 0.29, p[2] + 0.16), target=tuple(p))
    print("[VIEW] Camera focuses on LEFT gripper; right gripper also receives commands.", flush=True)

    stage = layout.omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage after sim.reset()")
    hierarchy, geom = usd_audit(stage)
    # Save the geometry diagnosis immediately. Even a later motor/physics error
    # cannot erase the evidence the user needs to send us.
    write_reports(hierarchy, geom, [], "USD AUDIT COMPLETE - jaw motion pending")
    validate_geometry(geom)

    measurements = []
    initial_q = {}
    print("\n=== SLOW, VISIBLE JAW TEST (0 -> 45 -> 90 -> 0 degrees) ===", flush=True)
    for target in MOVING_ANGLES_DEG:
        print(f"[MOTION] Setting both grippers to {target:.0f} degrees (paced ~2.4 seconds) ...", flush=True)
        command_grippers(left, right, target)
        steps(sim, scene, STEPS_PER_POSE, visible=not layout.args.headless)
        row = {"commanded_deg": target, "arms": {}}
        for label, arm in (("LEFT", left), ("RIGHT", right)):
            measured = angle_deg(arm)
            relative_p, relative_q = relative_jaw(arm)
            if label not in initial_q:
                initial_q[label] = relative_q
            rotation = rotation_diff_deg(initial_q[label], relative_q)
            row["arms"][label] = {
                "measured_joint_deg": measured,
                "jaw_rotation_relative_to_closed_deg": rotation,
                "jaw_origin_in_gripper_link_m": relative_p.detach().cpu().tolist(),
                "jaw_quat_wxyz_in_gripper_link": relative_q.detach().cpu().tolist(),
            }
            if label == "LEFT":
                row["authored_surface_sections"] = finger_sections(relative_p, relative_q)
            print(f"  [{label}] joint={measured:+.2f} deg; jaw link rotation={rotation:.2f} deg", flush=True)
            if abs(measured - target) > MAX_GRIPPER_TRACK_ERROR_DEG:
                raise RuntimeError(f"Jaw tracking failed ({label}); target={target}, got={measured}")
            if abs(rotation - target) > 5.0:
                raise RuntimeError(f"Jaw-link rotation failed ({label}); expected={target}, got={rotation}")
        drift_mm = float(torch.linalg.vector_norm(scene["cube"].data.root_pos_w[0] - baseline_cube).item() * 1000)
        row["cube_drift_mm"] = drift_mm
        print(f"[CUBE] Displacement {drift_mm:.2f} mm", flush=True)
        if drift_mm > MAX_CUBE_DRIFT_MM:
            raise RuntimeError(f"Cube moved {drift_mm:.2f} mm during gripper-only test")
        measurements.append(row)
        show_section(stage, left, row["authored_surface_sections"])
        if not layout.args.headless:
            steps(sim, scene, 100, visible=True)
        write_reports(hierarchy, geom, measurements, "USD AUDIT COMPLETE - jaw motion in progress")

    fixed = [r for r in geom if r["link"] == "gripper" and r["collision_schema"] == "enabled"]
    moving = [r for r in geom if r["link"] == "jaw" and r["collision_schema"] == "enabled"]
    if fixed and moving:
        status = "GEOMETRY FOUND - bounds recorded; TCP NOT CALIBRATED"
    else:
        status = "GEOMETRY INCOMPLETE - hierarchy exported; TCP NOT CALIBRATED"
    write_reports(hierarchy, geom, measurements, status)
    print(f"[GEOMETRY RESULT] {status}", flush=True)
    print("[IMPORTANT] This run does not certify contact surfaces, a 40 mm finger aperture, or a TCP.", flush=True)
    print("[INFO] Headless test complete." if layout.args.headless else "[INFO] Close the simulator or Ctrl+C to exit.", flush=True)
    while not layout.args.headless and layout.app.is_running():
        steps(sim, scene, 1)


if __name__ == "__main__":
    if audit_args.offline:
        offline_main()
    else:
        try:
            main()
        except BaseException:
            # Kit 5.1 fast shutdown may exit(0), masking a failed assertion.
            # On test failure, preserve a nonzero process status explicitly.
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        else:
            # Isaac Lab's standalone STOP callback otherwise enters a render
            # loop when app.close closes the stage. Use its public teardown,
            # matching build_simulation_context in pinned Isaac Lab 2.3.0.
            sim = layout.SimulationContext.instance()
            if sim is not None:
                sim.clear_all_callbacks()
                sim.clear_instance()
            print("[SHUTDOWN] Closing SimulationApp", flush=True)
            # No Replicator writers are used by this audit.
            layout.app.close(wait_for_replicator=False)
