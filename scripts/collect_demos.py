"""User-controlled collection using the existing SO-101 expert and recorder.

Preview does geometry/IK only. Physical success comes from the unchanged expert's
independent evaluator, never from this script's pose filter.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import sysconfig
import time

REPO = Path(__file__).resolve().parents[1]
EDGE_MARGIN_M = .01
MIN_POSITION_GAP_M = .005


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=positive, default=10, help="Physical successes requested")
    parser.add_argument("--x-offset-cm", type=float, nargs=2, default=(-3., 3.), metavar=("MIN", "MAX"))
    parser.add_argument("--y-offset-cm", type=float, nargs=2, default=(-2., 2.), metavar=("MIN", "MAX"))
    parser.add_argument("--yaw-deg", type=float, nargs=2, default=(-6., 6.), metavar=("MIN", "MAX"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-format", choices=("mp4", "png"), default="mp4", help="H.264 streaming; png is the legacy fallback")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", dest="headless", action="store_false", help="Visible Isaac Sim (default)")
    mode.add_argument("--headless", action="store_true", help="Record with cameras but no GUI")
    parser.set_defaults(headless=False)
    parser.add_argument("--preview-only", action="store_true", help="Print checked poses; no simulation, recording or output files")
    parser.add_argument("--max-attempts", type=positive, help="Simulation attempt cap; default: 3 times --count")
    args = parser.parse_args()
    if args.max_attempts is None:
        args.max_attempts = 3 * args.count
    if args.max_attempts < args.count:
        parser.error("--max-attempts must be at least --count")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    for name in ("x_offset_cm", "y_offset_cm", "yaw_deg"):
        low, high = getattr(args, name)
        if not all(math.isfinite(v) for v in (low, high, high - low)) or low > high:
            parser.error(f"--{name.replace('_', '-')} requires finite MIN <= MAX")
    return args


def usd_environment():
    """Expose the installed USD libraries without launching Kit or installing anything.

    Linux resolves shared libraries at process start, hence one re-exec if needed.
    The simulation child gets its normal launcher environment, without this path.
    """
    if Path(sys.prefix).name != "leisaac-sim":
        raise ValueError("Activate the existing leisaac-sim Conda environment first")
    paths = sorted((Path(sysconfig.get_path("purelib")) / "isaacsim/extscache").glob("omni.usd.libs-*.cp311"))
    if len(paths) != 1:
        raise ValueError("Expected the existing pinned Isaac Sim USD library directory")
    usd = paths[0]
    libraries = [str(usd / "bin"), str(Path(sys.prefix) / "lib")]
    if str(usd) not in sys.path or not all(p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) for p in libraries):
        env = dict(os.environ)
        env.setdefault("SO101_COLLECT_LAUNCH_ENV", json.dumps({name: os.environ.get(name) for name in ("PYTHONPATH", "LD_LIBRARY_PATH")}))
        env["PYTHONPATH"] = str(usd) + os.pathsep + env.get("PYTHONPATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    env = dict(os.environ)
    for name, value in json.loads(env.pop("SO101_COLLECT_LAUNCH_ENV", "{}")).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def planned_poses(args):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from benchmark_core import EVALUATION_CASES
    from so101_model import SO101Model, apply, calibrate, mesh_box_overlap
    from so101_planning import CollisionWorld, plan_lift

    # Read literal scene dimensions without importing its Isaac AppLauncher.
    names = {"TABLE_SIZE_M", "TABLE_TOP_Z_M", "CUBE_EDGE_M", "BIN_CENTER_XY_M", "BIN_OUTER_XY_M"}
    scene = {}
    tree = ast.parse((REPO / "scripts/inspect_task_layout.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                scene[node.targets[0].id] = ast.literal_eval(node.value)
    if scene.keys() != names:
        raise ValueError("Cannot read unchanged scene dimensions; no simulation started")
    nominal = np.array(EVALUATION_CASES["nominal"][0])
    ranges = [nominal[i] + np.array(bounds) / 100 for i, bounds in enumerate((args.x_offset_cm, args.y_offset_cm))]
    print(f"Nominal cube XY: {tuple(nominal)} m; offsets are centimetres.", flush=True)
    print(f"Requested X {ranges[0]} m; Y {ranges[1]} m; yaw {args.yaw_deg} degrees.", flush=True)
    print("Checking tabletop, HOME obstacles and open-gripper pregrasp/grasp IK; no physics.", flush=True)
    model = SO101Model()
    calibrate(model)
    z = scene["TABLE_TOP_Z_M"] + scene["CUBE_EDGE_M"] / 2
    reference = []
    for offset in (.035, .002):
        solutions = model.ik([*nominal, z + offset], rotation=np.eye(3), seeds=12)
        if not solutions:
            raise ValueError("Existing nominal IK reference unavailable")
        reference.append(solutions[0]["q"])
    home = np.zeros(6)
    right_base = model.base.copy()
    right_base[0, 3] = .25
    home_meshes = model.world_meshes(home) + model.world_meshes(home, right_base)
    rng = random.Random(args.seed)
    budget = max(40, 8 * args.max_attempts)
    axes = []
    for low, high in (*ranges, args.yaw_deg):
        values = [float(low + (high - low) * (i + rng.random()) / budget) for i in range(budget)]
        rng.shuffle(values)
        axes.append(values)
    feasible, rejected = [], Counter()
    for x, y, yaw in zip(*axes):
        xy = np.round([x, y], 6)
        yaw = round(yaw, 4)
        rotation = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        half = scene["CUBE_EDGE_M"] / 2
        footprint = np.abs(rotation[:2, :2]) @ np.full(2, half)
        if np.any(np.abs(xy) + footprint + EDGE_MARGIN_M > np.array(scene["TABLE_SIZE_M"][:2]) / 2):
            rejected["table edge / incomplete support"] += 1
            continue
        if np.all(np.abs(xy - scene["BIN_CENTER_XY_M"]) <= np.array(scene["BIN_OUTER_XY_M"]) / 2 + footprint + EDGE_MARGIN_M):
            rejected["bin clearance"] += 1
            continue
        cube = np.r_[xy, z]
        if any(mesh_box_overlap((apply(pose, mesh.triangles) - cube) @ rotation,
                                np.zeros(3), np.full(3, half + EDGE_MARGIN_M))
               for mesh, _, pose in home_meshes):
            rejected["robot HOME clearance"] += 1
            continue
        world = CollisionWorld(model, cube, cube_rotation=rotation)
        reachable = True;endpoints=[]
        for offset, seed in zip((.035, .002), reference):
            # Same full-rotation targets and USD bounds as plan_pickup. The
            # existing controller still validates the complete path at runtime.
            solutions = model.ik(cube + [0, 0, offset], rotation=rotation, reference=seed, seeds=2)
            valid=[s for s in solutions if s["limit_margin_deg"]>=3 and not world.check(s["q"],margin=.002)]
            if not valid:
                reachable = False
                break
            endpoints.append(valid[0]['q'])
        if not reachable:
            rejected["pregrasp/grasp IK or endpoint clearance"] += 1
            continue
        feasible.append(dict(cube_xy_m=xy.tolist(), yaw_deg=yaw,_endpoints=endpoints))

    # Farthest-position ordering spreads attempts across the feasible region.
    # Five millimetres is a minimum between attempts, not a substitute region.
    selected = []
    while feasible and len(selected) < args.max_attempts:
        if selected:
            distances = [min(math.dist(p["cube_xy_m"], q["cube_xy_m"]) for q in selected) for p in feasible]
            index = max(range(len(feasible)), key=distances.__getitem__)
            if distances[index] < MIN_POSITION_GAP_M:
                break
        else:
            index = 0
        candidate=feasible.pop(index);pre,grasp=candidate.pop('_endpoints')
        cube=np.r_[candidate['cube_xy_m'],z]
        rotation=Rotation.from_euler('z',candidate['yaw_deg'],degrees=True).as_matrix()
        world=CollisionWorld(model,cube,cube_rotation=rotation)
        if not world.segment(pre,grasp,margin=.001)['safe']:
            rejected['pregrasp descent collision']+=1;continue
        closed=grasp.copy();closed[-1]=np.deg2rad(22)
        cube_pose=np.eye(4);cube_pose[:3,:3]=rotation;cube_pose[:3,3]=cube
        local=np.linalg.inv(model.tcp_pose(closed))@cube_pose
        lift=plan_lift(model,closed,held=True,cube=cube,cube_rotation=rotation,cube_in_tcp=local,
                       target=[*candidate['cube_xy_m'],.08],other_q=home)
        if not lift['safe']:
            rejected['predicted lift IK/payload path']+=1;continue
        selected.append(candidate)
    print(f"Sampled {budget} stratified candidates; rejections: {dict(rejected)}", flush=True)
    print(f"Found {len(selected)} feasible positions at least 5 mm apart (cap {args.max_attempts}).", flush=True)
    if selected:
        positions = np.array([p["cube_xy_m"] for p in selected])
        print(f"Proposed coverage X {positions[:, 0].min():.6f} to {positions[:, 0].max():.6f} m; "
              f"Y {positions[:, 1].min():.6f} to {positions[:, 1].max():.6f} m.", flush=True)
        if any(np.ptp(positions[:, axis]) < .75 * (high - low) for axis, (low, high) in enumerate(ranges) if high > low):
            print("WARNING: feasible sampled coverage is narrower than requested. Inspect the printed positions; "
                  "the script has not substituted a smaller sampling region.", flush=True)
    print("Attempt | X world m | Y world m | X offset cm | Y offset cm | yaw deg", flush=True)
    for i, pose in enumerate(selected, 1):
        offsets = (np.array(pose["cube_xy_m"]) - nominal) * 100
        print(f"{i:7d} | {pose['cube_xy_m'][0]: .6f} | {pose['cube_xy_m'][1]: .6f} | "
              f"{offsets[0]: .3f} | {offsets[1]: .3f} | {pose['yaw_deg']: .4f}", flush=True)
    print("Geometry/IK feasibility is not a completed physical demonstration. No ranges were changed.", flush=True)
    if len(selected) < args.count:
        raise ValueError(f"Only {len(selected)} distinct feasible poses found for {args.count} requested successes. "
                         "Change your ranges/count or seed; no simulation or recording started.")
    if len(selected) < args.max_attempts:
        print(f"WARNING: only {len(selected)} distinct candidates available; collection will stop if they are exhausted.", flush=True)
    return selected


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}  # Interrupted/corrupt output cannot establish success.


def attempt_outcome(meta, result, marker):
    """Physical success and interruption are separate from process exit status."""
    evaluation=meta.get("evaluation", {})
    physical=(evaluation.get("success") is True and
              all(evaluation.get("stages", {}).get(s) is True for s in ("pickup","lift","handover","placement")))
    if meta.get("interrupted") or marker.get("status")=="interrupted" or result.get("status")=="interrupted":
        return "interrupted",physical
    if (physical and meta.get("success") is True and marker.get("status")=="completed"
            and result.get("status")=="passed" and result.get('full_task_success') is True):
        return "success",True
    return "failed",physical


def record(args, poses, env):
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root=REPO/"datasets/generated_demos"/f"{stamp}_seed{args.seed}"
    for name in ("successful","failed","interrupted","reviews","pending"):
        (root/name).mkdir(parents=True,exist_ok=False)
    cases=[dict(pose,name=f"demo_{i:04d}",seed=(args.seed+i)%2**32) for i,pose in enumerate(poses,1)]
    manifest=root/"cases.json"
    write_json(manifest,dict(schema="so101-cases-1.0",split="train",cases=cases))
    summary=dict(requested_successes=args.count,max_attempts=args.max_attempts,arguments=vars(args),
                 status="in_progress",successful=0,failed=0,interrupted=0,attempts=[],
                 note="Physical successes await human dataset acceptance; no automatic dataset QA or training.")
    write_json(root/"summary.json",summary)
    command=[sys.executable,str(REPO/"scripts/run_manipulation.py"),"--stage","full","--episodes","1",
             "--motion-mode","continuous","--expert-hz","20","--grip-feedback-from-lift",
             "--overview-profile","centered","--cases-file",str(manifest),"--split","train",
             "--record-dir",str(root/"pending"),"--record-format",args.record_format,"--authorize-recording",
             "--success-count",str(args.count),"--output-dir",str(root/"controller"),"--exit-after"]
    if args.headless:command.append("--headless")
    print(f"Run directory: {root}\nOne simulator for this batch. Ctrl+C stops collection safely.",flush=True)
    interrupted=False;started=time.perf_counter()
    with (root/"controller.log").open("w") as log:
        child=subprocess.Popen(command,cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in child.stdout:
                log.write(line);log.flush()
                if line.startswith(("[COLLECT]","[FAILURE]")):print(line.rstrip(),flush=True)
            code=child.wait()
        except KeyboardInterrupt:
            interrupted=True
            if child.poll() is None:
                try:child.send_signal(signal.SIGINT)
                except ProcessLookupError:pass
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:code=child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();code=child.wait()
            log.write(child.stdout.read());log.flush()
        finally:child.stdout.close()
    report=read_json(root/"controller/simulation.json")
    if report.get("history_file"):
        checkpoint=report;report=read_json(Path(report["history_file"]))
        report["interrupted_case"]=checkpoint.get("active_case")
    results={row["case"]:row for row in report.get("episodes",[])}
    interrupted=interrupted or code<0 or code==130 or report.get('status')=='interrupted'
    if interrupted:report.setdefault('interrupted_case',report.get('active_case'))
    raw={}
    for episode in (root/"pending").iterdir():
        meta=read_json(episode/"episode.json") or read_json(episode/"attempt.json")
        raw[meta.get("case",episode.name)]=(episode,meta)
    if not results and not raw and not interrupted:
        results[cases[0]['name']]=dict(status='failed',failure=f'Simulator initialization failed (exit {code}); see controller.log')
    stop_reviews=interrupted
    for number,case in enumerate(cases,1):
        name=case["name"];result=results.get(name,{})
        if name not in raw and not result and name!=report.get("interrupted_case"):continue
        episode,meta=raw.get(name,(None,{}))
        marker=read_json(episode/"attempt.json") if episode else {}
        if not result and (interrupted or report.get("status")=="interrupted"):
            result=dict(status="interrupted",failure="interrupted")
        status,physical=attempt_outcome(meta,result,marker)
        row=dict(attempt=number,**case,status=status,success=status=="success",physical_success=physical,
                 duration_s=meta.get("frames",meta.get("completed_policy_steps",0))/20,
                 wall_s=result.get("episode_wall_s"),episode_dir=None,review_path=None,
                 failure=None if status=="success" else result.get("failure") or "Incomplete recording; see controller.log")
        row['failure_category']=result.get('failure_category') if status!='success' else None
        if episode:
            destination=root/({"success":"successful","failed":"failed","interrupted":"interrupted"}[status])/episode.name
            shutil.move(str(episode),destination);row["episode_dir"]=str(destination)
            if not stop_reviews and (destination/"trajectory.npz").exists() and (destination/"episode.json").exists():
                review=root/"reviews"/f"attempt_{number:04d}.html"
                try:
                    reviewed=subprocess.run([sys.executable,str(REPO/"scripts/so101.py"),"dataset","review",
                        "--episode",str(destination),"--output",str(review)],cwd=REPO,env=env,capture_output=True,text=True)
                    if reviewed.returncode==0:row["review_path"]=str(review)
                    else:row["review_error"]=reviewed.stderr.strip() or reviewed.stdout.strip()
                except KeyboardInterrupt:
                    interrupted=True;stop_reviews=True;row["review_error"]="Review interrupted; recording retained"
            else:row["review_error"]="Interrupted/incomplete recording; HTML can be generated later where possible"
        summary["attempts"].append(row)
        summary[{"success":"successful","failed":"failed","interrupted":"interrupted"}[status]]+=1
        write_json(root/"summary.json",summary)
    summary["wall_s"]=time.perf_counter()-started;summary["returncode"]=code
    summary["status"]=("interrupted" if interrupted or report.get("status")=="interrupted" else
                       "target_reached" if summary["successful"]>=args.count else "attempt_limit_or_reset_failure")
    write_json(root/"summary.json",summary)
    print(f"Stopped: {summary['status']} | Successful: {summary['successful']}/{args.count} | "
          f"Failed: {summary['failed']} | Interrupted: {summary['interrupted']}\nSummary: {root/'summary.json'}",flush=True)
    return 0 if summary["status"]=="target_reached" else 130 if summary["status"]=="interrupted" else 1


def main():
    args = arguments()
    env = usd_environment()
    poses = planned_poses(args)
    if args.preview_only:
        print("Preview only: no simulator, recordings or files created.")
        return 0
    return record(args, poses, env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, ImportError) as error:
        raise SystemExit(f"Error: {error}")
