"""SO-101 benchmark commands. Help and command previews require only Python.

Run from the repository root. Simulator operations use leisaac-sim; conversion,
official training preflight and policy serving use so101-il-py312. Dispatch keeps
the validated controller and recording interface intact.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
PROTOCOL = REPO / "configs/demonstration_protocol.json"


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def simulation_options(parser, *, preview=True):
    parser.add_argument("--headless", action="store_true", help="GUI is the default")
    parser.add_argument("--cpu-threads", type=positive, default=8)
    if preview:
        parser.add_argument("--dry-run", action="store_true", help="Print launch command; do not start Isaac")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    p = commands.add_parser("scene", help="Stationary task scene, actual three-camera feeds and optical guides")
    simulation_options(p)
    p.add_argument("--frames", action="store_true", help="Also show world/base/TCP coordinate axes")
    p.add_argument("--duration", type=float, default=0, help="Exit after this many viewer seconds; 0 waits for window close")
    p.add_argument("--snapshot-dir", type=Path, help="Fresh diagnostic directory for RGB and measured camera poses")
    p = commands.add_parser("expert", help="Validated full-task continuous expert; no recording")
    simulation_options(p)
    p.add_argument("--case", default="nominal", help="Named benchmark scenario, e.g. nominal or yaw_minus_1deg")
    p.add_argument("--manifest", type=Path, help="Run all cases in an explicit scenario manifest")
    p.add_argument("--split", choices=("train", "validation", "heldout"), default="heldout")
    p.add_argument("--episodes", type=positive, default=1, help="Repetitions per case; every attempt is reported")
    p.add_argument("--output", type=Path, default=Path("outputs/expert"))
    p.add_argument("--camera-preview", type=Path, help="Diagnostic phase images, never demonstration recording")
    p.add_argument("--exit-after", action="store_true", help="Close after evaluation rather than leaving the final view")

    p = commands.add_parser("collect", help="Protocol planning and explicitly authorized collection")
    collection = p.add_subparsers(dest="operation", required=True)
    q = collection.add_parser("plan", help="Write reproducible manifests only; no simulator or authorization")
    q.add_argument("--protocol", type=Path, default=PROTOCOL)
    q.add_argument("--output", type=Path, required=True, help="Fresh manifest directory")
    q = collection.add_parser("run", help="Print launches by default; collection requires every existing approval gate")
    q.add_argument("--protocol", type=Path, default=PROTOCOL)
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--split", choices=("pilot", "train", "validation"), required=True, help="Must match manifest partition")
    q.add_argument("--raw", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True, help="Run ledger/log directory")
    q.add_argument("--approval", type=Path)
    q.add_argument("--pilot-report", type=Path, help="Human-accepted pilot QA, required for final collection")
    q.add_argument("--execute", action="store_true")
    q.add_argument("--authorize-collection", action="store_true", help="Assert separate explicit user authorization for this session")
    q.add_argument("--resume", action="store_true", help="Skip ALL previously started cases, including failures")
    q.add_argument("--headless", action="store_true")

    p = commands.add_parser("dataset", help="Inventory, synchronized raw review, full QA and LeRobot conversion")
    data = p.add_subparsers(dest="operation", required=True)
    q = data.add_parser("inspect", help="Read attempt metadata/QA/ledger; does not grant acceptance or revalidate bytes")
    q.add_argument("--raw", type=Path, default=Path("datasets/protocol_v1_pilot"))
    q.add_argument("--quality-report", type=Path)
    q.add_argument("--ledger", type=Path)
    q = data.add_parser("review", help="Write a seekable three-camera/raw-state/action HTML viewer")
    q.add_argument("--episode", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True, help="New HTML file; images remain in the raw directory")
    q = data.add_parser("validate", help="Exhaustive recorded-data QA; optionally bind human review to raw hashes")
    q.add_argument("--protocol", type=Path, default=PROTOCOL)
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--raw", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--human-review", type=Path)
    q.add_argument("--ledger", type=Path)
    q.add_argument("--review-video", action="store_true")
    q = data.add_parser("convert", help="Approved raw partition to LeRobot v3; preview by default")
    q.add_argument("--raw", type=Path, required=True)
    q.add_argument("--quality-report", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--execute", action="store_true")
    q = data.add_parser("viz", help="Official LeRobot viewer for a converted dataset; preview by default")
    q.add_argument("--dataset", type=Path, required=True)
    q.add_argument("--episode-index", type=int, default=0)
    q.add_argument("--execute", action="store_true")

    p = commands.add_parser("train", help="Official LeRobot 0.6.1 command + config/data preflight; never trains by default")
    p.add_argument("--policy", choices=("act", "diffusion"), required=True)
    p.add_argument("--config", default="matched-v1", help="Named recipe in configs/policy_training.json")
    p.add_argument("--dataset", type=Path, required=True, help="Accepted, converted TRAIN partition")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    p.add_argument("--resume", type=Path, help="Existing checkpoint pretrained_model/train_config.json")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--authorize-training", action="store_true", help="Assert separate explicit user training authorization")
    p.add_argument("--approval", type=Path, help="Hash-bound training approval; see docs/POLICIES.md")

    p = commands.add_parser("evaluate", help="Separate policy server and shared-physics client; checkpoints required")
    evaluation = p.add_subparsers(dest="operation", required=True)
    q = evaluation.add_parser("serve", help="Serve official trained checkpoint in so101-il-py312; preview by default")
    q.add_argument("--policy", choices=("act", "diffusion"), required=True)
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--socket", type=Path, required=True)
    q.add_argument("--execute", action="store_true")
    q = evaluation.add_parser("run", help="Shared simulator client; no expert assistance or recording")
    simulation_options(q)
    q.add_argument("--socket", type=Path, required=True)
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--split", choices=("validation", "heldout"), default="heldout")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--exit-after", action="store_true")

    p = commands.add_parser("replay", help="Physically replay actual recorded position targets at their original reset pose")
    simulation_options(p)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Fresh replay reports, never the raw recording")
    p.add_argument("--exit-after", action="store_true")
    return result


def launch(command: list[str], environment: str, *, dry_run=False) -> int:
    print(f"Environment: {environment}\n{shlex.join(command)}", flush=True)
    if dry_run:
        print("Preview only; nothing launched.")
        return 0
    if Path(sys.prefix).name != environment:
        raise ValueError(f"Activate {environment} in this shell first; running {sys.executable}")
    return subprocess.run(command, cwd=REPO, check=False).returncode


def controller_command(args) -> list[str]:
    """Pin the successful full-task recipe, without exposing recording or old modes."""
    command = [sys.executable, str(SCRIPTS / "run_manipulation.py"), "--stage", "full",
               "--motion-mode", "continuous", "--expert-hz", "20", "--grip-feedback-from-lift",
               "--cpu-threads", str(args.cpu_threads), "--output-dir", str(args.output.resolve())]
    if args.headless:
        command.append("--headless")
    if args.exit_after:
        command.append("--exit-after")
    return command


def append_options(command, args, names):
    for name in names:
        value = getattr(args, name)
        if value is None or value is False:
            continue
        command.append("--" + name.replace("_", "-"))
        if value is not True:
            command.append(str(value.resolve()) if isinstance(value, Path) else str(value))


def inspect_dataset(args) -> int:
    if not args.raw.is_dir():
        raise ValueError(f"Raw attempt root does not exist: {args.raw}")
    quality = json.loads(args.quality_report.read_text()) if args.quality_report else {}
    if quality and quality.get("raw_root") != str(args.raw.resolve()):
        raise ValueError("QA report belongs to a different raw root")
    proofs = {row["path"]: row for row in quality.get("episodes", []) if "path" in row}
    rows = []
    for root in sorted(p for p in args.raw.iterdir() if p.is_dir()):
        path = root / "episode.json"
        meta = json.loads(path.read_text()) if path.exists() else {}
        proof = proofs.get(str(root.resolve()), {})
        rows.append(dict(path=str(root), case=meta.get("case"), split=meta.get("split"),
                         completed=bool(meta), physical_success=meta.get("evaluation", {}).get("success", False),
                         automatic_qa=proof.get("automatic_pass"), human_accepted=proof.get("accepted"),
                         rejection=proof.get("reason"), frames=meta.get("frames")))
    starts = [json.loads(line) for line in args.ledger.read_text().splitlines()] if args.ledger else []
    print(json.dumps(dict(raw_root=str(args.raw.resolve()), raw_attempt_directories=len(rows),
                         ledger_started_attempts=sum(r.get("event") == "started" for r in starts) if args.ledger else None,
                         declared_attempts=quality.get("declared_attempts"),
                         note="Metadata inventory only; QA status is read from the supplied report, not revalidated here.",
                         episodes=rows), indent=2))
    return 0


def review_episode(args) -> int:
    """No image copies or simulation: scrub actual pre-action RGB, q, qd and targets."""
    import numpy as np
    from dataset_io import JOINT_ORDER, validate_episode

    if args.output.resolve().is_relative_to(args.episode.resolve()):
        raise ValueError("Review output must be outside the preserved raw episode")
    validate_episode(args.episode)
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads((args.episode / "episode.json").read_text())
    with np.load(args.episode / "trajectory.npz") as data:
        payload = {key: data[key].tolist() for key in ("state", "action", "time_s", "physics_step")}
    payload["cameras"] = list(meta["cameras"])
    payload["joints"] = JOINT_ORDER
    payload["root"] = Path(os.path.relpath(args.episode.resolve(), args.output.parent.resolve())).as_posix()
    payload["videos"] = {name: video["path"] for name,video in meta.get("videos",{}).items()}
    payload["fps"] = meta["fps"]
    captures = args.episode / "captures.jsonl"
    payload["phases"] = [json.loads(line)["phase"] for line in captures.read_text().splitlines()] if captures.exists() else []
    source = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    page = """<!doctype html><meta charset="utf-8"><title>SO-101 raw episode review</title>
<style>body{font:16px system-ui;margin:24px;background:#182027;color:#eee} .feeds{display:flex;gap:12px}
.feeds figure{margin:0;width:32%}img,canvas{width:100%}table{border-collapse:collapse}td,th{padding:4px 16px;border-bottom:1px solid #526}
input{width:75%}button{padding:8px}a{color:#9df}</style>
<h1>Raw pre-action camera / state / action review</h1><p>EPISODE</p>
<p>Diagnostic viewer only. Physical success and human data acceptance are separate. Angles: radians; velocities: rad/s.
Images and observations precede the displayed target, which is held for the next three 60 Hz physics ticks.</p>
<button id="play">Play / pause</button> <input id="seek" type="range" min="0" value="0"><p id="clock"></p>
<div class="feeds" id="feeds"></div><table><thead><tr><th>Joint</th><th>Measured q</th><th>Measured qd</th><th>Next target</th></tr></thead><tbody id="rows"></tbody></table>
<script>const d=PAYLOAD;const seek=document.querySelector('#seek');seek.max=d.action.length-1;
const feeds=document.querySelector('#feeds');for(const name of d.cameras){const f=document.createElement('figure');
const title=document.createElement('figcaption');title.textContent=name;f.append(title);
if(d.videos[name]){const v=document.createElement('video');v.id='video_'+name;v.preload='auto';v.muted=true;v.playsInline=true;v.hidden=true;v.src=d.root+'/'+d.videos[name];f.append(v);const c=document.createElement('canvas');c.id=name;c.width=640;c.height=480;f.append(c);}
else{const im=document.createElement('img');im.id=name;f.append(im);}feeds.append(f);}const rows=document.querySelector('#rows');
async function frame(name,i){if(!d.videos[name])return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve([name,im]);im.onerror=()=>reject(new Error('Missing RGB: '+name));im.src=d.root+'/'+name+'/'+String(i).padStart(6,'0')+'.png';});
const v=document.getElementById('video_'+name);if(v.readyState<1)await new Promise((resolve,reject)=>{v.addEventListener('loadedmetadata',resolve,{once:true});v.addEventListener('error',reject,{once:true});});
const t=(i+.25)/d.fps;if(Math.abs(v.currentTime-t)>1e-6)await new Promise((resolve,reject)=>{v.addEventListener('seeked',resolve,{once:true});v.addEventListener('error',reject,{once:true});v.currentTime=t;});return [name,v];}
let request=0,busy=false;async function show(){++request;if(busy)return;busy=true;try{while(true){const i=Number(seek.value),id=request;
const images=await Promise.all(d.cameras.map(name=>frame(name,i)));if(id!==request)continue;
for(const [name,im] of images){const view=document.getElementById(name);if(d.videos[name])view.getContext('2d').drawImage(im,0,0,view.width,view.height);else view.src=im.src;}
document.querySelector('#clock').textContent=`Row ${i} / ${d.action.length-1} | physics step ${d.physics_step[i]} | ${d.time_s[i].toFixed(3)} s | ${d.phases[i]||'phase unavailable'}`;
rows.replaceChildren();d.joints.forEach((name,j)=>{const tr=document.createElement('tr');for(const text of [name,d.state[i][j].toFixed(5),d.state[i][j+12].toFixed(5),d.action[i][j].toFixed(5)]){const td=document.createElement('td');td.textContent=text;tr.append(td);}rows.append(tr);});break;}}
catch(error){document.querySelector('#clock').textContent='RGB decode/seek error: '+error.message;}finally{busy=false;}}
seek.oninput=show;let playing=false;document.querySelector('#play').onclick=async()=>{playing=!playing;if(!playing)return;
while(playing&&Number(seek.value)<Number(seek.max)){await new Promise(resolve=>setTimeout(resolve,50));if(!playing)break;seek.value=Number(seek.value)+1;await show();}playing=false;};show();</script>"""
    page = page.replace("EPISODE", html.escape(str(args.episode))).replace("PAYLOAD", source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page)
    print(f"Open locally: {args.output.resolve().as_uri()}")
    return 0


def dispatch(args) -> int:
    if args.command == "scene":
        if args.duration < 0 or (args.headless and args.duration <= 0):
            raise ValueError("Headless scene inspection requires --duration > 0; duration cannot be negative")
        if args.dry_run:
            command = [sys.executable, str(SCRIPTS / "so101.py"), "scene", "--cpu-threads", str(args.cpu_threads)]
            append_options(command, args, ["headless", "frames", "duration", "snapshot_dir"])
            return launch(command, "leisaac-sim", dry_run=True)
        if Path(sys.prefix).name != "leisaac-sim":
            raise ValueError("Activate leisaac-sim before opening the scene")
        if args.snapshot_dir and args.snapshot_dir.exists():
            raise FileExistsError(args.snapshot_dir)
        # AppLauncher owns the simulator flags and must run before simulation imports.
        sys.argv = [sys.argv[0], "--enable_cameras", "--cpu-threads", str(args.cpu_threads)]
        if args.headless:
            sys.argv.append("--headless")
        import inspect_task_layout as layout
        from policy_cameras import inspect_scene
        inspect_scene(layout, args)
        return 0
    if args.command == "expert":
        command = controller_command(args)
        command += ["--case", args.case, "--episodes", str(args.episodes), "--split", args.split]
        if args.manifest:
            command += ["--cases-file", str(args.manifest.resolve())]
        append_options(command, args, ["camera_preview"])
        return launch(command, "leisaac-sim", dry_run=args.dry_run)
    if args.command == "replay":
        meta = json.loads((args.episode / "episode.json").read_text())
        if args.output.resolve().is_relative_to(args.episode.resolve()) or args.output.exists():
            raise ValueError("Replay output must be fresh and outside the preserved raw episode")
        xy, yaw, seed = meta["configuration"]
        case = dict(name=meta["case"], cube_xy_m=xy, yaw_deg=yaw, seed=seed)
        command = controller_command(args) + ["--replay", str(args.episode.resolve()),
                    "--cases-file", str((args.output / "replay_case.json").resolve())]
        if not args.dry_run:
            if Path(sys.prefix).name != "leisaac-sim":
                raise ValueError("Activate leisaac-sim before physical replay")
            args.output.mkdir(parents=True)
            (args.output / "replay_case.json").write_text(json.dumps(dict(schema="so101-cases-1.0",
                split=meta["split"], cases=[case]), indent=2) + "\n")
        return launch(command, "leisaac-sim", dry_run=args.dry_run)
    if args.command == "collect" or (args.command == "dataset" and args.operation == "validate"):
        command = [sys.executable, str(SCRIPTS / "demonstration_pipeline.py"), "--protocol", str(args.protocol.resolve())]
        operation = "collect" if args.command == "collect" and args.operation == "run" else args.operation
        command.append(operation)
        if args.command == "collect" and args.operation == "run":
            manifest = json.loads(args.manifest.read_text())
            if manifest.get("partition") != args.split:
                raise ValueError("--split must match the exact manifest partition")
            names = ["manifest", "raw", "output", "approval", "pilot_report", "execute", "authorize_collection", "resume", "headless"]
        elif args.operation == "plan":
            names = ["output"]
        else:
            names = ["manifest", "raw", "output", "human_review", "ledger", "review_video"]
        append_options(command, args, names)
        return launch(command, "leisaac-sim")
    if args.command == "dataset":
        if args.operation == "inspect":
            return inspect_dataset(args)
        if args.operation == "review":
            return review_episode(args)
        if args.operation == "convert":
            command = [sys.executable, str(SCRIPTS / "policy_pipeline.py"), "convert"]
            append_options(command, args, ["raw", "quality_report", "output"])
        else:
            if args.episode_index < 0:
                raise ValueError("episode-index cannot be negative")
            command = [sys.executable, "-m", "lerobot.scripts.lerobot_dataset_viz", "--repo-id", "local/so101",
                       "--root", str(args.dataset.resolve()), "--episode-index", str(args.episode_index)]
        return launch(command, "so101-il-py312", dry_run=not args.execute)
    if args.command == "train":
        command = [sys.executable, str(SCRIPTS / "policy_pipeline.py"), "train"]
        append_options(command, args, ["policy", "config", "dataset", "output", "seed", "resume", "execute", "authorize_training", "approval"])
        return launch(command, "so101-il-py312")
    if args.command == "evaluate":
        if args.operation == "serve":
            command = [sys.executable, str(SCRIPTS / "policy_pipeline.py"), "serve", "--method", args.policy]
            append_options(command, args, ["checkpoint", "socket"])
            return launch(command, "so101-il-py312", dry_run=not args.execute)
        if not args.dry_run and not args.socket.is_socket():
            raise ValueError("Start a real trained policy server first; inference socket is absent")
        manifest = json.loads(args.manifest.read_text())
        if manifest.get("partition") != args.split or manifest.get("split") != args.split:
            raise ValueError("Evaluation requires the selected validation/heldout protocol partition, never training cases")
        if not args.dry_run:
            from demonstration_pipeline import load_manifest
            load_manifest(args.manifest, PROTOCOL)
        command = controller_command(args) + ["--policy-socket", str(args.socket.resolve()),
                    "--cases-file", str(args.manifest.resolve()), "--split", args.split]
        return launch(command, "leisaac-sim", dry_run=args.dry_run)
    raise ValueError("Unknown operation")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    os.chdir(REPO)
    try:
        return dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
