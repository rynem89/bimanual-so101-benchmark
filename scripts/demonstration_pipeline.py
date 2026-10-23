"""Plan, explicitly authorized collection, full-file QA and human review for SO-101.

No training code. Importing and planning never start Isaac. Simulator launches use
the existing expert unchanged; each declared case is attempted at most once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from dataset_io import validate_episode
from policy_cameras import CALIBRATION, CALIBRATION_SHA256, NAMES, CaptureContinuity, check_attachment, quaternion_rotation

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO/'configs/demonstration_protocol.json'
REVIEW_CHECKS = ('three_streams','cube_at_grasp','cube_at_handover','cube_at_release','full_motion','synchronization')


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def make_plan(protocol_path: Path, output: Path) -> dict:
    """Fixed attempts, never a successful-episode quota. No simulator required."""
    protocol=json.loads(protocol_path.read_text());output.mkdir(parents=True,exist_ok=False)
    protocol_hash=sha(protocol_path)
    summary=dict(protocol_sha256=protocol_hash,camera_config_sha256=CALIBRATION_SHA256,
                 status='proposed, not collection authorization',partitions={})
    for part_index,(partition,count) in enumerate(protocol['counts'].items()):
        rng=np.random.default_rng(protocol['seed']+part_index)
        if partition=='pilot':
            # Eight support corners, nominal and +/-1 degree known poses, centre.
            from itertools import product
            poses=list(product(protocol['cube_x_m'],protocol['cube_y_m'],protocol['cube_yaw_deg']))
            poses += [(-.34,-.04,0.),(-.34,-.04,-1.),(-.34,-.04,1.),(-.34,-.048,0.)]
            if count!=len(poses):raise ValueError('Pilot count must match declared boundary design')
        else:
            prefix='stress_' if partition=='stress' else 'cube_'
            bounds=[protocol[prefix+k] for k in ('x_m','y_m','yaw_deg')]
            # Independently permuted strata give coverage without rejected draws.
            unit=np.column_stack([(rng.permutation(count)+rng.random(count))/count for _ in bounds])
            poses=np.array([b[0] for b in bounds])+unit*np.array([b[1]-b[0] for b in bounds])
        cases=[dict(name=f'{partition}_{i:03d}',cube_xy_m=list(map(float,pose[:2])),
                    yaw_deg=float(pose[2]),seed=protocol['seed']*100+part_index*1000+i)
               for i,pose in enumerate(poses)]
        split='validation' if partition=='pilot' else ('heldout' if partition=='stress' else partition)
        manifest=dict(schema='so101-cases-1.0',partition=partition,split=split,
            protocol_sha256=protocol_hash,camera_config_sha256=CALIBRATION_SHA256,
            initial_robot_state=protocol['robot_initial_state'],cases=cases)
        path=output/f'{partition}.json';write_json(path,manifest)
        summary['partitions'][partition]=dict(path=str(path.resolve()),sha256=sha(path),attempts=len(cases),
            recording_allowed_after_approval=partition in ('pilot','train','validation'))
    write_json(output/'plan.json',summary)
    return summary


def load_manifest(path: Path, protocol_path: Path) -> dict:
    manifest=json.loads(path.read_text())
    if manifest.get('protocol_sha256')!=sha(protocol_path) or manifest.get('camera_config_sha256')!=CALIBRATION_SHA256:
        raise ValueError('Manifest does not match current protocol/camera files; regenerate and review')
    if len({c['name'] for c in manifest['cases']})!=len(manifest['cases']):raise ValueError('Duplicate case names')
    # A mutable manifest cannot redefine the approved distribution/counts.
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)/'plan';make_plan(protocol_path,root)
        expected=json.loads((root/(manifest['partition']+'.json')).read_text())
    if manifest!=expected:raise ValueError('Manifest differs from reproducible protocol plan')
    return manifest


def require_approval(path: Path, manifest_path: Path, protocol_path: Path) -> None:
    approval=json.loads(path.read_text())
    required=dict(approved=True,protocol_sha256=sha(protocol_path),camera_config_sha256=CALIBRATION_SHA256,
                  manifest_sha256=sha(manifest_path))
    if any(approval.get(k)!=v for k,v in required.items()) or not approval.get('reviewer') or not approval.get('date'):
        raise ValueError('Collection approval must name reviewer/date and exact protocol, camera and manifest hashes')


def preflight(args) -> None:
    """Offline approach feasibility only; never labels a physical demonstration."""
    manifest=load_manifest(args.manifest,args.protocol)
    if manifest['partition']!='pilot':raise ValueError('Preflight the pilot; leave evaluation manifests untouched')
    if args.output.exists():raise FileExistsError(args.output)
    from scipy.spatial.transform import Rotation
    from so101_model import SO101Model, calibrate
    from so101_planning import plan_pickup
    model=SO101Model();calibrate(model);rows=[]
    for case in manifest['cases']:
        started=time.perf_counter()
        plan=plan_pickup(model,case['cube_xy_m']+[.02],Rotation.from_euler('z',case['yaw_deg'],degrees=True).as_matrix())
        safe=bool(plan['waypoints']) and all(check['safe'] for check in plan['segment_checks'])
        rows.append(dict(case=case['name'],approach_geometry_safe=safe,wall_s=time.perf_counter()-started,
                         rejection_count=len(plan['ik']['rejections'])))
        write_json(args.output,dict(protocol_sha256=sha(args.protocol),manifest_sha256=sha(args.manifest),
            physical_success_proven=False,scope='HOME/open/pregrasp/grasp geometry only',cases=rows))
        print(json.dumps(rows[-1]),flush=True)
    if not all(row['approach_geometry_safe'] for row in rows):raise SystemExit(1)


def collect(args) -> None:
    manifest=load_manifest(args.manifest,args.protocol)
    if manifest['partition'] not in ('pilot','train','validation'):
        raise ValueError('Held-out/stress cases are evaluation-only; never collect training demonstrations from them')
    if args.execute:
        if not args.authorize_collection or args.approval is None:
            raise ValueError('Execution requires session authorization and a reviewed --approval file')
        require_approval(args.approval,args.manifest,args.protocol)
        if manifest['partition']!='pilot':
            if args.pilot_report is None:raise ValueError('Final collection requires accepted pilot QA')
            pilot=json.loads(args.pilot_report.read_text())
            protocol=json.loads(args.protocol.read_text())
            if (pilot.get('partition')!='pilot' or pilot.get('protocol_sha256')!=sha(args.protocol)
                or pilot.get('camera_config_sha256')!=CALIBRATION_SHA256
                or not pilot.get('ready_for_conversion') or pilot.get('accepted')!=protocol['quality']['pilot_required_acceptances']):
                raise ValueError('Pilot has not passed automatic and human quality gates')
        dirty=subprocess.run(['git','status','--porcelain','--untracked-files=normal'],cwd=REPO,
                             check=True,capture_output=True,text=True).stdout
        if dirty.strip():raise ValueError('Commit/review source before collection; repository has uncommitted files')
    ledger_path=args.output/'attempts.jsonl'
    ledger=[json.loads(line) for line in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
    attempted={row['case'] for row in ledger}
    if attempted and not args.resume:raise ValueError('Existing ledger; use --resume to skip every previously attempted case')
    commands=[]
    for case in manifest['cases']:
        if case['name'] in attempted:continue  # Includes interrupted starts; never silently replace them.
        case_path=args.output/'cases'/(case['name']+'.json')
        command=[sys.executable,str(REPO/'scripts/run_manipulation.py'),'--cpu-threads','8','--stage','full',
            '--motion-mode','continuous','--expert-hz','20','--grip-feedback-from-lift',
            '--overview-profile','centered','--cases-file',str(case_path.resolve()),
            '--record-dir',str(args.raw.resolve()),'--authorize-recording',
            '--output-dir',str((args.output/case['name']).resolve()),'--exit-after']
        if args.headless:command.append('--headless')
        commands.append(shlex.join(command))
        if not args.execute:continue
        write_json(case_path,dict(manifest,cases=[case]))
        started=time.time()
        with ledger_path.open('a') as stream:
            stream.write(json.dumps(dict(case=case['name'],event='started',wall_unix_s=started,
                protocol_sha256=sha(args.protocol),manifest_sha256=sha(args.manifest),command=command))+'\n')
            stream.flush()
        # A startup/reset failure is counted even if no raw episode was created.
        args.output.mkdir(parents=True,exist_ok=True)
        with (args.output/(case['name']+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=False)
        with ledger_path.open('a') as stream:
            stream.write(json.dumps(dict(case=case['name'],event='finished',returncode=result.returncode,
                wall_s=time.time()-started))+'\n')
    print(json.dumps(dict(executed=args.execute,commands=commands,previously_attempted=len(attempted)),indent=2))


def episode_digest(root: Path) -> str:
    """Bind human acceptance to ALL input bytes, including RGB and diagnostic evidence."""
    digest=hashlib.sha256()
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        digest.update(str(path.relative_to(root)).encode());digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def check_camera(camera: dict, name: str) -> None:
    if camera['resolution']!=CALIBRATION['resolution_wh'] or camera['fps']!=20:raise ValueError('Camera sampling changed')
    if not np.allclose(camera['intrinsic'],CALIBRATION['intrinsic_expected'],atol=1e-3,rtol=0):
        raise ValueError('Frozen camera intrinsics changed')
    q=np.asarray(camera['quaternion_ros_wxyz'])
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-5:
        raise ValueError('Invalid rendered optical orientation')
    if name=='overview':
        if not np.allclose(camera['position_world_m'],CALIBRATION['overview_eye_world_m'],atol=2e-5,rtol=0):
            raise ValueError('Frozen overview position changed')
        if abs(np.dot(q,CALIBRATION['overview_quaternion_ros_wxyz']))<1-1e-5:
            raise ValueError('Frozen overview orientation changed')
    else:
        if camera['parent']!=name.split('_')[0]+'/gripper':raise ValueError('Wrong wrist parent')
        check_attachment(np.asarray(camera['position_world_m']),quaternion_rotation(q),
                         np.asarray(camera['parent_position_world_m']),np.asarray(camera['parent_rotation_world']))


def audit_episode(root: Path, meta: dict, case: dict, manifest: dict, quality: dict) -> dict:
    """Decode every RGB file and check independent physics + measured capture evidence."""
    checked=validate_episode(root)
    if not meta['success']:raise ValueError('Physical expert failure')
    evaluation=meta['evaluation']
    if (not evaluation['success'] or evaluation['failure'] is not None or evaluation['collision_violations']!=0
        or not all(evaluation['stages'].get(k) is True for k in ('pickup','lift','handover','placement'))):
        raise ValueError('Independent full-task/contact evidence failed')
    for key,limit in [('max_joint_speed_rad_s','maximum_speed_rad_s'),('max_effort_Nm','maximum_effort_Nm')]:
        if not np.isfinite(evaluation[key]) or not 0<=evaluation[key]<=quality[limit]:raise ValueError('Physical limit: '+key)
    for key,limit in [('max_stable_grasp_slip_mm','maximum_slip_mm'),('max_stable_grasp_rotation_deg','maximum_slip_deg')]:
        values=np.asarray(evaluation[key])
        if values.shape!=(2,) or not np.isfinite(values).all() or np.any(values>quality[limit]):raise ValueError('Unstable grasp: '+key)
    for key,limit in [('position_error_mm','maximum_reset_position_error_mm'),('orientation_error_deg','maximum_reset_orientation_error_deg')]:
        value=evaluation['reset'][key]
        if not np.isfinite(value) or not 0<=value<=quality[limit]:raise ValueError('Reset prerequisite failed')
    expected=[case['cube_xy_m'],case['yaw_deg'],case['seed']]
    if meta['configuration']!=expected or meta['case']!=case['name']:raise ValueError('Wrong reset case')
    for key in ('protocol_sha256','camera_config_sha256'):
        if meta.get(key)!=manifest[key]:raise ValueError('Provenance mismatch: '+key)
    one_case=dict(manifest,cases=[case])
    case_bytes=(json.dumps(one_case,indent=2)+'\n').encode()
    if meta.get('manifest_sha256')!=hashlib.sha256(case_bytes).hexdigest():raise ValueError('Reset manifest hash mismatch')
    if (meta.get('overview_profile')!='centered' or meta.get('physics_hz')!=60 or meta.get('expert_command_hz')!=20
        or meta.get('motion_mode')!='continuous' or not meta.get('grip_feedback_from_lift')
        or meta.get('tracked_source_dirty') is not False or not meta.get('source_revision') or not meta.get('source_sha256')):
        raise ValueError('Unreviewed source/controller/camera configuration')
    if set(meta['cameras'])!=set(NAMES) or meta['image_shape']!=[480,640,3]:raise ValueError('Require all three approved cameras')
    n=checked['frames']
    if not quality['minimum_frames']<=n<=quality['maximum_frames']:raise ValueError('Unexpected episode length')
    captures=[json.loads(line) for line in (root/'captures.jsonl').read_text().splitlines()]
    if len(captures) not in (n,n+1):raise ValueError('Missing capture/attachment evidence')
    with np.load(root/'trajectory.npz') as data:
        steps=data['physics_step'].copy()
    continuity=CaptureContinuity();previous={};runs={name:0 for name in NAMES};maximum_runs=dict(runs)
    phase_indices={};unique={name:set() for name in NAMES}
    for i,row in enumerate(captures[:n]):
        if row['index']!=i or row['physics_step']!=int(steps[i]):raise ValueError('Capture/state chronology mismatch')
        if not continuity.accept(row['physics_step'],int(steps[0]),row['sensor_frame_ids']):raise ValueError('Duplicate capture tick')
        if set(row['cameras'])!=set(NAMES):raise ValueError('Capture camera set changed')
        phase_indices[row['phase']]=i
        for name in NAMES:
            check_camera(row['cameras'][name],name)
            with Image.open(root/name/f'{i:06d}.png') as image:rgb=np.asarray(image)
            if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8 or rgb.std()<quality['minimum_rgb_std']:
                raise ValueError(f'Invalid/blank RGB {name}:{i}')
            digest=hashlib.sha256(rgb.tobytes()).hexdigest();unique[name].add(digest)
            runs[name]=runs[name]+1 if digest==previous.get(name) else 1
            maximum_runs[name]=max(maximum_runs[name],runs[name]);previous[name]=digest
    required=('close','hold','right_close','left_release','right_only_hold','bin_release','bin_settle')
    if any(phase not in phase_indices for phase in required):raise ValueError('Missing grasp/transfer/placement phase images')
    if max(maximum_runs.values())>quality['maximum_identical_frame_run']:
        raise ValueError('Suspicious frozen camera stream (>3 s); review before accepting, never auto-waive')
    return dict(checked,phase_indices=phase_indices,unique_rgb={k:len(v) for k,v in unique.items()},
                maximum_identical_frame_run=maximum_runs)


def review_sheet(root: Path, indices: dict, output: Path) -> None:
    selected={'initial':0,**{k:indices[k] for k in ('close','hold','right_close','left_release','right_only_hold','bin_release','bin_settle')}}
    sheet=Image.new('RGB',(320*3,270*len(selected)),(20,20,20));draw=ImageDraw.Draw(sheet)
    for row,(phase,index) in enumerate(selected.items()):
        for col,name in enumerate(NAMES):
            with Image.open(root/name/f'{index:06d}.png') as image:sheet.paste(image.resize((320,240)),(col*320,row*270+25))
            draw.text((col*320+5,row*270+5),f'{phase} | {name} | frame {index}',fill='white')
    sheet.save(output)


def review_video(root: Path, frames: int, output: Path) -> None:
    """Diagnostic derivative only; original lossless RGB remains the training source."""
    import imageio_ffmpeg
    writer=imageio_ffmpeg.write_frames(str(output),(960,240),fps=20,codec='libx264',quality=6,
                                      output_params=['-preset','veryfast','-threads','2'])
    writer.send(None)
    try:
        for i in range(frames):
            images=[]
            for name in NAMES:
                with Image.open(root/name/f'{i:06d}.png') as image:images.append(np.asarray(image.resize((320,240))))
            writer.send(np.concatenate(images,axis=1))
    finally:writer.close()


def validate(args) -> dict:
    manifest=load_manifest(args.manifest,args.protocol);quality=json.loads(args.protocol.read_text())['quality']
    if not args.raw.is_dir():raise ValueError('Missing raw attempt root')
    args.output.mkdir(parents=True,exist_ok=False)
    human=json.loads(args.human_review.read_text()) if args.human_review else {}
    if human and (human.get('protocol_sha256')!=manifest['protocol_sha256'] or not human.get('reviewer') or not human.get('date')):
        raise ValueError('Human review lacks protocol hash/reviewer/date')
    expected={case['name']:case for case in manifest['cases']};found={};rows=[]
    for root in sorted(p for p in args.raw.iterdir() if p.is_dir()):
        path=root/'episode.json'
        if not path.exists():
            rows.append(dict(path=str(root.resolve()),automatic_pass=False,accepted=False,reason='incomplete raw attempt'));continue
        try:meta=json.loads(path.read_text());case_name=meta['case']
        except (ValueError,KeyError):
            rows.append(dict(path=str(root.resolve()),automatic_pass=False,accepted=False,reason='corrupt episode metadata'));continue
        found.setdefault(case_name,[]).append((root,meta))
    for name,attempts in found.items():
        for root,meta in attempts:
            row=dict(case=name,path=str(root.resolve()),automatic_pass=False,accepted=False)
            try:
                if name not in expected or len(attempts)!=1:raise ValueError('Unknown or repeated case; no replacement sampling')
                evidence=audit_episode(root,meta,expected[name],manifest,quality)
                digest=episode_digest(root);review=human.get('episodes',{}).get(name,{})
                row.update(automatic_pass=True,digest=digest,evidence=evidence,
                           source_revision=meta['source_revision'],source_sha256=meta['source_sha256'])
                row['accepted']=(review.get('accept') is True and review.get('digest')==digest and bool(review.get('notes'))
                    and all(review.get('checks',{}).get(key) is True for key in REVIEW_CHECKS))
                row['reason']='accepted' if row['accepted'] else 'awaiting human RGB/motion review bound to digest'
                review_sheet(root,evidence['phase_indices'],args.output/(name+'.jpg'))
                if args.review_video:review_video(root,evidence['frames'],args.output/(name+'.mp4'))
            except (ValueError,KeyError,OSError,TypeError,RuntimeError) as error:
                row.update(automatic_pass=False,accepted=False,reason=str(error))
            rows.append(row)
    for name in expected.keys()-found.keys():rows.append(dict(case=name,automatic_pass=False,accepted=False,reason='not completed/missing'))
    source_signatures={json.dumps(row['source_sha256'],sort_keys=True) for row in rows if row['automatic_pass']}
    if len(source_signatures)>1:
        for row in rows:
            if row['automatic_pass']:row.update(automatic_pass=False,accepted=False,reason='Mixed expert/recorder source versions')
    ledger=[json.loads(line) for line in args.ledger.read_text().splitlines()] if args.ledger else []
    starts=[row for row in ledger if row.get('event')=='started']
    finishes=[row for row in ledger if row.get('event')=='finished']
    ledger_ok=(bool(starts) and len(starts)==len(expected) and {r['case'] for r in starts}==set(expected)
               and len(finishes)==len(expected) and {r['case'] for r in finishes}==set(expected)
               and all(r.get('protocol_sha256')==manifest['protocol_sha256'] and r.get('manifest_sha256')==sha(args.manifest) for r in starts))
    returncodes={row['case']:row['returncode'] for row in finishes}
    for row in rows:
        if row['accepted'] and returncodes.get(row.get('case'))!=0:
            row.update(accepted=False,reason='Missing successful process completion in attempt ledger')
    # Collector records original partition hash, even though runner receives a one-case manifest.
    accepted=sum(row['accepted'] for row in rows)
    threshold=1.0 if manifest['partition']=='pilot' else quality['minimum_final_acceptance_fraction']
    report=dict(schema='so101-quality-1.0',partition=manifest['partition'],protocol_sha256=manifest['protocol_sha256'],
        camera_config_sha256=CALIBRATION_SHA256,manifest_sha256=sha(args.manifest),raw_root=str(args.raw.resolve()),
        declared_attempts=len(expected),recorded_attempts=len(starts),ledger_complete=ledger_ok,
        automatic_pass=sum(row['automatic_pass'] for row in rows),accepted=accepted,
        ready_for_conversion=ledger_ok and accepted/len(expected)>=threshold and len(rows)==len(expected),
        human_review=human,episodes=rows)
    write_json(args.output/'quality.json',report)
    write_json(args.output/'human_review_template.json',dict(protocol_sha256=manifest['protocol_sha256'],reviewer='',date='',
        instructions='Review all streams and full motion, cube visibility at grasp/transfer/release, timing and reset. Explain accept/reject in notes. Do not mark from nonblank tests alone.',
        episodes={r['case']:dict(accept=False,digest=r['digest'],notes='',checks={key:False for key in REVIEW_CHECKS})
                  for r in rows if r['automatic_pass']}))
    print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','human_review')},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('plan');p.add_argument('--output',type=Path,required=True)
    p.set_defaults(function=lambda args:print(json.dumps(make_plan(args.protocol,args.output),indent=2)))
    p=commands.add_parser('preflight',help='Offline pilot approach geometry only; requires bundled USD libraries')
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.set_defaults(function=preflight)
    p=commands.add_parser('collect',help='Print commands by default; execution requires explicit session and hashed approval')
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--approval',type=Path)
    p.add_argument('--pilot-report',type=Path);p.add_argument('--execute',action='store_true')
    p.add_argument('--authorize-collection',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--headless',action='store_true');p.set_defaults(function=collect)
    p=commands.add_parser('validate');p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--human-review',type=Path);p.add_argument('--ledger',type=Path)
    p.add_argument('--review-video',action='store_true');p.set_defaults(function=validate)
    args=parser.parse_args();args.function(args)


if __name__=='__main__':main()
