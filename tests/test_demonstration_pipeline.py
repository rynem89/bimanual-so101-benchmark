"""Synthetic files test gates and corruption detection, never manipulation success."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dataset_io import EpisodeWriter
from demonstration_pipeline import (DEFAULT_PROTOCOL, CALIBRATION, CALIBRATION_SHA256, NAMES,
    REVIEW_CHECKS, audit_episode, check_camera, collect, episode_digest, load_manifest, make_plan,
    require_approval, sha, validate, write_json)
from policy_cameras import WRIST_POSITION, WRIST_ROTATION, WRIST_QUAT


def fixture(root,case,manifest):
    """Eight short synthetic transitions, distinctly below production length gate."""
    cameras={}
    for name in NAMES:
        cameras[name]=dict(position_world_m=CALIBRATION['overview_eye_world_m'] if name=='overview' else WRIST_POSITION.tolist(),
            quaternion_world_wxyz=[1,0,0,0],quaternion_ros_wxyz=CALIBRATION['overview_quaternion_ros_wxyz'] if name=='overview' else list(WRIST_QUAT),
            intrinsic=CALIBRATION['intrinsic_expected'],resolution=[640,480],fps=20,
            parent='world' if name=='overview' else name.split('_')[0]+'/gripper',
            parent_position_world_m=[0,0,0],parent_rotation_world=np.eye(3).tolist())
    one_case=dict(manifest,cases=[case])
    writer=EpisodeWriter(root,dict(case=case['name'],configuration=[case['cube_xy_m'],case['yaw_deg'],case['seed']],
        cameras=cameras,image_shape=[480,640,3],joint_limits_rad=[[-2,3]]*6,
        protocol_sha256=manifest['protocol_sha256'],camera_config_sha256=CALIBRATION_SHA256,
        manifest_sha256=hashlib.sha256((json.dumps(one_case,indent=2)+'\n').encode()).hexdigest(),
        overview_profile='centered',physics_hz=60,expert_command_hz=20,motion_mode='continuous',
        grip_feedback_from_lift=True,tracked_source_dirty=False,source_revision='synthetic-test-fixture',
        source_sha256={'synthetic':'test-only'},split=manifest['split']))
    phases=['initial','close','hold','right_close','left_release','right_only_hold','bin_release','bin_settle']
    rgb=np.arange(480*640*3,dtype=np.uint8).reshape(480,640,3)
    for i,phase in enumerate(phases):
        frame=SimpleNamespace(t_s=(180+i*3)/60,q=np.zeros((2,6)),qd=np.zeros((2,6)),
            cube=np.zeros(13),pads=np.ones((2,2)),environment=np.zeros(6))
        writer.begin(frame,{name:np.roll(rgb,i,axis=1) for name in NAMES},np.zeros(12),
            capture=dict(phase=phase,physics_step=180+i*3,sensor_frame_ids={name:i+10 for name in NAMES},cameras=cameras))
        for _ in range(3):writer.applied(np.zeros(12))
    evaluation=dict(success=True,failure=None,collision_violations=0,
        stages=dict(pickup=True,lift=True,handover=True,placement=True),
        max_joint_speed_rad_s=1.,max_effort_Nm=1.,max_stable_grasp_slip_mm=[0,0],
        max_stable_grasp_rotation_deg=[0,0],reset=dict(position_error_mm=0,orientation_error_deg=0))
    writer.finish(True,evaluation,frame)
    return json.loads((root/'episode.json').read_text())


class ProtocolTests(unittest.TestCase):
    def setup_plan(self,temp):
        root=Path(temp);protocol=json.loads(DEFAULT_PROTOCOL.read_text())
        protocol['counts'].update(train=2,validation=1,heldout=2,stress=2)
        protocol['quality']['minimum_frames']=8  # Explicitly synthetic; production is >=1,200.
        path=root/'protocol.json';write_json(path,protocol);make_plan(path,root/'plan')
        manifest_path=root/'plan/train.json';manifest=load_manifest(manifest_path,path)
        return root,path,manifest_path,manifest,protocol

    def test_reproducible_disjoint_plans_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root,path,mp,manifest,_=self.setup_plan(temp)
            make_plan(path,root/'second')
            self.assertEqual(mp.read_bytes(),(root/'second/train.json').read_bytes())
            seen=set()
            for part in ('pilot','train','validation','heldout','stress'):
                doc=json.loads((root/f'plan/{part}.json').read_text())
                for case in doc['cases']:
                    key=(*case['cube_xy_m'],case['yaw_deg']);self.assertNotIn(key,seen);seen.add(key)
            manifest['cases'][0]['yaw_deg']=90;write_json(mp,manifest)
            with self.assertRaisesRegex(ValueError,'reproducible'):load_manifest(mp,path)

    def test_execution_needs_exact_approval_and_no_hidden_evaluation_collection(self):
        with tempfile.TemporaryDirectory() as temp:
            root,path,mp,manifest,_=self.setup_plan(temp)
            args=SimpleNamespace(manifest=mp,protocol=path,raw=root/'raw',output=root/'run',execute=True,
                authorize_collection=False,approval=None,pilot_report=None,headless=False,resume=False)
            with patch('demonstration_pipeline.subprocess.run') as launch:
                with self.assertRaisesRegex(ValueError,'session authorization'):collect(args)
                launch.assert_not_called()
            approval=root/'approval.json';write_json(approval,dict(approved=True,reviewer='test',date='test'))
            with self.assertRaisesRegex(ValueError,'hashes'):require_approval(approval,mp,path)
            args.execute=False;args.manifest=root/'plan/heldout.json'
            with self.assertRaisesRegex(ValueError,'evaluation-only'):collect(args)

    def test_mock_collector_preserves_failures_and_resume_never_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            root,path,_,_,_=self.setup_plan(temp);mp=root/'plan/pilot.json'
            approval=root/'mock_approval.json'
            write_json(approval,dict(approved=True,reviewer='synthetic-test-only',date='test',
                protocol_sha256=sha(path),camera_config_sha256=CALIBRATION_SHA256,manifest_sha256=sha(mp)))
            args=SimpleNamespace(manifest=mp,protocol=path,raw=root/'raw',output=root/'run',execute=True,
                authorize_collection=True,approval=approval,pilot_report=None,headless=False,resume=False)
            calls=[]
            def fake_run(command,**kwargs):
                if command[0]=='git':return SimpleNamespace(stdout='',returncode=0)
                calls.append(command)
                return SimpleNamespace(returncode=1 if len(calls)==1 else 0)
            with patch('demonstration_pipeline.subprocess.run',side_effect=fake_run):
                collect(args)
                args.resume=True;collect(args)
            self.assertEqual(len(calls),12)
            self.assertTrue(all('--headless' not in command and '--authorize-recording' in command for command in calls))
            ledger=[json.loads(line) for line in (args.output/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual(len(ledger),24)
            self.assertEqual(sum(r.get('returncode')==1 for r in ledger),1)
            self.assertFalse(args.raw.exists())  # All simulator processes were mocked.

    def test_full_decode_attachment_clock_and_physics_faults(self):
        with tempfile.TemporaryDirectory() as temp:
            root,path,mp,manifest,protocol=self.setup_plan(temp);case=manifest['cases'][0]
            episode=root/'raw/example';meta=fixture(episode,case,manifest)
            quality=protocol['quality'];self.assertEqual(audit_episode(episode,meta,case,manifest,quality)['frames'],8)
            production=json.loads(DEFAULT_PROTOCOL.read_text())['quality']
            with self.assertRaisesRegex(ValueError,'length'):audit_episode(episode,meta,case,manifest,production)
            bad=copy.deepcopy(meta);bad['evaluation']['collision_violations']=1
            with self.assertRaisesRegex(ValueError,'evidence'):audit_episode(episode,bad,case,manifest,quality)
            captures=episode/'captures.jsonl';original=captures.read_text();rows=[json.loads(s) for s in original.splitlines()]
            rows[3]['sensor_frame_ids']['right_wrist']=rows[2]['sensor_frame_ids']['right_wrist']
            captures.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            with self.assertRaisesRegex(ValueError,'advance'):audit_episode(episode,meta,case,manifest,quality)
            captures.write_text(original)
            rows=[json.loads(s) for s in original.splitlines()];rows[3]['cameras']['left_wrist']['parent_position_world_m'][0]+=.001
            captures.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            with self.assertRaisesRegex(RuntimeError,'attachment'):audit_episode(episode,meta,case,manifest,quality)
            captures.write_text(original)
            # Not first/middle/last: old sparse decoding would miss this corruption.
            (episode/'overview/000002.png').write_bytes(b'corrupt')
            with self.assertRaises(OSError):audit_episode(episode,meta,case,manifest,quality)

    def test_qa_human_hash_gate_ledger_and_incomplete_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root,path,mp,manifest,_=self.setup_plan(temp);ledger=[]
            for case in manifest['cases']:
                fixture(root/'raw'/case['name'],case,manifest)
                ledger.extend([dict(case=case['name'],event='started',protocol_sha256=sha(path),manifest_sha256=sha(mp)),
                               dict(case=case['name'],event='finished',returncode=0)])
            lp=root/'attempts.jsonl';lp.write_text('\n'.join(json.dumps(row) for row in ledger)+'\n')
            args=SimpleNamespace(manifest=mp,protocol=path,raw=root/'raw',output=root/'qa1',human_review=None,ledger=lp,review_video=False)
            first=validate(args)
            self.assertEqual(first['automatic_pass'],2);self.assertFalse(first['ready_for_conversion'])
            human=json.loads((args.output/'human_review_template.json').read_text());human.update(reviewer='test',date='test')
            for entry in human['episodes'].values():
                entry.update(accept=True,notes='Synthetic gate fixture, not a human simulator observation',checks=dict.fromkeys(REVIEW_CHECKS,True))
            hp=root/'review.json';write_json(hp,human);args.human_review=hp;args.output=root/'qa2'
            self.assertTrue(validate(args)['ready_for_conversion'])
            # Alter bytes after review; stale human acceptance must no longer admit it.
            episode=root/'raw'/manifest['cases'][0]['name'];old=episode_digest(episode)
            (episode/'additional.txt').write_text('changed');self.assertNotEqual(old,episode_digest(episode))
            args.output=root/'qa3';third=validate(args)
            self.assertEqual(third['accepted'],1);self.assertFalse(third['ready_for_conversion'])
            args.ledger=None;args.output=root/'qa4';self.assertFalse(validate(args)['ledger_complete'])
            meta_path=episode/'episode.json';meta=json.loads(meta_path.read_text());meta['source_sha256']={'changed':'source'};write_json(meta_path,meta)
            args.ledger=lp;args.output=root/'qa5';mixed=validate(args)
            self.assertEqual(mixed['automatic_pass'],0)
            self.assertTrue(all('Mixed' in row['reason'] for row in mixed['episodes']))

    def test_frozen_overview_and_bad_intrinsics_rejected(self):
        camera=dict(resolution=[640,480],fps=20,intrinsic=CALIBRATION['intrinsic_expected'],
            quaternion_ros_wxyz=[0,1,0,0],position_world_m=[0,-.2,.78])
        check_camera(camera,'overview');camera['position_world_m']=[0,-.1,.72]
        with self.assertRaisesRegex(ValueError,'position'):check_camera(camera,'overview')


if __name__=='__main__':unittest.main()
