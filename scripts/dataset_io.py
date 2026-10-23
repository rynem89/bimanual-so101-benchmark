"""Versioned 20 Hz demonstration exchange format, independent of Kit and LeRobot.

The simulator writes pre-action RGB/proprioception and the absolute 12-joint
position target held for the next three physics ticks. Privileged task state is
in a separate diagnostic stream and never part of policy inputs. Conversion to
LeRobot v3 happens only in the isolated training environment.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

SCHEMA='so101-demonstration-1.1'
JOINT_ORDER=[f'{side}.{joint}' for side in ('left','right') for joint in
             ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')]
TASK='Left arm picks up the cube, transfers it to the right arm, which places it in the basket.'


def validate_episode(root):
    """Check chronology, exact held actions, dimensions and RGB on real files."""
    root=Path(root);meta=json.loads((root/'episode.json').read_text())
    if meta['schema'] not in (SCHEMA,'so101-demonstration-1.0') or meta['joint_order']!=JOINT_ORDER or meta['fps']!=20:
        raise ValueError('Dataset convention/version mismatch')
    if meta['success'] and not meta['evaluation'].get('success'):
        raise ValueError('Success lacks independent physical evidence')
    for name,camera in meta['cameras'].items():
        quaternion=np.array(camera.get('quaternion_world_wxyz',[]))
        if quaternion.shape!=(4,) or not np.isclose(np.linalg.norm(quaternion),1,atol=1e-5):
            raise ValueError('Invalid camera calibration: '+name)
    with np.load(root/'trajectory.npz') as data:
        n=len(data['action']);steps=data['physics_step']
        if n<2 or data['state'].shape!=(n,24) or data['action'].shape!=(n,12):
            raise ValueError('Invalid episode state/action shapes')
        if not all(np.isfinite(data[k]).all() for k in ('state','action','time_s')):
            raise ValueError('Nonfinite trajectory')
        if data['state'].dtype!=np.float32 or data['action'].dtype!=np.float32:
            raise ValueError('State/action must use float32 radians')
        if not np.allclose(data['time_s'],steps/60,atol=1e-6,rtol=0):
            raise ValueError('Absolute physics timestamps disagree')
        bounded='joint_limits_rad' in meta
        if meta['schema']==SCHEMA and not bounded:raise ValueError('Missing authored action bounds')
        if bounded:
            limits=np.asarray(meta['joint_limits_rad'])
            if limits.shape!=(6,2) or not np.isfinite(limits).all() or np.any(limits[:,0]>=limits[:,1]):
                raise ValueError('Invalid authored action bounds')
            # Targets sent to PhysX are float32; allow only its rounding error.
            if np.any(data['action']<np.tile(limits[:,0],2)-2e-7) or np.any(data['action']>np.tile(limits[:,1],2)+2e-7):
                raise ValueError('Action outside authored radian bounds')
        if not np.all(np.diff(steps)==3) or not np.allclose(np.diff(data['time_s']),.05,atol=1e-6,rtol=0):
            raise ValueError('Episode is not synchronized at 20 Hz')
        if data['action_60hz'].shape!=(n,3,12) or not np.array_equal(data['action_60hz'],np.repeat(data['action'][:,None],3,axis=1)):
            raise ValueError('Recorded target was not actually held for all three ticks')
        if data['image_step'].shape!=(n,len(meta['cameras'])) or not np.all(data['image_step']==steps[:,None]):
            raise ValueError('Camera/state physics timestamps disagree')
        for camera in meta['cameras']:
            if not all((root/camera/f'{i:06d}.png').is_file() for i in range(n)):
                raise ValueError('Missing RGB frame')
            for index in {0,n//2,n-1}:
                rgb=np.asarray(Image.open(root/camera/f'{index:06d}.png'))
                if rgb.shape!=tuple(meta['image_shape']) or rgb.dtype!=np.uint8 or rgb.std()<2:
                    raise ValueError('Invalid or blank RGB observation')
    return dict(frames=n,seconds=n/20,success=meta['success'],schema=meta['schema'],action_bounds_verified=bounded)


class EpisodeWriter:
    """One attempt, including failures. Only verified successes become training data."""
    def __init__(self,root,metadata):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.meta=dict(metadata,schema=SCHEMA,joint_order=JOINT_ORDER,fps=20,task=TASK,
                       state_layout='left q[6], right q[6], left qd[6], right qd[6]',
                       action_semantics='absolute joint position, radians, held for 3 physics ticks')
        self.rows=[];self.actual=[];self.privileged=[];self.pending=None;self.image_seconds=0.
        for name in self.meta['cameras']:(self.root/name).mkdir()
        self._checkpoint()

    def _checkpoint(self):
        """Make interrupted attempts visible without admitting them to training."""
        (self.root/'attempt.json').write_text(json.dumps(dict(self.meta,status='in_progress',completed_policy_steps=len(self.rows)),indent=2))
        if self.rows:
            np.savez(self.root/'trajectory.partial.npz',**{k:np.array([r[k] for r in self.rows]) for k in self.rows[0]})

    def begin(self,frame,images,action,capture=None):
        if self.pending is not None:raise RuntimeError('Previous policy transition is incomplete')
        if frame is None:raise RuntimeError('Reset before recording')
        if set(images)!=set(self.meta['cameras']):raise ValueError('Camera set changed')
        index=len(self.rows);started=time.perf_counter()
        for name,rgb in images.items():
            if rgb.shape!=tuple(self.meta['image_shape']) or rgb.dtype!=np.uint8:
                raise ValueError('RGB dimensions/type changed')
            Image.fromarray(rgb).save(self.root/name/f'{index:06d}.png')
        self.image_seconds+=time.perf_counter()-started
        self.pending=dict(state=np.r_[frame.q.ravel(),frame.qd.ravel()].astype(np.float32),
                          action=np.asarray(action,dtype=np.float32).reshape(12).copy(),time_s=frame.t_s,
                          physics_step=round(frame.t_s*60),image_step=[round(frame.t_s*60)]*len(images))
        self.actual=[]
        if capture is not None:
            if capture['physics_step']!=round(frame.t_s*60):
                raise ValueError('Rendered camera and measured state clocks differ')
            # Append-only timing/attachment evidence, excluded from policy features.
            with (self.root/'captures.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(capture,index=index))+'\n')
        self.privileged.append(dict(physics_step=round(frame.t_s*60),cube=frame.cube.tolist(),
                                    pads_N=frame.pads.tolist(),environment_N=frame.environment.tolist()))

    def applied(self,action):
        if self.pending is None:return
        self.actual.append(np.asarray(action,dtype=np.float32).reshape(12).copy())
        if len(self.actual)==3:
            self.pending['action_60hz']=np.stack(self.actual)
            self.rows.append(self.pending);self.pending=None
            if len(self.rows)%200==0:self._checkpoint()

    def finish(self,success,evaluation,terminal_frame):
        # Terminal partial transitions are diagnostics, never mislabeled as a
        # full 50 ms target hold. The final physical state is retained separately.
        self.meta.update(success=bool(success),evaluation=evaluation,frames=len(self.rows),
                         incomplete_terminal_ticks=len(self.actual) if self.pending is not None else 0,
                         terminal_physics_step=round(terminal_frame.t_s*60),
                         terminal_state=np.r_[terminal_frame.q.ravel(),terminal_frame.qd.ravel()].tolist(),
                         image_write_wall_s=self.image_seconds)
        if self.rows:
            np.savez_compressed(self.root/'trajectory.npz',**{k:np.array([r[k] for r in self.rows]) for k in self.rows[0]})
        (self.root/'privileged.json').write_text(json.dumps(self.privileged))
        (self.root/'episode.json').write_text(json.dumps(self.meta,indent=2))
        if success:validate_episode(self.root)
        (self.root/'attempt.json').write_text(json.dumps(dict(self.meta,status='completed'),indent=2))
        (self.root/'trajectory.partial.npz').unlink(missing_ok=True)
        return self.root
