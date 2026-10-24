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
import subprocess
import numpy as np
from PIL import Image

SCHEMA='so101-demonstration-1.2'
JOINT_ORDER=[f'{side}.{joint}' for side in ('left','right') for joint in
             ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')]
TASK='Left arm picks up the cube, transfers it to the right arm, which places it in the basket.'


def _video_capture(root,camera,meta):
    # OpenCV is already installed in both locked environments (unlike PyAV in
    # simulation or imageio_ffmpeg in learning). Always release its decoder.
    import cv2
    cap=cv2.VideoCapture(str(Path(root)/meta['videos'][camera]['path']))
    height,width,_=meta['image_shape']
    if (not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))!=width
            or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))!=height
            or abs(cap.get(cv2.CAP_PROP_FPS)-meta['fps'])>1e-6):
        cap.release();raise ValueError('Video dimensions/fps/open state disagree with metadata')
    return cap


def iter_rgb(root,camera,meta=None):
    """Decode old PNGs or new video to the same uint8 HWC RGB interface."""
    root=Path(root);meta=meta or json.loads((root/'episode.json').read_text())
    if meta.get('image_storage','png')=='png':
        for i in range(meta['frames']):
            with Image.open(root/camera/f'{i:06d}.png') as image:yield np.array(image)
    else:
        import cv2
        cap=_video_capture(root,camera,meta)
        try:
            while True:
                ready,bgr=cap.read()
                if not ready:break
                yield cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        finally:cap.release()


def read_rgb(root,camera,index,meta=None):
    root=Path(root);meta=meta or json.loads((root/'episode.json').read_text())
    if meta.get('image_storage','png')=='png':
        with Image.open(root/camera/f'{index:06d}.png') as image:return np.array(image)
    import cv2
    cap=_video_capture(root,camera,meta)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ready,bgr=cap.read()
        if not ready or round(cap.get(cv2.CAP_PROP_POS_FRAMES))!=index+1:
            raise ValueError(f'Missing RGB frame {camera}:{index}')
        return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    finally:cap.release()


def video_frame_count(root,camera,meta):
    cap=_video_capture(root,camera,meta);count=0
    try:
        while cap.grab():count+=1
    finally:cap.release()
    return count


def episode_rgb(root,meta=None):
    """Yield exactly the complete paired rows, excluding a terminal extra image."""
    root=Path(root);meta=meta or json.loads((root/'episode.json').read_text())
    streams={name:iter_rgb(root,name,meta) for name in meta['cameras']}
    try:
        for _ in range(meta['frames']):
            try:images={name:next(stream) for name,stream in streams.items()}
            except StopIteration:raise ValueError('Video shorter than paired trajectory') from None
            yield images
    finally:
        for stream in streams.values():stream.close()


def validate_episode(root):
    """Check chronology, exact held actions, dimensions and RGB on real files."""
    root=Path(root);meta=json.loads((root/'episode.json').read_text())
    if meta['schema'] not in (SCHEMA,'so101-demonstration-1.1','so101-demonstration-1.0') or meta['joint_order']!=JOINT_ORDER or meta['fps']!=20:
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
        if meta['schema']!='so101-demonstration-1.0' and not bounded:raise ValueError('Missing authored action bounds')
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
            if meta.get('image_storage','png')=='png':
                if not all((root/camera/f'{i:06d}.png').is_file() for i in range(n)):
                    raise ValueError('Missing RGB frame')
            else:
                video=meta['videos'][camera]
                expected=n+int(meta.get('terminal_image_recorded',False))
                if video['frames']!=expected or video_frame_count(root,camera,meta)!=expected:
                    raise ValueError('Video frame count disagrees with paired rows/terminal image')
            for index in {0,n//2,n-1}:
                rgb=read_rgb(root,camera,index,meta)
                if rgb.shape!=tuple(meta['image_shape']) or rgb.dtype!=np.uint8 or rgb.std()<2:
                    raise ValueError('Invalid or blank RGB observation')
    return dict(frames=n,seconds=n/20,success=meta['success'],schema=meta['schema'],action_bounds_verified=bounded)


class EpisodeWriter:
    """One attempt, including failures. Only verified successes become training data."""
    def __init__(self,root,metadata,image_storage='png'):
        if image_storage not in ('png','mp4'):raise ValueError('Use png or mp4 recording')
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.meta=dict(metadata,schema=SCHEMA if image_storage=='mp4' else 'so101-demonstration-1.1',joint_order=JOINT_ORDER,fps=20,task=TASK,
                       image_storage=image_storage,
                       state_layout='left q[6], right q[6], left qd[6], right qd[6]',
                       action_semantics='absolute joint position, radians, held for 3 physics ticks')
        self.rows=[];self.actual=[];self.privileged=[];self.pending=None;self.image_seconds=0.
        self.encoders={};self.image_count=0
        started=time.perf_counter()
        if image_storage=='png':
            for name in self.meta['cameras']:(self.root/name).mkdir()
        else:
            import imageio_ffmpeg
            height,width,_=self.meta['image_shape'];self.meta['videos']={}
            for name in self.meta['cameras']:
                self.meta['videos'][name]=dict(path=name+'.mp4',codec='h264',pixel_format='yuv420p',crf=18,frames=0)
                encoder=imageio_ffmpeg.write_frames(str(self.root/(name+'.mp4')),(width,height),fps=20,codec='libx264',quality=None,
                    output_params=['-preset','ultrafast','-crf','18','-threads','1','-g','20','-keyint_min','20',
                                   '-sc_threshold','0','-bf','0','-movflags','+frag_keyframe+empty_moov+default_base_moof'])
                encoder.send(None);self.encoders[name]=encoder
        self.image_seconds+=time.perf_counter()-started
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
        for name,rgb in images.items():
            if self.encoders:self.encoders[name].send(np.ascontiguousarray(rgb))
            else:Image.fromarray(rgb).save(self.root/name/f'{index:06d}.png')
        self.image_count+=1
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

    def finish(self,success,evaluation,terminal_frame,*,interrupted=False):
        # Terminal partial transitions are diagnostics, never mislabeled as a
        # full 50 ms target hold. The final physical state is retained separately.
        started=time.perf_counter();errors=[]
        for name,encoder in self.encoders.items():
            try:
                encoder.close()
                if self.image_count:
                    import imageio_ffmpeg
                    path=self.root/self.meta['videos'][name]['path'];temporary=path.with_suffix('.final.mp4')
                    # Fragmented streaming survives interruption. A lossless
                    # remux on close adds the browser's seek/duration index.
                    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-map','0:v:0',
                        '-c','copy','-movflags','+faststart',str(temporary)],check=True,capture_output=True)
                    temporary.replace(path)
            except Exception as error:errors.append(str(error))
            self.meta['videos'][name]['frames']=self.image_count
        self.encoders.clear();self.image_seconds+=time.perf_counter()-started
        self.meta.update(success=bool(success and not interrupted and not errors),evaluation=evaluation,frames=len(self.rows),
                         interrupted=interrupted,terminal_image_recorded=self.image_count>len(self.rows),
                         incomplete_terminal_ticks=len(self.actual) if self.pending is not None else 0,
                         terminal_physics_step=round(terminal_frame.t_s*60),
                         terminal_state=np.r_[terminal_frame.q.ravel(),terminal_frame.qd.ravel()].tolist(),
                         image_write_wall_s=self.image_seconds)
        if errors:self.meta['encoding_errors']=errors
        if self.rows:
            np.savez_compressed(self.root/'trajectory.npz',**{k:np.array([r[k] for r in self.rows]) for k in self.rows[0]})
        (self.root/'privileged.json').write_text(json.dumps(self.privileged))
        (self.root/'episode.json').write_text(json.dumps(self.meta,indent=2))
        if self.meta['success']:validate_episode(self.root)
        (self.root/'attempt.json').write_text(json.dumps(dict(self.meta,status='interrupted' if interrupted else 'completed'),indent=2))
        (self.root/'trajectory.partial.npz').unlink(missing_ok=True)
        return self.root
