"""Deterministic SO-101 manipulation evaluation in the existing task scene.

Run: python scripts/run_manipulation.py --stage full --episodes 1
Only episode reset writes object/joint state. Task motion uses actuator targets.
"""
from __future__ import annotations
import argparse
import copy
import json
import hashlib
import inspect
import math
import os
from pathlib import Path
import sys
import time
import traceback
import subprocess
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

PROCESS_STARTED=time.perf_counter()
PROCESS_CPU_STARTED=time.process_time()
MAIN_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PLAN_TIMINGS=[]
parser = argparse.ArgumentParser(description=__doc__,add_help=False)
parser.add_argument('--stage', choices=('model','approach','pickup','transport','handover','full'), default='approach')
parser.add_argument('--episodes',type=int,default=1)
parser.add_argument('--output-dir',type=Path,default=Path('outputs/controller'))
parser.add_argument('--exit-after',action='store_true',help='Close the GUI after evaluation instead of preserving the final view')
parser.add_argument('--handover-mode',choices=('baseline','staged'),default='staged')
parser.add_argument('--motion-mode',choices=('legacy','continuous'),default='legacy')
parser.add_argument('--viewer-delay',type=float,default=0.,help='Optional wall seconds per physics tick, GUI only')
parser.add_argument('--suite',choices=('nominal','perturb','negative'),default='nominal')
parser.add_argument('--case',default='nominal',help='Fixed cube pose from benchmark_core.EVALUATION_CASES')
parser.add_argument('--record-dir',type=Path,help='Write 20 Hz RGB expert attempts, successes and failures')
parser.add_argument('--expert-hz',type=int,choices=(20,60),default=60)
parser.add_argument('--grip-feedback-from-lift',action='store_true',help='Apply the existing bounded force servo during lift/hold as well as transport')
parser.add_argument('--cases-file',type=Path,help='Explicit named cube cases and split; all cases are attempted')
parser.add_argument('--split',choices=('train','validation','heldout'),default='train')
parser.add_argument('--policy-socket',type=Path,help='20 Hz learned policy in the isolated Python 3.12 environment')
parser.add_argument('--replay',type=Path,help='Replay raw 20 Hz commands through the shared physics/evaluator')
if '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
    print('\nIsaac launcher options also include --headless and --device cuda:0.\n'
          'See: python scripts/inspect_task_layout.py --help')
    raise SystemExit(0)
options, rest = parser.parse_known_args()
if options.record_dir or options.policy_socket:
    options.expert_hz=20
    if '--enable_cameras' not in rest:rest.append('--enable_cameras')
sys.argv=[sys.argv[0],*rest]
import inspect_task_layout as layout
import carb
import numpy as np
from scipy.spatial.transform import Rotation
from benchmark_env import BenchmarkEnv, configure_cameras
from benchmark_core import EVALUATION_CASES, JointPath, policy_boundary, quintic, trajectory_limits, transfer_metrics
from so101_model import SO101Model, calibrate, jsonable, transform, apply, box_mesh, LINK_NAMES
from so101_planning import (plan_pickup, plan_lift, plan_transport, plan_receiver, plan_donor_clearance, plan_placement,
                           transport_fingerprint, CollisionWorld, fingerprint, valid_cache, CACHE_STATE_ATOL)


def compute_plan(function,*args,**kwargs):
    """Keep the GUI responsive while CPU-only geometry planning runs.

    Rendering does not step physics, so measured held poses stay fixed while
    planning. The worker touches only the already-loaded NumPy robot model.
    """
    started=time.perf_counter()
    cpu_started=time.process_time()
    name=function.__qualname__
    print(f'[PLAN] {name}: computing CPU checks; physics paused.',flush=True)
    if layout.args.headless:
        result=function(*args,**kwargs)
        elapsed=time.perf_counter()-started
        PLAN_TIMINGS.append(dict(function=name,wall_s=elapsed,cpu_s=time.process_time()-cpu_started))
        print(f'[PLAN DONE] {name}: {elapsed:.2f}s',flush=True)
        return result
    with ThreadPoolExecutor(max_workers=1) as worker:
        future=worker.submit(function,*args,**kwargs)
        while not future.done():
            sim=layout.SimulationContext.instance()
            if sim is None:layout.app.update()
            else:sim.render()
            time.sleep(.02)
        result=future.result()
    elapsed=time.perf_counter()-started
    PLAN_TIMINGS.append(dict(function=name,wall_s=elapsed,cpu_s=time.process_time()-cpu_started))
    print(f'[PLAN DONE] {name}: {elapsed:.2f}s',flush=True)
    return result


class Episode:
    def __init__(self,sim,scene,model):
        self.sim,self.scene,self.model=sim,scene,model
        self.left,self.right,self.cube=scene['left_arm'],scene['right_arm'],scene['cube']
        self.env=BenchmarkEnv(sim,scene,model,layout)
        self.ids=self.env.ids
        self.dt=sim.get_physics_dt()
        self.cube_contact=self.env.cube_contact
        self.samples=[]
        self.tick=0
        self.baseline=None
        self.contact_allowed=False
        self.phase='reset'
        self.failure_state=None
        self.max_tracking_m=0.
        self.max_feedforward_Nm=0.
        self.max_speed_rad_s=0.
        self.max_applied_torque_Nm=0.
        self.transport_reference=None
        self.max_transport_slip_m=0.
        self.max_transport_rotation_deg=0.
        self.reference_robot='left'
        self.contact_arms={'Left_Robot'}
        self.right_target=np.zeros(6)
        self.left_target=np.zeros(6)
        self.pad_forces={}
        self.active_grips={}
        self.checkpoint=None
        self.max_carried_environment_force_N=0.
        self.transfer_trace=[]
        self.receiver_unloaded_ticks=0
        self.nominal_trajectory_peaks=[]
        self.case='nominal'
        self.fault=None
        self.plan_dir=options.output_dir
        self.motion_timing=[]
        self.writer=None
        self.held_action=None
        self.source_sha256={}
        self.source_revision=None
        self.source_dirty=False
        self.split='fixed_regression'

    def measured_q(self,side='left'):
        arm=self.left if side=='left' else self.right
        return arm.data.joint_pos[0,self.ids].detach().cpu().numpy().copy()

    def validate_segment(self,name,world,start,end,**parameters):
        """Reuse a checked segment only for matching geometry and measured state.

        Context includes both bases/arms, cube position/orientation, joint
        endpoints and contact/margin options. The state tolerance is far below
        the millimetre clearance margins; no endpoint-only checks are reused.
        """
        context=np.concatenate([world.model.base.ravel(),np.asarray(start),np.asarray(end),
                                world.cube.ravel(),world.cube_rotation.ravel(),
                                *[pose.ravel() for _,_,pose in world.right]])
        key=fingerprint(world.model)+json.dumps(parameters,sort_keys=True)
        directory=self.plan_dir/'segments';directory.mkdir(exist_ok=True)
        path=directory/f'{name}.json'
        saved=json.loads(path.read_text()) if path.exists() else {}
        valid=valid_cache(saved,key,context=context)
        if valid:
            print(f'[CACHE] {name}: validated segment reused',flush=True)
            return dict(saved['check'],cache_hit=True)
        check=compute_plan(world.segment,start,end,**parameters)
        path.write_text(json.dumps(jsonable({'fingerprint':key,'context':context,'check':check}),indent=2))
        return dict(check,cache_hit=False)

    def tcp(self):
        idx=layout.body_index(self.left,'gripper')
        p=self.left.data.body_pos_w[0,idx].cpu().numpy()
        q=self.left.data.body_quat_w[0,idx].cpu().numpy()
        return p+Rotation.from_quat(np.roll(q,-1)).apply(self.model.tcp[:3,3])

    def cube_in_tcp(self,side='left'):
        cube_pose=transform(self.cube.data.root_pos_w[0].cpu().numpy(),
                            Rotation.from_quat(np.roll(self.cube.data.root_quat_w[0].cpu().numpy(),-1)).as_matrix())
        base=self.model.base.copy()
        if side=='right':base[0,3]=.25
        return np.linalg.inv(self.model.tcp_pose(self.measured_q(side),base))@cube_pose

    def check_fk(self,side='left'):
        arm=self.left if side=='left' else self.right
        base=self.model.base.copy()
        if side=='right':base[0,3]=.25
        poses=self.model.fk(self.measured_q(side),base)
        errors=[]
        angles=[]
        for name in LINK_NAMES:
            idx=layout.body_index(arm,name)
            actual_p=arm.data.body_pos_w[0,idx].cpu().numpy()
            actual_q=arm.data.body_quat_w[0,idx].cpu().numpy()
            errors.append(np.linalg.norm(poses[name][:3,3]-actual_p))
            angles.append(np.rad2deg(Rotation.from_matrix(poses[name][:3,:3].T @ Rotation.from_quat(np.roll(actual_q,-1)).as_matrix()).magnitude()))
        if max(errors)>.0005 or max(angles)>.1:
            raise RuntimeError(f'FK disagreement: {max(errors)*1000:.3f}mm, {max(angles):.3f}deg')
        return {'max_link_position_error_mm':max(errors)*1000,'max_link_orientation_error_deg':max(angles)}

    def forces(self):
        forbidden=[]
        frame=self.env.frame
        for source,target,magnitude in frame.contacts:
            if target==source:continue
            side,body=source.split('/')
            allowed=target=='table' and body=='base'
            if '/' in target:
                other_side,other=target.split('/')
                allowed |= side==other_side and frozenset((body,other)) in self.model.adjacent
            allowed |= (self.contact_allowed and target=='cube' and body in ('gripper','jaw')
                        and ('Left_Robot' if side=='left' else 'Right_Robot') in self.contact_arms)
            if not allowed:forbidden.append((source,target,magnitude))
        if forbidden:
            self.failure_state={'phase':self.phase,'contacts':forbidden,'q_rad':self.measured_q(),
                                'tcp_m':self.tcp(),'cube_m':self.cube.data.root_pos_w[0].cpu().numpy().copy()}
            raise RuntimeError(f'Forbidden physical contact in {self.phase}: {forbidden[:4]}')
        if self.transport_reference is not None:
            maximum=float(frame.environment.max())
            self.max_carried_environment_force_N=max(self.max_carried_environment_force_N,maximum)
            if maximum>=.05:
                raise RuntimeError(f'Carried cube/environment contact in {self.phase}: {maximum:.3f}N')
        self.pad_forces={side:frame.pads[i].tolist() for i,side in enumerate(('left','right'))}
        return self.pad_forces['left']

    def step(self,target,check=True):
        target=target.copy()
        # Orientation changes alter gravity torque at the jaw. Retain measured
        # bilateral preload with bounded slow servo correction, not attachment.
        for side,command in list(self.active_grips.items()):
            force=self.pad_forces.get(side,[0.,0.])
            low,high=3.,4.
            if side=='right' and options.handover_mode=='staged' and self.phase=='left_release':
                if max(self.pad_forces.get('left',[99,99]))<.05:self.receiver_unloaded_ticks+=1
                else:self.receiver_unloaded_ticks=0
                accepted=max(0,self.receiver_unloaded_ticks-6)*self.dt
                low=1.+2.*quintic(min(accepted/1.,1.));high=low+1.
            if min(force)<low:command-=np.deg2rad(.02)
            elif min(force)>high:command+=np.deg2rad(.01)
            command=float(np.clip(command,np.deg2rad(15),np.deg2rad(30)))
            self.active_grips[side]=command
            if side=='left':target[-1]=command
            else:self.right_target[-1]=command
        self.left_target=target.copy()
        action=np.stack([target,self.right_target])
        if options.expert_hz==20:
            if policy_boundary(self.env.total_steps,self.env.policy_origin_step):
                self.held_action=action.copy()
                if self.writer:
                    self.writer.begin(self.env.frame,self.env.capture_images(),self.held_action)
            if self.held_action is None:raise RuntimeError('Expert policy clock must start after reset')
            action=self.held_action
        before=self.env.total_steps
        try:frame=self.env.advance(action,evaluate=check)
        finally:
            if self.writer and self.env.total_steps==before+1:self.writer.applied(action)
        self.max_feedforward_Nm=max(self.max_feedforward_Nm,float(np.abs(frame.gravity).max()))
        self.max_speed_rad_s=max(self.max_speed_rad_s,float(np.abs(frame.qd).max()))
        self.max_applied_torque_Nm=max(self.max_applied_torque_Nm,float(np.abs(frame.effort).max()))
        self.tick+=1
        forces=self.forces() if check else [0.,0.]
        cube=self.cube.data.root_pos_w[0].cpu().numpy().copy()
        if check and self.transport_reference is not None:
            local=self.cube_in_tcp(self.reference_robot)
            slip=float(np.linalg.norm(local[:3,3]-self.transport_reference[:3,3]))
            angle=float(np.rad2deg(Rotation.from_matrix(self.transport_reference[:3,:3].T@local[:3,:3]).magnitude()))
            self.max_transport_slip_m=max(self.max_transport_slip_m,slip)
            self.max_transport_rotation_deg=max(self.max_transport_rotation_deg,angle)
            if slip>.005 or angle>10:
                raise RuntimeError(f'Transport grasp slip: {slip*1000:.2f}mm / {angle:.2f}deg')
        if check and not self.contact_allowed and self.baseline is not None and np.linalg.norm(cube-self.baseline)>.002:
            raise RuntimeError(f'Cube disturbed before intended contact: {(cube-self.baseline).tolist()}')
        if self.phase in ('right_close','left_release','right_only_hold') or self.phase.startswith('donor_normal_clearance_'):
            local=np.linalg.inv(frame.tcp[1])@frame.cube_pose()
            self.transfer_trace.append(dict(t_s=self.tick*self.dt,phase=self.phase,cube_m=frame.cube[:3].tolist(),
                cube_quat_wxyz=frame.cube[3:7].tolist(),right_tcp_m=frame.tcp[1,:3,3].tolist(),
                right_tcp_rotation=frame.tcp[1,:3,:3].tolist(),right_wrist_m=frame.wrist[1].tolist(),
                cube_in_right_tcp_m=local[:3,3].tolist(),cube_in_right_tcp_rotation=local[:3,:3].tolist(),
                right_joint_tracking_deg=float(np.rad2deg(np.abs(frame.q[1,:5]-frame.target[1,:5]).max())),
                right_q_rad=frame.q[1].tolist(),right_effort_Nm=frame.effort[1].tolist(),
                right_target_q_rad=frame.target[1].tolist(),right_qd_rad_s=frame.qd[1].tolist(),
                right_gravity_Nm=frame.gravity[1].tolist(),pad_force_N=frame.pads.tolist(),
                cube_com_velocity_m_s=frame.cube[7:10].tolist(),cube_angular_velocity_rad_s=frame.cube[10:13].tolist(),
                cube_vertical_velocity_m_s=float(frame.cube[9])))
        if self.tick%10==0:
            tracking=float(np.linalg.norm(self.tcp()-self.model.tcp_pose(target)[:3,3]))
            self.max_tracking_m=max(self.max_tracking_m,tracking)
            self.samples.append({'t_s':self.tick*self.dt,'phase':self.phase,'cube_m':cube,'tcp_m':self.tcp(),
                                 'cube_quat_wxyz':self.cube.data.root_quat_w[0].cpu().numpy().copy(),
                                 'q_rad':self.measured_q(),'target_q_rad':target.copy(),'pad_force_N':forces,
                                 'right_q_rad':self.measured_q('right'),'right_pad_force_N':self.pad_forces.get('right',[0,0]),
                                 'tcp_tracking_error_mm':tracking*1000})
        if self.checkpoint is not None and self.tick%120==0:
            self.checkpoint()
        if not layout.args.headless and options.viewer_delay>0:
            time.sleep(options.viewer_delay)
        return cube,forces

    def move(self,start,end,phase,contact=False,jaw_target=None):
        self.phase=phase;self.contact_allowed=contact
        print('[PHASE]',phase,flush=True)
        duration=max(1.,np.max(np.abs(end-start))*1.875/.35)
        self.nominal_trajectory_peaks.append(dict(phase=phase,duration_s=duration,**trajectory_limits(end-start,duration)))
        self.motion_timing.append(dict(phase=phase,moving_s=(math.ceil(duration/self.dt)+1)*self.dt,settling_s=90*self.dt))
        for t in np.linspace(0,1,math.ceil(duration/self.dt)+1):
            blend=quintic(t)
            target=start+(end-start)*blend
            if jaw_target is not None:target[-1]=jaw_target
            self.step(target)
        target=end.copy()
        if jaw_target is not None:target[-1]=jaw_target
        for _ in range(90):self.step(target)
        error=np.max(np.abs(self.measured_q()[:5]-end[:5]))
        if error>np.deg2rad(3):
            raise RuntimeError(f'Joint tracking error {np.rad2deg(error):.2f} degrees in {phase}')
        return self.check_fk()

    def move_right(self,start,end,phase):
        self.phase=phase
        print('[PHASE]',phase,flush=True)
        duration=max(1.,np.max(np.abs(end-start))*1.875/.35)
        self.nominal_trajectory_peaks.append(dict(phase=phase,duration_s=duration,**trajectory_limits(end-start,duration)))
        self.motion_timing.append(dict(phase=phase,moving_s=(math.ceil(duration/self.dt)+1)*self.dt,settling_s=90*self.dt))
        for t in np.linspace(0,1,math.ceil(duration/self.dt)+1):
            blend=quintic(t)
            self.right_target=start+(end-start)*blend
            self.step(self.left_target)
        self.right_target=end.copy()
        for _ in range(90):self.step(self.left_target)
        if np.max(np.abs(self.measured_q('right')[:5]-end[:5]))>np.deg2rad(3):
            raise RuntimeError('Right arm joint tracking error >3 degrees')
        return self.check_fk('right')

    def follow(self,points,side,phase,speed):
        """Execute the independently curve-checked Cartesian waypoint sequence.

        Keep a single stabilization interval at the real contact/clearance
        boundary. Grip feedback continues at every 60 Hz physics tick.
        """
        path=JointPath(points,speed);self.phase=phase
        self.nominal_trajectory_peaks.append(dict(phase=phase,duration_s=path.duration,**path.limits()))
        self.motion_timing.append(dict(phase=phase,moving_s=(math.ceil(path.duration/self.dt)+1)*self.dt,settling_s=90*self.dt))
        print('[PHASE]',phase,flush=True)
        for u in np.linspace(0,1,math.ceil(path.duration/self.dt)+1):
            if side=='right':self.right_target=path.position(u);self.step(self.left_target)
            else:self.step(path.position(u))
        for _ in range(90):self.step(self.left_target)
        if np.max(np.abs(self.measured_q(side)[:5]-points[-1][:5]))>np.deg2rad(3):
            raise RuntimeError('Continuous endpoint joint tracking error >3 degrees')
        return self.check_fk(side)

    def reset(self):
        for name,expected in self.source_sha256.items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Source changed during this run; restart before reset: '+name)
        self.phase='reset';self.contact_allowed=False;self.baseline=None
        self.transport_reference=None
        self.reference_robot='left';self.contact_arms={'Left_Robot'}
        self.right_target=np.zeros(6);self.left_target=np.zeros(6)
        self.active_grips={}
        self.max_carried_environment_force_N=0.
        self.max_transport_slip_m=0.;self.max_transport_rotation_deg=0.
        self.transfer_trace=[];self.receiver_unloaded_ticks=0;self.nominal_trajectory_peaks=[]
        self.motion_timing=[]
        self.held_action=None
        xy,yaw,seed=EVALUATION_CASES[self.case]
        self.env.reset(xy,yaw,seed);self.tick+=180
        self.baseline=self.cube.data.root_pos_w[0].cpu().numpy().copy()
        if np.linalg.norm(self.baseline[:2]-np.array(xy))>.002 or abs(self.baseline[2]-.02)>.002:
            raise RuntimeError('Reset cube pose outside 2mm tolerance')
        self.forces()
        support=float(self.cube_contact.get_contact_force_matrix(self.dt)[0,0,2].item())
        if not .3<support<.7:
            raise RuntimeError(f'Contact sensor prerequisite: cube support {support:.3f}N, expected about 0.49N')
        self.max_tracking_m=0.;self.max_feedforward_Nm=0.
        self.max_speed_rad_s=0.;self.max_applied_torque_Nm=0.
        result=self.check_fk();result['cube_table_support_N']=support
        result['measured_reset']=dict(self.env.evaluator.reset_metrics)
        return result

    def run(self,plan):
        result={'case':self.case,'fault':self.fault,'configuration':EVALUATION_CASES[self.case],
                'reset_fk':self.reset(),'pickup_success':False,'lift_success':False,'handover_success':False,
                'placement_success':False,'full_task_success':False}
        self.result=result
        if options.record_dir and options.stage!='model':
            from dataset_io import EpisodeWriter
            root=options.record_dir/f'{self.case}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}'
            self.writer=EpisodeWriter(root,dict(case=self.case,configuration=EVALUATION_CASES[self.case],
                cameras=self.env.camera_metadata(),image_shape=[224,224,3],motion_mode=options.motion_mode,split=self.split,
                source_revision=self.source_revision,tracked_source_dirty=self.source_dirty,
                source_sha256=self.source_sha256,
                physics_hz=60,expert_command_hz=20,controller='privileged geometric expert',
                grip_feedback_from_lift=options.grip_feedback_from_lift,joint_limits_rad=self.model.limits.tolist()))
            result['recording_dir']=str(root)
        if options.stage=='model':
            before=self.env.total_steps
            observation,terminal,metrics=self.env.step(np.zeros((2,6)))
            if terminal or self.env.total_steps-before!=3:raise RuntimeError('Policy decimation/termination regression')
            expected={'joint_position_rad','joint_velocity_rad_s','previous_position_action_rad'}
            if set(observation)!=expected or any(value.shape!=(2,6) for value in observation.values()):
                raise RuntimeError('Policy observation schema regression')
            for invalid in (np.zeros(12),np.full((2,6),np.nan),np.full((2,6),99.)):
                before=self.env.total_steps
                try:self.env.step(invalid)
                except ValueError:pass
                else:raise RuntimeError('Invalid policy action accepted')
                if self.env.total_steps!=before:raise RuntimeError('Invalid action stepped physics')
            result['policy_adapter_check']=dict(physics_hz=60,policy_hz=20,observation_keys=sorted(expected),invalid_actions_rejected=3)
            return result
        waypoints=[np.array(q) for q in plan['waypoints']]
        if not waypoints:
            self.phase='plan_pickup'
            result['approach_plan_evidence_path']=str(self.plan_dir/'plan.json')
            raise RuntimeError('No validated approach path: '+plan.get('failure','IK/collision feasibility rejected'))
        for i,phase in enumerate(('open','pregrasp','grasp')):
            result[phase+'_fk']=self.move(waypoints[i],waypoints[i+1],phase)
        result['grasp_tcp_error_mm']=float(np.linalg.norm(self.tcp()-np.array(plan['targets'][1]))*1000)
        if result['grasp_tcp_error_mm']>3:raise RuntimeError('Grasp TCP error >3mm')
        result['approach_success']=True
        result['approach_max_tcp_tracking_error_mm']=self.max_tracking_m*1000
        result['max_gravity_feedforward_Nm']=self.max_feedforward_Nm
        result['precontact_cube_drift_mm']=float(np.linalg.norm(self.cube.data.root_pos_w[0].cpu().numpy()-self.baseline)*1000)
        if options.stage=='approach':return result
        closed=waypoints[-1].copy()
        closure_end=closed.copy();closure_end[-1]=np.deg2rad(-5)
        pose=self.env.frame.cube_pose()
        closure_check=self.validate_segment('left_closure',CollisionWorld(self.model,pose[:3,3],cube_rotation=pose[:3,:3]),closed,closure_end,cube_contact=True)
        if not closure_check['safe']:raise RuntimeError(f'Unsafe closure: {closure_check}')
        result['closure_path']=closure_check
        # Stop closing once both pads load the cube; do not drive the servo to
        # its end stop through a rigid object. Force is measured, not inferred.
        self.phase='close';self.contact_allowed=True
        stable=0
        for _ in range(500):
            cube,force=self.step(closed)
            if min(force)>3.:
                stable+=1
                if stable>=30:break
            else:
                stable=0
                if self.fault!='grasp':closed[-1]=max(np.deg2rad(-5),closed[-1]-np.deg2rad(.12))
        for _ in range(60):cube,force=self.step(closed)
        result['closure_pad_force_N']=force
        result['closure_cube_m']=cube.tolist()
        result['closure_joint_deg']=float(np.rad2deg(self.measured_q()[-1]))
        if min(force)<2.:raise RuntimeError(f'No sustained bilateral cube contact: {force}')
        print('[CLOSURE]',result['closure_joint_deg'],force,flush=True)
        measured=self.measured_q()
        lift_key=fingerprint(self.model)+hashlib.sha256(inspect.getsource(plan_lift).encode()).hexdigest()
        pose=self.env.frame.cube_pose();local=self.cube_in_tcp()
        context=np.r_[measured,pose.ravel(),local.ravel()]
        lift_path=self.plan_dir/'lift_plan.json'
        lift=json.loads(lift_path.read_text()) if lift_path.exists() else {}
        if not valid_cache(lift,lift_key,context=context):
            # Keep the task's intended lift target separate from contact seating.
            # Actual settled cube pose/transform still define collision geometry.
            xy=EVALUATION_CASES[self.case][0]
            lift=compute_plan(plan_lift,self.model,measured,held=True,cube=pose[:3,3],cube_rotation=pose[:3,:3],cube_in_tcp=local,target=[*xy,.08])
            lift['fingerprint']=lift_key;lift['context']=context
            lift_path.write_text(json.dumps(jsonable(lift),indent=2))
        result['lift_plan']=lift
        if not lift['safe']:raise RuntimeError(f'Lift path rejected: {lift}')
        if options.grip_feedback_from_lift:self.active_grips['left']=closed[-1]
        start=self.measured_q();end=np.array(lift['lift_q'])
        result['lift_fk']=self.move(start,end,'lift',contact=True,jaw_target=closed[-1])
        target=end.copy();target[-1]=closed[-1]
        self.phase='hold'
        hold=[]
        for _ in range(round(3/self.dt)):
            cube,forces=self.step(target)
            hold.append({'cube_m':cube.copy(),'tcp_m':self.tcp(),'force_N':forces})
        positions=np.array([h['cube_m'] for h in hold]);tcp=np.array([h['tcp_m'] for h in hold])
        relative=positions-tcp
        result['hold_min_cube_height_m']=float(positions[:,2].min())
        result['hold_cube_motion_mm']=float(np.max(np.linalg.norm(positions-positions[0],axis=1))*1000)
        result['hold_relative_drift_mm']=float(np.max(np.linalg.norm(relative-relative[0],axis=1))*1000)
        result['hold_min_pad_force_N']=np.min([h['force_N'] for h in hold],axis=0).tolist()
        result['hold_duration_s']=3.
        result['max_tcp_tracking_error_mm']=self.max_tracking_m*1000
        result['max_joint_speed_rad_s']=self.max_speed_rad_s
        result['max_applied_torque_Nm']=self.max_applied_torque_Nm
        result['hold_cube_quat_wxyz']=self.cube.data.root_quat_w[0].cpu().numpy().copy()
        result['hold_cube_m']=positions[-1]
        result['hold_q_rad']=self.measured_q()
        if positions[:,2].min()<.06 or result['hold_relative_drift_mm']>3 or min(result['hold_min_pad_force_N'])<.05:
            raise RuntimeError(f'Lift/hold failed: min_height={positions[:,2].min():.6f}m, '
                               f'drift={result["hold_relative_drift_mm"]:.3f}mm, pads={result["hold_min_pad_force_N"]}N')
        result['pickup_success']=True;result['lift_success']=True
        if options.stage in ('transport','handover','full'):
            local=self.cube_in_tcp();start=self.measured_q()
            path=self.plan_dir/'handover_plan.json'
            transport=json.loads(path.read_text()) if path.exists() else {}
            valid=transport.get('fingerprint')==transport_fingerprint(self.model)
            valid &= np.allclose(transport.get('start_q',np.full(6,99)),start,atol=CACHE_STATE_ATOL,rtol=0)
            if valid:
                valid=np.allclose(transport['cube_in_left_tcp'],local,atol=CACHE_STATE_ATOL,rtol=0)
            if not valid:
                transport=compute_plan(plan_transport,self.model,start,local)
                path.write_text(json.dumps(jsonable(transport),indent=2))
            result['transport_plan']=transport
            if transport['selected'] is None:raise RuntimeError('No validated transport path')
            end=np.array(transport['selected']['left_q'])
            self.transport_reference=local
            self.active_grips.setdefault('left',closed[-1])
            result['transport_fk']=self.move(start,end,'transport',contact=True,jaw_target=closed[-1])
            target=end.copy();target[-1]=closed[-1]
            self.phase='handover_hold'
            minimum_force=np.full(2,np.inf)
            for _ in range(round(3/self.dt)):
                cube,force=self.step(target)
                minimum_force=np.minimum(minimum_force,force)
            if cube[2]<.15 or min(minimum_force)<.05:raise RuntimeError('Transport hold lost support')
            result['transport_success']=True
            result['transport_max_slip_mm']=self.max_transport_slip_m*1000
            result['transport_max_rotation_deg']=self.max_transport_rotation_deg
            result['transport_hold_min_pad_force_N']=minimum_force
            result['transport_cube_m']=cube
            result['transport_q_rad']=self.measured_q()
            result['transport_cube_in_tcp']=self.cube_in_tcp()
            if options.stage in ('handover','full'):
                self.handover(result,transport)
            if options.stage=='full':self.place(result)
        result['max_joint_speed_rad_s']=self.max_speed_rad_s
        result['max_applied_torque_Nm']=self.max_applied_torque_Nm
        result['max_carried_environment_force_N']=self.max_carried_environment_force_N
        result['independent_evaluation']=self.env.evaluator.report()
        result['transfer_metrics']=transfer_metrics(self.transfer_trace)
        result['nominal_trajectory_peaks']=self.nominal_trajectory_peaks
        result['motion_timing']=self.motion_timing
        if options.stage=='full' and not self.env.evaluator.success:
            raise RuntimeError('Expert claimed completion without independent physical success')
        return result

    def handover(self,result,transport):
        """Receive on orthogonal faces, unload left, verify right-only support."""
        left_q=self.measured_q()
        cube_pose=self.model.tcp_pose(left_q)@self.cube_in_tcp()
        reference=np.array(transport['selected']['right_q'])
        path=self.plan_dir/'receiver_plan.json'
        key=transport_fingerprint(self.model)+hashlib.sha256(inspect.getsource(plan_receiver).encode()).hexdigest()
        receiver=json.loads(path.read_text()) if path.exists() else {}
        valid=receiver.get('fingerprint')==key
        valid &= np.allclose(receiver.get('left_q',np.full(6,99)),left_q,atol=CACHE_STATE_ATOL,rtol=0)
        if valid:valid=np.allclose(receiver['cube_pose'],cube_pose,atol=CACHE_STATE_ATOL,rtol=0)
        if not valid:
            print('[PLAN] Validating receiver path around the measured held cube',flush=True)
            receiver=compute_plan(plan_receiver,self.model,left_q,cube_pose,reference,continuous=options.motion_mode=='continuous')
            receiver['fingerprint']=key
            path.write_text(json.dumps(jsonable(receiver),indent=2))
        result['receiver_plan']=receiver
        if receiver['selected'] is None:raise RuntimeError('Receiver path rejected; see receiver_plan.json')
        waypoints=[np.array(q) for q in receiver['selected']['waypoints']]
        approach_start=receiver['selected'].get('approach_start_index',2)
        phases=['right_open']+[f'right_transit_{i}' for i in range(approach_start-1)]
        phases += [f'right_approach_{i}' for i in range(len(waypoints)-1-approach_start)]
        for i,phase in enumerate(phases[:approach_start] if options.motion_mode=='continuous' else phases):
            result[phase+'_fk']=self.move_right(waypoints[i],waypoints[i+1],phase)
        if options.motion_mode=='continuous':
            result['right_approach_fk']=self.follow(waypoints[approach_start:],'right','right_cartesian_approach',.12)
        self.contact_arms.add('Right_Robot')
        self.phase='right_close'
        stable=0
        for _ in range(500):
            self.step(self.left_target)
            force=self.pad_forces['right']
            # During dual support a strong receiving squeeze bends both
            # compliant arms. Establish light bilateral contact first; increase
            # right preload only as the donor releases.
            if min(force)>1.:
                stable+=1
                if stable>=30:break
            else:
                stable=0
                if self.fault!='transfer':self.right_target[-1]=max(np.deg2rad(15),self.right_target[-1]-np.deg2rad(.12))
        for _ in range(60):self.step(self.left_target)
        result['right_acquisition_force_N']=self.pad_forces['right'].copy()
        if min(self.pad_forces['right'])<.6:raise RuntimeError('Right grasp has insufficient bilateral preload')
        left_q=self.measured_q();right_q=self.measured_q('right')
        # Holding the previous unloaded arm target was tested and rejected:
        # its fixed finger retained contact after release. Preserve the proven
        # measured donor pose while staging only receiver preload.
        release_start=left_q.copy()
        release=release_start.copy();release[-1]=np.pi/4
        right_base=self.model.base.copy();right_base[0,3]=.25
        cube_pose=self.model.tcp_pose(left_q)@self.cube_in_tcp()
        world=CollisionWorld(self.model,cube_pose[:3,3],other_q=right_q,other_base=right_base,cube_rotation=cube_pose[:3,:3])
        release_check=self.validate_segment('left_release',world,left_q,release,margin=.001,cube_contact=True)
        result['left_release_check']=release_check
        if not release_check['safe']:raise RuntimeError('Left release would collide with receiving hand')
        self.reference_robot='right';self.transport_reference=self.cube_in_tcp('right')
        self.active_grips.pop('left',None)
        self.active_grips['right']=self.right_target[-1]
        self.max_transport_slip_m=0.;self.max_transport_rotation_deg=0.
        self.move(release_start,release,'left_release',contact=True)
        result['donor_clearance_attempts']=[]
        for attempt in range(2):
            if max(self.pad_forces['left'])<.05:break
            if min(self.pad_forces['right'])<.6 or self.env.frame.cube[2]<.15:
                raise RuntimeError('Receiver cannot support a donor clearance move')
            clearance=compute_plan(plan_donor_clearance,self.model,self.measured_q(),self.measured_q('right'),self.env.frame.cube_pose())
            result['donor_clearance_attempts'].append(clearance)
            if not clearance['safe']:raise RuntimeError('No checked donor fixed-pad clearance path')
            self.move(self.measured_q(),np.array(clearance['goal_q']),f'donor_normal_clearance_{attempt}',
                      contact=True,jaw_target=np.pi/4)
        self.phase='right_only_hold'
        minimum_right=np.full(2,np.inf);maximum_left=np.zeros(2);heights=[]
        for _ in range(round(3/self.dt)):
            cube,_=self.step(self.left_target)
            minimum_right=np.minimum(minimum_right,self.pad_forces['right'])
            maximum_left=np.maximum(maximum_left,self.pad_forces['left'])
            heights.append(cube[2])
        result['right_only_min_force_N']=minimum_right
        result['released_left_max_force_N']=maximum_left
        result['right_only_min_height_m']=float(min(heights))
        if min(minimum_right)<.05 or max(maximum_left)>.05 or min(heights)<.15:
            raise RuntimeError('Right-only support not established after left release')
        result['handover_success']=True
        result['handover_cube_in_right_tcp']=self.cube_in_tcp('right')
        result['handover_right_q_rad']=self.measured_q('right')
        result['handover_left_q_rad']=self.measured_q()
        result['handover_cube_m']=self.cube.data.root_pos_w[0].cpu().numpy().copy()

    def place(self,result):
        """Release over the bin, retreat, and measure sustained resting support."""
        self.phase='plan_placement'
        if self.checkpoint:self.checkpoint()
        left_q=self.measured_q();right_q=self.measured_q('right');local=self.cube_in_tcp('right')
        path=self.plan_dir/'placement_plan.json'
        key=transport_fingerprint(self.model)+hashlib.sha256(inspect.getsource(plan_placement).encode()).hexdigest()
        placement=json.loads(path.read_text()) if path.exists() else {}
        valid=placement.get('fingerprint')==key
        for name,value in (('left_start',left_q),('right_start',right_q),('cube_in_right_tcp',local)):
            if valid:valid=np.allclose(placement[name],value,atol=CACHE_STATE_ATOL,rtol=0)
        if not valid:
            print('[PLAN] Validating donor withdrawal and loaded bin travel',flush=True)
            placement=compute_plan(plan_placement,self.model,left_q,right_q,local,continuous=options.motion_mode=='continuous')
            placement['fingerprint']=key
            path.write_text(json.dumps(jsonable(placement),indent=2))
        result['placement_plan']=placement
        selected=placement['selected']
        if selected is None:raise RuntimeError('No validated withdrawal/bin path; see placement_plan.json')
        self.contact_arms.discard('Left_Robot')
        waypoints=[np.array(q) for q in selected['left_departure']['waypoints']]
        if options.motion_mode=='continuous' and selected['left_departure'].get('kind')!='joint_transit':
            self.move(waypoints[0],waypoints[1],'left_separation',contact=True)
            self.follow(waypoints[1:18],'left','left_cartesian_retreat',.2)
            for i in range(17,len(waypoints)-1):
                self.move(waypoints[i],waypoints[i+1],f'left_retreat_{i}',contact=True)
        else:
            for i in range(len(waypoints)-1):
                self.move(waypoints[i],waypoints[i+1],f'left_retreat_{i}',contact=True)
        self.move_right(self.measured_q('right'),np.array(selected['right_goal']),'right_to_bin')
        right_model=copy.copy(self.model);right_model.base=self.model.base.copy();right_model.base[0,3]=.25
        cube_pose=right_model.tcp_pose(self.measured_q('right'))@self.cube_in_tcp('right')
        vertices=apply(cube_pose,box_mesh(np.zeros(3),np.full(3,.04),'cube').vertices)
        if np.any(vertices[:,:2].min(0)<[.275,-.105]) or np.any(vertices[:,:2].max(0)>[.405,.025]):
            raise RuntimeError('Measured payload is not inside the bin release footprint')
        if min(self.pad_forces['right'])<.05:raise RuntimeError('Right grasp lost before bin release')
        opened=self.measured_q('right');opened[-1]=np.pi/4
        world=CollisionWorld(right_model,cube=(99,99,99),other_q=self.measured_q(),other_base=self.model.base)
        check=self.validate_segment('bin_opening',world,self.measured_q('right'),opened,margin=.001)
        result['bin_opening_check']=check
        if not check['safe']:raise RuntimeError('Measured bin opening path rejected')
        result['release_cube_m']=cube_pose[:3,3]
        self.active_grips.pop('right',None);self.transport_reference=None
        self.move_right(self.measured_q('right'),opened,'bin_release')
        cube_pose=right_model.tcp_pose(self.measured_q('right'))@self.cube_in_tcp('right')
        world=CollisionWorld(right_model,cube_pose[:3,3],other_q=self.measured_q(),
                             other_base=self.model.base,cube_rotation=cube_pose[:3,:3])
        home=np.zeros(6);home[-1]=np.pi/4
        self.phase='plan_right_retreat'
        if self.checkpoint:self.checkpoint()
        check=self.validate_segment('right_retreat',world,self.measured_q('right'),home,margin=.001)
        result['right_retreat_check']=check
        if not check['safe']:raise RuntimeError('Post-release retreat rejected')
        self.contact_arms.clear()
        self.move_right(self.measured_q('right'),home,'right_retreat')
        self.phase='bin_settle';metrics=[]
        for _ in range(round(3/self.dt)):
            self.step(self.left_target)
            state=self.cube.data.root_state_w[0].cpu().numpy()
            pose=transform(state[:3],Rotation.from_quat(np.roll(state[3:7],-1)).as_matrix())
            verts=apply(pose,box_mesh(np.zeros(3),np.full(3,.04),'cube').vertices)
            support=float(self.cube_contact.get_contact_force_matrix(self.dt)[0,1,2].item())
            inside=bool(np.all(verts.min(0)>=[.27,-.11,.009]) and np.all(verts.max(0)<=[.41,.03,.085]))
            metrics.append({'inside':inside,'floor_support_N':support,'linear_speed_m_s':float(np.linalg.norm(state[7:10])),
                            'angular_speed_rad_s':float(np.linalg.norm(state[10:13]))})
        result['bin_settle_metrics']=metrics
        result['placement_cube_m']=self.cube.data.root_pos_w[0].cpu().numpy().copy()
        result['bin_min_floor_support_N']=min(m['floor_support_N'] for m in metrics)
        result['bin_max_linear_speed_m_s']=max(m['linear_speed_m_s'] for m in metrics)
        result['bin_max_angular_speed_rad_s']=max(m['angular_speed_rad_s'] for m in metrics)
        if not all(m['inside'] and .3<m['floor_support_N']<.7 and m['linear_speed_m_s']<.01
                   and m['angular_speed_rad_s']<.1 for m in metrics):
            raise RuntimeError('Released cube did not remain settled inside bin on its floor')
        result['placement_success']=True
        result['full_task_success']=all(result[k] for k in ('pickup_success','lift_success','handover_success','placement_success'))


def evaluate_policy(env,case):
    """Same reset, actuators, physics and evaluator; no expert control or plans."""
    from multiprocessing.connection import Client
    xy,yaw,seed=EVALUATION_CASES[case]
    env.reset(xy,yaw,seed);latency=[];connection=None;actions=None
    result=dict(case=case,configuration=EVALUATION_CASES[case],controller='learned' if options.policy_socket else 'recorded_targets',
                reset=env.evaluator.report()['reset'])
    try:
        if options.policy_socket:
            connection=Client(str(options.policy_socket),family='AF_UNIX')
            connection.send(dict(command='reset',seed=seed));handshake=connection.recv()
            if not handshake.get('ok') or handshake.get('protocol')!='so101-policy-1.0':
                raise RuntimeError('Restart policy server with the matching versioned protocol')
            result['policy_metadata']=handshake['policy_metadata'];result['policy_seed']=handshake['seed']
        else:
            from dataset_io import validate_episode
            validate_episode(options.replay)
            recorded=json.loads((options.replay/'episode.json').read_text())
            if not np.allclose(np.r_[xy,yaw],np.r_[recorded['configuration'][0],recorded['configuration'][1]],rtol=0,atol=1e-9):
                raise ValueError('Replay reset configuration differs from the recording')
            with np.load(options.replay/'trajectory.npz') as data:actions=data['action'].copy()
        for step in range(round(env.evaluator.spec.timeout_s*20)+1):
            if connection:
                images=env.capture_images()
                state=np.r_[env.frame.q.ravel(),env.frame.qd.ravel()].astype(np.float32)
                # Bytes/lists avoid NumPy 1.x/2.x pickle implementation coupling.
                started=time.perf_counter();connection.send(dict(command='action',state=state.tolist(),
                    images={key:value.tobytes() for key,value in images.items()}))
                response=connection.recv();latency.append(dict(round_trip_s=time.perf_counter()-started,
                                                              model_s=response['inference_wall_s']))
                action=response['action']
            else:
                # Keep the last actual target for terminal dwell; never copy
                # recorded physical states into the running simulation.
                action=actions[min(step,len(actions)-1)]
                if step>=len(actions)+60:raise RuntimeError('Replay failed terminal dwell within 3 seconds')
            _,terminal,_=env.step(np.asarray(action).reshape(2,6))
            if step%200==0:
                print(f'[POLICY] {case}: {step+1} policy steps; stages={env.evaluator.flags}',flush=True)
            if terminal:break
        result['status']='passed' if env.evaluator.success else 'failed'
        result['failure_category']=env.evaluator.failure
    except Exception as exc:
        result.update(status='failed',failure=str(exc),failure_category=env.evaluator.failure or 'policy_adapter')
    finally:
        if connection:connection.close()
    evaluation=env.evaluator.report();result['independent_evaluation']=evaluation
    for key,value in evaluation['stages'].items():result[key+'_success']=value
    result['full_task_success']=evaluation['success']
    result['inference_samples']=len(latency)
    if latency:
        result['latency_s']={key:dict(mean=float(np.mean([r[key] for r in latency])),
            p95=float(np.percentile([r[key] for r in latency],95)),maximum=max(r[key] for r in latency)) for key in latency[0]}
    return result


def main():
    if options.episodes<1:raise ValueError('episodes must be positive')
    if options.policy_socket or options.replay:
        if options.stage!='full':raise ValueError('Policy/replay evaluation requires --stage full')
        if options.policy_socket and options.replay:raise ValueError('Choose policy inference or recorded replay')
        if options.record_dir:raise ValueError('--record-dir is for expert demonstrations')
    cases_document=None
    if options.cases_file:
        cases_document=json.loads(options.cases_file.read_text())
        if cases_document['schema']!='so101-cases-1.0':raise ValueError('Unsupported case schema')
        if 'splits' in cases_document:
            cases_document=dict(cases_document,cases=cases_document['splits'][options.split],split=options.split)
            cases_document.pop('splits')
        EVALUATION_CASES.update({row['name']:(tuple(row['cube_xy_m']),row['yaw_deg'],row['seed']) for row in cases_document['cases']})
    if options.case not in EVALUATION_CASES:raise ValueError(f'Unknown case {options.case}; choose {list(EVALUATION_CASES)}')
    if options.suite=='negative' and options.stage!='full':raise ValueError('Negative suite requires --stage full')
    if options.suite=='perturb' and options.stage not in ('model','full'):raise ValueError('Perturbation suite requires --stage model/full')
    cases=[(options.case,None)]
    if options.cases_file:cases=[(row['name'],None) for row in cases_document['cases']]
    if options.suite=='perturb':cases=[(name,None) for name in EVALUATION_CASES]
    if options.suite=='negative':cases=[('nominal','grasp'),('nominal','transfer')]
    print(f'[RUN] stage={options.stage}, suite={options.suite}, episodes/case={options.episodes}, headless={layout.args.headless}',flush=True)
    options.output_dir.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();model=SO101Model(layout.ROBOT_USD)
    geometry_s=time.perf_counter()-started
    started=time.perf_counter();calibration=calibrate(model)
    calibration_s=time.perf_counter()-started
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    sources=('run_manipulation.py','so101_model.py','so101_planning.py','benchmark_core.py','benchmark_env.py','inspect_task_layout.py','dataset_io.py')
    report={'stage':options.stage,'suite':options.suite,'requested_episodes_per_case':options.episodes,
            'requested_episodes':options.episodes*len(cases),'run_id':run_id,'headless':layout.args.headless,
            'handover_mode':options.handover_mode,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'motion_mode':options.motion_mode,'viewer_delay_s':options.viewer_delay,
            'expert_command_hz':options.expert_hz,
            'grip_feedback_from_lift':options.grip_feedback_from_lift,
            'runtime_threads':dict(requested_worker_limit=layout.args.cpu_threads,openblas_env=os.environ.get('OPENBLAS_NUM_THREADS'),
                                  torch_intraop=layout.torch.get_num_threads(),
                                  carb_workers=carb.settings.get_settings().get('/plugins/carb.tasking.plugin/threadCount'),
                                  tbb_workers=carb.settings.get_settings().get('/plugins/omni.tbb.globalcontrol/maxThreadCount')),
            'controller':'learned' if options.policy_socket else ('replay' if options.replay else 'expert'),
            'cases_document':cases_document,'policy_socket':str(options.policy_socket) if options.policy_socket else None,
            'replay':str(options.replay) if options.replay else None,
            'tracked_source_dirty':bool(subprocess.run(['git','diff','--quiet'],check=False).returncode),
            'source_sha256':{name:MAIN_SOURCE_SHA256 if name==Path(__file__).name else hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources},
            'calibration':calibration,'geometry_loading_s':geometry_s,'calibration_s':calibration_s,
            'planning_timings':PLAN_TIMINGS,'episodes':[],'status':'in_progress'}
    path=options.output_dir/'simulation.json'
    logging_s=0.
    history_path=options.output_dir/f'{options.stage}_{run_id}.json'
    def save():
        nonlocal logging_s
        started=time.perf_counter();path.write_text(json.dumps(jsonable(report),indent=2))
        history_path.write_text(json.dumps(jsonable(report),indent=2))
        logging_s+=time.perf_counter()-started
    save()
    sim=layout.SimulationContext(layout.sim_utils.SimulationCfg(device=layout.args.device))
    cfg=layout.BimanualSceneCfg(num_envs=1,env_spacing=2.)
    if options.record_dir or options.policy_socket:configure_cameras(cfg)
    cfg.left_arm.init_state.pos=layout.LEFT_BASE_M;cfg.right_arm.init_state.pos=layout.RIGHT_BASE_M
    cfg.left_arm.spawn.activate_contact_sensors=True;cfg.right_arm.spawn.activate_contact_sensors=True
    cfg.cube.spawn.activate_contact_sensors=True
    # Kinematic actors retain fixed geometry and expose GPU pairwise forces.
    for name in ('table','bin_floor','bin_left_wall','bin_right_wall','bin_front_wall','bin_back_wall'):
        getattr(cfg,name).spawn.rigid_props=layout.sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True)
    scene=layout.InteractiveScene(cfg);sim.reset()
    sim.set_camera_view(eye=(.8,.9,.7),target=(0,-.12,.10))
    episode=Episode(sim,scene,model)
    episode.source_sha256=report['source_sha256']
    episode.source_revision=report['source_revision'];episode.source_dirty=report['tracked_source_dirty']
    if cases_document:episode.split=cases_document.get('split',options.split)
    report['camera_calibration']=episode.env.camera_metadata()
    report['cold_start_wall_s']=time.perf_counter()-PROCESS_STARTED
    report['cube_com_offset_body_m']=episode.cube.data.body_com_pos_b.cpu().numpy()
    def checkpoint():
        nonlocal logging_s
        # Do not serialize all completed traces/plans every two simulated
        # seconds. Full history is persisted at episode boundaries; retain a
        # small interruption checkpoint with an explicit link to that history.
        started=time.perf_counter()
        active={k:v for k,v in getattr(episode,'result',{}).items()
                if not k.endswith(('_plan','_metrics','_trace','_peaks'))}
        state=dict(status='in_progress',run_id=run_id,history_file=str(history_path),
                   active_phase=episode.phase,active_case=episode.case,active_episode=active,
                   finished_attempts=len(report['episodes']),last_episode_samples=episode.samples[-12:])
        path.write_text(json.dumps(jsonable(state),indent=2))
        logging_s+=time.perf_counter()-started
    unexpected_failures=[]
    try:
        for case,fault in cases:
            episode.case=case;episode.fault=fault
            if options.policy_socket or options.replay:
                for repetition in range(options.episodes):
                    started=time.perf_counter();before=dict(episode.env.timings)
                    result=evaluate_policy(episode.env,case)
                    result.update(repetition=repetition+1,episode_wall_s=time.perf_counter()-started,
                                  environment_timing_s={k:v-before[k] for k,v in episode.env.timings.items()})
                    report['episodes'].append(result);save()
                    if not result['full_task_success']:unexpected_failures.append(dict(case=case,failure=result['failure_category']))
                continue
            episode.plan_dir=options.output_dir/'plans'/case/options.handover_mode/options.motion_mode
            episode.plan_dir.mkdir(parents=True,exist_ok=True)
            xy,yaw,seed=EVALUATION_CASES[case]
            cube=np.array([*xy,.02]);rotation=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
            plan_path=episode.plan_dir/'plan.json'
            plan_started=time.perf_counter()
            key=fingerprint(model)+json.dumps(EVALUATION_CASES[case])
            plan=json.loads(plan_path.read_text()) if plan_path.exists() else {}
            cache_hit=plan.get('fingerprint')==key
            if options.stage!='model' and not cache_hit:
                try:plan=compute_plan(plan_pickup,model,cube,rotation)
                except Exception as exc:plan={'waypoints':[],'failure':str(exc)}
                plan['fingerprint']=key;plan_path.write_text(json.dumps(jsonable(plan),indent=2))
            report.setdefault('approach_plan_timings',[]).append(dict(case=case,cache_hit=cache_hit,wall_s=time.perf_counter()-plan_started))
            for repetition in range(options.episodes):
                episode.samples=[];episode.result={};episode.failure_state=None;episode.checkpoint=checkpoint
                start_tick=episode.tick;started=time.perf_counter();timings_before=dict(episode.env.timings)
                plan_index=len(PLAN_TIMINGS)
                error=None
                try:
                    result=episode.run(plan)
                    result['status']='passed'
                except Exception as exc:
                    error=str(exc);result=getattr(episode,'result',{})
                    result.update(case=case,fault=fault,status='failed',failure=error,failure_phase=episode.phase)
                    frame=episode.env.frame
                    if frame is not None:
                        result['failure_observation']=dict(cube=frame.cube,pads_N=frame.pads,environment_N=frame.environment,q_rad=frame.q)
                    result['independent_evaluation']=episode.env.evaluator.report()
                    result['failure_category']=(episode.env.evaluator.failure or
                        {'reset':'reset_prerequisite','close':'failed_grasp','right_close':'failed_transfer',
                         'hold':'failed_lift','lift':'failed_lift'}.get(episode.phase)
                        or ('planning_rejection' if any(word in error.lower() for word in ('path','planning','collision','unsafe')) else 'controller_guard'))
                    result['transfer_trace']=list(episode.transfer_trace)
                    result['transfer_metrics']=transfer_metrics(episode.transfer_trace)
                    # Expected negatives must fail at the actual omitted grasp,
                    # never from an unrelated reset/planning/collision failure.
                    expected=(fault=='grasp' and 'No sustained bilateral cube contact' in error
                              or fault=='transfer' and 'Right grasp has insufficient bilateral preload' in error)
                    result['expected_failure_detected']=bool(expected and not episode.env.evaluator.success)
                    if episode.writer:
                        episode.writer.finish(False,result['independent_evaluation'],episode.env.frame)
                        episode.writer=None
                    if not expected:unexpected_failures.append(dict(case=case,repetition=repetition+1,failure=error))
                    try:
                        episode.checkpoint=None;result['failure_reset']=episode.reset()
                    except Exception as reset_exc:
                        result['failure_reset_error']=str(reset_exc)
                        unexpected_failures.append(dict(case=case,failure='failure_reset:'+str(reset_exc)))
                    print(f'[FAILURE] {case}: {error}',flush=True)
                result['episode_wall_s']=time.perf_counter()-started
                result['episode_duration_s']=(episode.tick-start_tick)*episode.dt
                result['planning_wall_s']=sum(row['wall_s'] for row in PLAN_TIMINGS[plan_index:])
                result['environment_timing_s']={name:value-timings_before[name] for name,value in episode.env.timings.items()}
                # Failure reset clears trace/reference state: capture before reset
                # in the exception path, retain normal trace here for successes.
                if error is None:
                    result['transfer_trace']=list(episode.transfer_trace)
                    result['transfer_metrics']=transfer_metrics(episode.transfer_trace)
                    if episode.writer:
                        episode.writer.finish(bool(result.get('full_task_success')),episode.env.evaluator.report(),episode.env.frame)
                        episode.writer=None
                result['repetition']=repetition+1
                report['episodes'].append(result);report['last_episode_samples']=episode.samples
                save()
                print(f'[EPISODE] {case}/{repetition+1}: {result["status"]}',flush=True)
                if error and options.suite=='nominal' and not options.cases_file:break
            if unexpected_failures and options.suite=='nominal' and not options.cases_file:break
        report['status']='failed' if unexpected_failures else ('passed_negative_regression' if options.suite=='negative' else 'passed_requested_stage')
    finally:
        keys=('approach_success','pickup_success','lift_success','transport_success','handover_success','placement_success','full_task_success')
        report['success_counts']={key:sum(bool(e.get(key)) for e in report['episodes']) for key in keys}
        report['attempted_episodes']=len(report['episodes'])
        report['completed_episodes']=sum(e['status']=='passed' for e in report['episodes'])
        report['unexpected_failures']=unexpected_failures
        report['case_counts']={case:dict(attempted=sum(e['case']==case for e in report['episodes']),
            full_task_success=sum(e['case']==case and bool(e.get('full_task_success')) for e in report['episodes'])) for case,_ in cases}
        report['logging_wall_s']=logging_s
        report['total_wall_s']=time.perf_counter()-PROCESS_STARTED
        report['process_cpu_s']=time.process_time()-PROCESS_CPU_STARTED
        save()
    print('[RESULT]',report['status'],flush=True)
    if unexpected_failures:raise RuntimeError(f'{len(unexpected_failures)} unexpected failure(s); see {path}')
    if not layout.args.headless and not options.exit_after:
        print('[VIEWER] Evaluation saved. Final state frozen for inspection; close the window to exit.',flush=True)
        sim.clear_all_callbacks()
        while layout.app.is_running():sim.render();time.sleep(.02)
    return sim,scene,episode


if __name__=='__main__':
    try:handles=main()
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    else:
        # Destroy RTX sensors while scene/PhysX views remain strongly owned.
        # Sim 5.1 prim-deletion callbacks contain weak proxies; collecting the
        # whole scene at once can otherwise abort during Replicator detach.
        import gc
        handles[2].env.cameras.clear();handles[1].sensors.clear();gc.collect()
        sim=layout.SimulationContext.instance()
        if sim is not None:sim.clear_all_callbacks();sim.clear_instance()
        layout.app.close(wait_for_replicator=False)
