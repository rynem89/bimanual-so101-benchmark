"""Deterministic SO-101 manipulation evaluation in the existing task scene.

Run: python scripts/run_manipulation.py --headless --stage approach --episodes 1
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
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--stage', choices=('model','approach','pickup','transport','handover','full'), default='approach')
parser.add_argument('--episodes',type=int,default=1)
parser.add_argument('--output-dir',type=Path,default=Path('outputs/controller'))
parser.add_argument('--exit-after',action='store_true',help='Close the GUI after evaluation instead of preserving the final view')
options, rest = parser.parse_known_args()
sys.argv=[sys.argv[0],*rest]
import inspect_task_layout as layout
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import omni.physics.tensors as tensors
from so101_model import SO101Model, calibrate, jsonable, transform, apply, box_mesh, JOINT_NAMES, LINK_NAMES
from so101_planning import plan_pickup, plan_lift, plan_transport, plan_receiver, plan_placement, transport_fingerprint, CollisionWorld, fingerprint

# Joint radians / transform metres or matrix entries. The 0.6 m radius bound
# gives <0.1 mm combined geometric change, below the >=1 mm planning margins.
CACHE_STATE_ATOL=1e-5


def compute_plan(function,*args,**kwargs):
    """Keep the GUI responsive while CPU-only geometry planning runs.

    Rendering does not step physics, so measured held poses stay fixed while
    planning. The worker touches only the already-loaded NumPy robot model.
    """
    started=time.perf_counter()
    name=function.__qualname__
    print(f'[PLAN] {name}: computing CPU checks; physics paused.',flush=True)
    if layout.args.headless:
        result=function(*args,**kwargs)
        print(f'[PLAN DONE] {name}: {time.perf_counter()-started:.2f}s',flush=True)
        return result
    with ThreadPoolExecutor(max_workers=1) as worker:
        future=worker.submit(function,*args,**kwargs)
        while not future.done():
            sim=layout.SimulationContext.instance()
            if sim is None:layout.app.update()
            else:sim.render()
            time.sleep(.02)
        result=future.result()
    print(f'[PLAN DONE] {name}: {time.perf_counter()-started:.2f}s',flush=True)
    return result


class Episode:
    def __init__(self,sim,scene,model):
        self.sim,self.scene,self.model=sim,scene,model
        self.left,self.right,self.cube=scene['left_arm'],scene['right_arm'],scene['cube']
        self.ids=[layout.joint_index(self.left,n) for n in JOINT_NAMES]
        if self.ids!=[layout.joint_index(self.right,n) for n in JOINT_NAMES]:
            raise RuntimeError('Left/right joint ordering differs; explicit remapping required')
        self.dt=sim.get_physics_dt()
        self.root='/World/envs/env_0/'
        self.filters=[self.root+n for n in ('Cube','Table','BinFloor','BinLeftWall','BinRightWall','BinFrontWall','BinBackWall')]
        self.filters += [self.root+arm+'/'+body for arm in ('Left_Robot','Right_Robot') for body in LINK_NAMES]
        self.physics=tensors.create_simulation_view('torch')
        self.physics.set_subspace_roots('/')
        self.contacts=[self.physics.create_rigid_contact_view([self.root+arm+'/'+n for n in LINK_NAMES],filter_patterns=[self.filters]*len(LINK_NAMES),
                                                            max_contact_data_count=4096)
                       for arm in ('Left_Robot','Right_Robot')]
        self.cube_contact=self.physics.create_rigid_contact_view(self.root+'Cube',filter_patterns=self.filters[1:7])
        if self.cube_contact.sensor_count!=1 or self.cube_contact.filter_count!=6:
            raise RuntimeError('Cube/environment force sensor prerequisites failed')
        for v in self.contacts:
            if v.sensor_count!=7 or v.filter_count!=len(self.filters):
                raise RuntimeError(f'Contact prerequisites: sensors={v.sensor_count}, filters={v.filter_count}')
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
        directory=options.output_dir/'segments';directory.mkdir(exist_ok=True)
        path=directory/f'{name}.json'
        saved=json.loads(path.read_text()) if path.exists() else {}
        valid=saved.get('fingerprint')==key
        if valid:valid=np.allclose(saved['context'],context,atol=CACHE_STATE_ATOL,rtol=0)
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
        matrices=[]
        forbidden=[]
        for view in self.contacts:
            forces=view.get_contact_force_matrix(self.dt).cpu().numpy()
            matrices.append(forces)
            for i,path in enumerate(view.sensor_paths):
                source=str(path); body=source.rsplit('/',1)[-1]
                for j,target in enumerate(self.filters):
                    magnitude=np.linalg.norm(forces[i,j])
                    if magnitude<.05 or target==source:
                        continue
                    other=target.rsplit('/',1)[-1]
                    same_robot=source.rsplit('/',1)[0]==target.rsplit('/',1)[0]
                    allowed=(other=='Table' and body=='base') or (same_robot and frozenset((body,other)) in self.model.adjacent)
                    allowed |= self.contact_allowed and other=='Cube' and body in ('gripper','jaw') and any(name in source for name in self.contact_arms)
                    if not allowed:
                        forbidden.append((source,target,float(magnitude)))
        if forbidden:
            self.failure_state={'phase':self.phase,'contacts':forbidden,'q_rad':self.measured_q(),
                                'tcp_m':self.tcp(),'cube_m':self.cube.data.root_pos_w[0].cpu().numpy().copy()}
            raise RuntimeError(f'Forbidden physical contact in {self.phase}: {forbidden[:4]}')
        if self.transport_reference is not None:
            environment=self.cube_contact.get_contact_force_matrix(self.dt)[0].cpu().numpy()
            maximum=float(np.linalg.norm(environment,axis=-1).max())
            self.max_carried_environment_force_N=max(self.max_carried_environment_force_N,maximum)
            if maximum>=.05:
                raise RuntimeError(f'Carried cube/environment contact in {self.phase}: {maximum:.3f}N')
        for i,v in enumerate(self.contacts):
            indices={str(p).rsplit('/',1)[-1]:j for j,p in enumerate(v.sensor_paths)}
            self.pad_forces[('left','right')[i]]=[float(np.linalg.norm(matrices[i][indices[n],0])) for n in ('gripper','jaw')]
        return self.pad_forces['left']

    def step(self,target,check=True):
        target=target.copy()
        # Orientation changes alter gravity torque at the jaw. Retain measured
        # bilateral preload with bounded slow servo correction, not attachment.
        for side,command in list(self.active_grips.items()):
            force=self.pad_forces.get(side,[0.,0.])
            if min(force)<3.:command-=np.deg2rad(.02)
            elif min(force)>4.:command+=np.deg2rad(.01)
            command=float(np.clip(command,np.deg2rad(15),np.deg2rad(30)))
            self.active_grips[side]=command
            if side=='left':target[-1]=command
            else:self.right_target[-1]=command
        self.left_target=target.copy()
        values=self.left.data.default_joint_pos.clone()
        values[0,self.ids]=torch.tensor(target,device=self.left.device,dtype=values.dtype)
        self.left.set_joint_position_target(values)
        right_values=self.right.data.default_joint_pos.clone()
        right_values[0,self.ids]=torch.tensor(self.right_target,device=self.right.device,dtype=right_values.dtype)
        self.right.set_joint_position_target(right_values)
        # Feed-forward actuator torque cancels robot gravity; gravity and cube
        # physics remain enabled. This avoids centimetre-scale passive sag.
        for arm in (self.left,self.right):
            effort=arm.root_physx_view.get_gravity_compensation_forces()
            if not torch.isfinite(effort).all() or effort.abs().max()>10:
                raise RuntimeError('Gravity compensation exceeds actuator limit')
            self.max_feedforward_Nm=max(self.max_feedforward_Nm,float(effort.abs().max().item()))
            arm.set_joint_effort_target(effort)
        layout.step_physics(self.sim,self.scene,1)
        for arm in (self.left,self.right):
            speed=float(arm.data.joint_vel.abs().max().item())
            torque=float(arm.data.applied_torque.abs().max().item())
            self.max_speed_rad_s=max(self.max_speed_rad_s,speed)
            self.max_applied_torque_Nm=max(self.max_applied_torque_Nm,torque)
            if speed>10.05 or torque>10.05:
                raise RuntimeError('Measured actuator speed/effort limit exceeded')
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
        if not layout.args.headless:
            time.sleep(.008)
        return cube,forces

    def move(self,start,end,phase,contact=False,jaw_target=None):
        self.phase=phase;self.contact_allowed=contact
        print('[PHASE]',phase,flush=True)
        duration=max(1.,np.max(np.abs(end-start))*1.875/.35)
        for t in np.linspace(0,1,math.ceil(duration/self.dt)+1):
            blend=t*t*t*(10-15*t+6*t*t)
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
        for t in np.linspace(0,1,math.ceil(duration/self.dt)+1):
            blend=t*t*t*(10-15*t+6*t*t)
            self.right_target=start+(end-start)*blend
            self.step(self.left_target)
        self.right_target=end.copy()
        for _ in range(90):self.step(self.left_target)
        if np.max(np.abs(self.measured_q('right')[:5]-end[:5]))>np.deg2rad(3):
            raise RuntimeError('Right arm joint tracking error >3 degrees')
        return self.check_fk('right')

    def reset(self):
        self.phase='reset';self.contact_allowed=False;self.baseline=None
        self.transport_reference=None
        self.reference_robot='left';self.contact_arms={'Left_Robot'}
        self.right_target=np.zeros(6);self.left_target=np.zeros(6)
        self.active_grips={}
        self.max_carried_environment_force_N=0.
        self.max_transport_slip_m=0.;self.max_transport_rotation_deg=0.
        for arm in (self.left,self.right):
            arm.write_joint_state_to_sim(arm.data.default_joint_pos,torch.zeros_like(arm.data.default_joint_pos))
            arm.reset()
        self.cube.write_root_state_to_sim(self.cube.data.default_root_state.clone())
        self.cube.reset()
        for _ in range(180):self.step(np.zeros(6),check=False)
        self.baseline=self.cube.data.root_pos_w[0].cpu().numpy().copy()
        if np.linalg.norm(self.baseline[:2]-np.array(layout.CUBE_START_M[:2]))>.002 or abs(self.baseline[2]-.02)>.002:
            raise RuntimeError('Reset cube pose outside 2mm tolerance')
        self.forces()
        support=float(self.cube_contact.get_contact_force_matrix(self.dt)[0,0,2].item())
        if not .3<support<.7:
            raise RuntimeError(f'Contact sensor prerequisite: cube support {support:.3f}N, expected about 0.49N')
        self.max_tracking_m=0.;self.max_feedforward_Nm=0.
        self.max_speed_rad_s=0.;self.max_applied_torque_Nm=0.
        result=self.check_fk();result['cube_table_support_N']=support
        return result

    def run(self,plan):
        result={'reset_fk':self.reset(),'pickup_success':False,'lift_success':False,'handover_success':False,
                'placement_success':False,'full_task_success':False}
        self.result=result
        if options.stage=='model':return result
        waypoints=[np.array(q) for q in plan['waypoints']]
        if not waypoints:raise RuntimeError('No validated approach path')
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
        closure_check=self.validate_segment('left_closure',CollisionWorld(self.model),closed,closure_end,cube_contact=True)
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
                closed[-1]=max(np.deg2rad(-5),closed[-1]-np.deg2rad(.12))
        for _ in range(60):cube,force=self.step(closed)
        result['closure_pad_force_N']=force
        result['closure_cube_m']=cube.tolist()
        result['closure_joint_deg']=float(np.rad2deg(self.measured_q()[-1]))
        if min(force)<2.:raise RuntimeError(f'No sustained bilateral cube contact: {force}')
        print('[CLOSURE]',result['closure_joint_deg'],force,flush=True)
        measured=self.measured_q()
        lift_key=fingerprint(self.model)+hashlib.sha256(inspect.getsource(plan_lift).encode()).hexdigest()
        lift_path=options.output_dir/'lift_plan.json'
        lift=json.loads(lift_path.read_text()) if lift_path.exists() else {}
        if lift.get('source_fingerprint')!=lift_key or not np.allclose(lift.get('start_q',np.full(6,99)),measured,atol=CACHE_STATE_ATOL,rtol=0):
            lift=compute_plan(plan_lift,self.model,measured,held=True)
            lift['source_fingerprint']=lift_key;lift['start_q']=measured
            lift_path.write_text(json.dumps(jsonable(lift),indent=2))
        result['lift_plan']=lift
        if not lift['safe']:raise RuntimeError(f'Lift path rejected: {lift}')
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
            raise RuntimeError(f'Lift/hold failed: {result}')
        result['pickup_success']=True;result['lift_success']=True
        if options.stage in ('transport','handover','full'):
            local=self.cube_in_tcp();start=self.measured_q()
            path=options.output_dir/'handover_plan.json'
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
            self.active_grips['left']=closed[-1]
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
        return result

    def handover(self,result,transport):
        """Receive on orthogonal faces, unload left, verify right-only support."""
        left_q=self.measured_q()
        cube_pose=self.model.tcp_pose(left_q)@self.cube_in_tcp()
        reference=np.array(transport['selected']['right_q'])
        path=options.output_dir/'receiver_plan.json'
        key=transport_fingerprint(self.model)+hashlib.sha256(inspect.getsource(plan_receiver).encode()).hexdigest()
        receiver=json.loads(path.read_text()) if path.exists() else {}
        valid=receiver.get('fingerprint')==key
        valid &= np.allclose(receiver.get('left_q',np.full(6,99)),left_q,atol=CACHE_STATE_ATOL,rtol=0)
        if valid:valid=np.allclose(receiver['cube_pose'],cube_pose,atol=CACHE_STATE_ATOL,rtol=0)
        if not valid:
            print('[PLAN] Validating receiver path around the measured held cube',flush=True)
            receiver=compute_plan(plan_receiver,self.model,left_q,cube_pose,reference)
            receiver['fingerprint']=key
            path.write_text(json.dumps(jsonable(receiver),indent=2))
        result['receiver_plan']=receiver
        if receiver['selected'] is None:raise RuntimeError('Receiver path rejected; see receiver_plan.json')
        waypoints=[np.array(q) for q in receiver['selected']['waypoints']]
        phases=['right_open','right_pregrasp']+[f'right_approach_{i}' for i in range(len(waypoints)-3)]
        for i,phase in enumerate(phases):
            result[phase+'_fk']=self.move_right(waypoints[i],waypoints[i+1],phase)
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
                self.right_target[-1]=max(np.deg2rad(15),self.right_target[-1]-np.deg2rad(.12))
        for _ in range(60):self.step(self.left_target)
        result['right_acquisition_force_N']=self.pad_forces['right'].copy()
        if min(self.pad_forces['right'])<.6:raise RuntimeError('Right grasp has insufficient bilateral preload')
        left_q=self.measured_q();right_q=self.measured_q('right')
        release=left_q.copy();release[-1]=np.pi/4
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
        self.move(left_q,release,'left_release',contact=True)
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
        path=options.output_dir/'placement_plan.json'
        key=transport_fingerprint(self.model)+hashlib.sha256(inspect.getsource(plan_placement).encode()).hexdigest()
        placement=json.loads(path.read_text()) if path.exists() else {}
        valid=placement.get('fingerprint')==key
        for name,value in (('left_start',left_q),('right_start',right_q),('cube_in_right_tcp',local)):
            if valid:valid=np.allclose(placement[name],value,atol=CACHE_STATE_ATOL,rtol=0)
        if not valid:
            print('[PLAN] Validating donor withdrawal and loaded bin travel',flush=True)
            placement=compute_plan(plan_placement,self.model,left_q,right_q,local)
            placement['fingerprint']=key
            path.write_text(json.dumps(jsonable(placement),indent=2))
        result['placement_plan']=placement
        selected=placement['selected']
        if selected is None:raise RuntimeError('No validated withdrawal/bin path; see placement_plan.json')
        self.contact_arms.discard('Left_Robot')
        waypoints=[np.array(q) for q in selected['left_departure']['waypoints']]
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


def main():
    if options.episodes<1:raise ValueError('episodes must be positive')
    print(f'[RUN] stage={options.stage}, episodes={options.episodes}, headless={layout.args.headless}',flush=True)
    options.output_dir.mkdir(parents=True,exist_ok=True)
    model=SO101Model(layout.ROBOT_USD)
    calibration=calibrate(model)
    plan_path=options.output_dir/'plan.json'
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else {}
    if options.stage!='model' and plan.get('fingerprint')!=fingerprint(model):
        plan=compute_plan(plan_pickup,model)
        plan['fingerprint']=fingerprint(model)
        plan_path.write_text(json.dumps(jsonable(plan),indent=2))
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report={'stage':options.stage,'requested_episodes':options.episodes,'run_id':run_id,'headless':layout.args.headless,
            'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ('run_manipulation.py','so101_model.py','so101_planning.py')},
            'calibration':calibration,'plan':plan,'episodes':[],'status':'in_progress'}
    path=options.output_dir/'simulation.json'
    def save():path.write_text(json.dumps(jsonable(report),indent=2))
    save()
    if options.stage!='model' and not plan['waypoints']:raise RuntimeError('Offline planning rejected all trajectories; see simulation.json')
    sim=layout.SimulationContext(layout.sim_utils.SimulationCfg(device=layout.args.device))
    cfg=layout.BimanualSceneCfg(num_envs=1,env_spacing=2.)
    cfg.left_arm.init_state.pos=layout.LEFT_BASE_M;cfg.right_arm.init_state.pos=layout.RIGHT_BASE_M
    cfg.left_arm.spawn.activate_contact_sensors=True;cfg.right_arm.spawn.activate_contact_sensors=True
    cfg.cube.spawn.activate_contact_sensors=True
    # GPU force filters require rigid actors. Kinematic bodies preserve the
    # fixed table/bin poses and shape dimensions while making forces observable.
    for name in ('table','bin_floor','bin_left_wall','bin_right_wall','bin_front_wall','bin_back_wall'):
        getattr(cfg,name).spawn.rigid_props=layout.sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True)
    scene=layout.InteractiveScene(cfg)
    sim.reset()
    sim.set_camera_view(eye=(.8,.9,.7),target=(0,-.12,.10))
    episode=Episode(sim,scene,model)
    def checkpoint():
        report['active_phase']=episode.phase
        report['active_episode']=getattr(episode,'result',{})
        report['last_episode_samples']=episode.samples
        save()
    episode.checkpoint=checkpoint
    try:
        for i in range(options.episodes):
            episode.samples=[]
            episode.result={};episode.failure_state=None
            start_tick=episode.tick
            result=episode.run(plan)
            result['episode_duration_s']=(episode.tick-start_tick)*episode.dt
            report['episodes'].append(result)
            report['last_episode_samples']=episode.samples
            save()
            print(f'[EPISODE {i+1}] '+json.dumps(jsonable({k:v for k,v in result.items() if not k.endswith(('_plan','_metrics'))})),flush=True)
        report['status']='passed_requested_stage'
    except Exception as exc:
        report['status']='failed';report['failure']=str(exc)
        report['failure_state']=episode.failure_state or {
            'phase':episode.phase,'left_q_rad':episode.measured_q(),'right_q_rad':episode.measured_q('right'),
            'cube_m':episode.cube.data.root_pos_w[0].cpu().numpy().copy(),'pad_force_N':episode.pad_forces}
        report['failed_episode']=getattr(episode,'result',{})
        report['last_episode_samples']=list(episode.samples)
        try:
            episode.checkpoint=None
            report['failure_reset']=episode.reset()
        except Exception as reset_exc:
            report['failure_reset_error']=str(reset_exc)
        raise
    finally:
        report['success_counts']={key:sum(bool(e.get(key)) for e in report['episodes'])
                                  for key in ('approach_success','pickup_success','lift_success','transport_success','handover_success','placement_success','full_task_success')}
        report['completed_episodes']=len(report['episodes'])
        save()
        (options.output_dir/f'{options.stage}_{run_id}.json').write_text(json.dumps(jsonable(report),indent=2))
    print('[RESULT]',report['status'],flush=True)
    if not layout.args.headless and not options.exit_after:
        print('[VIEWER] Evaluation saved. Final state frozen for inspection; close the window to exit.',flush=True)
        sim.clear_all_callbacks()
        while layout.app.is_running():
            sim.render()
            time.sleep(.02)


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    else:
        sim=layout.SimulationContext.instance()
        if sim is not None:sim.clear_all_callbacks();sim.clear_instance()
        layout.app.close(wait_for_replicator=False)
