"""Finite deterministic SO-101 pickup evaluation in the existing task scene.

Run: python scripts/run_manipulation.py --headless --stage approach --episodes 1
Only episode reset writes object/joint state. Task motion uses actuator targets.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import inspect
import math
import os
from pathlib import Path
import sys
import time
import traceback

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--stage', choices=('model','approach','pickup'), default='approach')
parser.add_argument('--episodes',type=int,default=1)
parser.add_argument('--output-dir',type=Path,default=Path('outputs/controller'))
options, rest = parser.parse_known_args()
sys.argv=[sys.argv[0],*rest]
import inspect_task_layout as layout
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import omni.physics.tensors as tensors
from so101_model import SO101Model, calibrate, jsonable, JOINT_NAMES, LINK_NAMES
from so101_planning import plan_pickup, plan_lift, CollisionWorld, fingerprint


class Episode:
    def __init__(self,sim,scene,model):
        self.sim,self.scene,self.model=sim,scene,model
        self.left,self.right,self.cube=scene['left_arm'],scene['right_arm'],scene['cube']
        self.ids=[layout.joint_index(self.left,n) for n in JOINT_NAMES]
        self.dt=sim.get_physics_dt()
        self.root='/World/envs/env_0/'
        self.filters=[self.root+n for n in ('Cube','Table','BinFloor','BinLeftWall','BinRightWall','BinFrontWall','BinBackWall')]
        self.filters += [self.root+arm+'/'+body for arm in ('Left_Robot','Right_Robot') for body in LINK_NAMES]
        self.physics=tensors.create_simulation_view('torch')
        self.physics.set_subspace_roots('/')
        self.contacts=[self.physics.create_rigid_contact_view([self.root+arm+'/'+n for n in LINK_NAMES],filter_patterns=[self.filters]*len(LINK_NAMES),
                                                            max_contact_data_count=4096)
                       for arm in ('Left_Robot','Right_Robot')]
        self.cube_contact=self.physics.create_rigid_contact_view(self.root+'Cube',filter_patterns=[self.root+'Table'])
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

    def measured_q(self):
        return self.left.data.joint_pos[0,self.ids].detach().cpu().numpy().copy()

    def tcp(self):
        idx=layout.body_index(self.left,'gripper')
        p=self.left.data.body_pos_w[0,idx].cpu().numpy()
        q=self.left.data.body_quat_w[0,idx].cpu().numpy()
        return p+Rotation.from_quat(np.roll(q,-1)).apply(self.model.tcp[:3,3])

    def check_fk(self):
        poses=self.model.fk(self.measured_q())
        errors=[]
        angles=[]
        for name in LINK_NAMES:
            idx=layout.body_index(self.left,name)
            actual_p=self.left.data.body_pos_w[0,idx].cpu().numpy()
            actual_q=self.left.data.body_quat_w[0,idx].cpu().numpy()
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
                    allowed |= self.contact_allowed and other=='Cube' and body in ('gripper','jaw') and 'Left_Robot' in source
                    if not allowed:
                        forbidden.append((source,target,float(magnitude)))
        if forbidden:
            self.failure_state={'phase':self.phase,'contacts':forbidden,'q_rad':self.measured_q(),
                                'tcp_m':self.tcp(),'cube_m':self.cube.data.root_pos_w[0].cpu().numpy().copy()}
            raise RuntimeError(f'Forbidden physical contact in {self.phase}: {forbidden[:4]}')
        v=self.contacts[0]
        indices={str(p).rsplit('/',1)[-1]:i for i,p in enumerate(v.sensor_paths)}
        return [float(np.linalg.norm(matrices[0][indices[n],0])) for n in ('gripper','jaw')]

    def step(self,target,check=True):
        values=self.left.data.default_joint_pos.clone()
        values[0,self.ids]=torch.tensor(target,device=self.left.device,dtype=values.dtype)
        self.left.set_joint_position_target(values)
        self.right.set_joint_position_target(self.right.data.default_joint_pos)
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
        if check and not self.contact_allowed and self.baseline is not None and np.linalg.norm(cube-self.baseline)>.002:
            raise RuntimeError(f'Cube disturbed before intended contact: {(cube-self.baseline).tolist()}')
        if self.tick%10==0:
            tracking=float(np.linalg.norm(self.tcp()-self.model.tcp_pose(target)[:3,3]))
            self.max_tracking_m=max(self.max_tracking_m,tracking)
            self.samples.append({'t_s':self.tick*self.dt,'phase':self.phase,'cube_m':cube,'tcp_m':self.tcp(),
                                 'cube_quat_wxyz':self.cube.data.root_quat_w[0].cpu().numpy().copy(),
                                 'q_rad':self.measured_q(),'target_q_rad':target.copy(),'pad_force_N':forces,
                                 'tcp_tracking_error_mm':tracking*1000})
        if not layout.args.headless:
            time.sleep(.008)
        return cube,forces

    def move(self,start,end,phase,contact=False,jaw_target=None):
        self.phase=phase;self.contact_allowed=contact
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

    def reset(self):
        self.phase='reset';self.contact_allowed=False;self.baseline=None
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
        closure_check=CollisionWorld(self.model).segment(closed,closure_end,cube_contact=True)
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
        if lift.get('source_fingerprint')!=lift_key or not np.allclose(lift.get('start_q',np.full(6,99)),measured,atol=1e-7,rtol=0):
            lift=plan_lift(self.model,measured,held=True)
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
        return result


def main():
    if options.episodes<1:raise ValueError('episodes must be positive')
    options.output_dir.mkdir(parents=True,exist_ok=True)
    model=SO101Model(layout.ROBOT_USD)
    calibration=calibrate(model)
    plan_path=options.output_dir/'plan.json'
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else {}
    if options.stage!='model' and plan.get('fingerprint')!=fingerprint(model):
        plan=plan_pickup(model)
        plan['fingerprint']=fingerprint(model)
        plan_path.write_text(json.dumps(jsonable(plan),indent=2))
    report={'calibration':calibration,'plan':plan,'episodes':[],'status':'in_progress'}
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
    sim.set_camera_view(eye=(-.8,-.6,.55),target=(-.30,-.05,.10))
    episode=Episode(sim,scene,model)
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
            print(f'[EPISODE {i+1}] '+json.dumps(jsonable({k:v for k,v in result.items() if k!='lift_plan'})),flush=True)
        report['status']='passed_requested_stage'
    except Exception as exc:
        report['status']='failed';report['failure']=str(exc)
        report['failure_state']=episode.failure_state
        report['failed_episode']=getattr(episode,'result',{})
        report['last_episode_samples']=episode.samples
        try:
            report['failure_reset']=episode.reset()
        except Exception as reset_exc:
            report['failure_reset_error']=str(reset_exc)
        raise
    finally:
        report['success_counts']={key:sum(bool(e.get(key)) for e in report['episodes'])
                                  for key in ('approach_success','pickup_success','lift_success','handover_success','placement_success','full_task_success')}
        report['completed_episodes']=len(report['episodes'])
        save()
    print('[RESULT]',report['status'],flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    else:
        sim=layout.SimulationContext.instance()
        if sim is not None:sim.clear_all_callbacks();sim.clear_instance()
        layout.app.close(wait_for_replicator=False)
