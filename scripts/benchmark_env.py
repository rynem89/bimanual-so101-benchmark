"""Common Isaac adapter: reset, position actions, observations and task evaluation.

Import after AppLauncher. No planner imports; expert plans/contact labels are
never included in the default policy observation. Physical state writes occur
only in reset(). The existing actuator model and gravity compensation are shared
by all policies using this interface.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import omni.physics.tensors as tensors

from so101_model import JOINT_NAMES, LINK_NAMES, transform
from benchmark_core import Frame, TaskEvaluator, TaskSpec

from policy_cameras import NAMES, HEIGHT, WIDTH, WRIST_POSITION, WRIST_ROTATION, configure_cameras, check_attachment, OVERVIEW_PROFILES

CAMERA_VIEWS = {'overview': OVERVIEW_PROFILES['baseline']}


class BenchmarkEnv:
    def __init__(self, sim, scene, model, layout, overview_profile: str="baseline"):
        self.sim, self.scene, self.model, self.layout = sim, scene, model, layout
        self.overview_view=OVERVIEW_PROFILES[overview_profile]
        self.arms = [scene['left_arm'], scene['right_arm']]
        self.cube = scene['cube']
        self.ids = [layout.joint_index(self.arms[0], name) for name in JOINT_NAMES]
        if self.ids != [layout.joint_index(self.arms[1], name) for name in JOINT_NAMES]:
            raise RuntimeError('Arm joint ordering differs')
        self.body_ids = [{name:layout.body_index(arm, name) for name in LINK_NAMES} for arm in self.arms]
        self.dt = sim.get_physics_dt()
        if abs(self.dt-1/60) > 1e-9:
            raise RuntimeError('Benchmark physics timestep must be 1/60 second')
        root='/World/envs/env_0/'
        self.filters=[root+n for n in ('Cube','Table','BinFloor','BinLeftWall','BinRightWall','BinFrontWall','BinBackWall')]
        self.filters += [root+arm+'/'+body for arm in ('Left_Robot','Right_Robot') for body in LINK_NAMES]
        self.physics=tensors.create_simulation_view('torch');self.physics.set_subspace_roots('/')
        self.contacts=[self.physics.create_rigid_contact_view([root+arm+'/'+n for n in LINK_NAMES],
                       filter_patterns=[self.filters]*len(LINK_NAMES),max_contact_data_count=4096)
                       for arm in ('Left_Robot','Right_Robot')]
        self.cube_contact=self.physics.create_rigid_contact_view(root+'Cube',filter_patterns=self.filters[1:7])
        if self.cube_contact.sensor_count!=1 or self.cube_contact.filter_count!=6:
            raise RuntimeError('Cube environment contact prerequisites failed')
        if any(v.sensor_count!=7 or v.filter_count!=len(self.filters) for v in self.contacts):
            raise RuntimeError('Robot pairwise contact prerequisites failed')
        self.evaluator=TaskEvaluator(model.limits,model.adjacent,TaskSpec(physics_dt=self.dt))
        self.total_steps=0
        self.policy_origin_step=0
        self.timings={name:0. for name in ('actuation','physics','rendering','observations','evaluation','reset','camera')}
        self.frame=None
        self.matrices=[]
        self.targets=np.zeros((2,6))
        self.gravity=np.zeros((2,6))
        self.cameras={name:scene.sensors[name] for name in NAMES if name in scene.sensors}
        for name,camera in self.cameras.items():
            if name != 'overview':continue
            eye,target=self.overview_view
            camera.set_world_poses_from_view(torch.tensor([eye],device=sim.device),
                                            torch.tensor([target],device=sim.device))
        for profile,(eye,target) in OVERVIEW_PROFILES.items():
            if 'review_'+profile in scene.sensors:
                scene.sensors['review_'+profile].set_world_poses_from_view(
                    torch.tensor([eye],device=sim.device),torch.tensor([target],device=sim.device))
        if 'front' in scene.sensors:
            scene.sensors['front'].set_world_poses_from_view(
                torch.tensor([[0.,.72,.56]],device=sim.device),torch.tensor([[0.,-.14,.12]],device=sim.device))
        if self.cameras:
            # Warm RTX/Replicator buffers without advancing physical time.
            for _ in range(4):self.capture_images()

    def capture_images(self):
        """RGB at the current physics state, rendered before the next action.

        Sensor buffers are recomputed after rendering (update_period=0), even if simulation time
        is frozen for planning. No camera advances physics or reads labels.
        """
        if not self.cameras:raise RuntimeError('Launch with cameras enabled')
        started=time.perf_counter();self.sim.forward()
        # Pinned Isaac Sim does not propagate these child camera world transforms
        # from PhysX links reliably. Synchronize SENSOR-ONLY Fabric transforms from
        # the measured articulation, preserving authored local mounting offsets.
        for name,camera in self.cameras.items():
            if name=='overview':continue
            side=0 if name=='left_wrist' else 1
            state=self.arms[side].root_physx_view.get_link_transforms()[0,self.body_ids[side]['gripper']].cpu().numpy()
            parent=Rotation.from_quat(state[3:7]).as_matrix()
            pos=state[:3]+parent@WRIST_POSITION
            # Core XForm view uses OpenGL (-Z forward,+Y up), unlike ROS optics.
            rot=parent@WRIST_ROTATION@np.diag([1.,-1.,-1.])
            quat=Rotation.from_matrix(rot).as_quat()[[3,0,1,2]]
            camera._view.set_world_poses(torch.tensor(pos[None],device=self.sim.device,dtype=torch.float32),
                torch.tensor(quat[None],device=self.sim.device,dtype=torch.float32),usd=False)
        self.sim.render();images={}
        for name,camera in self.cameras.items():
            camera.update(0.,force_recompute=True)
            rgb=camera.data.output['rgb'][0,:,:,:3].cpu().numpy().copy()
            if rgb.shape!=(HEIGHT,WIDTH,3) or rgb.dtype!=np.uint8 or rgb.std()<2:
                raise RuntimeError(f'Invalid camera buffer: {name}')
            images[name]=rgb
        self.timings['camera']+=time.perf_counter()-started
        return images

    def capture_overviews(self):
        """Read all comparison tensors after the SAME render as policy cameras."""
        images={}
        for profile in OVERVIEW_PROFILES:
            camera=self.scene.sensors['review_'+profile]
            camera.update(0.,force_recompute=True)
            rgb=camera.data.output['rgb'][0,:,:,:3].cpu().numpy().copy()
            if rgb.shape!=(HEIGHT,WIDTH,3) or rgb.dtype!=np.uint8:
                raise RuntimeError('Invalid overview comparison image')
            images[profile]=rgb
        return images

    def camera_metadata(self):
        """Verify actual sensor poses against current rigid-link poses, not desired joints."""
        result={}
        for name,camera in self.cameras.items():
            pos=camera.data.pos_w[0].cpu().numpy()
            quat=camera.data.quat_w_ros[0].cpu().numpy()
            rotation=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            if name != 'overview' and self.frame is not None:
                side=0 if name=='left_wrist' else 1
                body=self.body_ids[side]['gripper']
                # render() refreshes PhysX articulation kinematics; read the live view,
                # not the cached pre-render ArticulationData body pose.
                state=self.arms[side].root_physx_view.get_link_transforms()[0,body].cpu().numpy()
                parent_rotation=Rotation.from_quat(state[3:7]).as_matrix()
                check_attachment(pos,rotation,state[:3],parent_rotation)
            elif name == 'overview' and not np.allclose(pos,self.overview_view[0],atol=1e-6,rtol=0):
                raise RuntimeError('Overview camera position mismatch')
            intrinsic=camera.data.intrinsic_matrices[0].cpu().numpy()
            if not np.isfinite(intrinsic).all() or min(intrinsic[0,0],intrinsic[1,1])<=0:
                raise RuntimeError('Invalid camera intrinsics')
            result[name]=dict(position_world_m=pos.tolist(),quaternion_world_wxyz=camera.data.quat_w_world[0].cpu().tolist(),
                quaternion_ros_wxyz=quat.tolist(),intrinsic=intrinsic.tolist(),resolution=[WIDTH,HEIGHT],fps=20,
                channels='RGB uint8',parent='world' if name=='overview' else name.split('_')[0]+'/gripper',
                calibration_status='initial simulation estimate, not hardware measured')
        if self.cameras and set(result)!=set(NAMES):raise RuntimeError('Missing policy cameras')
        return result

    @staticmethod
    def label(path):
        if '/Left_Robot/' in path:return 'left/'+path.rsplit('/',1)[-1]
        if '/Right_Robot/' in path:return 'right/'+path.rsplit('/',1)[-1]
        return {'Cube':'cube','Table':'table'}.get(path.rsplit('/',1)[-1], 'bin')

    def observe(self, privileged=False):
        """Proprioceptive policy track; explicit oracle track for expert/evaluator.

        RGB is explicitly opt-in via capture_images(). Object/contact/mesh data is
        privileged and must not silently enter a future vision policy's inputs.
        """
        if self.frame is None:raise RuntimeError('Reset before observation')
        if privileged:return self.frame
        return dict(joint_position_rad=self.frame.q.copy(),joint_velocity_rad_s=self.frame.qd.copy(),
                    previous_position_action_rad=self.frame.target.copy())

    def _read(self):
        q=[];qd=[];effort=[];tcp=[];desired=[];wrist=[]
        for i,arm in enumerate(self.arms):
            q.append(arm.data.joint_pos[0,self.ids].cpu().numpy().copy())
            qd.append(arm.data.joint_vel[0,self.ids].cpu().numpy().copy())
            effort.append(arm.data.applied_torque[0,self.ids].cpu().numpy().copy())
            idx=self.body_ids[i]['gripper']
            position=arm.data.body_pos_w[0,idx].cpu().numpy()
            quaternion=arm.data.body_quat_w[0,idx].cpu().numpy()
            tcp.append(transform(position,Rotation.from_quat(np.roll(quaternion,-1)).as_matrix())@self.model.tcp)
            base=self.model.base.copy();base[0,3]=(-.25,.25)[i]
            desired.append(self.model.tcp_pose(self.targets[i],base))
            wrist.append(arm.data.body_pos_w[0,self.body_ids[i]['wrist']].cpu().numpy().copy())
        self.matrices=[v.get_contact_force_matrix(self.dt).cpu().numpy() for v in self.contacts]
        if not all(np.isfinite(matrix).all() for matrix in self.matrices):
            raise RuntimeError('Nonfinite robot contact sensor matrix')
        pads=[];contacts=[]
        for view,matrix in zip(self.contacts,self.matrices):
            names={str(p).rsplit('/',1)[-1]:i for i,p in enumerate(view.sensor_paths)}
            pads.append([np.linalg.norm(matrix[names[n],0]) for n in ('gripper','jaw')])
            magnitudes=np.linalg.norm(matrix,axis=-1)
            for i,j in np.argwhere(magnitudes>=.05):
                contacts.append((self.label(str(view.sensor_paths[i])),self.label(self.filters[j]),float(magnitudes[i,j])))
        environment=np.linalg.norm(self.cube_contact.get_contact_force_matrix(self.dt)[0].cpu().numpy(),axis=-1)
        return Frame(self.total_steps*self.dt,self.cube.data.root_state_w[0].cpu().numpy().copy(),
                     np.array(q),np.array(qd),self.targets.copy(),np.array(effort),self.gravity.copy(),
                     np.array(tcp),np.array(desired),np.array(wrist),np.array(pads),environment,tuple(contacts))

    def advance(self, action, evaluate=True):
        """One physics step with absolute [left,right] six-joint position targets."""
        action=np.asarray(action,dtype=float)
        if action.shape!=(2,6) or not np.isfinite(action).all():
            raise ValueError('Position action must be finite shape (2,6), radians')
        if np.any(action<self.model.limits[:,0]) or np.any(action>self.model.limits[:,1]):
            raise ValueError('Position action outside authored joint limits')
        started=time.perf_counter();self.targets=action.copy()
        for i,arm in enumerate(self.arms):
            values=arm.data.default_joint_pos.clone()
            values[0,self.ids]=torch.tensor(action[i],device=arm.device,dtype=values.dtype)
            arm.set_joint_position_target(values)
            gravity=arm.root_physx_view.get_gravity_compensation_forces()
            if not torch.isfinite(gravity).all() or gravity.abs().max()>10:
                raise RuntimeError('Gravity compensation exceeds actuator effort limit')
            arm.set_joint_effort_target(gravity)
            self.gravity[i]=gravity[0,self.ids].cpu().numpy()
        self.timings['actuation']+=time.perf_counter()-started
        if not self.layout.app.is_running():raise RuntimeError('Simulator closed during benchmark')
        started=time.perf_counter()
        self.scene.write_data_to_sim();self.sim.step(render=False);self.scene.update(self.dt)
        self.timings['physics']+=time.perf_counter()-started;self.total_steps+=1
        if not self.layout.args.headless:
            started=time.perf_counter();self.sim.render()
            self.timings['rendering']+=time.perf_counter()-started
        started=time.perf_counter();self.frame=self._read()
        self.timings['observations']+=time.perf_counter()-started
        if evaluate:
            started=time.perf_counter();self.evaluator.update(self.frame)
            self.timings['evaluation']+=time.perf_counter()-started
            if self.evaluator.failure:raise RuntimeError('Independent evaluator: '+self.evaluator.failure)
        return self.frame

    def step(self, action):
        """20 Hz policy interface, holding its target for three physics ticks.

        Returns policy observation, terminal flag and independently measured
        metrics. Call reset after termination; no learned policy is implemented.
        """
        if self.evaluator.success or self.evaluator.failure:
            raise RuntimeError('Reset required after episode termination')
        for _ in range(self.evaluator.spec.policy_decimation):
            try:self.advance(action)
            except RuntimeError:
                if self.evaluator.failure is None:raise
                break
            if self.evaluator.success:break
        return self.observe(),bool(self.evaluator.success or self.evaluator.failure),self.evaluator.report()

    def reset(self, cube_xy=(-.34,-.04), cube_yaw_deg=0., seed=101):
        """The sole direct physical state-write entry point; verify after settling."""
        started=time.perf_counter()
        self.evaluator.reset_state()
        torch.manual_seed(seed);np.random.seed(seed)
        for arm in self.arms:
            arm.write_joint_state_to_sim(arm.data.default_joint_pos,torch.zeros_like(arm.data.default_joint_pos))
            arm.reset()
        state=self.cube.data.default_root_state.clone()
        state[0,:3]=torch.tensor([*cube_xy,.025],device=self.cube.device)
        quaternion=Rotation.from_euler('z',cube_yaw_deg,degrees=True).as_quat()
        state[0,3:7]=torch.tensor(np.roll(quaternion,1).copy(),device=self.cube.device,dtype=state.dtype)
        state[0,7:13]=0
        self.cube.write_root_state_to_sim(state);self.cube.reset()
        for _ in range(180):self.advance(np.zeros((2,6)),evaluate=False)
        self.evaluator.reset(self.frame,np.array(cube_xy),np.roll(quaternion,1))
        self.policy_origin_step=self.total_steps
        self.timings['reset']+=time.perf_counter()-started
        return self.observe()
