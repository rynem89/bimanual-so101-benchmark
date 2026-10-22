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

CAMERA_VIEWS = {
    'overview': ((0., -.12, .92), (0., -.12, .04)),
    'front': ((0., .72, .56), (0., -.14, .12)),
}


def configure_cameras(scene_cfg):
    """Opt-in Isaac Lab 2.3 RGB sensors; never installed into oracle state."""
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg
    for name in CAMERA_VIEWS:
        setattr(scene_cfg, name, CameraCfg(
            prim_path='{ENV_REGEX_NS}/'+name, update_period=0.,
            height=224, width=224, data_types=['rgb'],update_latest_camera_pose=True,
            spawn=PinholeCameraCfg(focal_length=18., horizontal_aperture=20.955,
                                  clipping_range=(.01, 10.))))


class BenchmarkEnv:
    def __init__(self, sim, scene, model, layout):
        self.sim, self.scene, self.model, self.layout = sim, scene, model, layout
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
        self.timings={name:0. for name in ('actuation','physics','rendering','observations','evaluation','reset','camera')}
        self.frame=None
        self.matrices=[]
        self.targets=np.zeros((2,6))
        self.gravity=np.zeros((2,6))
        self.cameras={name:scene.sensors[name] for name in CAMERA_VIEWS if name in scene.sensors}
        for name,camera in self.cameras.items():
            eye,target=CAMERA_VIEWS[name]
            camera.set_world_poses_from_view(torch.tensor([eye],device=sim.device),
                                            torch.tensor([target],device=sim.device))
        if self.cameras:
            # Warm RTX/Replicator buffers without advancing physical time.
            for _ in range(4):self.capture_images()

    def capture_images(self):
        """RGB at the current physics state, rendered before the next action.

        Sensor buffers are recomputed after rendering (update_period=0), even if simulation time
        is frozen for planning. No camera advances physics or reads labels.
        """
        if not self.cameras:raise RuntimeError('Launch with cameras enabled')
        started=time.perf_counter();self.sim.render();images={}
        for name,camera in self.cameras.items():
            camera.update(0.,force_recompute=True)
            rgb=camera.data.output['rgb'][0,:,:,:3].cpu().numpy().copy()
            if rgb.shape!=(224,224,3) or rgb.dtype!=np.uint8 or rgb.std()<2:
                raise RuntimeError(f'Invalid camera buffer: {name}')
            images[name]=rgb
        self.timings['camera']+=time.perf_counter()-started
        return images

    def camera_metadata(self):
        for name,camera in self.cameras.items():
            if not torch.isclose(camera.data.quat_w_world[0].norm(),torch.tensor(1.,device=self.sim.device),atol=1e-5):
                raise RuntimeError('Uninitialized camera pose buffer: '+name)
            if not np.allclose(camera.data.pos_w[0].cpu().numpy(),CAMERA_VIEWS[name][0],atol=1e-6,rtol=0):
                raise RuntimeError('Camera eye differs from configuration: '+name)
        return {name:dict(eye_m=CAMERA_VIEWS[name][0],target_m=CAMERA_VIEWS[name][1],
                         intrinsic=camera.data.intrinsic_matrices[0].cpu().numpy().tolist(),
                         position_world_m=camera.data.pos_w[0].cpu().numpy().tolist(),
                         quaternion_world_wxyz=camera.data.quat_w_world[0].cpu().numpy().tolist(),
                         quaternion_ros_wxyz=camera.data.quat_w_ros[0].cpu().numpy().tolist(),
                         resolution=[224,224],fps=20,channels='RGB uint8',
                         focal_length_mm=18.,horizontal_aperture_mm=20.955)
                for name,camera in self.cameras.items()}

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
        self.timings['reset']+=time.perf_counter()-started
        return self.observe()
