"""Policy-independent task evaluation, trajectory math and measured transfer analysis.

No Kit, USD or expert planner imports. Frames are world transforms, metres,
radians and seconds. Contact forces are measured magnitudes in newtons.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline, PPoly

CORNERS = np.array(list(product((-1., 1.), repeat=3))) * .02

EVALUATION_CASES = {
    'nominal': ((-.34,-.04),0.,101),
    'x_plus_1mm': ((-.339,-.04),0.,102),
    'x_minus_1mm': ((-.341,-.04),0.,103),
    'yaw_plus_1deg': ((-.34,-.04),1.,104),
    'yaw_minus_1deg': ((-.34,-.04),-1.,105),
}


@dataclass(frozen=True)
class TaskSpec:
    physics_dt: float = 1 / 60
    policy_decimation: int = 3  # 20 Hz policy, 60 Hz physics/low-level expert.
    timeout_s: float = 300.
    contact_N: float = .05
    hold_s: float = 3.
    speed_limit_rad_s: float = 10.
    effort_limit_Nm: float = 10.


@dataclass
class Frame:
    t_s: float
    cube: np.ndarray  # xyz, quaternion wxyz, linear velocity, angular velocity.
    q: np.ndarray  # [left/right, six actuators], radians.
    qd: np.ndarray
    target: np.ndarray
    effort: np.ndarray
    gravity: np.ndarray
    tcp: np.ndarray  # Actual measured body transform @ calibrated link offset.
    desired_tcp: np.ndarray
    wrist: np.ndarray  # Actual wrist xyz for each arm.
    pads: np.ndarray  # [left/right, fixed/moving finger], N.
    environment: np.ndarray  # Cube versus table, floor, four walls, N.
    contacts: tuple  # (source "left/link", target "cube"/"right/link"/..., N).

    def cube_pose(self):
        pose = np.eye(4)
        pose[:3, 3] = self.cube[:3]
        pose[:3, :3] = Rotation.from_quat(np.roll(self.cube[3:7], -1)).as_matrix()
        return pose

    def cube_vertices(self):
        pose = self.cube_pose()
        return CORNERS @ pose[:3, :3].T + pose[:3, 3]


class TaskEvaluator:
    """Success is derived only from physics observations, never expert phases.

    The evaluator accepts any policy using the same actuator interface. Lift,
    right-only support, settled containment and arm retreat need continuous
    evidence. Fatal failures are sticky until an independently verified reset.
    """
    def __init__(self, limits, adjacent, spec=TaskSpec()):
        self.limits = np.asarray(limits)
        self.adjacent = set(adjacent)
        self.spec = spec
        self.reset_state()

    def reset_state(self):
        self.flags = dict(pickup=False, lift=False, handover=False, placement=False)
        self.dwell = dict(lift=0., handover=0., placement=0.)
        self.failure = None
        self.elapsed_s = 0.
        self.max_speed = 0.
        self.max_effort = 0.
        self.max_tracking_mm = np.zeros(2)
        self.max_tracking_deg = np.zeros(2)
        self.max_acceleration = 0.
        self.max_jerk = 0.
        self.previous_qd = self.previous_acceleration = None
        self.slip_reference = {}
        self.max_slip_mm = np.zeros(2)
        self.max_slip_deg = np.zeros(2)
        self.collision_violations = 0
        self.reset_metrics = {}

    def reset(self, frame, expected_xy, expected_quaternion=(1.,0,0,0)):
        self.reset_state()
        position_error=np.linalg.norm(frame.cube[:3]-np.r_[expected_xy,.02])
        angle=Rotation.from_matrix(Rotation.from_quat(np.roll(expected_quaternion,-1)).as_matrix().T
                    @frame.cube_pose()[:3,:3]).magnitude()
        self.reset_metrics=dict(cube_m=frame.cube[:3].tolist(),cube_quaternion_wxyz=frame.cube[3:7].tolist(),
                               position_error_mm=float(position_error*1000),orientation_error_deg=float(np.rad2deg(angle)))
        if (position_error > .0001
                or np.max(np.abs(frame.q[:, :5])) > np.deg2rad(1)
                or np.linalg.norm(frame.cube[7:10]) > .025
                or np.linalg.norm(frame.cube[10:13]) > .1
                or angle>np.deg2rad(.1)
                or not .3 < frame.environment[0] < .7):
            raise RuntimeError('Independent reset pose, velocity or weight prerequisite failed')
        self.initial_cube = frame.cube[:3].copy()

    @property
    def success(self):
        return self.failure is None and all(self.flags.values())

    def _fail(self, reason):
        if self.failure is None:
            self.failure = reason

    def update(self, frame):
        if self.failure is not None:
            return
        dt = self.spec.physics_dt
        self.elapsed_s += dt
        arrays = (frame.cube, frame.q, frame.qd, frame.target, frame.effort, frame.gravity,
                  frame.tcp, frame.desired_tcp, frame.wrist, frame.pads, frame.environment)
        if (not all(np.isfinite(a).all() for a in arrays)
                or not all(np.isfinite(force) for _,_,force in frame.contacts)):
            self._fail('nonfinite_physics_observation'); return
        if abs(np.linalg.norm(frame.cube[3:7])-1)>1e-3:
            self._fail('invalid_cube_quaternion');return
        self.max_speed = max(self.max_speed, float(np.abs(frame.qd).max()))
        self.max_effort = max(self.max_effort, float(np.abs(frame.effort).max()))
        if self.max_speed > self.spec.speed_limit_rad_s+.05 or self.max_effort > self.spec.effort_limit_Nm+.05:
            self._fail('actuator_limit')
        if np.any(frame.q < self.limits[:, 0] - .001) or np.any(frame.q > self.limits[:, 1] + .001):
            self._fail('joint_position_limit')
        if self.previous_qd is not None:
            acceleration = (frame.qd - self.previous_qd) / dt
            self.max_acceleration = max(self.max_acceleration, float(np.abs(acceleration).max()))
            if self.previous_acceleration is not None:
                self.max_jerk = max(self.max_jerk, float(np.abs((acceleration-self.previous_acceleration)/dt).max()))
            self.previous_acceleration = acceleration
        self.previous_qd = frame.qd.copy()
        self.max_tracking_mm = np.maximum(self.max_tracking_mm,
            np.linalg.norm(frame.tcp[:, :3, 3]-frame.desired_tcp[:, :3, 3], axis=1)*1000)
        self.max_tracking_deg = np.maximum(self.max_tracking_deg,
            np.rad2deg(np.abs(frame.q[:, :5]-frame.target[:, :5]).max(axis=1)))

        # Allowed contacts belong to the task, not the expert's phase labels.
        for source, target, force in frame.contacts:
            if force < self.spec.contact_N or source == target:
                continue
            side, body = source.split('/')
            allowed = target == 'table' and body == 'base'
            if '/' in target:
                other_side, other_body = target.split('/')
                allowed |= side == other_side and frozenset((body, other_body)) in self.adjacent
            if target == 'cube' and body in ('gripper', 'jaw'):
                allowed |= (side == 'left' and not self.flags['handover']) or (side == 'right' and self.flags['lift'])
            if not allowed:
                self.collision_violations += 1
                self._fail(f'forbidden_contact:{source}:{target}')
        vertices = frame.cube_vertices()
        inside_xy = bool(np.all(vertices[:, :2].min(0) >= [.27, -.11])
                         and np.all(vertices[:, :2].max(0) <= [.41, .03]))
        # Physical release is loss of both hands' contact, not an arbitrary jaw
        # angle. The cube can fall clear before a 45-degree opening is reached.
        released_at_bin = inside_xy and np.max(frame.pads) < self.spec.contact_N
        if self.flags['lift'] and np.max(frame.environment) >= self.spec.contact_N and not released_at_bin:
            self._fail('carried_object_environment_contact')
        if frame.cube[2] < -.02:
            self._fail('object_fell_off_table')

        left_support = min(frame.pads[0]) >= self.spec.contact_N
        right_support = min(frame.pads[1]) >= self.spec.contact_N
        lifted = frame.cube[2] >= .06 and left_support
        self.dwell['lift'] = self.dwell['lift'] + dt if lifted else 0.
        if self.dwell['lift'] + 1e-9 >= self.spec.hold_s:
            self.flags['pickup'] = self.flags['lift'] = True
        received = self.flags['lift'] and frame.cube[2] >= .15 and right_support and max(frame.pads[0]) < .05
        self.dwell['handover'] = self.dwell['handover'] + dt if received else 0.
        if self.dwell['handover'] + 1e-9 >= self.spec.hold_s:
            self.flags['handover'] = True
        contained = inside_xy and vertices[:, 2].min() >= .009 and vertices[:, 2].max() <= .085
        resting = (contained and .3 < frame.environment[1] < .7
                   and np.linalg.norm(frame.cube[7:10]) < .01 and np.linalg.norm(frame.cube[10:13]) < .1
                   and np.max(frame.pads) < .05 and np.max(np.abs(frame.q[:, :5])) < np.deg2rad(3))
        self.dwell['placement'] = self.dwell['placement'] + dt if resting and self.flags['handover'] else 0.
        if self.dwell['placement'] + 1e-9 >= self.spec.hold_s:
            self.flags['placement'] = True
        for side, enabled in enumerate((self.flags['lift'], self.flags['handover'])):
            # Carry stability excludes intentional dual-hand load transfer.
            # Transfer settlement is measured separately, without phase-based
            # permission to override physical success criteria.
            sole_support = min(frame.pads[side]) >= .05 and min(frame.pads[1-side]) < .05
            if enabled and sole_support and frame.q[side, -1] < np.deg2rad(35):
                local = np.linalg.inv(frame.tcp[side]) @ frame.cube_pose()
                reference = self.slip_reference.setdefault(side, local)
                self.max_slip_mm[side] = max(self.max_slip_mm[side], np.linalg.norm(local[:3, 3]-reference[:3, 3])*1000)
                self.max_slip_deg[side] = max(self.max_slip_deg[side], np.rad2deg(Rotation.from_matrix(reference[:3, :3].T@local[:3, :3]).magnitude()))
                if self.max_slip_mm[side] > 5 or self.max_slip_deg[side] > 10:
                    self._fail('stable_grasp_slip')
        if self.elapsed_s >= self.spec.timeout_s and not self.success:
            self._fail('timeout')

    def report(self):
        return dict(success=self.success, stages=dict(self.flags), failure=self.failure,
                    reset=dict(self.reset_metrics),
                    elapsed_s=self.elapsed_s, collision_violations=self.collision_violations,
                    max_joint_speed_rad_s=self.max_speed, max_effort_Nm=self.max_effort,
                    max_tcp_tracking_mm=self.max_tracking_mm.tolist(), max_joint_tracking_deg=self.max_tracking_deg.tolist(),
                    max_acceleration_rad_s2=self.max_acceleration, max_jerk_rad_s3=self.max_jerk,
                    acceleration_jerk_limits='Not authored in this USD/actuator configuration; measured, not certified',
                    max_stable_grasp_slip_mm=self.max_slip_mm.tolist(), max_stable_grasp_rotation_deg=self.max_slip_deg.tolist())


def quintic(u):
    return u**3 * (10 - 15*u + 6*u*u)


def policy_boundary(physics_step: int, episode_start_step: int, decimation: int = 3) -> bool:
    """20 Hz phase restarts after every reset, independent of global tick count."""
    if physics_step<episode_start_step or decimation<1:raise ValueError('Invalid policy clock')
    return (physics_step-episode_start_step)%decimation==0


def trajectory_limits(delta, duration):
    """Exact peak nominal derivatives for a rest-to-rest quintic segment."""
    delta = np.abs(delta)
    return dict(velocity_rad_s=1.875*delta/duration,
                acceleration_rad_s2=(10/np.sqrt(3))*delta/duration**2,
                jerk_rad_s3=60*delta/duration**3)


def receiver_final_contact(frame,phase,force):
    """Only light finger contact inside the aligned final acquisition region.

    Transit/ordinary approach never gain a contact exemption. Independent task
    success, force preload, slip and right-only support criteria are unchanged.
    """
    if phase!='right_final_insert' or not 0<=force<=.6:return False
    if frame.cube[2]<.15 or np.min(frame.pads[0])<.6 or frame.q[1,5]<np.deg2rad(40):return False
    pose=frame.cube_pose();tcp=frame.tcp[1]
    expected=pose[:3,:3]@np.array([[0,1,0],[1,0,0],[0,0,-1.]])
    aligned=Rotation.from_matrix(expected.T@tcp[:3,:3]).magnitude()<=np.deg2rad(2)
    return bool(aligned and np.linalg.norm(tcp[:3,3]-pose[:3,3])<=.003)


class JointPath:
    """C2 joint spline with rest-to-rest quintic time scaling, radians/seconds.

    Unlike stitching minimum-duration moves, internal Cartesian waypoints have
    continuous nonzero velocity. Geometry must be checked on this curve itself;
    validation of the original polyline is insufficient.
    """
    def __init__(self, waypoints, speed=.12):
        points=np.asarray(waypoints,dtype=float)
        if points.ndim!=2 or points.shape[1]!=6 or len(points)<2 or not np.isfinite(points).all():
            raise ValueError('Expected finite (N,6) joint waypoints')
        distances=np.abs(np.diff(points,axis=0)).sum(axis=1)
        points=points[np.r_[True,distances>1e-10]]
        if len(points)<2:raise ValueError('Path must move')
        knots=np.r_[0.,np.cumsum(np.abs(np.diff(points,axis=0)).sum(axis=1))]
        knots/=knots[-1]
        self.spline=CubicSpline(knots,points,bc_type='clamped')
        self.peaks=[];bounds=[]
        for order in range(4):
            low=[];high=[]
            for joint in range(6):
                polynomial=PPoly(self.spline.c[:,:,joint],knots).derivative(order)
                roots=polynomial.derivative().roots(extrapolate=False) if order<3 else []
                values=polynomial(np.r_[knots,np.asarray(roots)[np.isfinite(roots)]])
                low.append(values.min());high.append(values.max())
            if order==0:bounds=np.array([low,high]).T
            else:self.peaks.append(np.maximum(np.abs(low),np.abs(high)))
        self.bounds=bounds
        self.duration=max(1.,1.875*max(self.peaks[0])/speed)

    def position(self,u):
        return self.spline(quintic(u))

    def limits(self):
        d1,d2,d3=self.peaks;t=self.duration
        v=1.875/t;a=10/np.sqrt(3)/t**2;j=60/t**3
        return dict(velocity_rad_s=d1*v,acceleration_rad_s2=d2*v*v+d1*a,
                    jerk_rad_s3=d3*v**3+3*d2*v*a+d1*j)


class CoordinatedJointPath:
    """Checked donor clearance knots with uninterrupted receiver quintic travel.

    Donor legs retain their existing joint-linear geometry. Their rest-to-rest
    durations share one overall clock with receiver travel, with no intermediate
    settling pause. Optional receiver lead allows the payload to clear first.
    """
    def __init__(self, left_waypoints, right_start, right_goal, speed=.35, lead=0.):
        self.left=np.asarray(left_waypoints,dtype=float)
        self.right=np.asarray([right_start,right_goal],dtype=float)
        if (self.left.ndim!=2 or self.left.shape[1]!=6 or len(self.left)<2 or self.right.shape!=(2,6)
                or not np.isfinite(self.left).all() or not np.isfinite(self.right).all()
                or not 0<=lead<1 or speed<=0):raise ValueError('Valid paired path, speed and lead required')
        self.delta=np.diff(self.left,axis=0);self.right_delta=self.right[1]-self.right[0]
        durations=np.maximum(1.,1.875*np.abs(self.delta).max(axis=1)/speed)
        self.duration=max(float(durations.sum())/(1-lead),1.875*float(np.abs(self.right_delta).max())/speed,1.)
        self.knots=lead+(1-lead)*np.r_[0.,np.cumsum(durations)/durations.sum()]
        self.bounds=np.array([np.stack((self.left.min(0),self.left.max(0)),axis=1),
                              np.stack((self.right.min(0),self.right.max(0)),axis=1)])

    def position(self,u):
        u=float(np.clip(u,0,1))
        if u<=self.knots[0]:left=self.left[0].copy()
        else:
            index=min(len(self.delta)-1,int(np.searchsorted(self.knots,u,side='right')-1))
            local=(u-self.knots[index])/(self.knots[index+1]-self.knots[index])
            left=self.left[index]+self.delta[index]*quintic(local)
        return np.array([left,self.right[0]+self.right_delta*quintic(u)])

    def limits(self,side):
        if side=='right':return trajectory_limits(self.right_delta,self.duration)
        values=[trajectory_limits(delta,self.duration*(b-a)) for delta,a,b in zip(self.delta,self.knots[:-1],self.knots[1:])]
        return {key:np.max([row[key] for row in values],axis=0) for key in values[0]}

    def validation_grid(self,radii,spacing):
        if spacing<=0:raise ValueError('Positive geometry spacing required')
        def peak(a,b):
            x=.5 if a<=.5<=b else (a if a>.5 else b)
            return 30*x*x*(1-x)*(1-x)
        breaks=np.unique(np.r_[0.,self.knots]);grid=[0.]
        for start,end in zip(breaks[:-1],breaks[1:]):
            pieces=np.linspace(start,end,9)
            index=min(len(self.delta)-1,int(np.searchsorted(self.knots,(start+end)/2,side='right')-1))
            for a,b in zip(pieces[:-1],pieces[1:]):
                bound=float(radii[1]@np.abs(self.right_delta))*peak(a,b)
                if index>=0:
                    low,high=self.knots[index:index+2]
                    bound+=float(radii[0]@np.abs(self.delta[index]))*peak((a-low)/(high-low),(b-low)/(high-low))/(high-low)
                count=max(1,int(np.ceil(bound*(b-a)/spacing)))
                grid.extend(np.linspace(a,b,count+1)[1:])
        return np.asarray(grid)


def validation_grid(paths, radii, spacing):
    """Split at spline knots and bound relative geometry travel per interval.

    Exact derivative extrema give a conservative sample count in each polynomial
    interval. A brief high-slope section no longer forces the same dense sampling
    over the entire trajectory. All paths share the same monotone progress.
    """
    if spacing<=0 or len(paths)!=len(radii):raise ValueError('Positive spacing and matching paths/radii required')
    knots=np.unique(np.concatenate([path.spline.x for path in paths]))
    derivatives=[]
    for path in paths:
        group=[]
        for joint in range(6):
            poly=PPoly(path.spline.c[:,:,joint],path.spline.x).derivative()
            roots=poly.derivative().roots(extrapolate=False)
            group.append((poly,roots[np.isfinite(roots)]))
        derivatives.append(group)
    grid=[knots[0]]
    for low,high in zip(knots[:-1],knots[1:]):
        bound=0.
        for group,lengths in zip(derivatives,radii):
            for (poly,roots),radius in zip(group,lengths):
                points=np.r_[low,high,roots[(roots>low)&(roots<high)]]
                bound+=radius*np.abs(poly(points)).max()
        count=max(1,int(np.ceil(bound*(high-low)/spacing)))
        grid.extend(np.linspace(low,high,count+1)[1:])
    return np.asarray(grid)


def transfer_metrics(rows):
    if not rows:
        return {}
    release = [r for r in rows if r['phase'] in ('left_release','right_only_hold') or r['phase'].startswith('donor_normal_clearance_')]
    if not release:
        return {}
    before = [r for r in rows if r['phase']=='right_close']
    if before:release=[before[-1],*release]
    if len(release)<2:return dict(samples=len(release),valid=False,reason='insufficient_transfer_samples')
    t = np.array([r['t_s'] for r in release])
    cube = np.array([r['cube_m'] for r in release])
    tcp = np.array([r['right_tcp_m'] for r in release])
    relative = np.array([r['cube_in_right_tcp_m'] for r in release])
    velocity = np.array([r['cube_vertical_velocity_m_s'] for r in release])
    reference = tcp[-1, 2]
    stable_pose = np.abs(tcp[:, 2]-reference) < .0005
    # First suffix staying within 0.5 mm and with low cube vertical speed.
    stable = stable_pose & (np.abs(velocity) < .01)
    suffix = np.logical_and.accumulate(stable[::-1])[::-1]
    indices = np.flatnonzero(suffix)
    pose_velocity=np.gradient(cube[:,2],t)
    pose_suffix=np.logical_and.accumulate((stable_pose & (np.abs(pose_velocity)<.01))[::-1])[::-1]
    pose_indices=np.flatnonzero(pose_suffix)
    return dict(samples=len(release), dt_s=float(np.median(np.diff(t))),
                trace_max_gap_s=float(np.diff(t).max()),
                trace_uniform_sampling=bool(np.allclose(np.diff(t),np.median(np.diff(t)),atol=1e-7,rtol=0)),
                cube_drop_from_release_start_mm=float((cube[0, 2]-cube[:, 2].min())*1000),
                right_tcp_drop_from_release_start_mm=float((tcp[0, 2]-tcp[:, 2].min())*1000),
                right_wrist_drop_mm=float((release[0]['right_wrist_m'][2]-min(r['right_wrist_m'][2] for r in release))*1000),
                cube_relative_translation_change_mm=float(np.linalg.norm(relative-relative[0], axis=1).max()*1000),
                cube_relative_rotation_change_deg=float(np.rad2deg(Rotation.from_matrix(
                    np.array(release[0]['cube_in_right_tcp_rotation']).T@np.array([r['cube_in_right_tcp_rotation'] for r in release])).magnitude()).max()),
                minimum_cube_height_m=float(cube[:, 2].min()),
                maximum_downward_cube_speed_m_s=float(max(0., -velocity.min())),
                maximum_downward_pose_speed_m_s=float(max(0.,-pose_velocity.min())),
                settle_s=None if not len(indices) else float(t[indices[0]]-t[0]),
                pose_settle_s=None if not len(pose_indices) else float(t[pose_indices[0]]-t[0]),
                final_reported_vertical_velocity_m_s=float(velocity[-1]),
                final_pose_vertical_velocity_m_s=float(pose_velocity[-1]),
                max_right_joint_tracking_deg=max(r['right_joint_tracking_deg'] for r in release))
