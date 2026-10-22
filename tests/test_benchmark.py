"""Policy-independent evidence, failure gates and nominal trajectory regression."""
from pathlib import Path
import ast
import copy
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_core import Frame,TaskEvaluator,TaskSpec,JointPath,policy_boundary,quintic,trajectory_limits,transfer_metrics


def frame(position=(-.34,-.04,.02)):
    cube=np.r_[position,[1.,0,0,0],np.zeros(6)]
    q=np.zeros((2,6));q[:,-1]=np.pi/4
    tcp=np.repeat(np.eye(4)[None],2,axis=0)
    return Frame(0.,cube,q,np.zeros_like(q),q.copy(),np.zeros_like(q),np.zeros_like(q),
                 tcp,tcp.copy(),np.zeros((2,3)),np.zeros((2,2)),np.array([.4905,0,0,0,0,0]),())


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.e=TaskEvaluator(np.tile([-np.pi,np.pi],(6,1)),[])
        self.e.reset(frame(),np.array([-.34,-.04]))

    def hold(self,f,count=180):
        for _ in range(count):self.e.update(f)

    def acquire_and_transfer(self):
        f=frame((-.34,-.04,.073));f.environment[:]=0;f.pads[0]=2;f.q[0,-1]=.38
        self.hold(f);self.assertTrue(self.e.flags['lift'])
        f=frame((0,-.27,.17));f.environment[:]=0;f.pads[1]=2;f.q[1,-1]=.38
        self.hold(f);self.assertTrue(self.e.flags['handover'])

    def test_complete_task_requires_all_observed_stages(self):
        f=frame((.34,-.04,.03));f.environment[:]=[0,.4905,0,0,0,0]
        self.hold(f);self.assertFalse(self.e.success)
        self.acquire_and_transfer();self.hold(f,179);self.assertFalse(self.e.success)
        self.hold(f,1);self.assertTrue(self.e.success)
        self.e.reset(frame(),np.array([-.34,-.04]));self.assertFalse(any(self.e.flags.values()))

    def test_floating_fast_or_held_cube_cannot_pass_placement(self):
        for fault in ('floating','fast','held','wall'):
            with self.subTest(fault=fault):
                self.setUp();self.acquire_and_transfer()
                f=frame((.34,-.04,.03));f.environment[:]=[0,.4905,0,0,0,0]
                if fault=='floating':f.environment[1]=0
                if fault=='fast':f.cube[7]=.02
                if fault=='held':f.pads[1]=1;f.q[1,-1]=.38
                if fault=='wall':f.cube[0]=.405
                self.hold(f);self.assertFalse(self.e.success)

    def test_physically_released_cube_may_settle_before_jaw_fully_opens(self):
        self.acquire_and_transfer()
        f=frame((.34,-.04,.03));f.q[1,-1]=np.deg2rad(28)
        f.environment[:]=[0,.4905,0,0,0,0]
        self.hold(f);self.assertTrue(self.e.success)

    def test_failed_grasp_and_transfer_do_not_gain_success(self):
        f=frame((-.34,-.04,.08));f.environment[:]=0
        self.hold(f);self.assertFalse(self.e.flags['lift'])
        f.pads[0]=2;f.q[0,-1]=.38;self.hold(f)
        f.pads[:]=0;f.cube[2]=.10;self.hold(f)
        self.assertFalse(self.e.flags['handover']);self.assertFalse(self.e.success)

    def test_sole_holder_slip_is_measured_without_expert_phase_labels(self):
        f=frame((-.34,-.04,.073));f.environment[:]=0;f.pads[0]=2;f.q[0,-1]=.38
        self.hold(f);f.cube[0]+=.006;self.e.update(f)
        self.assertEqual(self.e.failure,'stable_grasp_slip')

    def test_contact_limits_and_nonfinite_are_sticky_failures(self):
        for fault in ('contact','effort','nan','target','contact_nan','quaternion'):
            self.setUp();f=frame()
            if fault=='contact':f.contacts=(('left/gripper','table',.2),)
            if fault=='effort':f.effort[1,2]=11
            if fault=='nan':f.cube[0]=np.nan
            if fault=='target':f.target[0,0]=np.nan
            if fault=='contact_nan':f.contacts=(('left/gripper','table',np.nan),)
            if fault=='quaternion':f.cube[3:7]=0
            self.e.update(f);self.assertIsNotNone(self.e.failure)
            self.e.update(frame());self.assertIsNotNone(self.e.failure)

    def test_invalid_reset_fails_prerequisite(self):
        f=frame();f.environment[:]=0
        with self.assertRaisesRegex(RuntimeError,'reset'):
            self.e.reset(f,np.array([-.34,-.04]))
        # The reset gate must distinguish the actual 1 mm / 1 degree cases.
        with self.assertRaisesRegex(RuntimeError,'reset'):
            self.e.reset(frame((-.339,-.04,.02)),np.array([-.34,-.04]))
        from scipy.spatial.transform import Rotation
        f=frame();f.cube[3:7]=np.roll(Rotation.from_euler('z',1,degrees=True).as_quat(),1)
        with self.assertRaisesRegex(RuntimeError,'reset'):
            self.e.reset(f,np.array([-.34,-.04]))

    def test_timeout_and_reset_yaw_are_independent_of_controller(self):
        self.e=TaskEvaluator(np.tile([-np.pi,np.pi],(6,1)),[],TaskSpec(timeout_s=.05))
        self.e.reset(frame(),np.array([-.34,-.04]));self.hold(frame(),3)
        self.assertEqual(self.e.failure,'timeout')
        with self.assertRaisesRegex(RuntimeError,'reset'):
            self.e.reset(frame(),np.array([-.34,-.04]),[0,0,0,1])


class TrajectoryTests(unittest.TestCase):
    def test_transfer_metrics_include_departure_and_expose_trace_gaps(self):
        rows=[]
        for i,height in enumerate((.18,.179,.172,.175,.175)):
            rows.append(dict(t_s=i/60,phase='donor_normal_clearance_0' if i==2 else 'left_release',
                cube_m=[0,0,height],right_tcp_m=[0,0,height],right_wrist_m=[0,0,height],
                cube_in_right_tcp_m=[0,0,0],cube_in_right_tcp_rotation=np.eye(3).tolist(),
                cube_vertical_velocity_m_s=0.,right_joint_tracking_deg=0.))
        measured=transfer_metrics(rows)
        self.assertAlmostEqual(measured['cube_drop_from_release_start_mm'],8)
        self.assertEqual(measured['samples'],5);self.assertTrue(measured['trace_uniform_sampling'])
        incomplete=transfer_metrics(rows[:2]+rows[3:])
        self.assertFalse(incomplete['trace_uniform_sampling'])
        self.assertAlmostEqual(incomplete['trace_max_gap_s'],2/60)

    def test_policy_clock_restarts_at_arbitrary_reset_tick(self):
        for origin in (180,181,182,9041):
            self.assertEqual([step for step in range(origin,origin+9) if policy_boundary(step,origin)],
                             [origin,origin+3,origin+6])
        with self.assertRaises(ValueError):policy_boundary(100,101)

    def test_joint_spline_is_c2_and_its_analytic_bounds_cover_execution(self):
        points=np.array([[0]*6,[.05,.1,0,0,0,0],[.12,.08,0,0,0,0],[.2,.14,0,0,0,0]])
        path=JointPath(points,.12)
        u=np.linspace(0,1,20001);s=quintic(u);t=path.duration
        sd=(30*u*u-60*u**3+30*u**4)/t
        sdd=(60*u-180*u*u+120*u**3)/t**2
        velocity=path.spline(s,1)*sd[:,None]
        acceleration=path.spline(s,2)*sd[:,None]**2+path.spline(s,1)*sdd[:,None]
        self.assertLessEqual(np.abs(velocity).max(),.12+1e-10)
        self.assertTrue(np.all(np.abs(acceleration).max(0)<=path.limits()['acceleration_rad_s2']+1e-10))
        for knot in path.spline.x[1:-1]:
            for derivative in (0,1,2):
                np.testing.assert_allclose(path.spline(knot-1e-9,derivative),path.spline(knot+1e-9,derivative),atol=1e-6)
        np.testing.assert_allclose(velocity[[0,-1]],0,atol=1e-12)
        np.testing.assert_allclose(acceleration[[0,-1]],0,atol=1e-12)

    def test_quintic_derivatives_and_speed_bound(self):
        delta=np.array([.1,-1.5,.3]);duration=1.875*np.abs(delta).max()/.35
        peaks=trajectory_limits(delta,duration)
        u=np.linspace(0,1,10001)
        velocity=(30*u*u-60*u**3+30*u**4)
        acceleration=60*u-180*u*u+120*u**3
        jerk=60-360*u+360*u*u
        np.testing.assert_allclose(peaks['velocity_rad_s'],velocity.max()*np.abs(delta)/duration)
        np.testing.assert_allclose(peaks['acceleration_rad_s2'],np.abs(acceleration).max()*np.abs(delta)/duration**2,rtol=1e-7)
        np.testing.assert_allclose(peaks['jerk_rad_s3'],np.abs(jerk).max()*np.abs(delta)/duration**3)
        self.assertLessEqual(peaks['velocity_rad_s'].max(),.35+1e-12)
        self.assertEqual((quintic(0),quintic(1)),(0,1))

    def test_state_writes_exist_only_in_common_reset(self):
        root=Path(__file__).resolve().parents[1]/'scripts'
        writes=[]
        for name in ('benchmark_env.py','run_manipulation.py'):
            tree=ast.parse((root/name).read_text())
            for method in ast.walk(tree):
                if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    for call in ast.walk(method):
                        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr in ('write_root_state_to_sim','write_joint_state_to_sim'):
                            writes.append((name,method.name));self.assertEqual(method.name,'reset')
        self.assertTrue(writes)


if __name__=='__main__':unittest.main()
