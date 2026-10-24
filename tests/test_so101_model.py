"""Offline geometry/FK/IK checks; use the bundled USD paths from README."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from so101_model import SO101Model,calibrate,box_mesh,mesh_box_overlap,triangle_box_overlap,intersect_hulls,transform,_lp_intersection


class GeometryTests(unittest.TestCase):
    def test_transport_skips_candidate_without_complete_receiver_path(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import so101_planning as planner
        start=np.zeros(6);start[-1]=.36;other=np.zeros(6)
        model=SimpleNamespace(base=np.eye(4),tcp_pose=lambda q,base=None:np.eye(4),
                              ik=lambda *args,**kw:[dict(q=np.zeros(6),limit_margin_deg=10.)])
        def candidates():
            return [dict(left_q=np.r_[x,np.zeros(5)],right_q=np.zeros(6),interarm_envelope_collisions=[])
                    for x in (.1,.2)]
        for answers,selected_x in (([None,dict(checked=True)],.2),([None,None],None)):
            with self.subTest(selected_x=selected_x), \
                 patch.object(planner,'handover_candidates',side_effect=lambda *a,**kw:candidates()), \
                 patch.object(planner,'transport_fingerprint',return_value='test'), \
                 patch.object(planner,'CollisionWorld') as world, \
                 patch.object(planner,'carried_segment',return_value=dict(safe=True)), \
                 patch.object(planner,'plan_receiver',side_effect=[dict(selected=a) for a in answers]) as receiver:
                world.return_value.check.return_value=[]
                result=planner.plan_transport(model,start,np.eye(4),other_q=other)
                self.assertEqual(receiver.call_count,2)
                if selected_x is None:self.assertIsNone(result['selected'])
                else:self.assertEqual(result['selected']['left_q'][0],selected_x)
                for call in receiver.call_args_list:
                    self.assertEqual(call.args[1][-1],start[-1])
                    np.testing.assert_array_equal(call.kwargs['start_q'],other)
                    self.assertTrue(call.kwargs['continuous'])

    def test_buffered_receiver_recovers_two_measured_oct24_failures(self):
        import copy
        from so101_planning import plan_receiver,CollisionWorld
        from benchmark_core import JointPath
        model=SO101Model();calibrate(model)
        right=copy.copy(model);right.base=model.base.copy();right.base[0,3]=.25
        # Measured receiver-plan inputs from 20261024T222926314730Z_seed42,
        # attempts 1 (early fixed-pad contact) and 5 (no receiver path).
        cases=[
            dict(left=[1.553349495,-.402709693,1.124639034,-.833011627,.169103354,.358153284],
                 start=[-1.602539778,-1.553123951,1.140147090,.567225397,-1.464999557,.785397530],
                 cube=None,
                 reference=[-1.602204175,-1.048153821,.992210695,.256754485,-1.719716614,.785398163],
                 local=[[.995343537,.020571450,-.094170369,-.001158011],[-.020449147,.999788332,.002263657,.000418664],[.094197002,-.000327413,.995553523,-.004662229],[0,0,0,1]],distance=.04),
            dict(left=[1.504367113,-.455395877,1.110870719,-.765070558,-1.266305447,.362165809],
                 start=[-1.602539778,-1.553129673,1.140153766,.567220688,-1.464993238,.785397470],
                 cube=[[-.054381607,-.125629256,-.990585650,-.000193799],[-.963472731,.267130558,.019014794,-.271409303],[.262226883,.955436317,-.135567351,.175796613],[0,0,0,1]],
                 reference=[-1.557758422,-1.097260239,1.180507927,.067259929,-.269982790,.785398163],distance=.035)]
        for case in cases:
            with self.subTest(distance=case['distance']):
                left=np.array(case['left']);start=np.array(case['start'])
                cube=model.tcp_pose(left)@case['local'] if case['cube'] is None else np.array(case['cube'])
                result=plan_receiver(model,left,cube,np.array(case['reference']),continuous=True,start_q=start)
                selected=result['selected'];self.assertIsNotNone(selected)
                self.assertEqual(selected['pregrasp_distance_m'],case['distance'])
                self.assertEqual(selected['fixed_pad_offset_m'],-.004)
                self.assertGreaterEqual(selected['pre_ik']['limit_margin_deg'],3.)
                self.assertTrue(all(c['safe'] for c in selected['checks']))
                self.assertTrue(selected['curve_check']['safe']);self.assertTrue(selected['final_curve_check']['safe'])
                points=np.array(selected['waypoints']);entry=selected['approach_start_index'];cut=selected['final_start_index']
                world=CollisionWorld(right,cube[:3,3],other_q=left,other_base=model.base,
                                     cube_rotation=cube[:3,:3],cube_clearance=.005)
                if case['distance']==.04:
                    contact_q=np.array([-1.601570606,-1.401727557,1.042313457,.563628733,-1.721386552,.785395920])
                    unbuffered=CollisionWorld(right,cube[:3,3],other_q=left,other_base=model.base,
                                              cube_rotation=cube[:3,:3])
                    self.assertEqual(unbuffered.check(contact_q,.001),[])
                    self.assertIn('cube:gripper',world.check(contact_q,.001))
                # Independently reconstruct the exact two curves the runtime
                # executes, including the final shared boundary waypoint.
                prefix=JointPath(points[entry:cut+1],.12);final=JointPath(points[cut:],.06)
                for u in np.linspace(0,1,31):
                    self.assertEqual(world.check(prefix.position(u),.001),[])
                self.assertLess(np.linalg.norm(right.tcp_pose(final.position(1))[:3,3]-cube[:3,3]),.001)
                for path in (prefix,final):
                    self.assertTrue(np.all(path.bounds[:,0]>=model.limits[:,0]))
                    self.assertTrue(np.all(path.bounds[:,1]<=model.limits[:,1]))
                if case['distance']==.035:
                    self.assertTrue(selected['transit_routed'])
                    rejected=[a for a in result['rejections'] if a['stage']=='pregrasp_limit' and a['lateral_m']==0]
                    self.assertAlmostEqual(rejected[0]['limit_margin_deg'],2.817,places=2)

    def test_lift_fallback_recovers_measured_empty_ik_cases_without_relaxing_limits(self):
        from so101_planning import plan_lift,lift_ik_groups
        model=SO101Model();calibrate(model)
        right=np.array([-1.60254,-1.55315,1.14018,.56720,-1.46501,.7853975])
        cases=[
            ([-.222691774,.543522835,-.312873572,1.338461876,-1.757272959,.380862623],
             [[.999179549,.012803829,-.038422526,-.000914779],[-.012685612,.999914026,.003318977,.000190187],[.038461718,-.002828840,.999256071,-.003229837],[0,0,0,1]],[-.310904,-.037141]),
            ([-.249083161,.359479457,-.023321932,1.235253096,-1.866891146,.382848710],
             [[.999228772,.014387754,-.036535643,-.000913145],[-.014367539,.999896449,.000815808,.000300903],[.036543598,-.000290252,.999332018,-.003108753],[0,0,0,1]],[-.312205,-.059197])]
        for q,local,xy in cases:
            q=np.array(q);local=np.array(local);pose=model.tcp_pose(q)@local;target=[*xy,.08]
            self.assertEqual(next(lift_ik_groups(model,q,target))[0],[])
            result=plan_lift(model,q,held=True,cube=pose[:3,3],cube_rotation=pose[:3,:3],cube_in_tcp=local,target=target,other_q=right)
            self.assertTrue(result['safe']);self.assertTrue(result['lift_check']['safe'])
            self.assertGreaterEqual(result['ik']['limit_margin_deg'],5.)
            self.assertTrue(any(row['reason']=='no_ik_candidate' for row in result['rejections'] if 'reason' in row))

    def test_handover_selection_avoids_saved_receiver_transit_trap(self):
        from so101_planning import plan_transport,plan_receiver
        model=SO101Model();calibrate(model)
        start=np.array([-.370307028,.060861755,.017680859,1.244230986,1.569031715,.362931818])
        right=np.array([-1.602540016,-1.553139210,1.140164971,.567211092,-1.465009570,.785397410])
        local=np.array([[.995602645,.024063786,-.090533459,-.000968606],[-.024394777,.999699148,-.002551080,.000659035],[.090444834,.004748405,.995890147,-.004473088],[0,0,0,1]])
        reference=np.array([1.539077686,-.306951289,1.096672929,-.943922883,-.105789612,-1.602541659,-1.078564655,1.081044490,.151756878,-1.465011305])
        transport=plan_transport(model,start,local,other_q=right,handover_reference=reference)
        self.assertIsNotNone(transport['selected'])
        selected=transport['selected'];self.assertTrue(selected['buffered_receiver_pregrasp'])
        self.assertIsNotNone(selected['predicted_receiver_plan']['selected'])
        cube=model.tcp_pose(selected['left_q'])@local
        receiver=plan_receiver(model,selected['left_q'],cube,selected['right_q'],continuous=True,start_q=right)
        self.assertIsNotNone(receiver['selected'])
        self.assertTrue(receiver['selected']['final_curve_check']['safe'])

    def test_off_axis_receiver_uses_buffered_alignment_before_final_insert(self):
        from so101_planning import plan_receiver
        model=SO101Model();calibrate(model)
        left=np.array([1.549856544,-.547241449,1.079022527,-.527689934,.076118104,.359776855])
        start=np.array([-1.602539897,-1.553147674,1.140174747,.567204297,-1.465002060,.785397232])
        cube=np.array([[-.087950596,-.027132604,-.995755248,-.000503337],[.052213064,.998129241,-.031809037,-.269935280],[.994755492,-.054789056,-.086369385,.177996054],[0,0,0,1]])
        reference=np.array([-1.602484849,-1.084257287,1.156616841,.014121214,-1.625797378,.785398163])
        result=plan_receiver(model,left,cube,reference,continuous=True,start_q=start)
        self.assertIsNotNone(result['selected']);selected=result['selected']
        self.assertFalse(selected.get('direct_insertion',False))
        self.assertTrue(selected['curve_check']['safe']);self.assertTrue(selected['final_curve_check']['safe'])

    def test_authored_motion_bound_covers_meshes_and_payload(self):
        from so101_model import apply
        model=SO101Model();calibrate(model)
        payload=apply(model.tcp@transform((.03,-.02,-.03)),box_mesh(np.zeros(3),(.04,.04,.04),'payload').vertices)
        radii=model.motion_radii(payload)
        rng=np.random.default_rng(24601)
        for _ in range(12):
            origin=rng.uniform(model.limits[:,0],model.limits[:,1])
            for joint in range(7):
                delta=rng.uniform(-.2,.2,6)
                if joint<6:delta[np.arange(6)!=joint]=0.
                goal=np.clip(origin+delta,model.limits[:,0],model.limits[:,1])
                bound=float(radii@np.abs(goal-origin))
                before=model.world_meshes(origin);after=model.world_meshes(goal)
                displacement=max(np.linalg.norm(b[1]-a[1],axis=1).max() for a,b in zip(before,after))
                p0=apply(model.fk(origin)['gripper'],payload)
                p1=apply(model.fk(goal)['gripper'],payload)
                displacement=max(displacement,np.linalg.norm(p1-p0,axis=1).max())
                self.assertLessEqual(displacement,bound+1e-10)

    def test_bounded_rrt_checks_edges_and_reports_no_path(self):
        from types import SimpleNamespace
        from so101_planning import CollisionWorld
        class AnalyticWorld(CollisionWorld):
            def __init__(self,blocked=False):
                self.model=SimpleNamespace(limits=np.array([[0,1],[-1,1],[-.1,.1],[-.1,.1],[-.1,.1],[0,1]]))
                self.wall=np.array([[.4,.6],[-1,1] if blocked else [-.15,.15]])
            def check(self,q,margin=.001):
                return ['wall'] if np.all(q[:2]>=self.wall[:,0]) and np.all(q[:2]<=self.wall[:,1]) else []
            def segment(self,start,end,margin=.001):
                # Exact line/rectangle slab intersection, independent of the
                # RRT and production mesh sampling. Endpoints alone are unsafe.
                low,high=0.,1.
                for axis in range(2):
                    delta=end[axis]-start[axis]
                    if abs(delta)<1e-12:
                        if not self.wall[axis,0]<=start[axis]<=self.wall[axis,1]:return dict(safe=True,samples=2)
                    else:
                        a,b=sorted((self.wall[axis]-start[axis])/delta);low=max(low,a);high=min(high,b)
                return dict(safe=bool(low>high),samples=2)
        start=np.zeros(6);goal=start.copy();goal[0]=1
        world=AnalyticWorld();self.assertFalse(world.segment(start,goal)['safe'])
        proof=world.joint_transit(start,goal,max_iterations=1000,max_seconds=2)
        self.assertTrue(proof['safe']);path=proof['waypoints']
        np.testing.assert_array_equal(path[0],start);np.testing.assert_array_equal(path[-1],goal)
        self.assertGreater(len(path),2)
        for a,b in zip(path[:-1],path[1:]):self.assertTrue(world.segment(a,b)['safe'])
        self.assertTrue(all(np.all(q>=world.model.limits[:,0]) and np.all(q<=world.model.limits[:,1]) for q in path))
        self.assertFalse(AnalyticWorld(True).joint_transit(start,goal,max_iterations=50,max_seconds=1)['safe'])

    def test_ambiguous_lp_requires_definitive_retry(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import so101_model
        original=so101_model.linprog
        unknown=SimpleNamespace(status=4,message='indeterminate numerical status')
        box=box_mesh(np.zeros(3),np.ones(3),'test')
        def presolve_unknown(*args,**kwargs):
            return unknown if kwargs['method']=='highs' else original(*args,**kwargs)
        with patch.object(so101_model,'linprog',side_effect=presolve_unknown):
            self.assertFalse(_lp_intersection(box,np.eye(4),box,transform([2,0,0],np.eye(3)),.001))
            self.assertTrue(intersect_hulls(box,np.eye(4),box,np.eye(4)))
        with patch.object(so101_model,'linprog',return_value=unknown):
            with self.assertRaisesRegex(RuntimeError,'Collision LP unresolved'):
                intersect_hulls(box,np.eye(4),box,np.eye(4))

    def test_convex_envelopes_with_rotated_box(self):
        from scipy.spatial.transform import Rotation
        box=box_mesh(np.zeros(3),np.ones(3),'test')
        rotation=Rotation.from_euler('z',45,degrees=True).as_matrix()
        # Analytic X support: 0.5 + sqrt(0.5) = 1.207106... metres.
        self.assertTrue(intersect_hulls(box,np.eye(4),box,transform([1.20,0,0],rotation)))
        self.assertFalse(intersect_hulls(box,np.eye(4),box,transform([1.22,0,0],rotation)))
        self.assertTrue(intersect_hulls(box,np.eye(4),box,np.eye(4)))

    def test_box_containment_and_separation(self):
        mesh=box_mesh(np.zeros(3),np.ones(3),'test')
        self.assertFalse(triangle_box_overlap(mesh.triangles,np.zeros(3),np.ones(3)*.1))
        self.assertTrue(mesh_box_overlap(mesh.triangles,np.zeros(3),np.ones(3)*.1))
        self.assertFalse(mesh_box_overlap(mesh.triangles,np.array([1.,0,0]),np.ones(3)*.1))
        self.assertTrue(mesh_box_overlap(mesh.triangles,np.array([.55,0,0]),np.ones(3)*.1))

    def test_prefilter_matches_lp_random_rotations_and_inflated_acute_corners(self):
        from scipy.spatial import ConvexHull
        from scipy.spatial.transform import Rotation
        from so101_model import Mesh
        points=np.array([[0,0,0],[1,0,0],[.999,.002,0],[.999,.001,.02]])
        hull=ConvexHull(points)
        acute=Mesh('acute','acute',points[hull.simplices],points,hull.equations)
        box=box_mesh(np.zeros(3),np.full(3,.02),'box')
        rng=np.random.default_rng(101)
        # Acute half-space intersections expand further than margin/2 along
        # some directions; a simple original-vertex radius test is unsound.
        for _ in range(60):
            pose=transform(rng.uniform([-.05,-.03,-.03],[1.05,.03,.05]),Rotation.random(random_state=rng).as_matrix())
            self.assertEqual(intersect_hulls(acute,np.eye(4),box,pose,.002),_lp_intersection(acute,np.eye(4),box,pose,.002))
        unit=box_mesh(np.zeros(3),np.ones(3),'unit')
        for gap in (-1e-5,0.,1e-5):
            pose=transform([1.002+gap,0,0])
            self.assertEqual(intersect_hulls(unit,np.eye(4),unit,pose,.002),_lp_intersection(unit,np.eye(4),unit,pose,.002))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=SO101Model();cls.calibration=calibrate(cls.model)

    def test_fk_agrees_with_independent_usd_rest_transforms(self):
        from pxr import Usd,UsdGeom
        s=Usd.Stage.Open(self.model.asset);root=s.GetDefaultPrim();cache=UsdGeom.XformCache()
        root_inverse=cache.GetLocalToWorldTransform(root).GetInverse()
        for name,pose in self.model.fk(np.zeros(6),np.eye(4)).items():
            usd=np.array(cache.GetLocalToWorldTransform(s.GetPrimAtPath(f'{root.GetPath()}/{name}'))*root_inverse).T
            np.testing.assert_allclose(pose,usd,atol=2e-5,rtol=0,err_msg=name)

    def test_fk_ik_roundtrips_and_limits(self):
        for q5 in ((-.3,.6,-.4,1.3,-1.9),(.2,-.2,.3,.4,.6),(-.4,.3,-.2,.7,-.5)):
            q=np.r_[q5,np.pi/4];pose=self.model.tcp_pose(q)
            solutions=self.model.ik(pose[:3,3],rotation=pose[:3,:3],seeds=3)
            self.assertTrue(solutions)
            for s in solutions:
                self.assertLess(s['position_error_m'],.001)
                self.assertTrue(np.all(s['q']>=self.model.limits[:,0]))
                self.assertTrue(np.all(s['q']<=self.model.limits[:,1]))

    def test_unreachable_target_is_rejected(self):
        self.assertEqual(self.model.ik([2,2,2],seeds=2),[])

    def test_donor_withdrawal_recovers_recorded_axial_ik_failure(self):
        from so101_planning import plan_placement
        # Measured right-only hold from the fixed -1 degree yaw regression.
        # Its 40 mm axial donor retreat exhausts IK; all later geometry still
        # needs checking, including the carried cube and bin release opening.
        left=np.array([1.5710519552,.1522724628,1.0316096544,-1.5694576502,.1755351722,.7853982449])
        right=np.array([-1.5963497162,-.8121492863,.5984927416,.7274165750,-1.7238614559,.3762028813])
        local=np.array([[.0085353368,.9952957157,.0965069234,-.0012996518],
                        [.9999025491,-.0095611827,.0101723185,.0009427735],
                        [.0110471854,.0964106946,-.9952803312,-.0017849291],[0,0,0,1]])
        # Simulator joint observations are float32, unlike JSON-loaded probes.
        result=plan_placement(self.model,left.astype(np.float32),right.astype(np.float32),local,continuous=True)
        self.assertIsNotNone(result['selected'])
        selected=result['selected'];departure=selected['left_departure']
        # A jointly departing receiver can now clear the formerly blocked
        # donor sweep directly; retain the global fallback if that is unsafe.
        self.assertIn(departure['kind'],('coordinated_direct','joint_transit'))
        if departure['kind']=='joint_transit':self.assertTrue(departure['transit']['safe'])
        self.assertTrue(all(check['safe'] for check in departure['checks']))
        self.assertTrue(all(check['safe'] for check in selected['coordinated_return']['checks']))
        np.testing.assert_allclose(departure['waypoints'][-1],np.r_[np.zeros(5),np.pi/4])
        self.assertTrue(selected['carry_check']['safe'])
        self.assertTrue(selected['opening_check']['safe'])

    def test_pickup_continuation_retains_collision_rejection(self):
        from scipy.spatial.transform import Rotation
        from so101_planning import plan_pickup
        # First preselected wider-distribution draw: zero/random initialization
        # formerly returned no IK. Continuation converges, but the near-limit
        # pregrasp still intersects the cube and must NEVER be executed.
        plan=plan_pickup(self.model,[-.3481787,-.0335215,.02],
                         Rotation.from_euler('z',.3261,degrees=True).as_matrix())
        self.assertTrue(plan['ik']['pre_candidates'])
        self.assertTrue(plan['ik']['grasp_candidates'])
        self.assertEqual(plan['waypoints'],[])
        self.assertIn('cube:gripper',plan['ik']['rejections'][0]['faults'])

    def test_cube_fits_open_but_intersects_closed_authored_mesh(self):
        rows=self.calibration['surface_checks']
        self.assertTrue(rows[0]['surface_intersections_with_42mm_box'])
        self.assertEqual(next(r for r in rows if r['angle_deg']==45)['surface_intersections_with_42mm_box'],[])

    def test_complete_approach_matches_original_lp_at_every_query(self):
        from unittest.mock import patch
        import so101_planning as planning
        count=0
        def compare(*args,**kwargs):
            nonlocal count
            actual=intersect_hulls(*args,**kwargs)
            self.assertEqual(actual,_lp_intersection(*args,**kwargs))
            count+=1
            return actual
        with patch.object(planning,'intersect_hulls',side_effect=compare):
            plan=planning.plan_pickup(self.model)
        self.assertTrue(plan['waypoints'])
        samples=sum(check['samples'] for check in plan['segment_checks'])
        # Sampling density is now derived from authored joint radii instead of
        # a uniform 0.6 m bound. Its geometric coverage is tested independently.
        self.assertGreater(samples,2*len(plan['segment_checks']))
        self.assertGreater(count,0)
        print(f'Collision equivalence: {count} LP queries, {samples} path samples',flush=True)

    def test_cache_invalidates_geometry_calibration_settings_and_changed_state(self):
        from unittest.mock import patch
        from so101_planning import fingerprint,valid_cache,LOADED_SOURCES
        key=fingerprint(self.model);q=np.zeros(6)
        saved=dict(fingerprint=key,q=q.tolist())
        self.assertTrue(valid_cache(saved,key,q=q+1e-6))
        self.assertFalse(valid_cache(saved,key,q=q+.006))
        self.assertFalse(valid_cache(saved,key,q=np.zeros(5)))
        self.assertFalse(valid_cache(saved,key+'changed-margin',q=q))
        self.model.tcp[0,3]+=.001
        try:self.assertNotEqual(fingerprint(self.model),key)
        finally:self.model.tcp[0,3]-=.001
        mesh=self.model.meshes[0];mesh.triangles[0,0,0]+=.001
        try:self.assertNotEqual(fingerprint(self.model),key)
        finally:mesh.triangles[0,0,0]-=.001
        with patch.dict(LOADED_SOURCES,{'benchmark_core.py':b'older loaded algorithm'}):
            with self.assertRaisesRegex(RuntimeError,'source changed'):fingerprint(self.model)


if __name__=='__main__':unittest.main()
