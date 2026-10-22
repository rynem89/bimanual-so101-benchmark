"""Conservative nominal-path checks for the fixed bimanual tabletop task.

No collision-free status is inferred from IK. Every joint segment is sampled
with a displacement bound; mesh envelopes reject self/opposite/bin collisions.
Cube tests use triangles because convexifying the entire gripper bridges its
empty grasp region. Runtime contacts remain an independent necessary check.
"""
from __future__ import annotations

import numpy as np
import hashlib
import inspect
from pathlib import Path
from so101_model import apply, box_mesh, intersect_hulls, mesh_box_overlap
from benchmark_core import JointPath

CACHE_STATE_ATOL=1e-5
CACHE_MOTION_RESERVE_M=.0001


def valid_cache(saved, key, **state):
    """Fail closed on stale, absent, malformed or materially changed context.

    Six 1e-5-radian joint differences at a 0.6 m radius plus rigid-transform
    entry differences move relevant geometry by <0.1 mm in this scene. Segment
    sampling explicitly reserves that distance; margins are not reduced.
    """
    if saved.get('fingerprint')!=key:return False
    for name,current in state.items():
        previous=np.asarray(saved.get(name,[]));current=np.asarray(current)
        if previous.shape!=current.shape or not np.isfinite(previous).all():return False
        if not np.allclose(previous,current,atol=CACHE_STATE_ATOL,rtol=0):return False
    return True


def fingerprint(model):
    """Invalidate generated plans after any asset/model/planner change."""
    h=hashlib.sha256()
    for path in (Path(model.asset),Path(__file__).with_name('so101_model.py'),Path(__file__),Path(__file__).with_name('benchmark_core.py')):
        h.update(path.read_bytes())
    h.update(Path(__file__).with_name('inspect_task_layout.py').read_bytes())
    h.update(model.tcp.tobytes());h.update(model.base.tobytes());h.update(model.limits.tobytes())
    # Hash resolved geometry, including meshes supplied by referenced USDs.
    for mesh in model.meshes:
        h.update(mesh.triangles.tobytes());h.update(mesh.planes.tobytes())
    for routine in (CollisionWorld,plan_pickup):
        h.update(inspect.getsource(routine).encode())
    return h.hexdigest()


class CollisionWorld:
    def __init__(self, model, cube=(-.34,-.04,.02), other_q=None, other_base=None, cube_rotation=None):
        self.model = model
        self.cube = np.array(cube)
        self.cube_rotation=np.eye(3) if cube_rotation is None else cube_rotation
        self.cube_vertices=box_mesh(np.zeros(3),np.full(3,.04),'cube').vertices @ self.cube_rotation.T + self.cube
        self.bin = [box_mesh(np.array(c), s, name) for c,s,name in (
            ((.34,-.04,.005),(.16,.16,.01),'bin_floor'),
            ((.265,-.04,.0475),(.01,.16,.075),'bin_left'),
            ((.415,-.04,.0475),(.01,.16,.075),'bin_right'),
            ((.34,-.115,.0475),(.14,.01,.075),'bin_front'),
            ((.34,.035,.0475),(.14,.01,.075),'bin_back'))]
        if other_base is None:
            other_base = model.base.copy()
            other_base[0,3] = .25 if model.base[0,3]<0 else -.25
        self.right = model.world_meshes(np.zeros(6) if other_q is None else other_q,other_base)
        self.right_bounds=np.array([(v.min(0),v.max(0)) for _,v,_ in self.right])
        self.bin_bounds=np.array([(m.vertices.min(0),m.vertices.max(0)) for m in self.bin])

    def check(self, q, margin=.002, cube_contact=False):
        model = self.model
        if np.any(q < model.limits[:,0]) or np.any(q > model.limits[:,1]):
            return ['joint_limit']
        meshes = model.world_meshes(q)
        faults = []
        bounds=np.array([(v.min(0),v.max(0)) for _,v,_ in meshes])
        def overlaps(other):
            # Same inclusive AABB rejection as the scalar checker, calculated
            # once per mesh instead of recomputing bounds for every pair.
            return (np.all(bounds[:,None,0]<=other[None,:,1]+margin,axis=-1)
                    & np.all(other[None,:,0]<=bounds[:,None,1]+margin,axis=-1))
        for i,(mesh, verts, pose) in enumerate(meshes):
            # Base/table contact is the documented fixed mounting interface.
            if mesh.link != 'base' and bounds[i,0,2] < margin:
                faults.append(f'table:{mesh.link}')
            if not (cube_contact and mesh.link in ('gripper','jaw')):
                if np.all(bounds[i,0] <= self.cube_vertices.max(0)+margin) and np.all(bounds[i,1] >= self.cube_vertices.min(0)-margin):
                    triangles = (apply(pose,mesh.triangles)-self.cube) @ self.cube_rotation
                    if mesh_box_overlap(triangles,np.zeros(3),np.full(3,.02+margin)):
                        faults.append(f'cube:{mesh.link}')
        for i,j in np.argwhere(overlaps(self.bin_bounds)):
            mesh,_,pose=meshes[i];obstacle=self.bin[j]
            if intersect_hulls(mesh,pose,obstacle,np.eye(4),margin):
                faults.append(f'{obstacle.link}:{mesh.link}')
        for i,j in np.argwhere(overlaps(self.right_bounds)):
            mesh,_,pose=meshes[i];right,_,rpose=self.right[j]
            if intersect_hulls(mesh,pose,right,rpose,margin):
                faults.append(f'opposite:{mesh.link}/{right.link}')
        for i,j in np.argwhere(np.triu(overlaps(bounds),1)):
            a,_,at=meshes[i];b,_,bt=meshes[j]
            if a.link == b.link or frozenset((a.link,b.link)) in model.adjacent:
                continue  # USD joints disable contact on these mechanical interfaces.
            if intersect_hulls(a,at,b,bt,margin):
                faults.append(f'self:{a.link}/{b.link}')
        return sorted(set(faults))

    def segment(self, start, end, margin=.002, cube_contact=False):
        # Radius 0.6 m bounds every downstream point of this <0.5m robot.
        # Maximum point movement from the nearest sample <= margin/2. This
        # covers nominal joint-linear motion, not tracking error or PhysX hulls.
        # Retain 0.1 mm for the bounded state difference allowed by plan caches.
        spacing=margin-CACHE_MOTION_RESERVE_M
        if spacing<=0:raise ValueError('Margin must exceed cache motion reserve')
        count = max(2, int(np.ceil(.6*np.abs(end-start).sum()/spacing))+1)
        for i,t in enumerate(np.linspace(0,1,count)):
            q = start+(end-start)*t
            faults = self.check(q,margin,cube_contact)
            if faults:
                return {'safe':False,'samples':count,'failed_sample':i,'q':q,'faults':faults}
        return {'safe':True,'samples':count,'cache_motion_reserve_m':CACHE_MOTION_RESERVE_M}

    def curve(self,points,speed=.12,margin=.001):
        path=JointPath(points,speed)
        if np.any(path.bounds[:,0]<self.model.limits[:,0]) or np.any(path.bounds[:,1]>self.model.limits[:,1]):
            return dict(safe=False,faults=['spline_joint_limit'])
        # Per-joint exact spline derivative extrema bound every point's travel
        # between nearest samples, including spline bulges away from the chords.
        count=max(2,int(np.ceil(.6*sum(path.peaks[0])/(margin-CACHE_MOTION_RESERVE_M)))+1)
        for i,s in enumerate(np.linspace(0,1,count)):
            faults=self.check(path.spline(s),margin)
            if faults:return dict(safe=False,samples=count,failed_sample=i,faults=faults)
        return dict(safe=True,samples=count,duration_s=path.duration,**path.limits())


def plan_pickup(model, cube=(-.34,-.04,.02), rotation=None):
    """Find reachable pregrasp/grasp with aligned cube faces and validate travel."""
    rotation=np.eye(3) if rotation is None else rotation
    world = CollisionWorld(model,cube,cube_rotation=rotation)
    home = np.zeros(6)
    opened = home.copy(); opened[-1] = np.deg2rad(45)
    pretarget = np.array(cube)+[0,0,.035]
    grasptarget = np.array(cube)+[0,0,.002]
    pres = model.ik(pretarget,rotation=rotation,seeds=12)
    grasps = model.ik(grasptarget,rotation=rotation,seeds=12)
    evidence = {'pre_candidates':pres,'grasp_candidates':grasps,'rejections':[]}
    for pre in pres:
        faults=world.check(pre['q'])
        if faults:
            evidence['rejections'].append({'phase':'pregrasp','faults':faults});continue
        for grasp in grasps:
            segments=[(home,opened),(opened,pre['q']),(pre['q'],grasp['q'])]
            results=[world.segment(a,b,margin=.001) for a,b in segments]
            if all(r['safe'] for r in results):
                return {'waypoints':[home,opened,pre['q'],grasp['q']], 'segment_checks':results,
                        'targets':[pretarget,grasptarget],'ik':evidence}
            evidence['rejections'].append(results)
    return {'waypoints':[], 'ik':evidence}


def plan_lift(model, grasp_q, held=False, cube=(-.34,-.04,.02), cube_rotation=None, cube_in_tcp=None, target=None):
    """Plan a 58 mm lift with achievable tilt and a predicted payload envelope.

    The payload transform is a planning assumption, never a simulator attachment.
    Execution must independently verify bilateral force and object following.
    """
    world=CollisionWorld(model,cube,cube_rotation=cube_rotation)
    closed=np.array(grasp_q).copy()
    if not held:closed[-1]=np.deg2rad(-5)
    closure=world.segment(np.array(grasp_q),closed,cube_contact=True)
    if not closure['safe']:
        return {'safe':False,'failure':'closure_collision','check':closure}
    candidates=[]
    target=np.asarray(cube)+[0,0,.06] if target is None else np.asarray(target)
    for heading in (-19.,-18.5,-19.5):
        h=np.deg2rad(heading);tilt=np.deg2rad(15)
        direction=[np.sin(h)*np.sin(tilt),np.cos(h)*np.sin(tilt),-np.cos(tilt)]
        candidates.extend(model.ik(target,approach=direction,opening_deg=np.rad2deg(closed[-1]),reference=closed,seeds=3))
    candidates.sort(key=lambda x:np.linalg.norm(x['q']-closed))
    rejections=[]
    payload=box_mesh(np.array([0,0,-.002]) if cube_in_tcp is None else np.zeros(3),np.full(3,.04),'payload')
    for candidate in candidates:
        if candidate['limit_margin_deg']<5:
            continue
        end=candidate['q']
        check=world.segment(closed,end,cube_contact=True)
        if not check['safe']:
            rejections.append(check);continue
        faults=[]
        # Same sample spacing as nominal robot path; payload is under 0.6 m
        # downstream of each joint. Extra 3 mm obstacle margin covers seating.
        for t in np.linspace(0,1,check['samples']):
            q=closed+t*(end-closed);pose=model.tcp_pose(q)
            if cube_in_tcp is not None:
                pose=pose@cube_in_tcp
            vertices=apply(pose,payload.vertices)
            if vertices[:,2].min()<-.0002:
                faults.append('predicted_payload_table');break
            for mesh,verts,mt in model.world_meshes(q)+world.right:
                if mesh.link in ('gripper','jaw') and mt[0,3]<0:
                    continue
                if np.all(vertices.min(0)<=verts.max(0)+.003) and np.all(verts.min(0)<=vertices.max(0)+.003):
                    if intersect_hulls(payload,pose,mesh,mt,.003):faults.append('payload_robot:'+mesh.link)
            for obstacle in world.bin:
                if np.all(vertices.min(0)<=obstacle.vertices.max(0)+.003) and np.all(obstacle.vertices.min(0)<=vertices.max(0)+.003):
                    if intersect_hulls(payload,pose,obstacle,np.eye(4),.003):faults.append('payload_'+obstacle.link)
            if faults:break
        if not faults:
            return {'safe':True,'closed_q':closed,'lift_q':end,'closure_check':closure,'lift_check':check,
                    'ik':candidate,'target_world_m':target,'rejections':rejections}
        rejections.append({'faults':faults})
    return {'safe':False,'failure':'no_lift_candidate','rejections':rejections}


def handover_candidates(model, target, seeds=12, alternate=False, cube_in_left_tcp=None):
    """Coupled 10-joint solve for opposing approaches on orthogonal cube faces.

    Feasibility study only: coincident nominal TCPs, not a transfer controller.
    Both hands' five-DOF limitations are solved together instead of prescribing
    two independently infeasible six-DOF poses.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    target=np.asarray(target)
    cube_local=np.eye(4) if cube_in_left_tcp is None else cube_in_left_tcp
    right_base=model.base.copy();right_base[0,3]=.25
    # Right approach reverses left approach; right closes on the other face pair.
    relative_rotation=np.array([[0,-1,0],[-1,0,0],[0,0,-1.]])
    if alternate:relative_rotation=np.array([[0,1,0],[1,0,0],[0,0,-1.]])
    lower=np.tile(model.limits[:5,0],2);upper=np.tile(model.limits[:5,1],2)
    initial=np.deg2rad([90,-30,55,-25,90,-90,-60,75,-15,0])
    if alternate:initial[4]=-np.pi/2
    rng=np.random.default_rng(101)
    starts=[initial]+[np.clip(initial+rng.normal(0,.45,10),lower+.01,upper-.01) for _ in range(seeds)]
    answers=[]
    def residual(q):
        left=model.tcp_pose(np.r_[q[:5],np.deg2rad(22)]) @ cube_local
        right=model.tcp_pose(np.r_[q[5:],np.deg2rad(45)],right_base)
        return np.r_[(left[:3,3]-target)/.1,(right[:3,3]-left[:3,3])/.1,
                     Rotation.from_matrix((left[:3,:3]@relative_rotation).T@right[:3,:3]).as_rotvec()]
    for start in starts:
        fit=least_squares(residual,start,bounds=(lower,upper),max_nfev=250,gtol=1e-9,ftol=1e-9,xtol=1e-9)
        if np.linalg.norm(residual(fit.x))>.005:continue
        ql=np.r_[fit.x[:5],np.deg2rad(22)];qr=np.r_[fit.x[5:],np.deg2rad(45)]
        left=model.tcp_pose(ql) @ cube_local;right=model.tcp_pose(qr,right_base)
        if -left[0,2]<.5 or right[0,2]<.5:continue
        if any(np.linalg.norm(fit.x-a['q10'])<.05 for a in answers):continue
        pair_faults=[]
        for a,av,at in model.world_meshes(ql):
            for b,bv,bt in model.world_meshes(qr,right_base):
                if np.all(av.min(0)<=bv.max(0)+.001) and np.all(bv.min(0)<=av.max(0)+.001):
                    if intersect_hulls(a,at,b,bt,.001):pair_faults.append(a.link+'/'+b.link)
        answers.append({'q10':fit.x,'left_q':ql,'right_q':qr,'target_m':target,
                        'left_position_error_m':float(np.linalg.norm(left[:3,3]-target)),
                        'tcp_separation_m':float(np.linalg.norm(right[:3,3]-left[:3,3])),
                        'relative_orientation_error_deg':float(np.rad2deg(np.linalg.norm(residual(fit.x)[6:]))),
                        'interarm_envelope_collisions':sorted(set(pair_faults))})
    return answers


def carried_segment(model, start, end, cube_in_tcp, other_q=None, other_base=None):
    """Check robot and measured rigid payload along a nominal joint segment.

    The payload is geometry for prediction only. No constraint/attachment is
    created in physics. Runtime must reject slip from this measured transform.
    """
    world=CollisionWorld(model,cube=(99,99,99),other_q=other_q,other_base=other_base)
    check=world.segment(start,end,margin=.002)
    if not check['safe']:return check
    payload=box_mesh(np.zeros(3),np.full(3,.04),'payload')
    for i,t in enumerate(np.linspace(0,1,check['samples'])):
        q=start+(end-start)*t
        pose=model.tcp_pose(q)@cube_in_tcp
        vertices=apply(pose,payload.vertices)
        if vertices[:,2].min()<.002:
            return {'safe':False,'failed_sample':i,'faults':['payload/table']}
        own=[m for m in model.world_meshes(q) if m[0].link not in ('gripper','jaw')]
        other=world.right+[(m,m.vertices,np.eye(4)) for m in world.bin]
        for mesh,verts,mt in own+other:
            if np.all(vertices.min(0)<=verts.max(0)+.003) and np.all(verts.min(0)<=vertices.max(0)+.003):
                if intersect_hulls(payload,pose,mesh,mt,.003):
                    return {'safe':False,'failed_sample':i,'faults':['payload/'+mesh.link]}
    return check


def transport_fingerprint(model):
    h=hashlib.sha256(fingerprint(model).encode())
    for routine in (handover_candidates,carried_segment,plan_transport):
        h.update(inspect.getsource(routine).encode())
    return h.hexdigest()


def plan_transport(model, start_q, cube_in_tcp):
    candidates=handover_candidates(model,[0,-.27,.18],seeds=12,alternate=True,cube_in_left_tcp=cube_in_tcp)
    candidates.sort(key=lambda a:np.linalg.norm(a['left_q']-start_q))
    report={'cube_in_left_tcp':cube_in_tcp,'start_q':start_q,'candidates':candidates,'selected':None,
            'fingerprint':transport_fingerprint(model)}
    right_base=model.base.copy();right_base[0,3]=.25
    attempts=0
    for candidate in candidates:
        if candidate['interarm_envelope_collisions']:continue
        right_q=np.array(candidate['right_q']);right_pose=model.tcp_pose(right_q,right_base)
        pres=model.ik(right_pose[:3,3]+right_pose[:3,2]*.04,rotation=right_pose[:3,:3],
                      reference=right_q,base=right_base,seeds=2)
        pres=[p for p in pres if p['limit_margin_deg']>5]
        candidate['receiver_pre_candidates']=pres
        if not pres:continue
        attempts+=1
        end=np.array(candidate['left_q']);end[-1]=start_q[-1]
        candidate['transport_check']=carried_segment(model,start_q,end,cube_in_tcp)
        if candidate['transport_check']['safe']:
            candidate['left_q']=end;report['selected']=candidate;break
        if attempts>=4:break
    return report


def plan_receiver(model, left_q, cube_pose, reference_q, continuous=False):
    """Plan the open right hand around the actual held cube and stationary left."""
    import copy
    right=copy.copy(model);right.base=model.base.copy();right.base[0,3]=.25
    swap=np.array([[0,1,0],[1,0,0],[0,0,-1.]])
    rotation=cube_pose[:3,:3]@swap
    goals=right.ik(cube_pose[:3,3],rotation=rotation,reference=reference_q,seeds=6)
    world=CollisionWorld(right,cube_pose[:3,3],other_q=left_q,other_base=model.base,cube_rotation=cube_pose[:3,:3])
    result={'left_q':left_q,'cube_pose':cube_pose,'selected':None,'rejections':[]}
    for goal in goals:
        faults=world.check(goal['q'],margin=.001)
        if faults:
            result['rejections'].append({'goal':faults});continue
        for distance in (.04,.02):
            pres=right.ik(cube_pose[:3,3]+rotation[:,2]*distance,rotation=rotation,reference=goal['q'],seeds=3)
            for pre in pres:
                if pre['limit_margin_deg']<3:continue
                opened=np.zeros(6);opened[-1]=np.pi/4
                closure=goal['q'].copy();closure[-1]=np.deg2rad(15)
                # Joint interpolation bows into the cube despite clear end
                # poses. Follow the approach axis with short Cartesian IK legs.
                approach=[pre['q']]
                for offset in np.linspace(distance,0,17)[1:]:
                    solutions=right.ik(cube_pose[:3,3]+rotation[:,2]*offset,rotation=rotation,
                                       reference=approach[-1],seeds=0)
                    if not solutions:break
                    approach.append(min(solutions,key=lambda s:np.linalg.norm(s['q']-approach[-1]))['q'])
                if len(approach)!=17:
                    result['rejections'].append({'cartesian_ik_failed':len(approach)});continue
                waypoints=[np.zeros(6),opened,*approach]
                pairs=list(zip(waypoints[:-1],waypoints[1:]))
                checks=[]
                for a,b in pairs:
                    check=world.segment(a,b,margin=.001)
                    checks.append(check)
                    if not check['safe']:break
                if len(checks)==len(pairs) and all(c['safe'] for c in checks):
                    curve_check=world.curve(approach) if continuous else None
                    if continuous and not curve_check['safe']:
                        result['rejections'].append({'curve':curve_check});continue
                    check=world.segment(goal['q'],closure,margin=.001,cube_contact=True)
                    checks.append(check)
                    if check['safe']:
                        result['selected']={'waypoints':waypoints,
                                            'checks':checks,'grasp_ik':goal,'pre_ik':pre,'curve_check':curve_check}
                        return result
                result['rejections'].append(checks)
    return result


def plan_placement(model, left_q, right_q, cube_in_right_tcp, continuous=False):
    """Withdraw the donor, then carry the physical payload over the bin.

    Initial donor separation starts at a former contact surface. Only that
    short departure permits nominal finger/cube overlap; runtime requires
    right-only support and rejects renewed donor force. All other obstacles
    and subsequent segments retain the regular collision checks.
    """
    import copy
    right=copy.copy(model);right.base=model.base.copy();right.base[0,3]=.25
    cube_pose=right.tcp_pose(right_q)@cube_in_right_tcp
    world=CollisionWorld(model,cube_pose[:3,3],other_q=right_q,other_base=right.base,
                         cube_rotation=cube_pose[:3,:3])
    home=np.zeros(6);home[-1]=np.pi/4
    result={'selected':None,'rejections':[],'left_start':left_q,'right_start':right_q,
            'cube_in_right_tcp':cube_in_right_tcp}
    departure=None
    for pan_clearance in (-2.,-4.,2.,4.):
        separated=left_q.copy();separated[0]+=np.deg2rad(pan_clearance)
        if world.check(separated,margin=.001):continue
        separation=world.segment(left_q,separated,margin=.001,cube_contact=True)
        if not separation['safe']:
            result['rejections'].append(separation);continue
        pose=model.tcp_pose(separated)
        distance=.04
        retreat=[left_q,separated];checks=[separation]
        for offset in np.linspace(0,distance,17)[1:]:
            solutions=model.ik(pose[:3,3]+pose[:3,2]*offset,rotation=pose[:3,:3],
                               reference=retreat[-1],seeds=0)
            if not solutions:break
            clear=solutions[0]['q']
            check=world.segment(retreat[-1],clear,margin=.001)
            checks.append(check)
            if not check['safe']:break
            retreat.append(clear)
        if len(retreat)!=18 or world.check(retreat[-1],margin=.001):
            result['rejections'].append({'departure':checks});continue
        curve_check=world.curve(retreat[1:],speed=.2) if continuous else None
        if continuous and not curve_check['safe']:
            result['rejections'].append({'departure_curve':curve_check});continue
        # Turning away first avoids the jaw sweeping back through the held
        # cube when elbow and shoulder are straightened simultaneously.
        turned=retreat[-1].copy();turned[0]=0.
        second=world.segment(retreat[-1],turned,margin=.001)
        if not second['safe']:result['rejections'].append(second);continue
        third=world.segment(turned,home,margin=.001)
        if third['safe']:
            departure={'waypoints':[*retreat,turned,home],'checks':[*checks,second,third],'curve_check':curve_check};break
        result['rejections'].append(third)
    if departure is None:return result
    for height in (.11,.105,.10):
        for heading in (19.,18.5,19.5):
            h=np.deg2rad(heading);tilt=np.deg2rad(15)
            direction=[np.sin(h)*np.sin(tilt),np.cos(h)*np.sin(tilt),-np.cos(tilt)]
            candidates=right.ik([.34,-.04,height],approach=direction,
                                opening_deg=np.rad2deg(right_q[-1]),reference=right_q,seeds=3)
            for candidate in candidates:
                if candidate['limit_margin_deg']<3:continue
                goal=candidate['q']
                pose=right.tcp_pose(goal)@cube_in_right_tcp
                # Cube centre remains well inside the 140 mm inner bin width.
                if np.linalg.norm(pose[:2,3]-[.34,-.04])>.015:continue
                opened=goal.copy();opened[-1]=np.pi/4
                right_world=CollisionWorld(right,cube=(99,99,99),other_q=home,other_base=model.base)
                opening=right_world.segment(goal,opened,margin=.001)
                if not opening['safe']:result['rejections'].append(opening);continue
                travel=carried_segment(right,right_q,goal,cube_in_right_tcp,other_q=home,other_base=model.base)
                if not travel['safe']:result['rejections'].append(travel);continue
                result['selected']={'left_departure':departure,'right_goal':goal,
                                    'right_open':opened,'carry_check':travel,'opening_check':opening,
                                    'predicted_release_cube_pose':pose,'ik':candidate}
                return result
    return result


def profile_cpu(output):
    """Reproducible geometry/IK/collision profile, without a Kit/GPU launch."""
    import cProfile
    import pstats
    import time
    from so101_model import SO101Model,calibrate,jsonable
    started=time.perf_counter();model=SO101Model();geometry_s=time.perf_counter()-started
    started=time.perf_counter();calibrate(model);calibration_s=time.perf_counter()-started
    profiler=cProfile.Profile();started=time.perf_counter();cpu=time.process_time()
    plan=profiler.runcall(plan_pickup,model)
    report={'geometry_preprocessing_s':geometry_s,'calibration_s':calibration_s,
            'cold_planning_wall_s':time.perf_counter()-started,'cold_planning_cpu_s':time.process_time()-cpu,
            'samples':sum(c['samples'] for c in plan['segment_checks']),
            'source_fingerprint':fingerprint(model),'functions':{}}
    for (file,line,name),(cc,nc,tt,ct,_) in pstats.Stats(profiler).stats.items():
        if name in ('fk','ik','intersect_hulls','_lp_intersection','linprog','world_meshes','check','segment'):
            report['functions'][Path(file).name+':'+name]={'calls':nc,'self_s':tt,'inclusive_s':ct}
    started=time.perf_counter();warm=plan_pickup(model)
    report['warm_geometry_revalidation_wall_s']=time.perf_counter()-started
    report['persisted_plan_cache_used']=False
    if not plan['waypoints'] or not warm['waypoints']:raise RuntimeError('Profile task was not feasible')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(__import__('json').dumps(jsonable(report),indent=2))
    print(__import__('json').dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Offline SO-101 planning CPU profile; use bundled USD paths from README.')
    parser.add_argument('--profile',action='store_true',required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/controller/profile_after.json'))
    profile_cpu(parser.parse_args().output)
