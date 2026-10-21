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


def fingerprint(model):
    """Invalidate generated plans after any asset/model/planner change."""
    h=hashlib.sha256()
    for path in (Path(model.asset),Path(__file__).with_name('so101_model.py')):
        h.update(path.read_bytes())
    for routine in (CollisionWorld,plan_pickup):
        h.update(inspect.getsource(routine).encode())
    return h.hexdigest()


class CollisionWorld:
    def __init__(self, model, cube=(-.34,-.04,.02)):
        self.model = model
        self.cube = np.array(cube)
        self.bin = [box_mesh(np.array(c), s, name) for c,s,name in (
            ((.34,-.04,.005),(.16,.16,.01),'bin_floor'),
            ((.265,-.04,.0475),(.01,.16,.075),'bin_left'),
            ((.415,-.04,.0475),(.01,.16,.075),'bin_right'),
            ((.34,-.115,.0475),(.14,.01,.075),'bin_front'),
            ((.34,.035,.0475),(.14,.01,.075),'bin_back'))]
        right_base = model.base.copy()
        right_base[0,3] = .25
        self.right = model.world_meshes(np.zeros(6),right_base)

    def check(self, q, margin=.002, cube_contact=False):
        model = self.model
        if np.any(q < model.limits[:,0]) or np.any(q > model.limits[:,1]):
            return ['joint_limit']
        meshes = model.world_meshes(q)
        faults = []
        def overlap(a, b):
            return np.all(a.min(0) <= b.max(0)+margin) and np.all(b.min(0) <= a.max(0)+margin)
        for mesh, verts, pose in meshes:
            # Base/table contact is the documented fixed mounting interface.
            if mesh.link != 'base' and verts[:,2].min() < margin:
                faults.append(f'table:{mesh.link}')
            for obstacle in self.bin:
                if overlap(verts,obstacle.vertices) and intersect_hulls(mesh,pose,obstacle,np.eye(4),margin):
                    faults.append(f'{obstacle.link}:{mesh.link}')
            for right,rverts,rpose in self.right:
                if overlap(verts,rverts) and intersect_hulls(mesh,pose,right,rpose,margin):
                    faults.append(f'opposite:{mesh.link}/{right.link}')
            if not (cube_contact and mesh.link in ('gripper','jaw')):
                if np.all(verts.min(0) <= self.cube+.02+margin) and np.all(verts.max(0) >= self.cube-.02-margin):
                    triangles = apply(pose,mesh.triangles)
                    if mesh_box_overlap(triangles,self.cube,np.full(3,.02+margin)):
                        faults.append(f'cube:{mesh.link}')
        for i,(a,av,at) in enumerate(meshes):
            for b,bv,bt in meshes[i+1:]:
                if a.link == b.link or frozenset((a.link,b.link)) in model.adjacent:
                    continue  # USD joints disable contact on these mechanical interfaces.
                if overlap(av,bv) and intersect_hulls(a,at,b,bt,margin):
                    faults.append(f'self:{a.link}/{b.link}')
        return sorted(set(faults))

    def segment(self, start, end, margin=.002, cube_contact=False):
        # Radius 0.6 m bounds every downstream point of this <0.5m robot.
        # Maximum point movement from the nearest sample <= margin/2. This
        # covers nominal joint-linear motion, not tracking error or PhysX hulls.
        count = max(2, int(np.ceil(.6*np.abs(end-start).sum()/margin))+1)
        for i,t in enumerate(np.linspace(0,1,count)):
            q = start+(end-start)*t
            faults = self.check(q,margin,cube_contact)
            if faults:
                return {'safe':False,'samples':count,'failed_sample':i,'q':q,'faults':faults}
        return {'safe':True,'samples':count}


def plan_pickup(model, cube=(-.34,-.04,.02)):
    """Find reachable pregrasp/grasp with aligned cube faces and validate travel."""
    world = CollisionWorld(model,cube)
    home = np.zeros(6)
    opened = home.copy(); opened[-1] = np.deg2rad(45)
    pretarget = np.array(cube)+[0,0,.035]
    grasptarget = np.array(cube)+[0,0,.002]
    pres = model.ik(pretarget,rotation=np.eye(3),seeds=12)
    grasps = model.ik(grasptarget,rotation=np.eye(3),seeds=12)
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


def plan_lift(model, grasp_q, held=False):
    """Plan a 58 mm lift with achievable tilt and a predicted payload envelope.

    The payload transform is a planning assumption, never a simulator attachment.
    Execution must independently verify bilateral force and object following.
    """
    world=CollisionWorld(model)
    closed=np.array(grasp_q).copy()
    if not held:closed[-1]=np.deg2rad(-5)
    closure=world.segment(np.array(grasp_q),closed,cube_contact=True)
    if not closure['safe']:
        return {'safe':False,'failure':'closure_collision','check':closure}
    candidates=[]
    for heading in (-19.,-18.5,-19.5):
        h=np.deg2rad(heading);tilt=np.deg2rad(15)
        direction=[np.sin(h)*np.sin(tilt),np.cos(h)*np.sin(tilt),-np.cos(tilt)]
        candidates.extend(model.ik([-.34,-.04,.08],approach=direction,opening_deg=np.rad2deg(closed[-1]),reference=closed,seeds=3))
    candidates.sort(key=lambda x:np.linalg.norm(x['q']-closed))
    rejections=[]
    payload=box_mesh(np.array([0,0,-.002]),np.full(3,.04),'payload')
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
            q=closed+t*(end-closed);pose=model.tcp_pose(q);vertices=apply(pose,payload.vertices)
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
                    'ik':candidate,'target_world_m':[-.34,-.04,.08],'rejections':rejections}
        rejections.append({'faults':faults})
    return {'safe':False,'failure':'no_lift_candidate','rejections':rejections}


def handover_candidates(model, target, seeds=12, alternate=False):
    """Coupled 10-joint solve for opposing approaches on orthogonal cube faces.

    Feasibility study only: coincident nominal TCPs, not a transfer controller.
    Both hands' five-DOF limitations are solved together instead of prescribing
    two independently infeasible six-DOF poses.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    target=np.asarray(target)
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
        left=model.tcp_pose(np.r_[q[:5],np.deg2rad(22)])
        right=model.tcp_pose(np.r_[q[5:],np.deg2rad(45)],right_base)
        return np.r_[(left[:3,3]-target)/.1,(right[:3,3]-left[:3,3])/.1,
                     Rotation.from_matrix((left[:3,:3]@relative_rotation).T@right[:3,:3]).as_rotvec()]
    for start in starts:
        fit=least_squares(residual,start,bounds=(lower,upper),max_nfev=250,gtol=1e-9,ftol=1e-9,xtol=1e-9)
        if np.linalg.norm(residual(fit.x))>.005:continue
        ql=np.r_[fit.x[:5],np.deg2rad(22)];qr=np.r_[fit.x[5:],np.deg2rad(45)]
        left=model.tcp_pose(ql);right=model.tcp_pose(qr,right_base)
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
