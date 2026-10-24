"""CPU SO-101 model: authored USD joints, triangles, bounded IK and collision checks.

Column-vector transforms T_parent_child; metres/radians internally. USD matrices
are transposed at the boundary. The model never launches Kit or changes assets.
Convex envelopes are conservative rejection tests, not cooked PhysX geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os

import numpy as np
from scipy.optimize import least_squares, linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError
from scipy.spatial.transform import Rotation

JOINT_NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
LINK_NAMES = ('base', 'shoulder', 'upper_arm', 'lower_arm', 'wrist', 'gripper', 'jaw')


def transform(position=(0, 0, 0), rotation=None):
    result = np.eye(4)
    result[:3, 3] = position
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def apply(t, points):
    return np.asarray(points) @ t[:3, :3].T + t[:3, 3]


def joint_frame(position, quaternion):
    return transform(position, Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix())


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    parent_frame: np.ndarray
    child_inverse: np.ndarray
    axis: np.ndarray
    limits: tuple[float, float]


@dataclass
class Mesh:
    link: str
    path: str
    triangles: np.ndarray
    vertices: np.ndarray
    planes: np.ndarray
    inflated_vertices: dict = field(default_factory=dict,repr=False)
    separating_axes: dict = field(default_factory=dict,repr=False)


class SO101Model:
    def __init__(self, asset: str | Path | None = None):
        from pxr import Usd, UsdGeom, UsdPhysics

        if asset is None:
            assets = Path(os.environ.get('LEISAAC_ASSETS_ROOT', Path(__file__).resolve().parents[2] / 'leisaac/assets'))
            asset = assets / 'robots/so101_follower.usd'
        self.asset = str(asset)
        stage = Usd.Stage.Open(self.asset)
        if UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError('Expected metre-scaled USD')
        root = stage.GetDefaultPrim()
        xforms = UsdGeom.XformCache()
        self.joints = []
        for name in JOINT_NAMES:
            j = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f'{root.GetPath()}/joints/{name}'))
            self.joints.append(Joint(name, j.GetBody0Rel().GetTargets()[0].name,
                j.GetBody1Rel().GetTargets()[0].name,
                joint_frame(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
                np.linalg.inv(joint_frame(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())),
                np.eye(3)['XYZ'.index(j.GetAxisAttr().Get())],
                tuple(np.deg2rad([j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()]))))
        self.limits = np.array([j.limits for j in self.joints])
        self.adjacent = {frozenset((j.parent, j.child)) for j in self.joints}
        self.meshes = []
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Mesh) or '/collisions/' not in str(p.GetPath()):
                continue
            cursor = p
            enabled = False
            while cursor != root:
                if cursor.HasAPI(UsdPhysics.CollisionAPI):
                    enabled = UsdPhysics.CollisionAPI(cursor).GetCollisionEnabledAttr().Get() is not False
                    break
                cursor = cursor.GetParent()
            if not enabled:
                continue
            link = p
            while not link.HasAPI(UsdPhysics.RigidBodyAPI):
                link = link.GetParent()
                if link == root:
                    raise ValueError(f'No rigid owner for {p.GetPath()}')
            m = UsdGeom.Mesh(p)
            points = np.asarray(m.GetPointsAttr().Get(), dtype=float)
            counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
            indices = np.asarray(m.GetFaceVertexIndicesAttr().Get())
            if not np.all(counts == 3) or not np.isfinite(points).all():
                raise ValueError('Expected finite triangle mesh')
            matrix = np.array(xforms.GetLocalToWorldTransform(p) * xforms.GetLocalToWorldTransform(link).GetInverse()).T
            points = apply(matrix, points)
            hull = ConvexHull(np.unique(points, axis=0))
            self.meshes.append(Mesh(link.GetName(), str(p.GetPath()), points[indices.reshape(-1, 3)],
                                    hull.points[hull.vertices], hull.equations))
        if {m.link for m in self.meshes} != set(LINK_NAMES):
            raise ValueError('Missing collision meshes')
        # Root placement follows LeIsaac init_state.rot=(w,x,y,z)=(0,0,0,1).
        self.base = transform((-.25, -.30, -.030081), Rotation.from_euler('z', np.pi).as_matrix())
        # Fixed-link grasp frame: X closes jaws, Y pad width, -Z approach.
        # Determined below from the complete 40 mm cube footprint, not one ray.
        self.tcp = transform((0.0084, 0, -.08))

    def fk(self, q, base=None):
        q = np.asarray(q)
        if q.shape != (6,):
            raise ValueError('Expected six joint coordinates')
        poses = {'base': self.base if base is None else base}
        for joint, angle in zip(self.joints, q):
            poses[joint.child] = poses[joint.parent] @ joint.parent_frame @ transform(
                rotation=Rotation.from_rotvec(joint.axis * angle).as_matrix()) @ joint.child_inverse
        return poses

    def tcp_pose(self, q, base=None):
        return self.fk(q, base)['gripper'] @ self.tcp

    def ik(self, target_m, approach=(0, 0, -1), opening_deg=45, seeds=16, reference=None, base=None, rotation=None):
        """Position + tool approach (five constraints); axial roll is free.

        Return all distinct converged candidates, including errors and joint-limit
        margin. A six-dimensional pose is never claimed from this 5-DOF solve.
        """
        target = np.asarray(target_m)
        direction = np.asarray(approach, dtype=float)
        direction /= np.linalg.norm(direction)
        lower, upper = self.limits[:5].T
        rng = np.random.default_rng(101)
        starts = [np.zeros(5)]
        if reference is not None:
            starts.insert(0, np.asarray(reference)[:5])
        starts += list(rng.uniform(lower + .02, upper - .02, size=(seeds, 5)))
        solutions = []
        def residual(q5):
            t = self.tcp_pose(np.r_[q5, np.deg2rad(opening_deg)], base)
            orient = (-t[:3, 2] - direction if rotation is None else
                      Rotation.from_matrix(rotation.T @ t[:3, :3]).as_rotvec())
            return np.r_[(t[:3, 3] - target) / .1, orient]
        for initial in starts:
            fit = least_squares(residual, np.clip(initial, lower + 1e-7, upper - 1e-7),
                                bounds=(lower, upper), max_nfev=250, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            q = np.r_[fit.x, np.deg2rad(opening_deg)]
            pose = self.tcp_pose(q, base)
            error = np.linalg.norm(pose[:3, 3] - target)
            angle = np.rad2deg(np.arccos(np.clip(np.dot(-pose[:3, 2], direction), -1, 1)))
            if rotation is not None:
                angle = np.rad2deg(Rotation.from_matrix(rotation.T @ pose[:3, :3]).magnitude())
            if fit.success and error < .001 and angle < 1 and not any(np.linalg.norm(q - s['q']) < .01 for s in solutions):
                solutions.append({'q': q, 'position_error_m': float(error), 'approach_error_deg': float(angle),
                                  'limit_margin_deg': float(np.rad2deg(np.minimum(fit.x-lower, upper-fit.x).min())),
                                  'evaluations': fit.nfev})
        return sorted(solutions, key=lambda s: np.linalg.norm(s['q'][:5]-(np.zeros(5) if reference is None else reference[:5])))

    def world_meshes(self, q, base=None):
        poses = self.fk(q, base)
        return [(m, apply(poses[m.link], m.vertices), poses[m.link]) for m in self.meshes]

    def motion_radii(self, extra_vertices=None, link='gripper'):
        """Conservative metres/radian bounds from authored frames and geometry.

        Walk from each rigid link back through its ancestor joints. Triangle
        inequality on the fixed translations bounds every downstream point at
        EVERY joint angle; this is not a sampled-pose estimate. An optional
        payload is supplied in its owning rigid link's coordinates.
        """
        extents={}
        for mesh in self.meshes:
            extents[mesh.link]=max(extents.get(mesh.link,0.),float(np.linalg.norm(mesh.vertices,axis=1).max()))
        if extra_vertices is not None:
            points=np.asarray(extra_vertices,dtype=float)
            if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
                raise ValueError('Finite payload vertices in rigid-link coordinates required')
            if link not in LINK_NAMES:raise ValueError('Unknown payload link')
            extents[link]=max(extents.get(link,0.),float(np.linalg.norm(points,axis=1).max()))
        parents={joint.child:(i,joint) for i,joint in enumerate(self.joints)}
        radii=np.zeros(len(self.joints))
        for child,reach in extents.items():
            while child in parents:
                index,joint=parents[child]
                reach+=np.linalg.norm(joint.child_inverse[:3,3])
                radii[index]=max(radii[index],reach)
                reach+=np.linalg.norm(joint.parent_frame[:3,3])
                child=joint.parent
        return radii


def intersect_hulls(a, ta, b, tb, margin=0.001):
    """Conservative convex-envelope overlap via half-space feasibility.

    Inflates each hull by margin/2. Infeasibility proves separation of these
    envelopes; overlap only rejects a candidate, never proves real contact.
    """
    # A separating face axis is a sufficient proof of nonintersection. Use the
    # actual half-space-inflated vertices, not original vertices plus a radius:
    # acute corners can expand by more than margin/2. Unresolved edge/edge cases
    # still use the original LP, preserving its conservative contact semantics.
    inflated=[]
    for mesh in (a,b):
        if margin not in mesh.inflated_vertices:
            planes=mesh.planes.copy();planes[:,3]-=margin/2
            try:
                mesh.inflated_vertices[margin]=HalfspaceIntersection(planes,mesh.vertices.mean(0)).intersections
            except QhullError:
                mesh.inflated_vertices[margin]=None
        inflated.append(mesh.inflated_vertices[margin])
    if all(v is not None for v in inflated):
        key=(b.path,margin)
        cached=a.separating_axes.get(key)
        if cached is not None:
            axis=ta[:3,:3]@cached
            pa=inflated[0]@(ta[:3,:3].T@axis)+ta[:3,3]@axis
            pb=inflated[1]@(tb[:3,:3].T@axis)+tb[:3,3]@axis
            # This is a new proof against CURRENT full inflated vertices,
            # never reuse the previous pose's collision/no-collision result.
            if pa.max()<pb.min()-1e-7 or pb.max()<pa.min()-1e-7:return False
        centre=ta[:3,3]-tb[:3,3]
        if np.linalg.norm(centre)>1e-12:
            axis=centre/np.linalg.norm(centre)
            pa=inflated[0]@(ta[:3,:3].T@axis)+ta[:3,3]@axis
            pb=inflated[1]@(tb[:3,:3].T@axis)+tb[:3,3]@axis
            if pa.max()<pb.min()-1e-7 or pb.max()<pa.min()-1e-7:
                a.separating_axes[key]=ta[:3,:3].T@axis
                return False
        # Curved CAD hulls have thousands of facets. Testing every face axis
        # creates huge matrices; a small deterministic subset proves many
        # separations cheaply. Missing an axis only falls back to the LP.
        axes=np.vstack([a.planes[::max(1,len(a.planes)//12),:3]@ta[:3,:3].T,
                        b.planes[::max(1,len(b.planes)//12),:3]@tb[:3,:3].T])
        pa=inflated[0]@(ta[:3,:3].T@axes.T)+ta[:3,3]@axes.T
        pb=inflated[1]@(tb[:3,:3].T@axes.T)+tb[:3,3]@axes.T
        separated=(pa.max(0)<pb.min(0)-1e-7)|(pb.max(0)<pa.min(0)-1e-7)
        if separated.any():
            a.separating_axes[key]=ta[:3,:3].T@axes[np.flatnonzero(separated)[0]]
            return False
    return _lp_intersection(a,ta,b,tb,margin)


def _lp_intersection(a,ta,b,tb,margin):
    """Definitive fallback for pairs without a separating face-axis proof."""
    matrices = []
    offsets = []
    for mesh, t in ((a, ta), (b, tb)):
        normals = mesh.planes[:, :3] @ t[:3, :3].T
        matrices.append(normals)
        offsets.append(-mesh.planes[:, 3] + normals @ t[:3, 3] + margin/2)
    result = linprog(np.zeros(3), A_ub=np.vstack(matrices), b_ub=np.concatenate(offsets),
                     bounds=[(None, None)]*3, method='highs')
    if result.status not in (0, 2):
        # HiGHS presolve can return an unknown status on almost tangent small
        # hulls. Retry dual simplex without presolve; accept only a definite
        # feasibility/infeasibility result, never the ambiguous primal status.
        first_message=result.message
        result=linprog(np.zeros(3),A_ub=np.vstack(matrices),b_ub=np.concatenate(offsets),
                       bounds=[(None,None)]*3,method='highs-ds',options={'presolve':False})
        print(f'[COLLISION RETRY] {a.link}/{b.link}: {first_message} -> {result.message}',flush=True)
    if result.status not in (0, 2):
        raise RuntimeError(f'Collision LP unresolved: {result.message}')
    return result.status == 0


def box_mesh(centre, size, name):
    from itertools import product
    points = np.array(list(product((-1, 1), repeat=3))) * np.asarray(size)/2 + centre
    hull = ConvexHull(points)
    return Mesh(name, name, points[hull.simplices], points, hull.equations)


def triangle_box_overlap(triangles, centre, halfsize):
    """Exact triangle/AABB separating-axis test (surface intersection only)."""
    tri = triangles - centre
    halfsize = np.asarray(halfsize)
    broad = np.all(tri.min(axis=1) <= halfsize, axis=1) & np.all(tri.max(axis=1) >= -halfsize, axis=1)
    tri = tri[broad]
    if not len(tri):
        return False
    edges = np.roll(tri, -1, axis=1) - tri
    axes = [np.cross(edges[:, 0], edges[:, 1])]
    axes += [np.cross(edges[:, i], unit) for i in range(3) for unit in np.eye(3)]
    overlap = np.ones(len(tri), dtype=bool)
    for axis in axes:
        projections = np.einsum('tvc,tc->tv', tri, axis)
        radius = np.abs(axis) @ halfsize
        overlap &= (projections.min(axis=1) <= radius + 1e-10) & (projections.max(axis=1) >= -radius - 1e-10)
    return bool(overlap.any())


def mesh_box_overlap(triangles, centre, halfsize):
    """Closed mesh versus box, including the box-inside-mesh containment case.

    Ray parity uses all face interiors; duplicate edge hits are deduplicated.
    Surface intersection is checked first, so a tangent box is rejected.
    """
    if triangle_box_overlap(triangles, centre, halfsize):
        return True
    a=triangles[:,0];u=triangles[:,1]-a;v=triangles[:,2]-a
    det=u[:,1]*v[:,2]-u[:,2]*v[:,1]
    valid=np.abs(det)>1e-15
    a,u,v,det=a[valid],u[valid],v[valid],det[valid]
    dy,dz=centre[1]-a[:,1],centre[2]-a[:,2]
    b=(dy*v[:,2]-dz*v[:,1])/det;c=(u[:,1]*dz-u[:,2]*dy)/det
    inside=(b>=-1e-9)&(c>=-1e-9)&(b+c<=1+1e-9)
    hits=np.unique(np.round((a[:,0]+b*u[:,0]+c*v[:,0])[inside],9))
    return bool(np.count_nonzero(hits>centre[0])%2)


def inner_surface_x(triangles, y_range, z_range):
    """Clip every relevant triangle to the cube's YZ footprint, then max X.

    This is the support boundary of the entire fixed-finger footprint, including
    ridges, not its AABB. It is suitable for defining a nonpenetrating cube face.
    """
    mask = (triangles[:, :, 1].max(1) >= y_range[0]) & (triangles[:, :, 1].min(1) <= y_range[1])
    mask &= (triangles[:, :, 2].max(1) >= z_range[0]) & (triangles[:, :, 2].min(1) <= z_range[1])
    clipped = []
    for triangle in triangles[mask]:
        poly = list(triangle)
        for axis, bound, sign in ((1,y_range[0],1),(1,y_range[1],-1),(2,z_range[0],1),(2,z_range[1],-1)):
            out = []
            for a, b in zip(poly, poly[1:]+poly[:1]):
                da, db = sign*(a[axis]-bound), sign*(b[axis]-bound)
                if da >= 0:
                    out.append(a)
                if (da >= 0) != (db >= 0):
                    out.append(a + da/(da-db)*(b-a))
            poly = out
            if not poly:
                break
        clipped.extend(poly)
    if not clipped:
        raise ValueError('No fixed-finger surface in footprint')
    return float(np.max(np.array(clipped)[:, 0]))


def calibrate(model, depth_m=.09):
    fixed = next(m for m in model.meshes if m.link == 'gripper' and 'wrist_roll_follower' in m.path)
    centre = np.array([0., 0., -depth_m])
    # Two millimetres of initial fixed-pad clearance; closure must be validated
    # dynamically because the cube moves to seat against the fixed finger.
    boundary = inner_surface_x(fixed.triangles, (-.02,.02), (-depth_m-.02,-depth_m+.02))
    centre[0] = boundary + .022
    model.tcp = transform(centre)
    q = np.zeros(6)
    rows = []
    for angle in (0, 10, 15, 20, 25, 30, 35, 45, 60, 90):
        q[-1] = np.deg2rad(angle)
        poses = model.fk(q, np.eye(4))
        grip_inv = np.linalg.inv(poses['gripper'])
        collisions = []
        moving_boundary = None
        for mesh in model.meshes:
            if mesh.link not in ('gripper','jaw'):
                continue
            tri = apply(grip_inv @ poses[mesh.link], mesh.triangles)
            if mesh.link == 'jaw':
                reflected = tri.copy();reflected[:,:,0] *= -1
                try:
                    moving_boundary = -inner_surface_x(reflected,(-.02,.02),(-depth_m-.02,-depth_m+.02))
                except ValueError:
                    pass  # No opposing surface in this footprint, not infinite aperture.
            if mesh_box_overlap(tri, centre, np.full(3,.021)):
                collisions.append(mesh.path)
        rows.append({'angle_deg': angle, 'surface_intersections_with_42mm_box': collisions,
                     'fixed_support_x_m':boundary,'moving_support_x_m':moving_boundary,
                     'footprint_aperture_mm':None if moving_boundary is None else (moving_boundary-boundary)*1000})
    return {'tcp_position_gripper_m': centre.tolist(), 'tcp_rotation_gripper': np.eye(3).tolist(),
            'fixed_surface_support_x_m': boundary, 'cube_edge_m': .04, 'surface_checks': rows,
            'qualification': 'Authored triangle estimate; 42mm box includes 1mm per side. Physical contact unverified.'}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


if __name__ == '__main__':
    model = SO101Model()
    calibration = calibrate(model)
    print(json.dumps(calibration, indent=2))
    targets = {'pregrasp': [-.34,-.04,.055], 'grasp': [-.34,-.04,.022]}
    solutions = {name: model.ik(target,rotation=np.eye(3)) for name,target in targets.items()}
    report = {'calibration': calibration, 'targets_world_m': targets, 'solutions': solutions}
    out = Path('outputs/controller')
    out.mkdir(parents=True, exist_ok=True)
    (out/'offline.json').write_text(json.dumps(jsonable(report), indent=2))
    for name, candidates in solutions.items():
        print(name, json.dumps(jsonable(candidates)))
    if any(not x for x in solutions.values()):
        raise SystemExit('FAIL: one or more targets have no feasible IK')
