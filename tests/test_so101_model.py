"""Offline geometry/FK/IK checks; use the bundled USD paths from README."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from so101_model import SO101Model,calibrate,box_mesh,mesh_box_overlap,triangle_box_overlap


class GeometryTests(unittest.TestCase):
    def test_box_containment_and_separation(self):
        mesh=box_mesh(np.zeros(3),np.ones(3),'test')
        self.assertFalse(triangle_box_overlap(mesh.triangles,np.zeros(3),np.ones(3)*.1))
        self.assertTrue(mesh_box_overlap(mesh.triangles,np.zeros(3),np.ones(3)*.1))
        self.assertFalse(mesh_box_overlap(mesh.triangles,np.array([1.,0,0]),np.ones(3)*.1))
        self.assertTrue(mesh_box_overlap(mesh.triangles,np.array([.55,0,0]),np.ones(3)*.1))


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

    def test_cube_fits_open_but_intersects_closed_authored_mesh(self):
        rows=self.calibration['surface_checks']
        self.assertTrue(rows[0]['surface_intersections_with_42mm_box'])
        self.assertEqual(next(r for r in rows if r['angle_deg']==45)['surface_intersections_with_42mm_box'],[])


if __name__=='__main__':unittest.main()
