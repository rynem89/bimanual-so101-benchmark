"""Camera timing and optical-frame checks without launching Isaac Sim."""
import unittest
import numpy as np
from policy_cameras import NAMES, HEIGHT, WIDTH, WRIST_POSITION, WRIST_TARGET, WRIST_ROTATION, check_capture, check_attachment

class CameraTests(unittest.TestCase):
    def test_optical_frame_looks_at_grasp(self):
        z=WRIST_TARGET-WRIST_POSITION;z/=np.linalg.norm(z)
        np.testing.assert_allclose(WRIST_ROTATION[:,2],z)
        np.testing.assert_allclose(WRIST_ROTATION.T@WRIST_ROTATION,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(WRIST_ROTATION),1.)

    def test_wrong_parent_and_wrong_orientation_fail(self):
        parent_position=np.array([.25,-.3,.2])
        parent_rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        position=parent_position+parent_rotation@WRIST_POSITION
        rotation=parent_rotation@WRIST_ROTATION
        check_attachment(position,rotation,parent_position,parent_rotation)
        with self.assertRaisesRegex(RuntimeError,'translation'):
            check_attachment(position,rotation,parent_position+np.array([.001,0.,0.]),parent_rotation)
        with self.assertRaisesRegex(RuntimeError,'rotation'):
            check_attachment(position,np.eye(3),parent_position,parent_rotation)

    def test_missing_camera_bad_rgb_and_bad_clock_fail(self):
        rgb=np.arange(HEIGHT*WIDTH*3,dtype=np.uint8).reshape(HEIGHT,WIDTH,3)
        images={name:rgb for name in NAMES}
        check_capture(images,184,181)
        with self.assertRaisesRegex(ValueError,'Missing'):check_capture({'overview':rgb},184,181)
        with self.assertRaisesRegex(ValueError,'timestamp'):check_capture(images,185,181)
        with self.assertRaisesRegex(ValueError,'RGB'):check_capture({**images,'left_wrist':rgb[:2]},184,181)
