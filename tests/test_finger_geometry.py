"""CPU tests for triangle sections and fail-closed geometry prerequisites.

Run with the existing Isaac USD Python/library paths (README command).
"""
import runpy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.argv = [sys.argv[0], '--offline']
audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/audit_so101_finger_geometry_v2.py'))


class SurfaceGeometryTests(unittest.TestCase):
    def test_intersects_face_interior_not_just_vertices(self):
        # Two triangles form a 40 mm square in the plane x=30 mm.
        triangles = np.array([[[.03, -.02, -.02], [.03, .02, -.02], [.03, .02, .02]],
                              [[.03, -.02, -.02], [.03, .02, .02], [.03, -.02, .02]]])
        hits = audit['x_surface_intersections'](triangles, 0, 0)
        np.testing.assert_allclose(hits, [.03], atol=1e-9)
        self.assertEqual(len(audit['x_surface_intersections'](triangles, .03, 0)), 0)

    def test_parallel_and_degenerate_triangles_do_not_create_hits(self):
        triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 0, 1]], [[0, 0, 0]] * 3], dtype=float)
        self.assertEqual(len(audit['x_surface_intersections'](triangles, 0, 0)), 0)

    def test_missing_or_invalid_geometry_fails(self):
        with self.assertRaises(RuntimeError):
            audit['validate_geometry']([])
        records = [{'link': link, 'collision_schema': 'enabled', 'problem': '', 'triangle_count': 1}
                   for link in ('gripper', 'jaw')]
        audit['validate_geometry'](records)
        records[1]['problem'] = 'no bound'
        with self.assertRaises(RuntimeError):
            audit['validate_geometry'](records)


if __name__ == '__main__':
    unittest.main()
