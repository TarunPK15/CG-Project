"""
test_raybox.py - Unit test for Slab Method Ray-Box Intersection.
"""

import numpy as np
from core.raycaster import intersect_ray_box


def test_ray_box_hit_miss():
    # Box from (-1, -1, -1) to (1, 1, 1)
    bmin = (-1.0, -1.0, -1.0)
    bmax = (1.0, 1.0, 1.0)

    # 1. Straight hit along Z
    hit, t0, t1 = intersect_ray_box(0.0, 0.0, -3.0, 0.0, 0.0, 1.0, *bmin, *bmax)
    assert hit is True
    np.testing.assert_allclose(t0, 2.0, atol=1e-5)
    np.testing.assert_allclose(t1, 4.0, atol=1e-5)

    # 2. Oblique diagonal hit
    dir_diag = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    dir_diag /= np.linalg.norm(dir_diag)
    hit, t0, t1 = intersect_ray_box(-2.0, -2.0, -2.0, *dir_diag, *bmin, *bmax)
    assert hit is True
    assert t0 > 0.0
    assert t1 > t0

    # 3. Complete miss
    hit, _, _ = intersect_ray_box(5.0, 5.0, -3.0, 0.0, 0.0, 1.0, *bmin, *bmax)
    assert hit is False

    # 4. Ray starting inside the box
    hit, t0, t1 = intersect_ray_box(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, *bmin, *bmax)
    assert hit is True
    np.testing.assert_allclose(t0, 0.0, atol=1e-5)  # Clamped to 0
    np.testing.assert_allclose(t1, 1.0, atol=1e-5)


if __name__ == "__main__":
    test_ray_box_hit_miss()
    print("All ray-box intersection tests passed!")
