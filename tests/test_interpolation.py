"""
test_interpolation.py - Unit test for Trilinear Interpolation correctness.
"""

import numpy as np
from core.interpolation import sample_trilinear_scalar, sample_trilinear_vec3


def test_trilinear_scalar_linear_field():
    """
    On a linear function f(x, y, z) = ax + by + cz + d,
    trilinear interpolation is mathematically exact.
    """
    nx, ny, nz = 10, 10, 10
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    
    # f(x, y, z) = 1.5*x - 2.0*y + 0.5*z + 3.0
    field = (1.5 * x - 2.0 * y + 0.5 * z + 3.0).astype(np.float32)

    # Test random non-integer sample points
    test_points = [
        (2.3, 4.7, 1.2),
        (5.5, 5.5, 5.5),
        (0.1, 8.9, 3.4),
        (7.8, 1.1, 6.9),
    ]

    for tx, ty, tz in test_points:
        expected = 1.5 * tx - 2.0 * ty + 0.5 * tz + 3.0
        val = sample_trilinear_scalar(field, tx, ty, tz, nx, ny, nz)
        np.testing.assert_allclose(val, expected, rtol=1e-4, atol=1e-4)


def test_trilinear_out_of_bounds():
    """Tests that out-of-bounds queries safely return zero."""
    nx, ny, nz = 8, 8, 8
    field = np.ones((nz, ny, nx), dtype=np.float32)

    assert sample_trilinear_scalar(field, -0.5, 2.0, 2.0, nx, ny, nz) == 0.0
    assert sample_trilinear_scalar(field, 2.0, 8.5, 2.0, nx, ny, nz) == 0.0
    assert sample_trilinear_scalar(field, 2.0, 2.0, 10.0, nx, ny, nz) == 0.0


def test_trilinear_vec3():
    """Tests interpolation of 3D vector array."""
    nx, ny, nz = 6, 6, 6
    vectors = np.zeros((nz, ny, nx, 3), dtype=np.float32)
    vectors[:, :, :, 0] = 1.0
    vectors[:, :, :, 1] = 2.0
    vectors[:, :, :, 2] = 3.0

    out = np.zeros(3, dtype=np.float32)
    sample_trilinear_vec3(vectors, 2.5, 3.1, 4.8, nx, ny, nz, out)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0], atol=1e-5)


if __name__ == "__main__":
    test_trilinear_scalar_linear_field()
    test_trilinear_out_of_bounds()
    test_trilinear_vec3()
    print("All interpolation tests passed!")
