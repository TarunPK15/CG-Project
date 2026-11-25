"""
test_gradients.py - Unit test for 3D Central Difference Gradients.
"""

import numpy as np
from core.gradients import compute_volume_gradients


def test_quadratic_field_gradients():
    """
    On a quadratic function f(x, y, z) = x^2 + 2*y^2 + 3*z^2,
    the continuous gradient is (2x, 4y, 6z).
    The central difference approximation (f(x+h) - f(x-h)) / (2h)
    for quadratic functions is mathematically exact in the interior!
    """
    nx, ny, nz = 15, 15, 15
    spacing = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    coords_x = (np.arange(nx) - nx // 2) * spacing[0]
    coords_y = (np.arange(ny) - ny // 2) * spacing[1]
    coords_z = (np.arange(nz) - nz // 2) * spacing[2]

    z, y, x = np.meshgrid(coords_z, coords_y, coords_x, indexing="ij")

    field = (x**2 + 2.0 * y**2 + 3.0 * z**2).astype(np.float32)

    grads, mags, norms = compute_volume_gradients(field, spacing)

    # Check interior points (away from boundaries)
    for zi in range(2, nz - 2):
        for yi in range(2, ny - 2):
            for xi in range(2, nx - 2):
                expected_gx = 2.0 * coords_x[xi]
                expected_gy = 4.0 * coords_y[yi]
                expected_gz = 6.0 * coords_z[zi]

                np.testing.assert_allclose(
                    grads[zi, yi, xi, 0], expected_gx, rtol=1e-4, atol=1e-4
                )
                np.testing.assert_allclose(
                    grads[zi, yi, xi, 1], expected_gy, rtol=1e-4, atol=1e-4
                )
                np.testing.assert_allclose(
                    grads[zi, yi, xi, 2], expected_gz, rtol=1e-4, atol=1e-4
                )

                # Check normal unit length when magnitude > 0
                if mags[zi, yi, xi] > 1e-4:
                    n_len = np.linalg.norm(norms[zi, yi, xi])
                    np.testing.assert_allclose(n_len, 1.0, atol=1e-4)


if __name__ == "__main__":
    test_quadratic_field_gradients()
    print("All gradient tests passed!")
