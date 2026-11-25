"""
interpolation.py - JIT-compiled Trilinear Interpolation for 3D scalar and vector fields.
"""

from numba import njit
import numpy as np


@njit(fastmath=True, inline="always")
def sample_trilinear_scalar(
    data: np.ndarray,
    x: float,
    y: float,
    z: float,
    nx: int,
    ny: int,
    nz: int,
) -> float:
    """
    Samples 3D scalar array `data` of shape (nz, ny, nx) at continuous coordinates (x, y, z).
    Coordinates (x, y, z) are in voxel index units: x in [0, nx-1], y in [0, ny-1], z in [0, nz-1].
    Returns interpolated scalar value (clamped outside boundaries to 0.0).
    """
    if x < 0.0 or x > float(nx - 1) or y < 0.0 or y > float(ny - 1) or z < 0.0 or z > float(nz - 1):
        return 0.0

    x0 = int(x)
    y0 = int(y)
    z0 = int(z)

    x1 = x0 + 1 if x0 < nx - 1 else x0
    y1 = y0 + 1 if y0 < ny - 1 else y0
    z1 = z0 + 1 if z0 < nz - 1 else z0

    fx = x - float(x0)
    fy = y - float(y0)
    fz = z - float(z0)

    # 8 corner values
    c000 = data[z0, y0, x0]
    c100 = data[z0, y0, x1]
    c010 = data[z0, y1, x0]
    c110 = data[z0, y1, x1]
    c001 = data[z1, y0, x0]
    c101 = data[z1, y0, x1]
    c011 = data[z1, y1, x0]
    c111 = data[z1, y1, x1]

    # Interpolate along X
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx

    # Interpolate along Y
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    # Interpolate along Z
    return c0 * (1.0 - fz) + c1 * fz


@njit(fastmath=True, inline="always")
def sample_trilinear_vec3(
    vectors: np.ndarray,
    x: float,
    y: float,
    z: float,
    nx: int,
    ny: int,
    nz: int,
    out: np.ndarray,
):
    """
    Samples 3D vector array `vectors` of shape (nz, ny, nx, 3) at continuous coordinates (x, y, z).
    Writes interpolated 3D vector into `out`.
    """
    if x < 0.0 or x > float(nx - 1) or y < 0.0 or y > float(ny - 1) or z < 0.0 or z > float(nz - 1):
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        return

    x0 = int(x)
    y0 = int(y)
    z0 = int(z)

    x1 = x0 + 1 if x0 < nx - 1 else x0
    y1 = y0 + 1 if y0 < ny - 1 else y0
    z1 = z0 + 1 if z0 < nz - 1 else z0

    fx = x - float(x0)
    fy = y - float(y0)
    fz = z - float(z0)

    for k in range(3):
        c000 = vectors[z0, y0, x0, k]
        c100 = vectors[z0, y0, x1, k]
        c010 = vectors[z0, y1, x0, k]
        c110 = vectors[z0, y1, x1, k]
        c001 = vectors[z1, y0, x0, k]
        c101 = vectors[z1, y0, x1, k]
        c011 = vectors[z1, y1, x0, k]
        c111 = vectors[z1, y1, x1, k]

        c00 = c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx

        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy

        out[k] = c0 * (1.0 - fz) + c1 * fz
