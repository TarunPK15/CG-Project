"""
gradients.py - 3D Central Difference Gradient and Surface Normal Computation.
Implements the central difference approximation described in Drebin et al. (1988) and Levoy (1988).
"""

from typing import Tuple
from numba import njit, prange
import numpy as np


@njit(parallel=True, fastmath=True)
def _compute_gradients_kernel(
    data: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    gradients: np.ndarray,
    magnitudes: np.ndarray,
    normals: np.ndarray,
):
    nz, ny, nx = data.shape
    inv_2dx = 0.5 / dx
    inv_2dy = 0.5 / dy
    inv_2dz = 0.5 / dz
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dz = 1.0 / dz

    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                # Central difference in X
                if x == 0:
                    gx = (data[z, y, 1] - data[z, y, 0]) * inv_dx
                elif x == nx - 1:
                    gx = (data[z, y, nx - 1] - data[z, y, nx - 2]) * inv_dx
                else:
                    gx = (data[z, y, x + 1] - data[z, y, x - 1]) * inv_2dx

                # Central difference in Y
                if y == 0:
                    gy = (data[z, 1, x] - data[z, 0, x]) * inv_dy
                elif y == ny - 1:
                    gy = (data[z, ny - 1, x] - data[z, ny - 2, x]) * inv_dy
                else:
                    gy = (data[z, y + 1, x] - data[z, y - 1, x]) * inv_2dy

                # Central difference in Z
                if z == 0:
                    gz = (data[1, y, x] - data[0, y, x]) * inv_dz
                elif z == nz - 1:
                    gz = (data[nz - 1, y, x] - data[nz - 2, y, x]) * inv_dz
                else:
                    gz = (data[z + 1, y, x] - data[z - 1, y, x]) * inv_2dz

                gradients[z, y, x, 0] = gx
                gradients[z, y, x, 1] = gy
                gradients[z, y, x, 2] = gz

                mag = np.sqrt(gx * gx + gy * gy + gz * gz)
                magnitudes[z, y, x] = mag

                if mag > 1e-6:
                    inv_m = 1.0 / mag
                    normals[z, y, x, 0] = gx * inv_m
                    normals[z, y, x, 1] = gy * inv_m
                    normals[z, y, x, 2] = gz * inv_m
                else:
                    normals[z, y, x, 0] = 0.0
                    normals[z, y, x, 1] = 0.0
                    normals[z, y, x, 2] = 0.0


def compute_volume_gradients(
    data: np.ndarray,
    spacing: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes gradient vector field, gradient magnitude, and unit normal vector field.
    Returns:
        gradients: shape (Nz, Ny, Nx, 3) float32
        magnitudes: shape (Nz, Ny, Nx) float32
        normals: shape (Nz, Ny, Nx, 3) float32
    """
    nz, ny, nx = data.shape
    dx, dy, dz = float(spacing[0]), float(spacing[1]), float(spacing[2])

    gradients = np.zeros((nz, ny, nx, 3), dtype=np.float32)
    magnitudes = np.zeros((nz, ny, nx), dtype=np.float32)
    normals = np.zeros((nz, ny, nx, 3), dtype=np.float32)

    _compute_gradients_kernel(
        data, dx, dy, dz, gradients, magnitudes, normals
    )
    return gradients, magnitudes, normals
