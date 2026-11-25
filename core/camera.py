"""
camera.py - 3D Pinhole Perspective and Orthographic Camera.
Provides orbit controls and ray generation for volume ray casting.
"""

from typing import Tuple
from numba import njit, prange
import numpy as np


class Camera:
    """
    3D Camera supporting spherical orbit controls, perspective, and orthographic rays.
    """

    def __init__(
        self,
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        azimuth: float = 45.0,    # degrees
        elevation: float = 30.0,  # degrees
        distance: float = 2.5,
        fov_y: float = 45.0,      # degrees (vertical field-of-view)
        is_perspective: bool = True,
        up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    ):
        self.target = np.array(target, dtype=np.float32)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)
        self.distance = float(distance)
        self.fov_y = float(fov_y)
        self.is_perspective = is_perspective
        self.up_guide = np.array(up, dtype=np.float32)

    @property
    def eye(self) -> np.ndarray:
        """Computes camera eye position from spherical orbit coordinates."""
        theta = np.radians(self.azimuth)
        phi = np.radians(self.elevation)

        # Spherical coordinate convention
        x = self.distance * np.cos(phi) * np.sin(theta)
        y = self.distance * np.sin(phi)
        z = self.distance * np.cos(phi) * np.cos(theta)

        return self.target + np.array([x, y, z], dtype=np.float32)

    def get_view_basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (eye, forward, right, up) unit vectors.
        """
        eye = self.eye
        forward = self.target - eye
        dist = np.linalg.norm(forward)
        if dist > 1e-6:
            forward /= dist
        else:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        right = np.cross(forward, self.up_guide)
        r_len = np.linalg.norm(right)
        if r_len > 1e-6:
            right /= r_len
        else:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        return eye, forward, right, up

    def orbit(self, delta_azimuth: float, delta_elevation: float):
        """Orbit camera around target."""
        self.azimuth = (self.azimuth + delta_azimuth) % 360.0
        self.elevation = np.clip(self.elevation + delta_elevation, -89.0, 89.0)

    def zoom(self, factor: float):
        """Zoom in/out by adjusting distance."""
        self.distance = max(0.2, self.distance * factor)

    def pan(self, delta_x: float, delta_y: float):
        """Pan target in the camera plane."""
        _, _, right, up = self.get_view_basis()
        shift = right * delta_x + up * delta_y
        self.target += shift


@njit(parallel=True, fastmath=True)
def generate_camera_rays(
    width: int,
    height: int,
    eye: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    fov_y_deg: float,
    is_perspective: bool,
    ray_origins: np.ndarray,     # shape (height, width, 3)
    ray_directions: np.ndarray,  # shape (height, width, 3)
):
    """
    Computes primary viewing rays for every pixel in parallel.
    """
    aspect = float(width) / float(height)
    tan_half_fov = np.tan(np.radians(fov_y_deg * 0.5))

    for j in prange(height):
        # Normalized Device Coordinates: v in [-1, 1], u in [-aspect, aspect]
        v = (1.0 - 2.0 * (float(j) + 0.5) / float(height)) * tan_half_fov
        for i in range(width):
            u = (2.0 * (float(i) + 0.5) / float(width) - 1.0) * aspect * tan_half_fov

            if is_perspective:
                # Perspective ray: all rays originate at eye
                ray_origins[j, i, 0] = eye[0]
                ray_origins[j, i, 1] = eye[1]
                ray_origins[j, i, 2] = eye[2]

                # Direction = normalize(forward + u * right + v * up)
                dx = forward[0] + u * right[0] + v * up[0]
                dy = forward[1] + u * right[1] + v * up[1]
                dz = forward[2] + u * right[2] + v * up[2]
                inv_len = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)

                ray_directions[j, i, 0] = dx * inv_len
                ray_directions[j, i, 1] = dy * inv_len
                ray_directions[j, i, 2] = dz * inv_len
            else:
                # Orthographic ray: parallel directions, origins on view plane
                ray_origins[j, i, 0] = eye[0] + u * right[0] + v * up[0]
                ray_origins[j, i, 1] = eye[1] + u * right[1] + v * up[1]
                ray_origins[j, i, 2] = eye[2] + u * right[2] + v * up[2]

                ray_directions[j, i, 0] = forward[0]
                ray_directions[j, i, 1] = forward[1]
                ray_directions[j, i, 2] = forward[2]
