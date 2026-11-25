"""
volume.py - 3D Scalar Volume representation and geometric spatial bounds.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class VolumeBoundingBox:
    """Axis-Aligned Bounding Box (AABB) in world space."""
    min_pt: np.ndarray  # shape (3,), float32: [x_min, y_min, z_min]
    max_pt: np.ndarray  # shape (3,), float32: [x_max, y_max, z_max]

    @property
    def size(self) -> np.ndarray:
        return self.max_pt - self.min_pt

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.min_pt + self.max_pt)


class Volume:
    """
    Represents a 3D scalar volume on a regular Cartesian grid.
    Data is stored in (Nz, Ny, Nx) order as float32 in range [0, 1] or calibrated.
    """

    def __init__(
        self,
        data: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: str = "Volume",
    ):
        """
        Parameters:
            data: 3D numpy array of shape (Nz, Ny, Nx).
            spacing: Voxel spacing (dx, dy, dz) in world units.
            origin: World coordinates of voxel (0, 0, 0).
            name: Descriptive dataset name.
        """
        if data.ndim != 3:
            raise ValueError(f"Volume data must be 3D, got shape {data.shape}")

        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.nz, self.ny, self.nx = self.data.shape
        self.spacing = np.array(spacing, dtype=np.float32)
        self.origin = np.array(origin, dtype=np.float32)
        self.name = name

        # Physical extent in world space
        extent_x = (self.nx - 1) * self.spacing[0]
        extent_y = (self.ny - 1) * self.spacing[1]
        extent_z = (self.nz - 1) * self.spacing[2]

        self.bbox = VolumeBoundingBox(
            min_pt=self.origin.copy(),
            max_pt=self.origin + np.array([extent_x, extent_y, extent_z], dtype=np.float32),
        )

        # Precompute gradient volume and normals on demand
        self._gradients: Optional[np.ndarray] = None
        self._gradient_magnitudes: Optional[np.ndarray] = None
        self._normals: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Returns (Nz, Ny, Nx)."""
        return self.nz, self.ny, self.nx

    @property
    def gradients(self) -> np.ndarray:
        if self._gradients is None:
            self.compute_gradients()
        return self._gradients

    @property
    def gradient_magnitudes(self) -> np.ndarray:
        if self._gradient_magnitudes is None:
            self.compute_gradients()
        return self._gradient_magnitudes

    @property
    def normals(self) -> np.ndarray:
        if self._normals is None:
            self.compute_gradients()
        return self._normals

    def compute_gradients(self):
        """Computes gradients using central differences."""
        from .gradients import compute_volume_gradients
        grad, mag, norm = compute_volume_gradients(self.data, self.spacing)
        self._gradients = grad
        self._gradient_magnitudes = mag
        self._normals = norm

    def center_at_origin(self):
        """Translates origin such that volume center is at (0, 0, 0)."""
        c = self.bbox.center
        self.origin -= c
        self.bbox.min_pt -= c
        self.bbox.max_pt -= c

    def normalize_to_unit_cube(self):
        """
        Normalizes world coordinate bounds so that the maximum physical extent is 2.0
        (i.e. bounds fit within [-1, 1]^3), preserving physical isotropic aspect ratio,
        and centering at (0, 0, 0).
        """
        # Current extent
        ext_x = (self.nx - 1) * float(self.spacing[0])
        ext_y = (self.ny - 1) * float(self.spacing[1])
        ext_z = (self.nz - 1) * float(self.spacing[2])
        max_extent = max(ext_x, ext_y, ext_z)
        if max_extent < 1e-6:
            max_extent = 1.0

        scale = 2.0 / max_extent
        self.spacing = self.spacing * scale

        half_x = 0.5 * (self.nx - 1) * self.spacing[0]
        half_y = 0.5 * (self.ny - 1) * self.spacing[1]
        half_z = 0.5 * (self.nz - 1) * self.spacing[2]

        self.origin = np.array([-half_x, -half_y, -half_z], dtype=np.float32)
        self.bbox = VolumeBoundingBox(
            min_pt=self.origin.copy(),
            max_pt=self.origin + np.array([(self.nx - 1) * self.spacing[0],
                                          (self.ny - 1) * self.spacing[1],
                                          (self.nz - 1) * self.spacing[2]], dtype=np.float32),
        )
        # Clear cached gradients if any
        self._gradients = None
        self._gradient_magnitudes = None
        self._normals = None

