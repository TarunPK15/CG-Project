"""
shading.py - Phong / Blinn-Phong Illumination Model.
Implements the volume shading formulations of Levoy (1988) and Drebin et al. (1988).
"""

from dataclasses import dataclass
from numba import njit
import numpy as np


@dataclass
class LightingConfig:
    """Configurable illumination parameters."""
    light_dir: np.ndarray = None        # shape (3,), pointing towards light source
    light_color: np.ndarray = None      # shape (3,), RGB in [0, 1]
    ambient_color: np.ndarray = None    # shape (3,), RGB in [0, 1]
    ka: float = 0.2                     # Ambient reflection coefficient
    kd: float = 0.6                     # Diffuse reflection coefficient
    ks: float = 0.3                     # Specular reflection coefficient
    shininess: float = 30.0             # Specular exponent (n)
    two_sided: bool = True              # Flip normal if facing away from view direction

    def __post_init__(self):
        if self.light_dir is None:
            # Default directional light from upper-right front
            ld = np.array([0.5, 0.7, 1.0], dtype=np.float32)
            self.light_dir = ld / np.linalg.norm(ld)
        else:
            self.light_dir = np.ascontiguousarray(self.light_dir, dtype=np.float32)
            self.light_dir /= np.linalg.norm(self.light_dir)

        if self.light_color is None:
            self.light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            self.light_color = np.ascontiguousarray(self.light_color, dtype=np.float32)

        if self.ambient_color is None:
            self.ambient_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            self.ambient_color = np.ascontiguousarray(self.ambient_color, dtype=np.float32)


@njit(fastmath=True, inline="always")
def apply_blinn_phong_shading(
    diffuse_color: np.ndarray,  # (3,)
    normal: np.ndarray,         # (3,)
    view_dir: np.ndarray,       # (3,) unit vector pointing towards camera
    light_dir: np.ndarray,      # (3,) unit vector pointing towards light
    light_color: np.ndarray,    # (3,)
    ambient_color: np.ndarray,  # (3,)
    ka: float,
    kd: float,
    ks: float,
    shininess: float,
    two_sided: bool,
    out_color: np.ndarray,      # (3,)
):
    """
    Computes Blinn-Phong shaded RGB color.
    Formula:
        N' = N if (N . V >= 0) else -N (if two_sided)
        H = normalize(L + V)
        C = ka * Ia * Kd + kd * Ip * max(0, N' . L) * Kd + ks * Ip * max(0, N' . H)^n * Ks
    """
    nx = normal[0]
    ny = normal[1]
    nz = normal[2]

    # Check for degenerate normal
    n_len_sq = nx * nx + ny * ny + nz * nz
    if n_len_sq < 1e-6:
        # No surface normal: ambient illumination only
        for c in range(3):
            out_color[c] = diffuse_color[c] * (ka * ambient_color[c] + kd * 0.5)
        return

    inv_n = 1.0 / np.sqrt(n_len_sq)
    nx *= inv_n
    ny *= inv_n
    nz *= inv_n

    # Two-sided normal check relative to view direction
    dot_nv = nx * view_dir[0] + ny * view_dir[1] + nz * view_dir[2]
    if two_sided and dot_nv < 0.0:
        nx = -nx
        ny = -ny
        nz = -nz

    # Halfway vector H = normalize(L + V)
    hx = light_dir[0] + view_dir[0]
    hy = light_dir[1] + view_dir[1]
    hz = light_dir[2] + view_dir[2]
    h_len = np.sqrt(hx * hx + hy * hy + hz * hz)
    if h_len > 1e-6:
        inv_h = 1.0 / h_len
        hx *= inv_h
        hy *= inv_h
        hz *= inv_h

    # Diffuse term: max(0, N . L)
    dot_nl = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    diff = dot_nl if dot_nl > 0.0 else 0.0

    # Specular term: max(0, N . H)^shininess
    dot_nh = nx * hx + ny * hy + nz * hz
    if dot_nh > 0.0:
        spec = dot_nh ** shininess
    else:
        spec = 0.0

    for c in range(3):
        ambient = ka * ambient_color[c] * diffuse_color[c]
        diffuse = kd * light_color[c] * diffuse_color[c] * diff
        specular = ks * light_color[c] * spec
        total = ambient + diffuse + specular
        out_color[c] = 1.0 if total > 1.0 else (0.0 if total < 0.0 else total)
