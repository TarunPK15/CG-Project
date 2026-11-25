"""
levoy.py - Implementation of Marc Levoy's 1988 Algorithm.
Reference:
    Marc Levoy (1988). "Display of Surfaces from Volume Data".
    IEEE Computer Graphics and Applications, 8(3), 29-37.
"""

from dataclasses import dataclass
from typing import List, Tuple
from numba import njit
import numpy as np


@dataclass
class LevoySurface:
    """Parameters for an individual isosurface in Levoy's model."""
    isovalue: float           # Target scalar value f_v (e.g. 0.3 for skin, 0.8 for bone)
    thickness: float = 1.5    # Surface thickness parameter r (in voxels)
    alpha_v: float = 0.8      # Maximum opacity alpha_v in [0, 1]
    color: np.ndarray = None  # RGB color (3,) in [0, 1]

    def __post_init__(self):
        if self.color is None:
            self.color = np.array([0.9, 0.7, 0.5], dtype=np.float32)
        else:
            self.color = np.ascontiguousarray(self.color, dtype=np.float32)


@njit(fastmath=True, inline="always")
def compute_levoy_single_surface_opacity(
    scalar_val: float,
    grad_mag: float,
    isovalue: float,
    thickness: float,
    alpha_v: float,
) -> float:
    """
    Computes Levoy's continuous isosurface opacity:
        alpha(x) = alpha_v * (1 - (1/r) * |(f_v - f(x)) / |grad f(x)||)
        when |grad f(x)| > 0 and |(f_v - f(x)) / |grad f(x)|| <= r; else 0.
    """
    if grad_mag <= 1e-5 or thickness <= 1e-5:
        return 0.0

    dist = np.abs(isovalue - scalar_val) / grad_mag
    if dist <= thickness:
        return alpha_v * (1.0 - dist / thickness)
    return 0.0


@njit(fastmath=True, inline="always")
def evaluate_levoy_sample(
    scalar_val: float,
    grad_mag: float,
    isovalues: np.ndarray,   # (M,)
    thicknesses: np.ndarray, # (M,)
    alpha_vs: np.ndarray,    # (M,)
    colors: np.ndarray,      # (M, 3)
    out_diffuse: np.ndarray, # (3,)
) -> float:
    """
    Evaluates multi-surface opacity and blended diffuse color.
    Formula:
        alpha_total = 1 - prod_m (1 - alpha_m)
        C_diffuse = sum_m (alpha_m * C_m) / (sum_m alpha_m + eps)
    """
    m_count = isovalues.shape[0]
    inv_alpha_prod = 1.0
    sum_alpha = 0.0

    out_diffuse[0] = 0.0
    out_diffuse[1] = 0.0
    out_diffuse[2] = 0.0

    for m in range(m_count):
        a_m = compute_levoy_single_surface_opacity(
            scalar_val, grad_mag, isovalues[m], thicknesses[m], alpha_vs[m]
        )
        if a_m > 0.0:
            inv_alpha_prod *= (1.0 - a_m)
            sum_alpha += a_m
            out_diffuse[0] += a_m * colors[m, 0]
            out_diffuse[1] += a_m * colors[m, 1]
            out_diffuse[2] += a_m * colors[m, 2]

    alpha_total = 1.0 - inv_alpha_prod
    if sum_alpha > 1e-6:
        inv_sum = 1.0 / sum_alpha
        out_diffuse[0] *= inv_sum
        out_diffuse[1] *= inv_sum
        out_diffuse[2] *= inv_sum

    return alpha_total


class LevoyRendererConfig:
    """Manages multi-surface parameters for Levoy's algorithm."""

    def __init__(self, surfaces: List[LevoySurface] = None):
        if surfaces is None:
            # Default preset: Dual surface (Skin + Bone)
            self.surfaces = [
                LevoySurface(isovalue=0.25, thickness=1.5, alpha_v=0.35, color=np.array([0.95, 0.75, 0.65], dtype=np.float32)), # Skin
                LevoySurface(isovalue=0.65, thickness=2.0, alpha_v=0.90, color=np.array([0.95, 0.95, 0.90], dtype=np.float32)), # Bone
            ]
        else:
            self.surfaces = surfaces

    def to_numba_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = len(self.surfaces)
        isovalues = np.zeros(m, dtype=np.float32)
        thicknesses = np.zeros(m, dtype=np.float32)
        alpha_vs = np.zeros(m, dtype=np.float32)
        colors = np.zeros((m, 3), dtype=np.float32)

        for i, s in enumerate(self.surfaces):
            isovalues[i] = s.isovalue
            thicknesses[i] = s.thickness
            alpha_vs[i] = s.alpha_v
            colors[i, :] = s.color

        return isovalues, thicknesses, alpha_vs, colors
