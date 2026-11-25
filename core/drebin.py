"""
drebin.py - Implementation of Robert Drebin et al.'s 1988 Algorithm.
Reference:
    Robert A. Drebin, Loren Carpenter, Pat Hanrahan (1988). "Volume Rendering".
    ACM SIGGRAPH Computer Graphics, 22(4), 65-74.
"""

from dataclasses import dataclass
from typing import List, Tuple
from numba import njit
import numpy as np


@dataclass
class DrebinMaterial:
    """Represents a material classification with trapezoidal membership."""
    name: str
    s_min: float               # Start of transition in
    s_peak_start: float        # Start of 100% membership
    s_peak_end: float          # End of 100% membership
    s_max: float               # End of transition out
    color: np.ndarray          # RGB color (3,) in [0, 1]
    opacity: float             # Base material opacity alpha_i in [0, 1]
    density_rho: float         # Density characteristic rho_i (e.g. Air=0, Tissue=1, Bone=2)


@njit(fastmath=True, inline="always")
def trapezoid_membership(s: float, s_min: float, sp_start: float, sp_end: float, s_max: float) -> float:
    """Computes continuous trapezoidal membership function in [0, 1]."""
    if s <= s_min or s >= s_max:
        return 0.0
    if s >= sp_start and s <= sp_end:
        return 1.0
    if s < sp_start:
        denom = sp_start - s_min
        return (s - s_min) / denom if denom > 1e-6 else 1.0
    else: # s > sp_end
        denom = s_max - sp_end
        return (s_max - s) / denom if denom > 1e-6 else 1.0


@njit(fastmath=True, inline="always")
def evaluate_drebin_classification(
    scalar_val: float,
    mat_params: np.ndarray,  # shape (M, 4): [s_min, sp_start, sp_end, s_max]
    mat_colors: np.ndarray,  # shape (M, 3): RGB
    mat_opacities: np.ndarray, # shape (M,): base opacities
    mat_rhos: np.ndarray,    # shape (M,): rho density weights
    out_percentages: np.ndarray, # shape (M,)
    out_diffuse_color: np.ndarray, # shape (3,)
) -> Tuple[float, float]:
    """
    Computes material percentage fractions, composite volumetric diffuse color,
    composite volumetric opacity, and effective density rho.
    Returns:
        (composite_opacity, effective_rho)
    """
    m_count = mat_params.shape[0]
    total_pct = 0.0

    # 1. Compute unnormalized percentages
    for m in range(m_count):
        p = trapezoid_membership(
            scalar_val,
            mat_params[m, 0],
            mat_params[m, 1],
            mat_params[m, 2],
            mat_params[m, 3],
        )
        out_percentages[m] = p
        total_pct += p

    # Normalize if total exceeds 1.0 to preserve conservation of volume
    if total_pct > 1.0:
        inv_tot = 1.0 / total_pct
        for m in range(m_count):
            out_percentages[m] *= inv_tot

    # 2. Accumulate composite color, opacity, and density
    comp_r = 0.0
    comp_g = 0.0
    comp_b = 0.0
    comp_alpha = 0.0
    comp_rho = 0.0

    for m in range(m_count):
        pct = out_percentages[m]
        comp_r += pct * mat_colors[m, 0]
        comp_g += pct * mat_colors[m, 1]
        comp_b += pct * mat_colors[m, 2]
        comp_alpha += pct * mat_opacities[m]
        comp_rho += pct * mat_rhos[m]

    out_diffuse_color[0] = comp_r
    out_diffuse_color[1] = comp_g
    out_diffuse_color[2] = comp_b

    return comp_alpha, comp_rho


@njit(fastmath=True, inline="always")
def evaluate_matte_mask(
    world_x: float,
    world_y: float,
    world_z: float,
    clip_plane_nx: float,
    clip_plane_ny: float,
    clip_plane_nz: float,
    clip_plane_d: float,
    wedge_cut: bool,
) -> float:
    """
    Drebin's Matte Volume evaluation:
    M(x, y, z) = 0 inside cutaway region, 1 outside.
    Supports clipping plane (nx*x + ny*y + nz*z - d >= 0) and 3D wedge cut-away.
    """
    # Clipping plane check
    dot = world_x * clip_plane_nx + world_y * clip_plane_ny + world_z * clip_plane_nz - clip_plane_d
    if dot < 0.0:
        return 0.0

    # Drebin 3D wedge cut-away (e.g. cutting front-right quadrant to reveal internal organs)
    if wedge_cut:
        if world_x > 0.0 and world_y > 0.0 and world_z > 0.0:
            return 0.0

    return 1.0


class DrebinClassifier:
    """Manages material list and produces arrays for JIT ray tracing."""

    def __init__(self, materials: List[DrebinMaterial] = None):
        if materials is None:
            # Default medical CT classification: Air, Soft Tissue / Muscle, Bone
            self.materials = [
                DrebinMaterial(
                    name="Air/Sinus",
                    s_min=-0.1, s_peak_start=-0.05, s_peak_end=0.10, s_max=0.18,
                    color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    opacity=0.0, density_rho=0.0,
                ),
                DrebinMaterial(
                    name="Soft Tissue/Muscle",
                    s_min=0.15, s_peak_start=0.25, s_peak_end=0.45, s_max=0.55,
                    color=np.array([0.85, 0.35, 0.35], dtype=np.float32), # Reddish/pink tissue
                    opacity=0.25, density_rho=1.0,
                ),
                DrebinMaterial(
                    name="Fat/Intermediate",
                    s_min=0.42, s_peak_start=0.48, s_peak_end=0.58, s_max=0.68,
                    color=np.array([0.95, 0.85, 0.50], dtype=np.float32), # Yellowish fat
                    opacity=0.35, density_rho=1.3,
                ),
                DrebinMaterial(
                    name="Dense Bone/Teeth",
                    s_min=0.60, s_peak_start=0.72, s_peak_end=1.05, s_max=1.10,
                    color=np.array([0.98, 0.96, 0.92], dtype=np.float32), # Ivory bone
                    opacity=0.95, density_rho=3.0,
                ),
            ]
        else:
            self.materials = materials

    def to_numba_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = len(self.materials)
        params = np.zeros((m, 4), dtype=np.float32)
        colors = np.zeros((m, 3), dtype=np.float32)
        opacities = np.zeros(m, dtype=np.float32)
        rhos = np.zeros(m, dtype=np.float32)

        for i, mat in enumerate(self.materials):
            params[i, :] = [mat.s_min, mat.s_peak_start, mat.s_peak_end, mat.s_max]
            colors[i, :] = mat.color
            opacities[i] = mat.opacity
            rhos[i] = mat.density_rho

        return params, colors, opacities, rhos
