"""
raycaster.py - Unified High-Performance Parallel Ray Caster.
Specialized kernels for:
    - Marc Levoy (1988) Isosurface Ray Casting
    - Robert Drebin et al. (1988) Material Mixture Classification
    - Maximum Intensity Projection (MIP)
    - Back-to-Front and Front-to-Back Compositing with Early Ray Termination
    - 3D Matte Cutaways and Arbitrary Clipping Planes
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple
from numba import njit, prange
import numpy as np

from .volume import Volume
from .camera import Camera, generate_camera_rays
from .interpolation import sample_trilinear_scalar, sample_trilinear_vec3
from .shading import apply_blinn_phong_shading, LightingConfig
from .levoy import (
    evaluate_levoy_sample,
    LevoyRendererConfig,
)
from .drebin import (
    evaluate_drebin_classification,
    evaluate_matte_mask,
    DrebinClassifier,
)


class RenderMethod(IntEnum):
    LEVOY = 0
    DREBIN = 1
    MIP = 2


class CompositingMode(IntEnum):
    BACK_TO_FRONT = 0   # Proposal specification
    FRONT_TO_BACK = 1   # Early ray termination optimization


@dataclass
class RaycastSettings:
    """Ray casting configuration settings."""
    method: RenderMethod = RenderMethod.LEVOY
    compositing: CompositingMode = CompositingMode.FRONT_TO_BACK
    step_size: float = 0.5          # Sampling step in voxel units (0.5 = Nyquist sampling)
    bg_color: np.ndarray = None     # RGBA background
    early_term_alpha: float = 0.99  # Early ray termination threshold (front-to-back)
    
    # Matte clipping plane: nx*x + ny*y + nz*z - d >= 0
    clip_plane_normal: np.ndarray = None
    clip_plane_d: float = -9999.0
    wedge_cut: bool = False

    def __post_init__(self):
        if self.bg_color is None:
            self.bg_color = np.array([0.05, 0.07, 0.10, 1.0], dtype=np.float32)
        else:
            self.bg_color = np.ascontiguousarray(self.bg_color, dtype=np.float32)

        if self.clip_plane_normal is None:
            self.clip_plane_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            self.clip_plane_normal = np.ascontiguousarray(self.clip_plane_normal, dtype=np.float32)
            norm = np.linalg.norm(self.clip_plane_normal)
            if norm > 1e-6:
                self.clip_plane_normal /= norm


@njit(fastmath=True, inline="always")
def intersect_ray_box(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    min_x: float, min_y: float, min_z: float,
    max_x: float, max_y: float, max_z: float,
) -> Tuple[bool, float, float]:
    """Slab method for ray - AABB intersection."""
    inv_dx = 1.0 / dx if np.abs(dx) > 1e-8 else (1e8 if dx >= 0.0 else -1e8)
    inv_dy = 1.0 / dy if np.abs(dy) > 1e-8 else (1e8 if dy >= 0.0 else -1e8)
    inv_dz = 1.0 / dz if np.abs(dz) > 1e-8 else (1e8 if dz >= 0.0 else -1e8)

    t1_x = (min_x - ox) * inv_dx
    t2_x = (max_x - ox) * inv_dx
    t_min_x = min(t1_x, t2_x)
    t_max_x = max(t1_x, t2_x)

    t1_y = (min_y - oy) * inv_dy
    t2_y = (max_y - oy) * inv_dy
    t_min_y = min(t1_y, t2_y)
    t_max_y = max(t1_y, t2_y)

    t1_z = (min_z - oz) * inv_dz
    t2_z = (max_z - oz) * inv_dz
    t_min_z = min(t1_z, t2_z)
    t_max_z = max(t1_z, t2_z)

    t_enter = max(max(t_min_x, t_min_y), t_min_z)
    t_exit = min(min(t_max_x, t_max_y), t_max_z)

    t_enter = max(0.0, t_enter)
    hit = t_enter < t_exit
    return hit, t_enter, t_exit


@njit(parallel=True, fastmath=True, cache=True)
def _render_mip_kernel(
    width: int,
    height: int,
    ray_origins: np.ndarray,      # (H, W, 3)
    ray_directions: np.ndarray,   # (H, W, 3)
    volume_data: np.ndarray,      # (Nz, Ny, Nx)
    bbox_min: np.ndarray,         # (3,)
    bbox_max: np.ndarray,         # (3,)
    origin: np.ndarray,           # (3,)
    inv_spacing: np.ndarray,      # (3,)
    step_world: float,
    bg_rgba: np.ndarray,          # (4,)
    clip_plane_normal: np.ndarray,# (3,)
    clip_plane_d: float,
    wedge_cut: bool,
    out_image: np.ndarray,        # (H, W, 4)
):
    nz, ny, nx = volume_data.shape
    inv_dx = inv_spacing[0]
    inv_dy = inv_spacing[1]
    inv_dz = inv_spacing[2]

    for j in prange(height):
        for i in range(width):
            ox = ray_origins[j, i, 0]
            oy = ray_origins[j, i, 1]
            oz = ray_origins[j, i, 2]
            dx = ray_directions[j, i, 0]
            dy = ray_directions[j, i, 1]
            dz = ray_directions[j, i, 2]

            hit, t_enter, t_exit = intersect_ray_box(
                ox, oy, oz, dx, dy, dz,
                bbox_min[0], bbox_min[1], bbox_min[2],
                bbox_max[0], bbox_max[1], bbox_max[2],
            )

            if not hit:
                out_image[j, i, 0] = bg_rgba[0]
                out_image[j, i, 1] = bg_rgba[1]
                out_image[j, i, 2] = bg_rgba[2]
                out_image[j, i, 3] = bg_rgba[3]
                continue

            ray_len = t_exit - t_enter
            num_steps = int(ray_len / step_world) + 1
            max_val = 0.0

            for s in range(num_steps):
                t = t_enter + float(s) * step_world
                wx = ox + t * dx
                wy = oy + t * dy
                wz = oz + t * dz

                if clip_plane_d > -9000.0 or wedge_cut:
                    m_mask = evaluate_matte_mask(
                        wx, wy, wz,
                        clip_plane_normal[0], clip_plane_normal[1], clip_plane_normal[2],
                        clip_plane_d, wedge_cut
                    )
                    if m_mask < 0.5:
                        continue

                vx = (wx - origin[0]) * inv_dx
                vy = (wy - origin[1]) * inv_dy
                vz = (wz - origin[2]) * inv_dz

                val = sample_trilinear_scalar(volume_data, vx, vy, vz, nx, ny, nz)
                if val > max_val:
                    max_val = val
                    if max_val >= 0.99:
                        break

            val_c = min(1.0, max(0.0, max_val))
            out_image[j, i, 0] = val_c
            out_image[j, i, 1] = val_c
            out_image[j, i, 2] = val_c
            out_image[j, i, 3] = 1.0


@njit(parallel=True, fastmath=True, cache=True)
def _render_levoy_kernel(
    width: int,
    height: int,
    ray_origins: np.ndarray,      # (H, W, 3)
    ray_directions: np.ndarray,   # (H, W, 3)
    volume_data: np.ndarray,      # (Nz, Ny, Nx)
    normals_data: np.ndarray,     # (Nz, Ny, Nx, 3)
    mag_data: np.ndarray,         # (Nz, Ny, Nx)
    bbox_min: np.ndarray,         # (3,)
    bbox_max: np.ndarray,         # (3,)
    origin: np.ndarray,           # (3,)
    inv_spacing: np.ndarray,      # (3,)
    compositing_mode: int,        # 0: Back-to-Front, 1: Front-to-Back
    step_world: float,
    bg_rgba: np.ndarray,          # (4,)
    early_term_alpha: float,
    light_dir: np.ndarray,        # (3,)
    light_color: np.ndarray,      # (3,)
    ambient_color: np.ndarray,    # (3,)
    ka: float, kd: float, ks: float, shininess: float, two_sided: bool,
    clip_plane_normal: np.ndarray,# (3,)
    clip_plane_d: float,
    wedge_cut: bool,
    levoy_isovalues: np.ndarray,
    levoy_thicknesses: np.ndarray,
    levoy_alpha_vs: np.ndarray,
    levoy_colors: np.ndarray,
    out_image: np.ndarray,        # (H, W, 4)
):
    nz, ny, nx = volume_data.shape
    inv_dx = inv_spacing[0]
    inv_dy = inv_spacing[1]
    inv_dz = inv_spacing[2]

    for j in prange(height):
        sampled_normal = np.zeros(3, dtype=np.float32)
        diffuse_color = np.zeros(3, dtype=np.float32)
        shaded_color = np.zeros(3, dtype=np.float32)

        for i in range(width):
            ox = ray_origins[j, i, 0]
            oy = ray_origins[j, i, 1]
            oz = ray_origins[j, i, 2]
            dx = ray_directions[j, i, 0]
            dy = ray_directions[j, i, 1]
            dz = ray_directions[j, i, 2]

            hit, t_enter, t_exit = intersect_ray_box(
                ox, oy, oz, dx, dy, dz,
                bbox_min[0], bbox_min[1], bbox_min[2],
                bbox_max[0], bbox_max[1], bbox_max[2],
            )

            if not hit:
                out_image[j, i, 0] = bg_rgba[0]
                out_image[j, i, 1] = bg_rgba[1]
                out_image[j, i, 2] = bg_rgba[2]
                out_image[j, i, 3] = bg_rgba[3]
                continue

            ray_len = t_exit - t_enter
            num_steps = int(ray_len / step_world) + 1
            view_dir = np.array([-dx, -dy, -dz], dtype=np.float32)

            if compositing_mode == 1:
                # Front-to-Back with Early Ray Termination
                accum_r = 0.0
                accum_g = 0.0
                accum_b = 0.0
                accum_a = 0.0

                for s in range(num_steps):
                    t = t_enter + float(s) * step_world
                    wx = ox + t * dx
                    wy = oy + t * dy
                    wz = oz + t * dz

                    if clip_plane_d > -9000.0 or wedge_cut:
                        m_mask = evaluate_matte_mask(
                            wx, wy, wz,
                            clip_plane_normal[0], clip_plane_normal[1], clip_plane_normal[2],
                            clip_plane_d, wedge_cut
                        )
                        if m_mask < 0.5:
                            continue

                    vx = (wx - origin[0]) * inv_dx
                    vy = (wy - origin[1]) * inv_dy
                    vz = (wz - origin[2]) * inv_dz

                    scalar_val = sample_trilinear_scalar(volume_data, vx, vy, vz, nx, ny, nz)
                    if scalar_val < 0.005:
                        continue

                    grad_mag = sample_trilinear_scalar(mag_data, vx, vy, vz, nx, ny, nz)
                    alpha_sample = evaluate_levoy_sample(
                        scalar_val, grad_mag,
                        levoy_isovalues, levoy_thicknesses, levoy_alpha_vs, levoy_colors,
                        diffuse_color
                    )

                    if alpha_sample > 0.001:
                        sample_trilinear_vec3(normals_data, vx, vy, vz, nx, ny, nz, sampled_normal)
                        apply_blinn_phong_shading(
                            diffuse_color, sampled_normal, view_dir,
                            light_dir, light_color, ambient_color,
                            ka, kd, ks, shininess, two_sided,
                            shaded_color
                        )

                        weight = (1.0 - accum_a) * alpha_sample
                        accum_r += weight * shaded_color[0]
                        accum_g += weight * shaded_color[1]
                        accum_b += weight * shaded_color[2]
                        accum_a += weight

                        if accum_a >= early_term_alpha:
                            break

                final_r = accum_r + (1.0 - accum_a) * bg_rgba[0]
                final_g = accum_g + (1.0 - accum_a) * bg_rgba[1]
                final_b = accum_b + (1.0 - accum_a) * bg_rgba[2]
                final_a = accum_a + (1.0 - accum_a) * bg_rgba[3]

                out_image[j, i, 0] = min(1.0, max(0.0, final_r))
                out_image[j, i, 1] = min(1.0, max(0.0, final_g))
                out_image[j, i, 2] = min(1.0, max(0.0, final_b))
                out_image[j, i, 3] = min(1.0, max(0.0, final_a))

            else:
                # Back-to-Front Compositing
                accum_r = 0.0
                accum_g = 0.0
                accum_b = 0.0
                accum_a = 0.0

                for s in range(num_steps - 1, -1, -1):
                    t = t_enter + float(s) * step_world
                    wx = ox + t * dx
                    wy = oy + t * dy
                    wz = oz + t * dz

                    if clip_plane_d > -9000.0 or wedge_cut:
                        m_mask = evaluate_matte_mask(
                            wx, wy, wz,
                            clip_plane_normal[0], clip_plane_normal[1], clip_plane_normal[2],
                            clip_plane_d, wedge_cut
                        )
                        if m_mask < 0.5:
                            continue

                    vx = (wx - origin[0]) * inv_dx
                    vy = (wy - origin[1]) * inv_dy
                    vz = (wz - origin[2]) * inv_dz

                    scalar_val = sample_trilinear_scalar(volume_data, vx, vy, vz, nx, ny, nz)
                    if scalar_val < 0.005:
                        continue

                    grad_mag = sample_trilinear_scalar(mag_data, vx, vy, vz, nx, ny, nz)
                    alpha_sample = evaluate_levoy_sample(
                        scalar_val, grad_mag,
                        levoy_isovalues, levoy_thicknesses, levoy_alpha_vs, levoy_colors,
                        diffuse_color
                    )

                    if alpha_sample > 0.001:
                        sample_trilinear_vec3(normals_data, vx, vy, vz, nx, ny, nz, sampled_normal)
                        apply_blinn_phong_shading(
                            diffuse_color, sampled_normal, view_dir,
                            light_dir, light_color, ambient_color,
                            ka, kd, ks, shininess, two_sided,
                            shaded_color
                        )

                        accum_r = shaded_color[0] * alpha_sample + (1.0 - alpha_sample) * accum_r
                        accum_g = shaded_color[1] * alpha_sample + (1.0 - alpha_sample) * accum_g
                        accum_b = shaded_color[2] * alpha_sample + (1.0 - alpha_sample) * accum_b
                        accum_a = alpha_sample + (1.0 - alpha_sample) * accum_a

                final_r = accum_r + (1.0 - accum_a) * bg_rgba[0]
                final_g = accum_g + (1.0 - accum_a) * bg_rgba[1]
                final_b = accum_b + (1.0 - accum_a) * bg_rgba[2]
                final_a = accum_a + (1.0 - accum_a) * bg_rgba[3]

                out_image[j, i, 0] = min(1.0, max(0.0, final_r))
                out_image[j, i, 1] = min(1.0, max(0.0, final_g))
                out_image[j, i, 2] = min(1.0, max(0.0, final_b))
                out_image[j, i, 3] = min(1.0, max(0.0, final_a))


@njit(parallel=True, fastmath=True, cache=True)
def _render_drebin_kernel(
    width: int,
    height: int,
    ray_origins: np.ndarray,      # (H, W, 3)
    ray_directions: np.ndarray,   # (H, W, 3)
    volume_data: np.ndarray,      # (Nz, Ny, Nx)
    normals_data: np.ndarray,     # (Nz, Ny, Nx, 3)
    mag_data: np.ndarray,         # (Nz, Ny, Nx)
    bbox_min: np.ndarray,         # (3,)
    bbox_max: np.ndarray,         # (3,)
    origin: np.ndarray,           # (3,)
    inv_spacing: np.ndarray,      # (3,)
    compositing_mode: int,        # 0: Back-to-Front, 1: Front-to-Back
    step_world: float,
    bg_rgba: np.ndarray,          # (4,)
    early_term_alpha: float,
    light_dir: np.ndarray,        # (3,)
    light_color: np.ndarray,      # (3,)
    ambient_color: np.ndarray,    # (3,)
    ka: float, kd: float, ks: float, shininess: float, two_sided: bool,
    clip_plane_normal: np.ndarray,# (3,)
    clip_plane_d: float,
    wedge_cut: bool,
    drebin_params: np.ndarray,
    drebin_colors: np.ndarray,
    drebin_opacities: np.ndarray,
    drebin_rhos: np.ndarray,
    out_image: np.ndarray,        # (H, W, 4)
):
    nz, ny, nx = volume_data.shape
    inv_dx = inv_spacing[0]
    inv_dy = inv_spacing[1]
    inv_dz = inv_spacing[2]
    m_drebin_count = drebin_params.shape[0]

    for j in prange(height):
        sampled_normal = np.zeros(3, dtype=np.float32)
        diffuse_color = np.zeros(3, dtype=np.float32)
        shaded_color = np.zeros(3, dtype=np.float32)
        drebin_percentages = np.zeros(m_drebin_count, dtype=np.float32)

        for i in range(width):
            ox = ray_origins[j, i, 0]
            oy = ray_origins[j, i, 1]
            oz = ray_origins[j, i, 2]
            dx = ray_directions[j, i, 0]
            dy = ray_directions[j, i, 1]
            dz = ray_directions[j, i, 2]

            hit, t_enter, t_exit = intersect_ray_box(
                ox, oy, oz, dx, dy, dz,
                bbox_min[0], bbox_min[1], bbox_min[2],
                bbox_max[0], bbox_max[1], bbox_max[2],
            )

            if not hit:
                out_image[j, i, 0] = bg_rgba[0]
                out_image[j, i, 1] = bg_rgba[1]
                out_image[j, i, 2] = bg_rgba[2]
                out_image[j, i, 3] = bg_rgba[3]
                continue

            ray_len = t_exit - t_enter
            num_steps = int(ray_len / step_world) + 1
            view_dir = np.array([-dx, -dy, -dz], dtype=np.float32)

            if compositing_mode == 1:
                # Front-to-Back with Early Ray Termination
                accum_r = 0.0
                accum_g = 0.0
                accum_b = 0.0
                accum_a = 0.0

                for s in range(num_steps):
                    t = t_enter + float(s) * step_world
                    wx = ox + t * dx
                    wy = oy + t * dy
                    wz = oz + t * dz

                    if clip_plane_d > -9000.0 or wedge_cut:
                        m_mask = evaluate_matte_mask(
                            wx, wy, wz,
                            clip_plane_normal[0], clip_plane_normal[1], clip_plane_normal[2],
                            clip_plane_d, wedge_cut
                        )
                        if m_mask < 0.5:
                            continue

                    vx = (wx - origin[0]) * inv_dx
                    vy = (wy - origin[1]) * inv_dy
                    vz = (wz - origin[2]) * inv_dz

                    scalar_val = sample_trilinear_scalar(volume_data, vx, vy, vz, nx, ny, nz)
                    if scalar_val < 0.005:
                        continue

                    grad_mag = sample_trilinear_scalar(mag_data, vx, vy, vz, nx, ny, nz)
                    comp_alpha, comp_rho = evaluate_drebin_classification(
                        scalar_val, drebin_params, drebin_colors, drebin_opacities, drebin_rhos,
                        drebin_percentages, diffuse_color
                    )

                    # Drebin surface strength weighting: S proportional to grad magnitude
                    surface_strength = grad_mag
                    alpha_sample = min(1.0, comp_alpha * (0.4 + 1.6 * surface_strength))

                    if alpha_sample > 0.001:
                        sample_trilinear_vec3(normals_data, vx, vy, vz, nx, ny, nz, sampled_normal)
                        apply_blinn_phong_shading(
                            diffuse_color, sampled_normal, view_dir,
                            light_dir, light_color, ambient_color,
                            ka, kd, ks, shininess, two_sided,
                            shaded_color
                        )

                        weight = (1.0 - accum_a) * alpha_sample
                        accum_r += weight * shaded_color[0]
                        accum_g += weight * shaded_color[1]
                        accum_b += weight * shaded_color[2]
                        accum_a += weight

                        if accum_a >= early_term_alpha:
                            break

                final_r = accum_r + (1.0 - accum_a) * bg_rgba[0]
                final_g = accum_g + (1.0 - accum_a) * bg_rgba[1]
                final_b = accum_b + (1.0 - accum_a) * bg_rgba[2]
                final_a = accum_a + (1.0 - accum_a) * bg_rgba[3]

                out_image[j, i, 0] = min(1.0, max(0.0, final_r))
                out_image[j, i, 1] = min(1.0, max(0.0, final_g))
                out_image[j, i, 2] = min(1.0, max(0.0, final_b))
                out_image[j, i, 3] = min(1.0, max(0.0, final_a))

            else:
                # Back-to-Front Compositing
                accum_r = 0.0
                accum_g = 0.0
                accum_b = 0.0
                accum_a = 0.0

                for s in range(num_steps - 1, -1, -1):
                    t = t_enter + float(s) * step_world
                    wx = ox + t * dx
                    wy = oy + t * dy
                    wz = oz + t * dz

                    if clip_plane_d > -9000.0 or wedge_cut:
                        m_mask = evaluate_matte_mask(
                            wx, wy, wz,
                            clip_plane_normal[0], clip_plane_normal[1], clip_plane_normal[2],
                            clip_plane_d, wedge_cut
                        )
                        if m_mask < 0.5:
                            continue

                    vx = (wx - origin[0]) * inv_dx
                    vy = (wy - origin[1]) * inv_dy
                    vz = (wz - origin[2]) * inv_dz

                    scalar_val = sample_trilinear_scalar(volume_data, vx, vy, vz, nx, ny, nz)
                    if scalar_val < 0.005:
                        continue

                    grad_mag = sample_trilinear_scalar(mag_data, vx, vy, vz, nx, ny, nz)
                    comp_alpha, comp_rho = evaluate_drebin_classification(
                        scalar_val, drebin_params, drebin_colors, drebin_opacities, drebin_rhos,
                        drebin_percentages, diffuse_color
                    )

                    surface_strength = grad_mag
                    alpha_sample = min(1.0, comp_alpha * (0.4 + 1.6 * surface_strength))

                    if alpha_sample > 0.001:
                        sample_trilinear_vec3(normals_data, vx, vy, vz, nx, ny, nz, sampled_normal)
                        apply_blinn_phong_shading(
                            diffuse_color, sampled_normal, view_dir,
                            light_dir, light_color, ambient_color,
                            ka, kd, ks, shininess, two_sided,
                            shaded_color
                        )

                        accum_r = shaded_color[0] * alpha_sample + (1.0 - alpha_sample) * accum_r
                        accum_g = shaded_color[1] * alpha_sample + (1.0 - alpha_sample) * accum_g
                        accum_b = shaded_color[2] * alpha_sample + (1.0 - alpha_sample) * accum_b
                        accum_a = alpha_sample + (1.0 - alpha_sample) * accum_a

                final_r = accum_r + (1.0 - accum_a) * bg_rgba[0]
                final_g = accum_g + (1.0 - accum_a) * bg_rgba[1]
                final_b = accum_b + (1.0 - accum_a) * bg_rgba[2]
                final_a = accum_a + (1.0 - accum_a) * bg_rgba[3]

                out_image[j, i, 0] = min(1.0, max(0.0, final_r))
                out_image[j, i, 1] = min(1.0, max(0.0, final_g))
                out_image[j, i, 2] = min(1.0, max(0.0, final_b))
                out_image[j, i, 3] = min(1.0, max(0.0, final_a))


class VolumeRenderer:
    """High-level rendering coordinator."""

    def __init__(
        self,
        volume: Volume,
        settings: Optional[RaycastSettings] = None,
        lighting: Optional[LightingConfig] = None,
        levoy_config: Optional[LevoyRendererConfig] = None,
        drebin_classifier: Optional[DrebinClassifier] = None,
    ):
        self.volume = volume
        self.settings = settings if settings is not None else RaycastSettings()
        self.lighting = lighting if lighting is not None else LightingConfig()
        self.levoy_config = levoy_config if levoy_config is not None else LevoyRendererConfig()
        self.drebin_classifier = drebin_classifier if drebin_classifier is not None else DrebinClassifier()

    def render(self, camera: Camera, width: int = 400, height: int = 400) -> np.ndarray:
        """
        Renders an RGBA image using the current configuration.
        Returns uint8 RGBA image of shape (height, width, 4).
        """
        eye, forward, right, up = camera.get_view_basis()

        ray_origins = np.zeros((height, width, 3), dtype=np.float32)
        ray_directions = np.zeros((height, width, 3), dtype=np.float32)

        generate_camera_rays(
            width, height, eye, forward, right, up,
            camera.fov_y, camera.is_perspective,
            ray_origins, ray_directions
        )

        out_float = np.zeros((height, width, 4), dtype=np.float32)
        inv_spacing = np.array([1.0 / self.volume.spacing[0],
                                1.0 / self.volume.spacing[1],
                                1.0 / self.volume.spacing[2]], dtype=np.float32)

        step_world = float(self.settings.step_size * np.min(self.volume.spacing))

        if self.settings.method == RenderMethod.MIP:
            _render_mip_kernel(
                width, height,
                ray_origins, ray_directions,
                self.volume.data,
                self.volume.bbox.min_pt,
                self.volume.bbox.max_pt,
                self.volume.origin,
                inv_spacing,
                step_world,
                self.settings.bg_color,
                self.settings.clip_plane_normal,
                float(self.settings.clip_plane_d),
                bool(self.settings.wedge_cut),
                out_float,
            )
        elif self.settings.method == RenderMethod.LEVOY:
            levoy_iso, levoy_th, levoy_av, levoy_col = self.levoy_config.to_numba_arrays()
            _render_levoy_kernel(
                width, height,
                ray_origins, ray_directions,
                self.volume.data,
                self.volume.normals,
                self.volume.gradient_magnitudes,
                self.volume.bbox.min_pt,
                self.volume.bbox.max_pt,
                self.volume.origin,
                inv_spacing,
                int(self.settings.compositing),
                step_world,
                self.settings.bg_color,
                float(self.settings.early_term_alpha),
                self.lighting.light_dir,
                self.lighting.light_color,
                self.lighting.ambient_color,
                float(self.lighting.ka),
                float(self.lighting.kd),
                float(self.lighting.ks),
                float(self.lighting.shininess),
                bool(self.lighting.two_sided),
                self.settings.clip_plane_normal,
                float(self.settings.clip_plane_d),
                bool(self.settings.wedge_cut),
                levoy_iso, levoy_th, levoy_av, levoy_col,
                out_float,
            )
        else: # DREBIN
            drebin_p, drebin_c, drebin_o, drebin_r = self.drebin_classifier.to_numba_arrays()
            _render_drebin_kernel(
                width, height,
                ray_origins, ray_directions,
                self.volume.data,
                self.volume.normals,
                self.volume.gradient_magnitudes,
                self.volume.bbox.min_pt,
                self.volume.bbox.max_pt,
                self.volume.origin,
                inv_spacing,
                int(self.settings.compositing),
                step_world,
                self.settings.bg_color,
                float(self.settings.early_term_alpha),
                self.lighting.light_dir,
                self.lighting.light_color,
                self.lighting.ambient_color,
                float(self.lighting.ka),
                float(self.lighting.kd),
                float(self.lighting.ks),
                float(self.lighting.shininess),
                bool(self.lighting.two_sided),
                self.settings.clip_plane_normal,
                float(self.settings.clip_plane_d),
                bool(self.settings.wedge_cut),
                drebin_p, drebin_c, drebin_o, drebin_r,
                out_float,
            )

        return (np.clip(out_float, 0.0, 1.0) * 255.0).astype(np.uint8)
