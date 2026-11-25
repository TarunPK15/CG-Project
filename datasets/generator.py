"""
generator.py - Procedural 3D Scalar Field Benchmark Generators.
Generates:
    1. Marschner-Lobb Benchmark (1994)
"""

import numpy as np
from core.volume import Volume


def create_marschner_lobb(res: int = 64) -> Volume:
    """
    Marschner-Lobb benchmark function (Marschner & Lobb, 1994).
    Tests high-frequency spatial reconstruction filters and gradient calculation.
    """
    alpha = 0.25
    f_m = 6.0

    coords = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")

    r = np.sqrt(x * x + y * y)

    # Cosine wave filter
    rho = np.zeros_like(r)
    mask = r <= 1.0
    rho[mask] = np.cos(2.0 * np.pi * f_m * np.cos(np.pi * r[mask] * 0.5))

    val = (1.0 - np.sin(np.pi * z * 0.5) + alpha * (1.0 + rho)) / (2.0 * (1.0 + alpha))
    val = np.clip(val, 0.0, 1.0)

    # Voxel spacing so world extent is [-1, 1]^3
    spacing = 2.0 / float(res - 1)
    vol = Volume(val, spacing=(spacing, spacing, spacing), origin=(-1.0, -1.0, -1.0), name="Marschner-Lobb")
    return vol
