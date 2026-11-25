"""
transfer_function.py - 1D Transfer Function representation and color mapping.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class TFControlPoint:
    """Control point for 1D transfer function piecewise linear interpolation."""
    scalar_val: float  # Scalar coordinate in [0, 1]
    color_rgb: Tuple[float, float, float]  # RGB in [0, 1]
    opacity: float     # Alpha in [0, 1]


class TransferFunction1D:
    """
    1D Transfer Function mapping scalar value -> RGBA.
    """

    def __init__(self, points: List[TFControlPoint] = None):
        if points is None:
            self.points = [
                TFControlPoint(0.0, (0.0, 0.0, 0.0), 0.0),
                TFControlPoint(0.2, (0.9, 0.6, 0.5), 0.2),
                TFControlPoint(0.6, (0.9, 0.8, 0.7), 0.8),
                TFControlPoint(1.0, (1.0, 1.0, 1.0), 1.0),
            ]
        else:
            self.points = sorted(points, key=lambda p: p.scalar_val)

    def evaluate_lut(self, resolution: int = 256) -> np.ndarray:
        """
        Samples the transfer function into a Look-Up Table (LUT) of shape (resolution, 4).
        """
        scalars = np.array([p.scalar_val for p in self.points], dtype=np.float32)
        r_vals = np.array([p.color_rgb[0] for p in self.points], dtype=np.float32)
        g_vals = np.array([p.color_rgb[1] for p in self.points], dtype=np.float32)
        b_vals = np.array([p.color_rgb[2] for p in self.points], dtype=np.float32)
        a_vals = np.array([p.opacity for p in self.points], dtype=np.float32)

        lut_scalars = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
        lut_r = np.interp(lut_scalars, scalars, r_vals)
        lut_g = np.interp(lut_scalars, scalars, g_vals)
        lut_b = np.interp(lut_scalars, scalars, b_vals)
        lut_a = np.interp(lut_scalars, scalars, a_vals)

        lut = np.stack([lut_r, lut_g, lut_b, lut_a], axis=1).astype(np.float32)
        return lut
