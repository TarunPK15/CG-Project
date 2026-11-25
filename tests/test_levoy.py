"""
test_levoy.py - Unit test for Marc Levoy (1988) Isosurface Formulation.
"""

import numpy as np
from core.levoy import (
    compute_levoy_single_surface_opacity,
    evaluate_levoy_sample,
)


def test_levoy_exact_isovalue():
    """At f(x) = f_v, opacity must equal alpha_v exactly."""
    isovalue = 0.5
    thickness = 2.0
    alpha_v = 0.8
    grad_mag = 1.0

    a = compute_levoy_single_surface_opacity(isovalue, grad_mag, isovalue, thickness, alpha_v)
    np.testing.assert_allclose(a, alpha_v, atol=1e-5)


def test_levoy_linear_falloff():
    """Opacity must decrease linearly with distance to isovalue."""
    isovalue = 0.5
    thickness = 2.0
    alpha_v = 0.8
    grad_mag = 1.0

    # Halfway (dist = 1.0, r = 2.0) -> (1 - 0.5) * alpha_v = 0.4
    a_half = compute_levoy_single_surface_opacity(0.5 + 1.0, grad_mag, isovalue, thickness, alpha_v)
    np.testing.assert_allclose(a_half, 0.4, atol=1e-5)

    # Beyond thickness -> 0.0
    a_out = compute_levoy_single_surface_opacity(0.5 + 2.5, grad_mag, isovalue, thickness, alpha_v)
    assert a_out == 0.0


def test_levoy_multi_surface_combination():
    """Tests multi-surface combining rule: alpha = 1 - prod(1 - a_m)."""
    isovalues = np.array([0.3, 0.7], dtype=np.float32)
    thicknesses = np.array([1.0, 1.0], dtype=np.float32)
    alpha_vs = np.array([0.5, 0.5], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    diffuse_out = np.zeros(3, dtype=np.float32)

    # Test sample where both surfaces overlap equally (e.g. at 0.5 where dist=0.2 from both with grad=1)
    # a1 = 0.5 * (1 - 0.2/1.0) = 0.4
    # a2 = 0.5 * (1 - 0.2/1.0) = 0.4
    # alpha_total = 1 - (1 - 0.4)*(1 - 0.4) = 1 - 0.36 = 0.64
    a_comb = evaluate_levoy_sample(0.5, 1.0, isovalues, thicknesses, alpha_vs, colors, diffuse_out)
    np.testing.assert_allclose(a_comb, 0.64, atol=1e-5)
    np.testing.assert_allclose(diffuse_out, [0.5, 0.5, 0.0], atol=1e-5)


if __name__ == "__main__":
    test_levoy_exact_isovalue()
    test_levoy_linear_falloff()
    test_levoy_multi_surface_combination()
    print("All Levoy tests passed!")
