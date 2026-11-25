"""
test_drebin.py - Unit test for Robert Drebin et al. (1988) Material Classification.
"""

import numpy as np
from core.drebin import (
    trapezoid_membership,
    evaluate_drebin_classification,
    evaluate_matte_mask,
    DrebinClassifier,
)


def test_trapezoid_membership():
    # s_min=0.2, sp_start=0.4, sp_end=0.6, s_max=0.8
    assert trapezoid_membership(0.1, 0.2, 0.4, 0.6, 0.8) == 0.0
    np.testing.assert_allclose(trapezoid_membership(0.3, 0.2, 0.4, 0.6, 0.8), 0.5, atol=1e-5)
    np.testing.assert_allclose(trapezoid_membership(0.5, 0.2, 0.4, 0.6, 0.8), 1.0, atol=1e-5)
    np.testing.assert_allclose(trapezoid_membership(0.7, 0.2, 0.4, 0.6, 0.8), 0.5, atol=1e-5)
    assert trapezoid_membership(0.9, 0.2, 0.4, 0.6, 0.8) == 0.0


def test_drebin_classification_composite():
    classifier = DrebinClassifier()
    params, colors, opacities, rhos = classifier.to_numba_arrays()

    percentages = np.zeros(len(classifier.materials), dtype=np.float32)
    diffuse_col = np.zeros(3, dtype=np.float32)

    # Test Bone region (scalar = 0.80)
    alpha, rho = evaluate_drebin_classification(
        0.80, params, colors, opacities, rhos, percentages, diffuse_col
    )

    # Bone should have high opacity and high density rho
    assert alpha > 0.8
    assert rho > 2.0
    assert diffuse_col[0] > 0.8  # Ivory/white bone


def test_matte_mask():
    # Plane x >= 0: normal = (1, 0, 0), d = 0
    assert evaluate_matte_mask(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, False) == 1.0
    assert evaluate_matte_mask(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, False) == 0.0

    # Wedge cut
    assert evaluate_matte_mask(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -10.0, True) == 0.0
    assert evaluate_matte_mask(-1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -10.0, True) == 1.0


if __name__ == "__main__":
    test_trapezoid_membership()
    test_drebin_classification_composite()
    test_matte_mask()
    print("All Drebin tests passed!")
