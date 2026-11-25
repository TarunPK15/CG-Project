"""
run_tests.py - Test Runner for all unit tests.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_interpolation import (
    test_trilinear_scalar_linear_field,
    test_trilinear_out_of_bounds,
    test_trilinear_vec3,
)
from tests.test_gradients import test_quadratic_field_gradients
from tests.test_raybox import test_ray_box_hit_miss
from tests.test_levoy import (
    test_levoy_exact_isovalue,
    test_levoy_linear_falloff,
    test_levoy_multi_surface_combination,
)
from tests.test_drebin import (
    test_trapezoid_membership,
    test_drebin_classification_composite,
    test_matte_mask,
)
from tests.test_compositing import test_compositing_mathematical_equivalence


def run_all():
    print("=" * 60)
    print("RUNNING IS F311 COMPUTER GRAPHICS VOLUME RENDERING TEST SUITE")
    print("=" * 60)

    print("\n[1/6] Testing Trilinear Interpolation...")
    test_trilinear_scalar_linear_field()
    test_trilinear_out_of_bounds()
    test_trilinear_vec3()
    print(" -> PASSED")

    print("\n[2/6] Testing Central Difference Gradients & Normal Vectors...")
    test_quadratic_field_gradients()
    print(" -> PASSED")

    print("\n[3/6] Testing Slab Method Ray-Box Intersection...")
    test_ray_box_hit_miss()
    print(" -> PASSED")

    print("\n[4/6] Testing Marc Levoy (1988) Isosurface & Multi-Surface Blending...")
    test_levoy_exact_isovalue()
    test_levoy_linear_falloff()
    test_levoy_multi_surface_combination()
    print(" -> PASSED")

    print("\n[5/6] Testing Robert Drebin (1988) Material Classification & Mattes...")
    test_trapezoid_membership()
    test_drebin_classification_composite()
    test_matte_mask()
    print(" -> PASSED")

    print("\n[6/6] Testing Back-to-Front vs Front-to-Back Compositing Equivalence...")
    test_compositing_mathematical_equivalence()
    print(" -> PASSED")

    print("\n" + "=" * 60)
    print("ALL 6 TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
