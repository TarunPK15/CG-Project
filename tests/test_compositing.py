"""
test_compositing.py - Verification of Back-to-Front vs Front-to-Back Compositing Equivalence.
"""

import numpy as np


def test_compositing_mathematical_equivalence():
    """
    Proves that Front-to-Back and Back-to-Front compositing formulations
    produce identical accumulated color and opacity along viewing rays.
    """
    np.random.seed(42)
    n_samples = 20

    colors = np.random.uniform(0.0, 1.0, (n_samples, 3)).astype(np.float32)
    alphas = np.random.uniform(0.05, 0.35, n_samples).astype(np.float32)

    # 1. Back-to-Front recurrence
    # Index 0 is front (near eye), Index n_samples-1 is back (far)
    b2f_c = np.zeros(3, dtype=np.float32)
    b2f_a = 0.0

    for s in range(n_samples - 1, -1, -1):
        c_s = colors[s]
        a_s = alphas[s]
        b2f_c = c_s * a_s + (1.0 - a_s) * b2f_c
        b2f_a = a_s + (1.0 - a_s) * b2f_a

    # 2. Front-to-Back recurrence
    f2b_c = np.zeros(3, dtype=np.float32)
    f2b_a = 0.0

    for s in range(n_samples):
        c_s = colors[s]
        a_s = alphas[s]
        weight = (1.0 - f2b_a) * a_s
        f2b_c += weight * c_s
        f2b_a += weight

    np.testing.assert_allclose(b2f_c, f2b_c, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b2f_a, f2b_a, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_compositing_mathematical_equivalence()
    print("All compositing equivalence tests passed!")
