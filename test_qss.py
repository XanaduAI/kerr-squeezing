import pytest
import numpy as np
from kerrlib import expected_squeezing_g
from QSS import qss

zf = 5
n = 501
L = 0.03
G = 0.0
T0 = 0.17 * 10**-12
P0 = 20
ng = 2.1
b2 = 1.7 * 10**-25
g = 1
v = 299792458 / ng


def test_squeezing():
    r"""Tests that maximal squeezing for a Gaussian pulse undergoing lossless, dispersionless
    propagation coincides with what's predicted analytically.
    """
    solver = qss(zf, n)
    u_ev, M, N = solver.evolution(L, G, T0, P0, ng, b2, g, TN=L / (0.25 * v) * 10**10, TD=5000)
    phi, sq = solver.squeezing(u_ev, M, N)
    assert np.allclose(np.min(sq), expected_squeezing_g(0.25), atol=1e-2, rtol=1e-2)
