# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Numerical example homodyne
==========================
This script can be used to calculate homodyne variance reduction for
different types of pulses as reported in Fig. 1 of arXiv:1910.06873

To run it simply do
```
python homodyne_plots.py
```
"""



import numpy as np
from kerrlib import (
    expected_squeezing_g,
    expected_squeezing_r,
    expected_squeezing_s,
    expected_squeezing_l,
)
from QSS import qss
import matplotlib.pyplot as plt

zf = 5
n = 501
L = 0.03
G = 0.0
T0 = 0.17 * 10 ** -12
P0 = 20
ng = 2.1
b2 = 1.7 * 10 ** -25
g = 1
v = 299792458 / ng

solver = qss(zf, n)

phis = np.array([0.25, 0.5, 0.75, 1])
gsq = np.zeros(4)
rsq = np.zeros(4)
ssq = np.zeros(4)
lsq = np.zeros(4)

for i, phi in enumerate(phis):
    u, M, N = solver.evolution(
        L, G, T0, P0, ng, b2, g, u="Gaussian", TN=L / (phi * v) * 10 ** 10, TD=5000
    )
    phi, sq = solver.squeezing(u, M, N)
    gsq[i] = np.min(sq)

for i, phi in enumerate(phis):
    u, M, N = solver.evolution(
        L, G, T0, P0, ng, b2, g, u="Sech", TN=L / (phi * v) * 10 ** 10, TD=5000
    )
    phi, sq = solver.squeezing(u, M, N)
    ssq[i] = np.min(sq)

zf = 40
solver = qss(zf, n)
for i, phi in enumerate(phis):
    u, M, N = solver.evolution(
        L, G, T0, P0, ng, b2, g, u="Lorentzian", TN=L / (phi * v) * 10 ** 10, TD=5000
    )
    phi, sq = solver.squeezing(u, M, N)
    lsq[i] = np.min(sq)

n = 1501  # For the rectangular pulse we have to use a lot more points, this partly related
# to the Gibbs phenomenon for piecewise continuously differentiable functions.
zf = 5
solver = qss(zf, n)
for i, phi in enumerate(phis):
    u, M, N = solver.evolution(
        L, G, T0, P0, ng, b2, g, u="Rect", TN=L / (phi * v) * 10 ** 10, TD=5000
    )
    phi, sq = solver.squeezing(u, M, N)
    rsq[i] = np.min(sq)

phis2 = np.linspace(0, 2)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


plt.plot(phis2, expected_squeezing_l(phis2), color=colors[0])
plt.plot(phis2, expected_squeezing_s(phis2), color=colors[1])
plt.plot(phis2, expected_squeezing_g(phis2), color=colors[2])
plt.plot(phis2, expected_squeezing_r(phis2), color=colors[3])
plt.legend(["Lorentzian", "Sech", "Gaussian", "Rect"])
plt.plot(phis, lsq, "o", color=colors[0])
plt.plot(phis, ssq, "o", color=colors[1])
plt.plot(phis, gsq, "o", color=colors[2])
plt.plot(phis, rsq, "o", color=colors[3])
plt.ylim(-10, 0)
plt.xlim(0, 1.1)
plt.xlabel("Pulse peak nonlinear phase (rad)")
plt.ylabel("Reduced noise level in dB (below SNL)")
plt.savefig("Figure.svg")
