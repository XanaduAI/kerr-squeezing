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
Numerical example dual pump
===========================
This code generates .npz files containing the mean field and phase
sensitive and phase insensitive moments and their k discretization.
To use it type in the terminal

```

python numerical_example_dual_pump.py x y

```

where x is the relative strength of the pump and y is the number of
grid points. For the results presented in the paper we used
x = 0.1, 1.4
y = 1000
"""


import sys
import numpy as np
from QSS import qss
from kerrlib import myfft


# The following lines provide a convenient function to evaluate the Fourier transform
# of the function f(z) = \exp(-z**4) by loading precalculated values and the using the
# interpolation capabilities of numpy
valsxy = np.load("zfourierexp-z4.npy")
func = lambda x0: np.interp(x0, valsxy[0], valsxy[1], left=0.0, right=0.0)


L = 0.014  # length of waveguide [m]
G = 0  # loss rate [Hz]
T0 = 796 * 10 ** -15  # pulse FWHM [s]
P0 = float(sys.argv[1])  # input power [W]
ng = 4.27  # group index
b2 = -13.62 * 10 ** -24  # group velocity dispersion [s^2/m]
g = 97  # nonlinear parameter [/m/W]

scale_factor = 80 * 10 ** 10

n = int(sys.argv[2])  # number of points in real-space array
zf = 30 * 2  # end points (-zf,+zf) of real-space array

solver = qss(zf, n)

# Define mean-field in z-space
# The cosine of 3*zz will give to peks 3 units away in k space
u = func(solver.zz) * np.cos(3 * solver.zz)


# Perform Evolution
u, M, N = solver.evolution(L, G, T0, P0, ng, b2, g, u=u, scale_factor=scale_factor)


# Save final mean-field in and moments in k as well as k discretization.
np.save("lsq_mmomentSc" + sys.argv[1] + "n" + sys.argv[2], M)
np.save("lsq_nmomentSc" + sys.argv[1] + "n" + sys.argv[2], N)
np.save("lsq_meanSc" + sys.argv[1] + "n" + sys.argv[2], myfft(u, solver.dz))
np.save("lsq_ksSc" + sys.argv[1] + "n" + sys.argv[2], solver.ks)
print(sys.argv, np.trace(N), solver.tf, solver.tf * solver.dt)
