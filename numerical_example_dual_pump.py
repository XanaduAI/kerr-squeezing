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
from kerrlib import P_no_loss, myfft

# pylint: disable=invalid-name


# The following lines provide a convenient function to evaluate the Fourier transform
# of the function f(z) = \exp(-z**4) by loading precalculated values and the using the
# interpolation capabilities of numpy
y = np.loadtxt("fourierexpz4.dat")
x = np.loadtxt("z.dat")
func = lambda x0: np.interp(x0, x, y, left=0.0, right=0.0)


c = 299792458  # speed of light [m/s]
L = 0.014  # length of waveguide [m]
T0 = 796 * 10 ** -15  # pulse FWHM [s]
v = c / 4.27  # group velocity
b2 = -13.62 * 10 ** -24  # group velocity dispersion [s^2/m]


P0 = float(sys.argv[1])  # input power [W]
g = 97  # nonlinear parameter [/m/W]


Z0 = T0 * v

scale_factor = 80 * 10 ** 10

TN = 1 / (g * P0 * v) * scale_factor  # scaled nonlinear time

TD = Z0 ** 2 / (b2 * v ** 3) * scale_factor  # scaled dispersion time

n = int(sys.argv[2])  # number of points in real-space array
zf = 30 * 2  # end points (-zf,+zf) of real-space array

# Set up z- and k-space arrays
zz = np.linspace(-zf, zf, n)
dz = zz[1] - zz[0]
kk = np.fft.fftfreq(n, d=dz) * (2.0 * np.pi)
ks = np.fft.fftshift(kk)
dk = kk[1] - kk[0]
dt = dz
tf = np.rint(L / (v * dt) * scale_factor).astype(
    int
)  # number of points in time (final time=dt*tf)

xx, yy = np.meshgrid(ks, ks) / dk
im = np.rint(xx - yy) + (n - 1) / 2
ip = np.rint(xx + yy) + (n - 1) / 2
im = np.clip(im, 0, n - 1).astype(int)
ip = np.clip(ip, 0, n - 1).astype(int)

# Define mean-field in z-space
# The cosine of 3*zz will give to peks 3 units away in k space
u = func(zz) * np.cos(3 * zz)


# Perform Evolution
u, M, N = P_no_loss(u, TD, TN, dz, kk, ks, dk, im, ip, tf, dt, n)


# Save final mean-field in and moments in k as well as k discretization.
np.save("lsq_mmomentSc" + sys.argv[1] + "n" + sys.argv[2], M)
np.save("lsq_nmomentSc" + sys.argv[1] + "n" + sys.argv[2], N)
np.save("lsq_meanSc" + sys.argv[1] + "n" + sys.argv[2], myfft(u, dz))
np.save("lsq_ksSc" + sys.argv[1] + "n" + sys.argv[2], ks)
print(sys.argv, np.trace(N), tf, tf * dt)
