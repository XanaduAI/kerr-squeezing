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
Kerrlib
=======
Provides numerical routines for the propagation of mean fields and phase sensitive
and insensitive moments in a (linearized) Kerr medium and some extra utility function.
"""


import numpy as np
from scipy.linalg import expm
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

# Pulse Shapes
def gaussian(z):
    r"""Returns a Gaussian function in z

    Args:
        z (array): Input values

    Returns:
        (array): Output array, element-wise exponential of negative z**2/2.
          This is a scalar if x is a scalar.
    """
    return np.exp(-z ** 2 / 2.0)


def sech(z):
    r"""Returns a hyperbolic secant function in z

    Args:
        z (array): Input values

    Returns:
        (array): Output array, element-wise sech of z. This is a scalar if x is a scalar.
    """
    return 1.0 / (np.cosh(z))


def rect(z, l=2 * np.sqrt(2 * np.log(2))):
    r"""Returns a Gaussian function in z

    Args:
        z (array): Input values
        l (array): Width of the top hat function

    Returns:
        (array): Output array, element-wise top hat function of z. This is a scalar if x is a
        scalar.
    """
    return np.where(abs(z) <= 0.5 * l, 1, 0)


def lorentzian(z):
    r"""Returns a Lorentzian function in z

    Args:
        z (array): Input values

    Returns:
        (array): Output array, element-wise lorentzian of z. This is a scalar if x is a scalar.
    """
    return 1.0 / np.sqrt(1.0 + z ** 2)


# Helper For Determining Mean-Field Widths
def FWHM(X, Y):
    r""" Calculates the Full Width at Half Maximum of the function Y=f(X)

    Args:
        X (array): Abscissae in which the function f( ) was sampled
        Y (array): Ordinate values, Y=f(X)

    Returns:
        (float): FWHM of the function Y=f(X)
    """
    half_max = np.max(Y) / 2.0
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]
    return X[right_idx] - X[left_idx]


# Fourier Transform Functions
def myfft(z, dz):
    r""" Numerical fourier transform of z=f(t) with t sampled at intervals dz

     Args:
        z (array): The function evaluated on a real space grid of points
        dz (float): The spacing between the grid points

    Returns:
        (array): The fourier transform of z=f(t)
    """
    
    return np.fft.fftshift(np.fft.fft(z) * dz / np.sqrt(2.0 * np.pi))


def myifft(k, dk, n):
    r""" Numerical inverse fourier transform of k=f(s) with s sampled at intervals dk
    for a total of n grid points

     Args:
        k (array): The function evaluated on a real space grid of points
        dk (float): The spacing between the grid points
        n (int): Number of sampling points

    Returns:
        (array): The fourier transform of z=f(t)
    """
    return np.fft.ifftshift(np.fft.ifft(k) * dk * n / np.sqrt(2.0 * np.pi))


# Split-Step Fourier Operators For Mean-Field Evolution
def opD(u, TD, G, kk, dt):
    r"""Short time "kinetic" or "dispersive" propagator. It applies exp(1j dt*(1/2*TD) d^2/dx^2) to
    u(x). The differential operator is applied as multiplication in reciprocal space using fast
    Fourier transforms.

     Args:
        u (array): The function evaluated on a real space grid of points
        TD (float): Dispersion time
        G (float): Loss rate
        kk (array): Grid of reciprocal space points with DC point at start
        dt (float): Size of time steps

    Returns:
        (array): The propagated array u by amount dt/2 (note the factor of 1/2)

    """
    k = np.fft.fft(u)
    return np.fft.ifft(np.exp(dt / 2.0 * (1j * kk ** 2 / (2.0 * TD))) * k) * np.exp(
        dt / 2.0 * (-G / 2.0)
    )


def opN(u, TN, ui, dt):
    r"""Short time "potential" or "nonlinear" propagator. It applies exp(1j dt*(TN) |ui(x)|^2) to
    u(x).

    Args:
        u (array): The initial function evaluated on a real space grid of points
        TN (float): Nonlinear time
        ui (array): Square root of the potential
        dt (float): Size of time steps

    Returns:
        (array): The propagated array u by amount dt

    """
    return np.exp(dt * 1j / TN * np.abs(ui) ** 2) * u


# Mean-Field Evolution
def P_mean_field(u, TD, TN, G, zz, dz, kk, N, dt):
    r"""Propagates the wavefunction u by time N*dt under both dispersion and nonlinearity.

    Args:
        u (array): The initial function evaluated on a real space grid of points
        TD (float): Dispersion time
        TN (float): Nonlinear time
        G (float): Loss rate
        zz (array): Grid of real space points
        dz (float): Size of discretization in real space
        kk (array): Grid of reciprocal space points with DC point at start
        N (int): Number of time steps
        dt (float): Size of time steps

    Returns:
        (array): The time evolved wavefunction after N*dt time.
    """
    for _ in range(N):
        ui = u
        u = opD(u, TD, G, kk, dt)
        u = opN(u, TN, ui, dt)
        u = opD(u, TD, G, kk, dt)
    return u


# Matrices For Fluctuation Evolution
def s(u, TN, dz):
    r""" Helper function to construct the S array for fluctuation propagation

    Args:
        u (array): Mean field values evaluated on a real space grid of points
        TN (float): Nonlinear time
        dz (float): Size of discretization in real space
    Returns:
        (array): S array
    """
    return myfft(u ** 2, dz) / TN


def m(u, TN, dz):
    r""" Helper function to construct the M array for fluctuation propagation

    Args:
        u (array): Mean field values evaluated on a real space grid of points
        TN (float): Nonlinear time
        dz (float): Size of discretization in real space
    Returns:
        (array): S array
    """
    return myfft(np.abs(u) ** 2, dz) / TN


def A(u, TD, TN, dz, ks, dk, im, n):
    r""" Construct the A matrix for fluctuation propagation

    Args:
        u (array): Mean field values evaluated on a real space grid of points
        TD (float): Dispersion time
        TN (float): Nonlinear time
        dz (float): Size of discretization in real space
        ks (array): Grid of reciprocal space points with DC point at centre
        dk (float): Size of discretization in reciprocal space
        im (int(n,n)): 2D array of integers (i,j) corresponding to the k-space gridpoints associated
          with i-j (clipped to be between 0 and n-1 so as not to fall off the grid).
        n (int): Size of the output matrix A
    Returns:
        (array): A matrix
    """
    mk = m(u, TN, dz)
    D = np.diag(np.full(n, ks ** 2 / (2.0 * TD)))
    return D + 2.0 * dk * mk[im] / np.sqrt(2.0 * np.pi)


def B(u, TN, dz, dk, ip):
    r""" Construct the B matrix for fluctuation propagation

    Args:
        u (array): Mean field values evaluated on a real space grid of points
        TN (float): Nonlinear time
        dz (float): Size of discretization in real space
        dk (float): Size of discretization in reciprocal space
        ip (int(n,n)): 2D array of integers (i,j) corresponding to the k-space gridpoints associated
          with i+j (clipped to be between 0 and n-1 so as not to fall off the grid).

    Returns:
        (array): B array
    """
    sk = s(u, TN, dz)
    return dk * sk[ip] / np.sqrt(2.0 * np.pi)


def Q(u, TD, TN, dz, ks, dk, im, ip, n):
    r""" Construct the Q matrix for fluctuation propagation

    Args:
        u (array): Mean field values evaluated on a real space grid of points
        TD (float): Dispersion time
        TN (float): Nonlinear time
        dz (float): Size of discretization in real space
        ks (array): Grid of reciprocal space points with DC point at centre
        dk (float): Size of discretization in reciprocal space
        im (int(n,n)): 2D array of integers (i,j) corresponding to the k-space gridpoints associated
          with i-j (clipped to be between 0 and n-1 so as not to fall off the grid).
        ip (int(n,n)): 2D array of integers (i,j) corresponding to the k-space gridpoints associated
          with i+j (clipped to be between 0 and n-1 so as not to fall off the grid).
        n (int): Size of the output matrix Q

    Returns:
        (array): Q matrix
    """
    a = A(u, TD, TN, dz, ks, dk, im, n)
    b = B(u, TN, dz, dk, ip)
    return np.block([[a, b], [-b.conj().T, -a.conj()]])


# Lossless Propagation
def P_no_loss(u, TD, TN, dz, kk, ks, dk, im, ip, tf, dt, n):
    r""" Lossless propagation of the mean and fluctuations in a Kerr medium

    Args:
        u (array): Mean field values evaluated on a real space grid of points
        TD (float): Dispersion time
        TN (float): Nonlinear time
        dz (float): Size of discretization in real space
        kk (array): Grid of reciprocal space points with DC point at start
        ks (array): Grid of reciprocal space points with DC point at centre
        dk (float): Size of discretization in reciprocal space
        im (int(n,n)): 2D array of integers (i,j) corresponding to the k-space gridpoints associated
          with i-j (clipped to be between 0 and n-1 so as not to fall off the grid).
        ip (int(n,n)): 2D array of integers (i,j) corresponding to the k-space gridpoints associated
          with i+j (clipped to be between 0 and n-1 so as not to fall off the grid).
        tf (int): Number of time steps
        dt (int): Size of time steps
        n (int): Size of the output matrices

    Returns:
        (tuple): (u,M,N), the first (u) and second order moments (M,N).
    """
    M = np.zeros(n)
    N = np.zeros(n)
    K = np.identity(2 * n)
    for _ in range(tf):
        ui = u
        u = opD(u, TD, 0, kk, dt)
        u = opN(u, TN, ui, dt)
        u = opD(u, TD, 0, kk, dt)
        K = expm(1j * dt * Q(u, TD, TN, dz, ks, dk, im, ip, n)) @ K
    U = K[0:n, 0:n]
    W = K[0:n, n:2 * n]
    M = U @ W.T
    N = W.conj() @ W.T
    return u, M, N


# Lossy Propagation
def P_loss(u, TD, TN, G, dz, kk, ks, dk, im, ip, tf, dt, n):
    r""" Lossy propagation of the mean and fluctuations in a Kerr medium

    Args:
        u (array): Mean field values evaluated on a real space grid of points
        TD (float): Dispersion time
        TN (float): Nonlinear time
        G (float): Loss rate
        dz (float): Size of discretization in real space
        kk (array): Grid of reciprocal space points with DC point at start
        ks (array): Grid of reciprocal space points with DC point at centre
        dk (float): Size of discretization in reciprocal space
        im (int(n,n)): 2D array of integers (i,j) corresponding to the k-space gridpoints associated
          with i-j (clipped to be between 0 and n-1 so as not to fall off the grid).
        ip (int(n,n)): 2D array of integers (i,j) corresponding to the k-space gridpoints associated
          with i+j (clipped to be between 0 and n-1 so as not to fall off the grid).
        tf (int): Number of time steps
        dt (int): Size of time steps
        n (int): Size of the output matrices

    Returns:
        (tuple): (u,M,N), the first (u) and second order moments (M,N).
    """
    M = np.zeros(n)
    N = np.zeros(n)
    for _ in range(tf):
        ui = u
        u = opD(u, TD, G, kk, dt)
        u = opN(u, TN, ui, dt)
        u = opD(u, TD, G, kk, dt)
        K = expm(1j * dt * Q(u, TD, TN, dz, ks, dk, im, ip, n))
        U = K[0:n, 0:n]
        W = K[0:n, n:2 * n]
        M = U @ M @ (U.T) + W @ (M.conj()) @ (W.T) + W @ N @ (U.T) + U @ (N.T) @ (W.T) + U @ (W.T)

        N = (
            W.conj() @ M @ (U.T)
            + U.conj() @ (M.conj()) @ (W.T)
            + U.conj() @ N @ (U.T)
            + W.conj() @ (N.T) @ (W.T)
            + W.conj() @ (W.T)
        )
        M = (1 - G * dt) * M
        N = (1 - G * dt) * N
    return u, M, N


def expected_squeezing_g(n_phi):
    r"""Calculate expected squeezing for Gaussian pulse for lossless, dispersionless propagation,
    with a maximum nonlinear phase shift of n_phi according to JOSA B 7, 30 (1990).

    Args:
        n_phi (float): Maximal nonlinear phase shift.

    Returns:
        Associated squeezing in dB.
    """
    return 10*np.log10(1 + 2 * n_phi**2 / np.sqrt(3) - (np.sqrt(2) * n_phi + 2 * np.sqrt(2) * n_phi**3 / 3) / np.sqrt(1 + 2 * n_phi**2 / 3))
