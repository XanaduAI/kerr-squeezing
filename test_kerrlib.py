import pytest
import numpy as np
from kerrlib import gaussian, myfft, opD, opN, FWHM

G = 0  # loss parameter

zf = 8  # end points (-zf,+zf) of real-space array
n = 6001  # number of points in real-space array

# Set up z- and k-space arrays
zz = np.linspace(-zf, zf, n)
hz = zz[1] - zz[0]
kk = np.fft.fftfreq(n, d=hz) * (2.0 * np.pi)
hk = kk[1] - kk[0]


def test_dispersion():
    r""" Test that dispersion is propagated correctly"""

    TN = 5000  # nonlinear time
    TD = 1  # dispersion time
    # Define mean-field in z-space and perform evolution for 1000 time steps
    U = gaussian(zz)
    factor = 200
    n_steps = int(1000 / factor)
    dt = hz * factor
    for i in range(n_steps):
        Ui = U
        U = opD(U, TD, G, kk, dt)
        U = opN(U, TN, Ui, dt)
        U = opD(U, TD, G, kk, dt)

    FWHM1 = FWHM(zz, gaussian(zz) ** 2)
    FWHM2 = FWHM(zz, np.abs(U) ** 2)
    found = FWHM2 / FWHM1
    expected = np.sqrt(1 + (dt * n_steps / TD) ** 2)
    # print(found, expected)
    np.allclose(found, expected, atol=1e-2, rtol=1e-2)


def test_nonlinearity_and_dispersion():
    r""" Test that nonlinearity and dispersion are propagated correctly"""

    TD = 4
    TN = 4
    U = gaussian(zz)
    factor = 1
    n_steps = int(1000 / factor)
    dt = hz * factor

    for i in range(n_steps):
        Ui = U
        U = opD(U, TD, G, kk, hz)
        U = opN(U, TN, Ui, hz)
        U = opD(U, TD, G, kk, dt)

    FWHM1 = FWHM(zz, gaussian(zz) ** 2)
    FWHM2 = FWHM(zz, np.abs(U) ** 2)
    phiN = hz * n_steps / TN
    phiD = hz * n_steps / TD
    found = FWHM2 / FWHM1
    expected = np.sqrt(
        1 + np.sqrt(2) * phiN * phiD + (1 + 4 / (3 * np.sqrt(3)) * phiN ** 2) * phiD ** 2
    )
    # print(found, expected)
    np.allclose(found, expected, atol=1e-2, rtol=1e-2)
