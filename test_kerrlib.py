import pytest
import numpy as np
from scipy.signal import find_peaks
from scipy.linalg import expm
from kerrlib import gaussian, myfft, opD, opN, FWHM, A, B, Q

G = 0  # loss parameter

zf = 8  # end points (-zf,+zf) of real-space array
n = 501  # number of points in real-space array

# Set up z- and k-space arrays
zz = np.linspace(-zf, zf, n)
hz = zz[1] - zz[0]
kk = np.fft.fftfreq(n, d=hz) * (2.0 * np.pi)
ks = np.fft.fftshift(kk)
hk = kk[1] - kk[0]

# Set up k-space grid
xx, yy = np.meshgrid(ks, ks) / hk
im = np.rint(xx - yy) + (n - 1) / 2
ip = np.rint(xx + yy) + (n - 1) / 2
im = np.clip(im, 0, n - 1).astype(int)
ip = np.clip(ip, 0, n - 1).astype(int)


def test_dispersion():
    r""" Test that dispersion is propagated correctly"""

    factor = 1
    n_steps = int(1000 / factor)
    dt = hz * factor
    TN = 5000  # large nonlinear time such that propagation is effectively without nonlinearity
    TD = 1  # dispersion time

    # Define mean-field in z-space and perform evolution for 1000 time steps
    U = gaussian(zz)
    for i in range(n_steps):
        Ui = U
        U = opD(U, TD, G, kk, dt)
        U = opN(U, TN, Ui, dt)
        U = opD(U, TD, G, kk, dt)

    # Confirm that FWHM in z changes as expected
    FWHM1 = FWHM(zz, gaussian(zz) ** 2)
    FWHM2 = FWHM(zz, np.abs(U) ** 2)
    found = FWHM2 / FWHM1
    expected = np.sqrt(1 + (dt * n_steps / TD) ** 2)
    np.allclose(found, expected, atol=1e-2, rtol=1e-2)


def test_nonlinearity():
    r""" Test that nonlinearity is propagated correctly"""

    factor = 1
    n_steps = int(1000 / factor)
    dt = hz * factor
    TN = n_steps * dt / (2.5*np.pi)  # nonlinear time corresponding to peak nonlinear phase shift of 2.5pi
    TD = 5000  # large dispersion time such that propagation is effectively dispersionless

    # Define mean-field in z-space and perform evolution for n_steps time steps
    U = gaussian(zz)
    for i in range(n_steps):
        Ui = U
        U = opD(U, TD, G, kk, dt)
        U = opN(U, TN, Ui, dt)
        U = opD(U, TD, G, kk, dt)

    # Convert to k-space and window to area of interest
    ks = np.fft.fftshift(kk)
    Uk = myfft(U, hz)[(ks > -1.5*zf) & (ks < 1.5*zf)]

    # Confirm number of peaks equals 3 for peak nonlinear phase shift of 2.5pi
    num_peaks = len(find_peaks(np.abs(Uk)**2)[0])
    assert num_peaks == 3


def test_nonlinearity_and_dispersion():
    r""" Test that nonlinearity and dispersion are propagated correctly"""

    factor = 1
    n_steps = int(1000 / factor)
    dt = hz * factor
    TN = 4  # Define dispersion time and nonlinear time equal and not too small
    TD = 4

    # Define mean-field in z-space and perform evolution for n_steps time steps
    U = gaussian(zz)
    for i in range(n_steps):
        Ui = U
        U = opD(U, TD, G, kk, hz)
        U = opN(U, TN, Ui, hz)
        U = opD(U, TD, G, kk, dt)

    # Confirm  that FWHM in z changes as expected
    FWHM1 = FWHM(zz, gaussian(zz) ** 2)
    FWHM2 = FWHM(zz, np.abs(U) ** 2)
    phiN = hz * n_steps / TN
    phiD = hz * n_steps / TD
    found = FWHM2 / FWHM1
    expected = np.sqrt(
        1 + np.sqrt(2) * phiN * phiD + (1 + 4 / (3 * np.sqrt(3)) * phiN ** 2) * phiD ** 2
    )
    np.allclose(found, expected, atol=1e-2, rtol=1e-2)


def test_submatrices():
    r"""Test submatrices of Q. A should be Hermitian and B should be symmetric."""
    U = gaussian(zz)
    TN = 4
    TD = 4

    a = A(U, TD, TN, hz, kk, hk, im, n)
    b = B(U, TN, hz, hk, ip)

    np.testing.assert_almost_equal(np.linalg.norm(a - a.conj().T), 0)
    np.testing.assert_almost_equal(np.linalg.norm(b - b.T), 0)


def test_Q_matrix():
    r"""Test that Q is a symplectic matrix."""
    U = gaussian(zz)
    TN = 4
    TD = 4

    Qtemp = Q(U, TD, TN, hz, ks, hk, im, ip, n)
    atemp = np.identity(n)
    R = np.block([[atemp, atemp * 0], [atemp * 0, -atemp]])

    np.testing.assert_almost_equal(np.linalg.norm(Qtemp @ R - R @ Qtemp.conj().T), 0)


def test_propagation_submatrices():
    r"""Test U and V matrices. Together they should satisfy UU^dag - WW^dag =I and UW^T = WU^T"""
    factor = 1
    n_steps = int(50 / factor)
    dt = hz * factor
    TN = 4
    TD = 4

    u = gaussian(zz)
    K = np.identity(2 * n)
    for _ in range(n_steps):
        ui = u
        u = opD(u, TD, 0, kk, dt)
        u = opN(u, TN, ui, dt)
        u = opD(u, TD, 0, kk, dt)
        K = expm(1j * dt * Q(u, TD, TN, hz, ks, hk, im, ip, n)) @ K
    U = K[0:n, 0:n]
    W = K[0:n, n:2 * n]
    np.testing.assert_almost_equal(np.linalg.norm(U @ (U.conj().T) - W @ (W.conj().T) - np.identity(n)), 0)
    np.testing.assert_almost_equal(np.linalg.norm(U @ (W.T) - W @ (U.T)), 0)
