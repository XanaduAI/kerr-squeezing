import numpy as np
import matplotlib.pyplot as plt
import glob

from strawberryfields.decompositions import takagi


def jsa_from_m(m):
    """Given a phase sensitive moment m returns the joint spectral amplitude associated with it.

    Args:
        m (array): phase sentive moment

    Returns:
        (array): joint spectral amplitude

    """
    ls, u = takagi(m)
    return u @ np.diag(0.5 * np.arcsinh(2 * ls)) @ u.T


n = 1501  # Size of discretization in k
files = glob.glob("*" + str(n) + "*.npy")
files.sort()

l = 2  # Number of parameter settings
mfiles = files[2 * l : 3 * l]
nfiles = files[3 * l : 4 * l]
meanfiles = files[1 * l : 2 * l]
ks = np.load(files[0])

totp = l
localms = [np.load(mfiles[i]) for i in range(totp)]
localns = [np.load(nfiles[i]) for i in range(totp)]

N = np.empty([totp])


## Generating plot for the Schmidt number occupations
fig, ax = plt.subplots(totp, 1, sharey=False, sharex=False, figsize=(4, 3))
for i in range(totp):
    y = 5
    ns = np.linalg.eigvalsh(localns[i])[::-1]
    K = (np.sum(ns) ** 2) / np.sum(ns ** 2)
    N[i] = np.sum(ns)
    ax[i].bar(
        np.arange(y),
        ns[0:y],
        label=r"$K$=" + str(np.round(K, 4)) + r", $\langle n \rangle=$" + str(np.round(N[i], 4)),
    )
    ax[i].legend()
    if i == 0:
        ax[i].set_xlabel(r"Schmidt mode $j$")

    ax[i].set_ylabel(r"$\langle n_j \rangle $")
    fig.savefig("schmidt_occ.pdf")
plt.close()

## Generating plot for the energy density in the fluctuations
for i in range(totp):
    x = np.load(nfiles[i])
    plt.semilogy(
        ks,
        np.diag(x).real / (ks[1] - ks[0]),
        label=r"$\langle n \rangle =$ " + str(np.round(N[i], 4)),
    )
    plt.xlim([-18, 18])
plt.xlabel(r"$(k-k_p)/\Delta k$")
plt.title(r"$\langle \delta b(k)^\dagger \delta b(k) \rangle$")
plt.legend()
plt.savefig("squeezed_energydensity.pdf")
plt.close()


## Generating plot for the energy density in the mean
for i in range(totp):
    x = np.load(meanfiles[i])
    plt.plot(ks, np.abs(x) ** 2, label=r"$\langle n \rangle =$ " + str(np.round(N[i], 4)))
    plt.xlim([-10, 10])
plt.xlabel(r"$(k-k_p)/\Delta k$")

plt.ylabel(r"arb. units")
plt.title(r"$|\langle b(k) \rangle|^2$")
plt.legend()
plt.savefig("mean_energydensity.pdf")
plt.close()


## Generating plot for the joint spectral amplitude
fig, ax = plt.subplots(1, totp, sharex=False, sharey=True, figsize=(12, 12))
for i in range(totp):
    localm = np.load(mfiles[i])
    ax[i].contour(ks, ks, np.abs(jsa_from_m(localm)), origin="lower", cmap="Greens")
    ax[i].set_aspect(aspect=1)
    ax[i].set_xlim([-10, 10])
    ax[i].set_ylim([-10, 10])
    ax[i].set_xlabel(r"$(k-k_p)/\Delta k$")
    if i == 0:
        ax[i].set_ylabel(r"$(k'-k_p)/\Delta k$")
    ax[i].set_title(r"$J(k,k')$")
    fig.savefig("jsas.pdf")
plt.close()
