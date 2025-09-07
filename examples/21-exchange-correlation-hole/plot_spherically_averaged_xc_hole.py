#!/usr/bin/env python
"""
Compute the energies of the lowest triplet (³P) and singlet states (¹D and ¹S)
of the carbon atom with full configuration interaction, exploting spin symmetry
but without using spatial symmetry.

The spherically averaged exchange-correlation hole around some point r,

is defined as

    H^{xc}ᵢⱼ(r,|u|) = 1/(4 π) ∫ dΩ H^{xc}ᵢⱼ(r,r+|u|*e(Ω))

where e(Ω) is a unit vector in the direction Ω, such that u = r'-r = |u|*e(Ω).

H^{xc}ᵢⱼ(r,|u|) is plotted as a function of the distance |u| from r
(similarly to figures in [Becke/Roussel1989]).

References
----------
[Becke1983] Becke, A. D.
    "Hartree-Fock exchange energy of an inhomogeneous electron gas."
    International journal of quantum chemistry 23.6 (1983): 1915-1922.
    doi:10.1002/qua.560230605
[Becke/Roussel1989] A. Becke, A, M. Roussel.
    "Exchange holes in inhomogeneous systems: A coordinate-space model."
    Phys. Rev. A, 39(8), 3761-3767.
    doi:10.1103/PhysRevA.39.3761
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy
from pyscf.data.nist import HARTREE2EV
import pyscf.fci
import pyscf.gto
import pyscf.scf
import scipy.special

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

def xc_hole_taylor_expansion(
    msmd: MultistateMatrixDensityFCI,
    center_r: numpy.ndarray,
    distances_u: numpy.ndarray
    ) -> numpy.ndarray:
    """
    Compute the Taylor expansion of the spherically averaged exchange-correlation hole
    as a function of the distance from the reference point.

    For a system with a single electron, the Taylor expansion up to second order
    of the spherical average of the exchange-correlation hole H^{xc}ᵢⱼ(r,r+u)
    over the coordinate u about the reference point r is

        <H^{xc,sr}ᵢⱼ(r,r+u)> = -Dᵢⱼ(r) - 1/6 ∇²Dᵢⱼ(r) u² + ...

    The exact exchange-correlation hole for a one-electron system however is different
    from the Taylor expansion. It is independent of the reference point and is given by

        H^{xc,sr}ᵢⱼ(r,r') = -Dᵢⱼ(r')     (note that the argument of Dᵢⱼ is r', not r!)

    :param msmd: matrix density and its derivatives at the reference point
    :type msmd: instance of MultistateMatrixDensityFCI

    :param center_r: reference point
    :type center_r: numpy.ndarray of shape (3,)

    :param distances_u: distances u from reference point
    :type distances_u: numpy.ndarray of shape (Ndist,)

    :return spherical_xc_hole_sr: short-range Taylor expansion of spherical
        exchange-correlation hole as a function of the distance from the reference point,
        xc_hole_sr[u,:,:] = <H^{xc,sr}ᵢⱼ(r)>(distances_u[u])
    :rtype spherical_xc_hole_sr: numpy.ndarray of shape (Ndist,Nstate,Nstate)
    """
    # Evaluate matrix density and its derivatives at the reference point r
    coords = numpy.reshape(center_r, (1,3))
    # Only Dᵢⱼ(r) and its Laplacian ∇²Dᵢⱼ(r) are needed
    D, _, lapl_D = msmd.evaluate(coords)
    # sum over spin and select reference point
    D = numpy.sum(D, axis=0)[:,:,0]
    lapl_D = numpy.sum(lapl_D, axis=0)[:,:,0]

    # Taylor expansion around u=0 up to quadratic order
    spherical_xc_hole_sr = (
        # -Dᵢⱼ(r)
        -numpy.expand_dims(D, 0)
        # - 1/6 ∇²Dᵢⱼ(r) u²
        -1.0/6.0 * numpy.einsum('ij,u->uij', lapl_D, distances_u**2)
    )

    # The exchange energy should be calculated for spin up and spin down
    # separately. Since only the total charge density is given, we have to
    # divide it by two.
    spherical_xc_hole_sr *= 0.5

    return spherical_xc_hole_sr

def exchange_hole_homogeneous(
    msmd: MultistateMatrixDensityFCI,
    center_r: numpy.ndarray,
    distances_u: numpy.ndarray
    ) -> numpy.ndarray:
    """
    Compute the (spherically symmetric) exchange hole of the homogeneous electron gas (HEG)
    as a function of the distance from the reference point.

    For the ground state density, the exchange hole is given by (see [Becke1983], without minus sign)

        ρₓ^{HEG}(u) = -9 ρ [j1(k u)/(k u)]²

    with the Fermi momentum

        k = (6π²ρ)¹ᐟ³

    and the spherical Bessel function of 1st order

        j1(x) = sin(x)/x² - cos(x)/x

    Since the hole only depends on the density it can be turned into a matrix function
    by diagonalizing the matrix density, applying ρₓ^{HEG}(u) to each eigenvalue and
    transforming back.

    :param msmd: matrix density at the reference point
    :type msmd: instance of MultistateMatrixDensityFCI

    :param center_r: reference point
    :type center_r: numpy.ndarray of shape (3,)

    :param distances_u: distances u from reference point
    :type distances_u: numpy.ndarray of shape (Ndist,)

    :return x_hole_heg: exchange hole of the homogeneous electron gas
        as a function of the distance from the reference point,
    :rtype x_hole_heg: numpy.ndarray of shape (Ndist,Nstate,Nstate)

    References
    ----------
    [Becke1983] Becke, A. D.
        "Hartree-Fock exchange energy of an inhomogeneous electron gas."
        International journal of quantum chemistry 23.6 (1983): 1915-1922.
        doi:10.1002/qua.560230605
    """
    # Evaluate matrix density at the reference point r
    coords = numpy.reshape(center_r, (1,3))
    # Only Dᵢⱼ(r) is needed
    D, _, _ = msmd.evaluate(coords)
    # sum over spin and select reference point
    D = numpy.sum(D, axis=0)[:,:,0]

    # The exchange energy should be calculated for spin up and spin down
    # separately. Since only the total charge density is given, we have to
    # divide it by two.
    D *= 0.5

    def exchange_hole_heg(density, u):
        # Fermi momentum
        kF = pow(6.0 * numpy.pi**2 * density, 1.0/3.0)
        x = kF*u
        # spherical Bessel function of 1st order
        j1 = scipy.special.spherical_jn(1, x)
        rho_x_heg = -9.0 * density * pow(j1/x, 2)
        return rho_x_heg

    # Compute eigenvalues Λ and eigenvectors U of the symmetric matrix D.
    L, U = numpy.linalg.eigh(D)

    # number of electronic states
    nstate = msmd.number_of_states
    # number of distances u
    ndist = distances_u.shape[0]

    # Empty output array
    x_hole_heg = numpy.zeros((ndist,nstate,nstate))

    for i,u in enumerate(distances_u):
        x_hole_heg[i,:,:] = numpy.einsum(
            'ia,a,ja->ij',
            # X = U ρₓ(Λ,u) U⁻¹
            U,
            # apply ρₓ(Λ,u) to eigenvalues Λ
            exchange_hole_heg(L, u),
            U
        )

    return x_hole_heg


#
# taken from https://github.com/pyscf/pyscf/blob/master/examples/scf/31-v_atom_rohf.py
#
# Spherical symmetry needs to be carefully treated in the atomic calculation.
# The default initial guess may break the spherical symmetry.  To preserve the
# spherical symmetry in the atomic calculation, it is often needed to tune the
# initial guess and SCF model.
#
# Construct the atomic initial guess from cation.
#
mol = pyscf.gto.Mole()
mol.verbose = 4
mol.atom = 'C'
mol.basis = 'cc-pvdz' #'roosdz' #'cc-pvdz' #aug-cc-pvdz' #'cc-pvtz' #'cc-pvdz' #'aug-cc-pvtz' #'aug-cc-pvqz'
# Do not use spatial symmetry.
mol.symmetry = False
# The electronic configuration of neutral carbon is 1s²2s²2p²
# The dication C^{2+} has the closed-shell configuration 1s²2s², which is spherically symmetric.
mol.spin = 0
mol.charge = 2
mol.build()

rohf = pyscf.scf.ROHF(mol)
rohf.kernel()
rohf.analyze()

# Restore the neutral atom, the ground state is a ³P
# spin = Sz
mol.spin = 2
mol.charge = 0
mol.build()

# full CI for ³P, ¹D and ¹S
norb = rohf.mo_energy.size
nelec = (3,3)
mol.nelec = nelec
fci = pyscf.fci.FCI(mol, rohf.mo_coeff, singlet=False)
# Multiplicity of ³P is 3 (2*L+1=3),
# multiplicity of ¹D is (2*L+1)=5 and multiplicity of ¹S is 1.
fci.nroots = 3+5+1
fci_energies, fcivecs = fci.kernel(nelec=nelec)
e = fci_energies
# spin multiplicities 2*S+1
multiplicities = numpy.array([fci.spin_square(x, norb, nelec)[1] for x in fcivecs])
multiplicities = numpy.round(multiplicities, decimals=2)
for i in range(0, len(fcivecs)):
    print(
        'state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
        (i, e[i], (e[i]-e[0])*HARTREE2EV, multiplicities[i])
    )
# Check that we got the expected states.
numpy.testing.assert_allclose(multiplicities, 3*[3.0] + 5*[1.0] + 1*[1.0])
state_labels = (
    [r"³P(1)", "³P(2)", "³P(3)"] +
    [r"¹D(1)","¹D(2)","¹D(3)","¹D(4)","¹D(5)"] +
    ["¹S"])

# Compute xc-hole
msmd = MultistateMatrixDensityFCI(
    mol, rohf, fci, fcivecs,
    compute_pair_density=True)

# First electron is put somewhat close to the origin r=0
center_r = numpy.array([0.0, 0.0, 0.5])

# Plot xc-hole as a function of te distance from the electron.
distances_u = numpy.linspace(0.0, 2.0, 100)

# Exact spherical xc-hole
spherical_xc_hole = msmd.spherically_averaged_xc_hole(center_r, distances_u)

# Quadratic Taylor expansion around u=0 derived for one-electron system
spherical_xc_hole_approximate = xc_hole_taylor_expansion(msmd, center_r, distances_u)

# exchange hole of homogeneous electron gas
x_hole_heg = exchange_hole_homogeneous(msmd, center_r, distances_u)

# Figure, axes, labels
fig, axes = plt.subplots(1,2, figsize=(10,5))

axes[0].text(
    0.4, 0.1, r"Electron position $\vec{r}$=%s" % str(center_r),
    horizontalalignment='center',
    verticalalignment='center',
    transform = axes[0].transAxes
)

# Diagonal elements of xc-hole
axes[0].set_ylabel(r"spherically averaged xc-hole / $e a_0^{-3}$")
axes[0].set_xlabel(r"$\vert \vec{u} \vert$ / $a_0$")
## Diagonal elements of xc-hole are plotted on a log-scale.
#axes[0].set_yscale("symlog")

nstate = msmd.number_of_states
for i in range(0, nstate):
    line, = axes[0].plot(
        distances_u, spherical_xc_hole[:,i,i],
        lw=2, alpha=0.5,
        label=state_labels[i]
    )
    axes[0].plot(
        distances_u, spherical_xc_hole_approximate[:,i,i],
        ls="--", color=line.get_color()
    )
    axes[0].plot(
        distances_u, x_hole_heg[:,i,i],
        ls="-.", color=line.get_color()
    )

axes[0].legend(title="$\mathbf{(a)}$ diagonal")

# Off-diagonal elements of xc-hole
axes[1].set_ylabel(r"spherically averaged xc-hole / $e a_0^{-3}$")
axes[1].set_xlabel(r"$\vert \vec{u} \vert$ / $a_0$")

for i in range(0, nstate):
    for j in range(i+1, nstate):
        if multiplicities[i] == multiplicities[j]:
            line, = axes[1].plot(
                distances_u, spherical_xc_hole[:,i,j],
                lw=2, alpha=0.5,
                label=state_labels[i]+","+state_labels[j]
            )
            axes[1].plot(
                distances_u, spherical_xc_hole_approximate[:,i,j],
                ls="--", color=line.get_color()
            )
            axes[1].plot(
                distances_u, x_hole_heg[:,i,j],
                ls="-.", color=line.get_color()
            )
        else:
            # Transition matrix elements between states with different spin are zero.
            numpy.testing.assert_allclose(
                spherical_xc_hole[:,i,j], numpy.zeros_like(spherical_xc_hole[:,i,j]),
                atol=1.0e-5)

axes[1].yaxis.set_label_position("right")
axes[1].yaxis.set_ticks_position("right")
axes[1].legend(title="$\mathbf{(b)}$ off-diagonal")

# Create the invisible solid and dashed
# black lines that are shown in the figure legend.
solid_line = matplotlib.lines.Line2D([], [], ls="-", color="black")
dashed_line = matplotlib.lines.Line2D([], [], ls="--", color="black")
dashdot_line = matplotlib.lines.Line2D([], [], ls="-.", color="black")

fig.legend(
    [solid_line, dashed_line, dashdot_line],
    [
        r"$<H^{\text{xc}}_{IJ}(\vec{r},\vert \vec{u} \vert )> = \frac{1}{4 \pi} \int H^{\text{xc}}_{IJ}(\vec{r},\vec{u}) d\Omega_u$",
        r"$<H^{\text{xc,sr}}_{IJ}(\vec{r},\vert \vec{u} \vert )> = -D_{IJ}(\vec{r}) -\frac{1}{6} \nabla^2 D_{IJ}(\vec{r}) u^2$",
        r"$H^{\text{x,HEG}}_{IJ}(\vec{r},\vert \vec{u} \vert )>$"
    ],
    fontsize='large',
    frameon=False,
    loc='outside upper center',
    ncol=2
)

# Otherwise the x-labels are partly cut off.
plt.subplots_adjust(bottom=0.15, wspace=0.05, left=0.1, right=0.86)

#plt.savefig("carbon_spherical_averaged_xc_hole.svg")
#plt.savefig("carbon_spherical_averaged_xc_hole.png", dpi=300)

plt.show()
