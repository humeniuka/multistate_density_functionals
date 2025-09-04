#!/usr/bin/env python
"""
Compute the energies of the lowest triplet (³P) and singlet states (¹D and ¹S)
of the carbon atom with full configuration interaction, exploting spin symmetry
but without using spatial symmetry.

The spherically averaged exchange-correlation hole around some point r,

is defined as

    H^{xc}ᵢⱼ(r,|u|) = 1/(4 π) ∫ dΩ H^{xc}ᵢⱼ(r,r+|u|*e(Ω))

where e(Ω) is a unit vector in the direction Ω, such that u = r'-r = |u|*e(Ω).

H^{xc}ᵢⱼ(r,|u|) is plotted as a function of the distance |u| from r.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy
from pyscf.data.nist import HARTREE2EV
import pyscf.fci
import pyscf.gto
import pyscf.scf

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

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
center_r = numpy.array([0.0, 0.0, 1.0])

# Plot xc-hole as a function of te distance from the electron.
distances_u = numpy.linspace(0.0, 5.0, 1000)

spherical_xc_hole = msmd.spherically_averaged_xc_hole(center_r, distances_u)

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
#dashed_line = matplotlib.lines.Line2D([], [], ls="--", color="black")

fig.legend(
    [solid_line,],#, dashed_line],
    [
        r"$H^{\text{xc}}_{IJ}(\vec{r},\vert \vec{u} \vert ) = \frac{1}{4 \pi} \int H^{\text{xc}}_{IJ}(\vec{r},\vec{u}) d\Omega_u$",
        #r"$\text{xc}^{\text{MSDFT}}_{IJ}(r) = \text{x}^{\text{B88}}[\mathbf{D}]_{IJ}(r) + \text{c}^{\text{LYP}}[\mathbf{D}]_{IJ}(r)$"
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
