#!/usr/bin/env python
# coding: utf-8
"""
compare LDA exchange-correlation energy densities XC(r) computed with the Chachiyo
correlation and Dirac exchange with the exact values obtained from full CI.
"""
import matplotlib.pyplot as plt
import numpy

import pyscf.gto

# Chachiyo correlation
from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional
# Dirac exchange
from msdft.ElectronRepulsionOperators import LDAExchangeLikeFunctional
from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

def compare_xc_energy_densities(mol, nstate=2):
    """

    :param nstate: Number of electronic states in the subspace.
       The full CI problem is solved for the lowest nstate states.
    :type nstate: int > 0
    """
    msmd = MultistateMatrixDensityFCI.create_matrix_density(
        mol, nstate=nstate,
        # To compute the xc-energy density we need the pair-density matrix Dᵢⱼ(r,r').
        compute_pair_density=True)

    # Plot exchange-correlation energy density along z-axis
    ncoord = 1000
    r = numpy.linspace(-1.0, 1.0, ncoord)
    coords = numpy.zeros((ncoord, 3))
    coords[:,2] = r

    # LDA functionals for exchange and correlation
    exchange_lda = LDAExchangeLikeFunctional(mol)
    correlation_lda = LDACorrelationLikeFunctional(mol)

    # Evaluate XC-energy density XCᵢⱼ(r) = - xᵢⱼ[D](r) + cᵢⱼ[D](r) along the cut
    # ... with the approximate functionals
    # There is only a single spin component, since the LDA functional operate on the
    # total charge density.
    xed_lda = exchange_lda.energy_density(msmd, coords)[0,...]
    ced_lda = correlation_lda.energy_density(msmd, coords)[0,...]
    xced_lda = -xed_lda + ced_lda

    # ... and exactly using the pair-density.
    xced_fci = msmd.exchange_correlation_energy_density(coords)

    # Plot XC
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,2, sharex=True, sharey=True)

    # The first row is for exchange only, the second one for correlation
    # and the third for the sum of the two.
    for row in [0,1,2]:
        axes[row,0].set_xlabel(r"r / $a_0$")
        axes[row,0].set_ylabel(r"state XCED(r) / Hartree")

        axes[row,1].set_xlabel(r"r / $a_0$")
        axes[row,1].set_ylabel(r"transition XCED(r) / Hartree")

    for column in [0,1]:
        axes[0,column].set_title(r"exchange")
        axes[1,column].set_title(r"correlation")
        axes[2,column].set_title(r"exchange-correlation")

    # Plot XC-energy density between different states.
    for istate in range(0, nstate):
        for jstate in range(istate, nstate):
            # Diagonal (state) and off-diagonal (transition) kinetic
            # energy densities are plotted separately.
            if istate == jstate:
                column = 0
            else:
                column = 1

            # exact XCED as reference
            label = r"xc$_{%d,%d}(r)$ (exact)" % (istate, jstate)
            line, = axes[2,column].plot(
                r, xced_fci[istate,jstate,:],
                lw=3, alpha=0.25,
                label=label)

            # approximate exchange
            axes[0,column].plot(
                r, -xed_lda[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls='--', label=r"x$_{%d,%d}(r)$ (LDA)" % (istate, jstate))
            # approximate correlation
            axes[1,column].plot(
                r, ced_lda[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls="-.", label=r"c$_{%d,%d}(r)$ (LDA)" % (istate, jstate))
            # approximate exchange-correlation energy density
            axes[2,column].plot(
                r, xced_lda[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls="-.", label=r"xc$_{%d,%d}(r)$ (LDA)" % (istate, jstate))

    for row in [0,1,2]:
        for column in [0,1]:
            axes[row,column].legend()

    plt.show()

if __name__ == "__main__":
    # hydrogen molecule, closed shell
    mol = pyscf.gto.M(
        atom = 'H 0 0 -0.35; H 0 0 0.35',
        basis = '6-31g', #'cc-pvdz',
        # singlet
        spin = 0)

    compare_xc_energy_densities(mol, nstate=3)
