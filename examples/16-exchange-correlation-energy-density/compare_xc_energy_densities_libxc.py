#!/usr/bin/env python
# coding: utf-8
"""
The libxc implementation of Becke's 88 exchange and Chachiyo's correlation
functionals is compared with the EigenFunctional implementation.
"""
import matplotlib.pyplot as plt
import numpy

import pyscf.gto

# Chachiyo correlation
from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional
# Dirac exchange
from msdft.ElectronRepulsionOperators import LDAExchangeLikeFunctional
# Becke 1988 exchange
from msdft.ElectronRepulsionOperators import GGABecke88ExchangeFunctional
from msdft.ElectronRepulsionOperators import LibxcFunctional
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
        compute_pair_density=True,
        # Spin symmetry is turned off, since we just want the lowest
        # electronic states no matter what spin state.
        spin_symmetry=False
    )

    # Plot exchange-correlation energy density along z-axis
    ncoord = 2000
    r = numpy.linspace(-3.0, 3.0, ncoord)
    coords = numpy.zeros((ncoord, 3))
    coords[:,2] = r

    # LDA functional for correlation
    correlation_lda = LDACorrelationLikeFunctional(mol)
    # GGA functional for exchange
    exchange_gga = GGABecke88ExchangeFunctional(mol)
    # libxc implementation of the same functionals
    correlation_lda_libxc = LibxcFunctional(
        msmd.mol, xc_code=',LDA_C_CHACHIYO', spin=0, level=4)
    exchange_gga_libxc = LibxcFunctional(
        msmd.mol, xc_code='GGA_X_B88,', spin=1, level=4)

    # Evaluate XC-energy density XCᵢⱼ(r) = - xᵢⱼ[D](r) + cᵢⱼ[D](r) along the cut
    # ... with my own implementation
    xed_gga = numpy.sum(
        # sum over spins
        exchange_gga.energy_density(msmd, coords), axis=0)
    ced_lda = numpy.sum(
        # sum over spins
        correlation_lda.energy_density(msmd, coords), axis=0)
    # Minus sign is not contained in our definition of exchange functional.
    xced = -xed_gga + ced_lda
    # ... and with libxc's implementation
    xed_gga_libxc = numpy.sum(
        # sum over spins
        exchange_gga_libxc.energy_density(msmd, coords), axis=0)
    ced_lda_libxc = numpy.sum(
        # sum over spins
        correlation_lda_libxc.energy_density(msmd, coords), axis=0)
    # Minus sign is already contained in libxc's implementation of exchange functional.
    xced_libxc = xed_gga_libxc + ced_lda_libxc

    # ... and exactly using the pair-density.
    xced_fci = msmd.exchange_correlation_energy_density(coords)

    # Plot XC
    fig, axes = plt.subplots(3,2, sharex=True, sharey=True)

    # The first row is for exchange only, the second one for correlation
    # and the third for the sum of the two.
    for row in [0,1]:
        axes[row,0].set_xlabel(r"r / $a_0$")
        axes[row,0].set_ylabel(r"state energy density / Hartree")

        axes[row,1].set_xlabel(r"r / $a_0$")
        axes[row,1].set_ylabel(r"transition energy density / Hartree")

    for column in [0,1]:
        axes[0,column].set_title(r"exchange")
        axes[1,column].set_title(r"correlation")
        axes[2,column].set_title(r"exchange and correlation")

    # Plot XC-energy density between different states.
    for istate in range(0, nstate):
        for jstate in range(istate, nstate):
            # Diagonal (state) and off-diagonal (transition) kinetic
            # energy densities are plotted separately.
            if istate == jstate:
                column = 0
            else:
                column = 1

            # approximate exchange
            # ... exchange
            line, = axes[0,column].plot(
                r, -xed_gga[istate,jstate,:],
                lw=1,
                ls='-', label=r"x$_{%d,%d}(r)$" % (istate, jstate))
            # ... exchange (libxc)
            axes[0,column].plot(
                # libxc's exchange functional already contains the correct minus sign.
                r,  xed_gga_libxc[istate,jstate,:],
                lw=2, color=line.get_color(),
                ls='-.', label=r"x$_{%d,%d}(r)$ libxc" % (istate, jstate))
            # ... correlation
            axes[1,column].plot(
                r, ced_lda[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls="-", label=r"c$_{%d,%d}(r)$" % (istate, jstate))
            # ... correlation (libxc)
            axes[1,column].plot(
                r, ced_lda_libxc[istate,jstate,:],
                lw=2, color=line.get_color(),
                ls="-.", label=r"c$_{%d,%d}(r)$ libxc" % (istate, jstate))
            # ... exchange and correlation
            axes[2,column].plot(
                r, xced[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls=":", label=r"xc$_{%d,%d}(r)$" % (istate, jstate))
            # ... libxc
            axes[2,column].plot(
                r, xced_libxc[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls="-.", label=r"xc$_{%d,%d}(r)$ libxc" % (istate, jstate))
            # ... full CI reference
            axes[2,column].plot(
                r, xced_fci[istate,jstate,:],
                lw=2, color=line.get_color(),
                ls="-", label=r"xc$_{%d,%d}(r)$ fci" % (istate, jstate))


    for row in [0,1,2]:
        for column in [0,1]:
            axes[row,column].legend()

    plt.show()

if __name__ == "__main__":
    # hydrogen molecule
    rHH = 1.4
    mol = pyscf.gto.M(
        atom = f'H 0 0 {-rHH/2.0}; H 0 0 {rHH/2.0}',
        basis = 'cc-pvdz',
        unit = 'Bohr'
    )

    compare_xc_energy_densities(mol, nstate=4)
