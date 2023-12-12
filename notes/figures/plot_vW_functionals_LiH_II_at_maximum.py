#!/usr/bin/env python
# coding: utf-8
"""
compare kinetic energy densities KED(r) computed with the von-Weizsaecker functionals
for the lithium hydride molecule.
"""
import matplotlib.pyplot as plt
import numpy

import pyscf.fci
import pyscf.scf

from msdft.KineticOperatorFunctional import ThomasFermiFunctional
from msdft.KineticOperatorFunctional import VonWeizsaecker1eFunctional
from msdft.KineticOperatorFunctional import VonWeizsaecker1eFunctionalII
from msdft.KineticOperatorFunctional import VonWeizsaeckerFunctional
from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI


def compare_vW_kinetic_energy_densities(mol, nstate=2):
    """
    The kinetic energy density is plotted for different functionals
    and is compared with the exact one.

    :param nstate: Number of electronic states in the subspace.
       The full CI problem is solved for the lowest nstate states.
    :type nstate: int > 0
    """
    # compute D(r) from full CI
    msmd = MultistateMatrixDensityFCI.create_matrix_density(
        mol, nstate=nstate, spin_symmetry=False, raise_error=False)

    # Plot T(0,0,z), cut along z-axis
    Ncoord = 5000
    coords = numpy.zeros((Ncoord, 3))
    r = numpy.linspace(-3.0, 3.0, Ncoord)
    coords[:,2] = r

    # Functionals for kinetic energy matrix.
    kinetic_vW = VonWeizsaeckerFunctional(mol)
    kinetic_vW1eII = VonWeizsaecker1eFunctionalII(mol)

    # Evalute kinetic energy density along the cut ...
    # ... with the approximate functionals from the matrix density
    KED_vW = kinetic_vW.kinetic_energy_density(msmd, coords)
    KED_vW1eII = kinetic_vW1eII.kinetic_energy_density(msmd, coords)
    # ... and exactly from the wavefunction.
    KED_lap, KED_gg = msmd.kinetic_energy_density(coords)

    # Sum over spin.
    KED_vW = numpy.sum(KED_vW, axis=0)
    KED_vW1eII = numpy.sum(KED_vW1eII, axis=0)

    KED_lap = numpy.sum(KED_lap, axis=0)
    KED_gg = numpy.sum(KED_gg, axis=0)

    # Plot KED(r).
    fig, axes = plt.subplots(2,2, figsize=(8,7), sharex=True)

    for row in [0,1]:
        axes[row,0].set_ylabel(r"state KED / $E_h$")
        axes[row,1].set_ylabel(r"transition KED / $E_h$")

        axes[row,1].yaxis.set_label_position("right")
        axes[row,1].yaxis.tick_right()

        # State KED can be quite large, so make the axis logarithmic.
        axes[row,0].set_yscale('log')

    for column in [0,1]:

        axes[0,column].text(
            0.05, 0.95, 'vW',
            fontsize='x-large',
            bbox=dict(facecolor='grey', alpha=0.1),
            horizontalalignment='left', verticalalignment='top', transform=axes[0,column].transAxes)
        axes[1,column].text(
            0.05, 0.95, 'vW1eII',
            fontsize='x-large',
            bbox=dict(facecolor='grey', alpha=0.1),
            horizontalalignment='left', verticalalignment='top', transform=axes[1,column].transAxes)

        axes[1,column].set_xlabel(r"r / $a_0$")

    # Plot kinetic energy density between different states.
    for istate in range(0, nstate):
        for jstate in range(istate, nstate):
            label = r"$(%d,%d)$" % (istate, jstate)
            # Diagonal (state) and off-diagonal (transition) kinetic
            # energy densities are plotted separately.
            if istate == jstate:
                column = 0
            else:
                column = 1

            # exact KED as reference
            """
            axes[0,column].plot(
                r, KED_lap[istate,jstate,:],
                lw=3, alpha=0.25, label=label+" exact (-1/2 ∇²f)")
            axes[1,column].plot(
                r, KED_lap[istate,jstate,:],
                lw=3, alpha=0.25, label=label+" exact (-1/2 ∇²f)")
            """
            line, = axes[0,column].plot(
                r, KED_gg[istate,jstate,:],
                lw=3, alpha=0.25,
                label=label)
            axes[1,column].plot(
                r, KED_gg[istate,jstate,:],
                lw=3, color=line.get_color(), alpha=0.25,
                label=label)

            # approximate KED
            axes[0,column].plot(
                r, KED_vW[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls='--')
            axes[1,column].plot(
                r, KED_vW1eII[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls="-.")

    for row in [0,1]:
        for column in [0,1]:
            axes[row,column].legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    # only show curves around maximum of KED
    for row in [0,1]:
        axes[row,0].set_xlim((-1.4878, -1.47406))
        axes[row,0].set_ylim((54.0963, 59.8301))

    #plt.savefig("vW_kinetic_energy_LiH_II_at_maximum.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # lithium hydride LiH
    mol = pyscf.gto.M(
        atom = 'Li 0 0 -0.79745; H 0 0 0.79745',
        basis = 'cc-pvdz',
        # singlet
        spin = 0)

    plt.style.use('./latex.mplstyle')
    compare_vW_kinetic_energy_densities(mol, nstate=4)
