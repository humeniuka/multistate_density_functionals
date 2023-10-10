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
from msdft.KineticOperatorFunctional import VonWeizsaeckerFunctional
from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

def create_matrix_density(mol, nstate=4):
    """
    Compute multistate matrix density for the lowest few excited states
    of a small molecule using full configuration interaction.
    
    :param mol: A test molecule
    :type mol: gto.Mole

    :param nstate: number of excited states to calculate
    :type nstate: positive int

    :return: multistate matrix density
    :rtype: MultistateMatrixDensity
    """
    assert nstate > 0
    hf = pyscf.scf.RHF(mol)
    # supress printing of SCF energy
    hf.verbose = 0
    # compute self-consistent field
    hf.kernel()

    cisolver = pyscf.fci.FCI(mol, hf.mo_coeff)
    # Solve for one state more than requested to avoid
    # problems when nstate == 1.
    cisolver.nroots = nstate+1
    fci_energies, fcivecs = cisolver.kernel()
    # Remove the additional state again. For small basis sets,
    # there can be fewer states than requested.
    if len(fcivecs) == nstate+1:
        fcivecs = fcivecs[:-1]
            
    msmd = MultistateMatrixDensityFCI(mol, hf, cisolver, fcivecs)

    return msmd


def compare_vW_kinetic_energy_densities(mol, nstate=2):
    """
    The kinetic energy density is plotted for different functionals
    (vW1e and vW) and is compared with the exact one.
    
    :param nstate: Number of electronic states in the subspace.
       The full CI problem is solved for the lowest nstate states.
    :type nstate: int > 0
    """
    # compute D(r) from full CI
    msmd = create_matrix_density(mol, nstate=nstate)
    
    # Plot T(0,0,z), cut along z-axis
    Ncoord = 5000
    coords = numpy.zeros((Ncoord, 3))
    r = numpy.linspace(-3.0, 3.0, Ncoord)
    coords[:,2] = r
        
    # Functionals for kinetic energy matrix.
    kinetic_vW1e = VonWeizsaecker1eFunctional(mol)
    kinetic_vW = VonWeizsaeckerFunctional(mol)

    # Evalute kinetic energy density along the cut ...
    # ... with the approximate functionals from the matrix density
    KED_vW1e = kinetic_vW1e.kinetic_energy_density(msmd, coords)
    KED_vW = kinetic_vW.kinetic_energy_density(msmd, coords)
    # ... and exactly from the wavefunction.
    KED_lap, KED_gg = msmd.kinetic_energy_density(coords)

    # Sum over spin.
    KED_vW1e = numpy.sum(KED_vW1e, axis=0)
    KED_vW = numpy.sum(KED_vW, axis=0)
    
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
        #axes[0,column].set_title(r'von Weizsäcker 1-electron (\textbf{vW1e})')
        #axes[1,column].set_title(r'von Weizsäcker (\textbf{vW})')

        axes[0,column].text(
            0.05, 0.95, 'vW1e',
            fontsize='x-large',
            bbox=dict(facecolor='grey', alpha=0.1),
            horizontalalignment='left', verticalalignment='top', transform=axes[0,column].transAxes)
        axes[1,column].text(
            0.05, 0.95, 'vW',
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
                r, KED_vW1e[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls='--')
            axes[1,column].plot(
                r, KED_vW[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls="-.")

    for row in [0,1]:
        for column in [0,1]:
            axes[row,column].legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    #plt.savefig("vW_kinetic_energy_LiH.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # lithium hydride LiH
    mol = pyscf.gto.M(
        atom = 'Li 0 0 -0.79745; H 0 0 0.79745',
        basis = '6-31g',
        # singlet
        spin = 0)

    plt.style.use('./latex.mplstyle')
    compare_vW_kinetic_energy_densities(mol, nstate=4)
