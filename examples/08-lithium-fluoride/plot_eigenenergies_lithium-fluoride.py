#!/usr/bin/env python
# coding: utf-8
"""
plot the eigenenergies of the lowest few electronic states
as a function of the Li-F bond length.
"""
import json
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy
from pyscf.data.nist import HARTREE2EV

if __name__ == "__main__":
    plt.style.use('./latex.mplstyle')
    # Load scan data
    with open('electron_repulsion_energies_lithium-fluoride.json', 'r') as filehandle:
        scan_data = json.load(filehandle)

    # Bond lengths in Å.
    bond_length = numpy.array(scan_data['bond_length'])
    # All energies are in Hartree.
    eigenenergies = numpy.array(scan_data['eigenenergies'])

    # number of electronic states
    nstate = eigenenergies[0].shape[0]

    plt.ylabel(r"adiab. energies / eV")
    plt.xlabel(r"bond length / $\AA$")

    # At the last scan geometry the bond length should be large enough
    # that the lowest state corresponds to the dissociated, neutral Li
    # and F atoms.
    dissociation_limit = eigenenergies[-1,0]

    # Both states belong to the
    symmetry_label = [r'$1^1\!\Sigma^+$', r'$2^1\!\Sigma^+$']
    for i in range(0, nstate):
        line, = plt.plot(
            bond_length,
            (eigenenergies[:,i] - dissociation_limit) * HARTREE2EV,
            lw=2, alpha=0.5,
            label=symmetry_label[i]
        )

    plt.legend(reverse=True)

    # Otherwise the x-labels are partly cut off.
    plt.tight_layout()

    plt.savefig("eigenenergies_lithium-fluoride.svg")
    plt.savefig("eigenenergies_lithium-fluoride.png", dpi=300)

    plt.show()
