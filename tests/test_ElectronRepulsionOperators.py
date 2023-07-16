#!/usr/bin/env python
# coding: utf-8
import unittest

import numpy
import numpy.linalg as la
import numpy.testing

import pyscf.dft
import pyscf.fci
import pyscf.gto
import pyscf.scf

from tqdm import tqdm

from msdft.ElectronRepulsionOperators import HartreeLikeOperatorFunctional
from msdft.MultistateMatrixDensity import MultistateMatrixDensity

class TestHartreeLikeOperatorFunctional(unittest.TestCase):
    def create_test_molecules(self):
        """ dictionary with different molecules to run the tests on """
        molecules = {
            # 1-electron systems
            'hydrogen atom': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = '6-31g',
                # doublet
                spin = 1),
            # many electrons
            'water': pyscf.gto.M(
                atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
                basis = 'sto-3g',
                # singlet
                spin = 0),
        }
        return molecules

    def create_matrix_density(self, mol, nstate=4):
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

        msmd = MultistateMatrixDensity(mol, hf, cisolver, fcivecs)

        return msmd

    def check_exact_hartree_energy(self, mol, nstate=1):
        """
        The Hartree-like energy is calculated in two different ways:
         1) By solving the Poisson equation for each (transition) density and
            integrating the product of the resulting electrostatic potential
            V(r) with the matrix density D(r) numerically on a grid
         2) By contracting the electroc repulsion integrals (ab|cd) in the
            AO basis with the AO (transition) density matrices (exact).

        :param mol: A test molecule
        :type mol: gto.Mole

        :param nstate: Number of electronic states in the subspace.
           The full CI problem is solved for the lowest nstate states.
        :type nstate: int > 0
        """
        # functional for Hartree-like energy J[D(r)]
        hartree_like_functional = HartreeLikeOperatorFunctional(mol)

        # compute D(r) from full CI
        msmd = self.create_matrix_density(mol, nstate=nstate)

        # Evaluate J[D(r)] by solving the Poisson equation and integration on the grid.
        J_msdft = hartree_like_functional(msmd)

        # The exact potential energy matrix is calculated by contracting the
        # (transition) density matrices in the AO basis with the electron
        # repulsion integrals.

        # Electron repulsion integrals (ab|cd)
        coulomb_integrals = msmd.exact_coulomb_energy()
        J_exact = 0.5 * numpy.einsum('ikkj->ij', coulomb_integrals)

        numpy.testing.assert_almost_equal(J_msdft, J_exact, decimal=3)

    def test_hartree_like_matrix(self):
        """
        Compare nuclear potential from numerical integration with exact matrix elements
        """
        for name, mol in tqdm(
                self.create_test_molecules().items()):
            for nstate in tqdm([1,2]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_exact_hartree_energy(mol, nstate=nstate)

if __name__ == "__main__":
    unittest.main()
