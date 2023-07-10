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

from msdft.MultistateMatrixDensity import MultistateMatrixDensity

class TestMultistateMatrixDensity(unittest.TestCase):
    def create_matrix_density(self):
        """
        Compute multistate matrix density for the lowest few excited states
        of a small molecule using full configuration interaction.
        """
        mol = pyscf.gto.Mole()
        mol.build(
            atom = 'Li 0 0 0; H 0 0 0.74',  # in Angstrom
            basis = '6-31g',
            # singlet
            spin = 0,
        )
        hf = pyscf.scf.RHF(mol)
        hf.kernel()

        # number of orbitals
        norb = hf.mo_coeff.shape[1]

        cisolver = pyscf.fci.FCI(mol, hf.mo_coeff)
        cisolver.nroots = 4
        fci_energies, fcivecs = cisolver.kernel()

        msmd = MultistateMatrixDensity(mol, hf, cisolver, fcivecs)

        return msmd

    def test_integrals(self):
        """
        check that the state density integrates to the correct number of electrons
        and that the transition density integrates to 0.
        """
        # Example density.
        msmd = self.create_matrix_density()
        # integration grid
        grids = pyscf.dft.gen_grid.Grids(msmd.mol)
        grids.level = 8
        grids.build()

        D, grad_D, trace_D, grad_trace_D = msmd.evaluate(grids.coords)
        # The integral also involves a sum over spins.
        integrals = numpy.einsum('r,sijr->ij', grids.weights, D)

        nstate = msmd.number_of_states
        number_of_electrons = sum(msmd.mol.nelec)
        for i in range(0, nstate):
            for j in range(0, nstate):
                with self.subTest(i=i, j=j):
                    if i == j:
                        # State densities should integrate to the number of electrons.
                        self.assertAlmostEqual(integrals[i,i], number_of_electrons, places=3)
                    else:
                        # Integrating the transition density, just gives the overlap between
                        # the states, which should be zero for different eigenstates.
                        self.assertAlmostEqual(integrals[i,j], 0.0, places=4)

    def test_gradients(self):
        """
        compare analytical gradients ∇D(r) and ∇tr(D)(r)
        with numerical ones from finite differences
        """
        # Example density.
        msmd = self.create_matrix_density()
        # Gradients are checked at random coordinates.
        ncoord = 100
        coords = 5.0*(numpy.random.rand(ncoord,3) - 0.5)

        # Analytical gradients of D and tr(D)
        D, grad_D, trace_D, grad_trace_D = msmd.evaluate(coords)

        # Numerical gradients of D and tr(D)
        grad_D_numerical = numpy.zeros_like(grad_D)
        grad_trace_D_numerical = numpy.zeros_like(grad_trace_D)

        # dD/dx = [D(x+h) - D(x-h)]/(2 h)
        h = 0.001
        for xyz in [0,1,2]:
            # unit vector in the x,y or z-direction
            unit_vector = numpy.zeros(3)
            unit_vector[xyz] = 1.0

            # D(r+h*e_x)
            D_plus, _, trace_D_plus, _ = msmd.evaluate(coords + h*unit_vector)
            # D(r-h*e_x)
            D_minus, _, trace_D_minus, _ = msmd.evaluate(coords - h*unit_vector)

            # finite difference gradient
            grad_D_numerical[:,:,:,xyz,:] = (D_plus - D_minus)/(2*h)
            grad_trace_D_numerical[:,xyz,:] = (trace_D_plus - trace_D_minus)/(2*h)

        # Compare analytical and numerical gradients
        with self.subTest("gradient of D(r)"):
            # relative error |∇D-∇D(numerical)|/|∇D(numerical)|
            relative_error = (
                la.norm(grad_D - grad_D_numerical)/la.norm(grad_D_numerical))
            self.assertLess(relative_error, 1.0e-3)
            numpy.testing.assert_almost_equal(grad_D, grad_D_numerical, decimal=3)
        with self.subTest("gradient of tr(D)"):
            # relative error |∇trD-∇trD(numerical)|/|∇trD(numerical)|
            relative_error = (
                la.norm(grad_trace_D - grad_trace_D_numerical)/la.norm(grad_trace_D_numerical))
            self.assertLess(relative_error, 1.0e-3)
            numpy.testing.assert_almost_equal(grad_trace_D, grad_trace_D_numerical, decimal=3)

if __name__ == "__main__":
    unittest.main()

