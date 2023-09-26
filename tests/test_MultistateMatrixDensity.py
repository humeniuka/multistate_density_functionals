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

from msdft.MultistateMatrixDensity import MultistateMatrixDensity

class TestMultistateMatrixDensity(unittest.TestCase):
    def create_test_molecules(self):
        """ dictionary with different molecules to run the tests on """
        molecules = {
            # 1-electron systems
            'hydrogen atom': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = '6-31g',
                # doublet
                spin = 1),
            'hydrogen atom (large basis set)': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = 'aug-cc-pvtz',
                # doublet
                spin = 1),
            'hydrogen molecular ion': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 1,
                spin = 1),
            # 2-electron systems, paired spins
            'hydrogen molecule': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 0,
                spin = 0),
            # 3-electron systems, one unpaired spin
            'lithium atom': pyscf.gto.M(
                atom = 'Li 0 0 0',
                basis = '6-31g',
                # doublet
                spin = 1),
            # 4-electron system, closed shell
            'lithium hydride': pyscf.gto.M(
                atom = 'Li 0 0 0; H 0 0 1.60',
                basis = '6-31g',
                # singlet
                spin = 0),
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
        hf = pyscf.scf.RHF(mol)
        # supress printing of SCF energy
        hf.verbose = 0
        # compute self-consistent field
        hf.kernel()

        cisolver = pyscf.fci.FCI(mol, hf.mo_coeff)
        cisolver.nroots = nstate
        fci_energies, fcivecs = cisolver.kernel()

        fcivecs = numpy.asarray(fcivecs)
        msmd = MultistateMatrixDensity(mol, hf, cisolver, fcivecs)

        return msmd

    def check_integrals(self, mol):
        """
        check that the state density integrates to the correct number of electrons
        and that the transition density integrates to 0.

        :param mol: A test molecule
        :type mol: gto.Mole
        """
        # Example density.
        msmd = self.create_matrix_density(mol)
        # integration grid
        grids = pyscf.dft.gen_grid.Grids(msmd.mol)
        grids.level = 8
        grids.build()

        D, grad_D, lapl_D = msmd.evaluate(grids.coords)
        trace_D = numpy.einsum('siir->sr', D)

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
                        self.assertAlmostEqual(integrals[i,j], 0.0)

        # The trace over spin and electronic states should be equal to
        # (number of electrons) x (number of states)
        integral_trace_D = numpy.einsum('r,sr->', grids.weights, trace_D)
        self.assertAlmostEqual(integral_trace_D, number_of_electrons*nstate)

    def test_integrals(self):
        """ Check integrals of D(r) for all test molecules """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_integrals(mol)

    def check_derivatives(self, mol):
        """
        compare analytical gradients ∇D(r) and ∇tr(D)(r) and the Laplacian ∇²D(r)
        with numerical ones from finite differences
        """
        # Example density.
        msmd = self.create_matrix_density(mol)
        # Gradients are checked at random coordinates.
        ncoord = 100
        coords = 5.0*(numpy.random.rand(ncoord,3) - 0.5)

        # Analytical gradients and Laplacian of D
        D, grad_D, lapl_D = msmd.evaluate(coords)

        # Trace out electronic states to get tr(D) and ∇tr(D)
        trace_D = numpy.einsum('siir->sr', D)
        grad_trace_D = numpy.einsum('siiar->sar', grad_D)

        # Numerical gradients of D and tr(D)
        grad_D_numerical = numpy.zeros_like(grad_D)
        lapl_D_numerical = numpy.zeros_like(lapl_D)
        grad_trace_D_numerical = numpy.zeros_like(grad_trace_D)

        # dD/dx = [D(x+h) - D(x-h)]/(2 h)
        h = 0.001
        for xyz in [0,1,2]:
            # unit vector in the x,y or z-direction
            unit_vector = numpy.zeros(3)
            unit_vector[xyz] = 1.0

            # D(r+h*e_x)
            D_plus, _, _ = msmd.evaluate(coords + h*unit_vector)
            trace_D_plus = numpy.einsum('siir->sr', D_plus)
            # D(r-h*e_x)
            D_minus, _, _ = msmd.evaluate(coords - h*unit_vector)
            trace_D_minus = numpy.einsum('siir->sr', D_minus)

            # finite difference gradient
            grad_D_numerical[:,:,:,xyz,:] = (D_plus - D_minus)/(2*h)
            grad_trace_D_numerical[:,xyz,:] = (trace_D_plus - trace_D_minus)/(2*h)

            # Add finite difference approximation for second derivative to
            # numerical Laplacian.
            lapl_D_numerical += (D_plus - 2*D + D_minus)/pow(h,2)

        # Compare analytical and numerical gradients
        with self.subTest("gradient of D(r)"):
            # relative error |∇D-∇D(numerical)|/|∇D(numerical)|
            relative_error = (
                la.norm(grad_D - grad_D_numerical)/la.norm(grad_D_numerical))
            self.assertLess(relative_error, 1.0e-3)
            numpy.testing.assert_almost_equal(grad_D, grad_D_numerical, decimal=2)
        with self.subTest("gradient of tr(D)"):
            # relative error |∇trD-∇trD(numerical)|/|∇trD(numerical)|
            relative_error = (
                la.norm(grad_trace_D - grad_trace_D_numerical)/la.norm(grad_trace_D_numerical))
            self.assertLess(relative_error, 1.0e-3)
            numpy.testing.assert_almost_equal(grad_trace_D, grad_trace_D_numerical, decimal=2)
        with self.subTest("Laplacian of D(r)"):
            # relative error |∇²D-∇²D(numerical)|/|∇²D(numerical)|
            relative_error = (
                la.norm(lapl_D - lapl_D_numerical)/la.norm(lapl_D_numerical))
            self.assertLess(relative_error, 1.0e-3)
            numpy.testing.assert_almost_equal(lapl_D, lapl_D_numerical, decimal=2)


    def test_derivatives(self):
        """ Compare numerical and analytical derivatives of D(r) for all test molecules """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_derivatives(mol)

    def check_kinetic_energy_density(self, mol):
        """
        Check that the kinetic energy density integrates
        to the correct kinetic energy.
        """
        msmd = self.create_matrix_density(mol)
        # Compute the kinetic energy matrix exactly
        kinetic_matrix_exact = msmd.exact_1e_operator(intor='int1e_kin')

        # Generate the multicenter integration grid.
        grids = pyscf.dft.gen_grid.Grids(mol)
        grids.level = 8
        grids.build()

        # Evaluate the two types of kinetic energy densities on the grid.
        T_lap, T_gg = msmd.kinetic_energy_density(grids.coords)
        # Integrate over spin and space, Tᵢⱼ = ∫ Tᵢⱼ(r) dr
        kinetic_matrix_lap = numpy.einsum('r,sijr->ij', grids.weights, T_lap)
        kinetic_matrix_gg = numpy.einsum('r,sijr->ij', grids.weights, T_gg)

        # Compare with the exact matrix elements
        numpy.testing.assert_almost_equal(
            kinetic_matrix_gg, kinetic_matrix_exact, decimal=6)
        numpy.testing.assert_almost_equal(
            kinetic_matrix_lap, kinetic_matrix_exact, decimal=6)

    def test_kinetic_energy_density(self):
        """ compare integral of kinetic energy density with exact matrix elements """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_kinetic_energy_density(mol)


if __name__ == "__main__":
    unittest.main()
