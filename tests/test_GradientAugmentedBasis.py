#!/usr/bin/env python
# coding: utf-8

import numpy

import pyscf.gto

from tqdm import tqdm
import unittest

from msdft.GradientAugmentedBasis import augment_basis_with_gradients
from msdft.GradientAugmentedBasis import orbital_gradient_projection

class TestGradientAugmentedBasis(unittest.TestCase):
    def create_test_molecules(self):
        """ dictionary with different molecules to run the tests on """
        molecules = {
            # Lithium dimer
            'lithium dimer': pyscf.gto.M(
                atom = 'Li 0 0 -0.55; Li 0 0 0.55',  # in Angstrom
                basis = '6-31g',
                symmetry = False,
                # gradient-augmented basis set has to use Cartesian basis functions.
                cart = True
            )
        }
        return molecules

    def check_resolution_of_identity(self, mol):
        """
        In the gradient-augmented basis both the basis function of the original basis
        a(r) and the their gradients ∇a(r) can be represented exactly.

        The orbital gradient ∇a(r) is projected onto the augmented basis using the
        resolution of identity,

          ∇a_projected(r) = ∑ₘ∑ₙ <r|m> S⁻¹ₘₙ <n|∇a>.

        and compared with ∇a. This test checks that the projection error is zero,

          ||∇a - ∇a_projected|| = 0.
        """
        # Compare ∇a and ∇a_projected on the z-axis.
        Ncoord = 2000
        coords = numpy.zeros((Ncoord, 3))
        r = numpy.linspace(-5.0, 5.0, Ncoord)
        coords[:,2] = r
        # differential dr for integration
        dr = numpy.ediff1d(r, to_end=r[-1]-r[-2])

        # Compute gradients of the basis functions and their projections onto
        # the gradient-augmented basis set.
        grad_ao, projected_grad_ao = orbital_gradient_projection(mol, coords, level=8)

        # Projection errors for each orbital and gradient component (x,y,z).
        # ||∇a - ∇a_projected||
        projection_errors = numpy.sqrt(
            numpy.einsum('r,dra->da', dr, abs(grad_ao - projected_grad_ao)**2)
            )
        numpy.testing.assert_almost_equal(
            projection_errors, numpy.zeros_like(projection_errors), decimal=3)

    def test_resolution_of_idensity(self):
        """
        Verify that the gradients of the atomic orbitals can be represented in the
        gradient-augmented basis set.
        """
        for name, mol in tqdm(
                    self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_resolution_of_identity(mol)


if __name__ == "__main__":
    unittest.main()
