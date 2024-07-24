#!/usr/bin/env python
# coding: utf-8
import unittest

import numpy

from msdft.SchroedingerEquation1D import SchroedingerEquation1D
from msdft.NoninteractingElectronGas1D import NoninteractingElectronGas1D


class TestNoninteractingElectronGas1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # equidistant grid and potential energy
        nx = 120
        x = numpy.linspace(-6.0, 6.0, nx)
        V = 0.5 * pow(x,2)

        # Compute the eigenfunctions numerically
        cls.se = SchroedingerEquation1D(x, V)
        cls.se.solve()

    def check_electron_density(self, nelec=2):
        """
        Check that the electron density integrates to the number of electrons.
        """
        # electron gas with N electrons
        electron_gas = NoninteractingElectronGas1D(self.se, nelec)
        density = electron_gas.electron_density()
        # compute N = ∫ ρ(x) dx
        nelec_integ = numpy.sum(density * self.se.dx)
        # compare with expected number of electrons
        self.assertAlmostEqual(nelec_integ, nelec)

    def test_electron_density(self):
        """ Check electron density for different number of electrons. """
        for nelec in [1,2,3]:
            with self.subTest(nelec=nelec):
                self.check_electron_density(nelec=nelec)

    def check_total_energy(self, nelec=2):
        # electron gas with N electrons
        electron_gas = NoninteractingElectronGas1D(self.se, nelec)
        density = electron_gas.electron_density()

        #
        ked = electron_gas.kinetic_energy_density()

        # compute the total energy by integration
        #    E = ∫ { t[ρ](x) + ρ(x) V(x) } dx
        total_energy = numpy.sum((ked + density * self.se.potential_x) * self.se.dx)
        # and by summing over the lowest N eigenvalues
        #    E = ∑ᵢ εᵢ
        total_energy_ref = numpy.sum(self.se.energies[:nelec])

        self.assertAlmostEqual(total_energy, total_energy_ref)

    def test_total_energy(self):
        """ Check total energy for different number of electrons. """
        for nelec in [1,2,3]:
            with self.subTest(nelec=nelec):
                self.check_total_energy(nelec=nelec)


if __name__ == "__main__":
    unittest.main()
