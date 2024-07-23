#!/usr/bin/env python
# coding: utf-8
import unittest

import numpy
import numpy.polynomial
import numpy.testing
import scipy.special

from msdft.SchroedingerEquation1D import SchroedingerEquation1D


class TestSchroedingerEquation1D(unittest.TestCase):
    def check_harmonic_oscillator(self, omega=1.0, hbar=1.0, mass=1.0, nstates=3):
        """
        Compare the numerical solution of the Schrödinger equation on a grid
        with the exact values for the lowest `nstates` eigenstates.
        """
        # equidistant grid and potential energy
        nx = 120
        x = numpy.linspace(-6.0, 6.0, nx)
        V = 0.5 * mass * pow(omega * x,2)

        # Compute the lower few exact eigenfunctions
        y = numpy.sqrt(mass * omega / hbar) * x
        exponent = numpy.exp(-0.5 * pow(y,2))

        wavefunctions_ref = numpy.zeros((nx, nstates))
        energies_ref = numpy.zeros(nstates)
        for n in range(0, nstates):
            # energy level
            energies_ref[n] = hbar * omega * (n + 0.5)

            # n-th order physicist's Hermite polynomial
            coef_n = [0] * n + [1]
            Hn = numpy.polynomial.hermite.Hermite(coef_n)

            wavefunctions_ref[:,n] = (
                1.0/numpy.sqrt(pow(2,n) * scipy.special.factorial(n)) *
                pow(mass * omega / (numpy.pi * hbar), 0.25) *
                exponent * Hn(y))

        # Compute the eigenfunctions numerically
        se = SchroedingerEquation1D(x, V, hbar=hbar, mass=mass)
        se.solve()

        # compare eigenvalues
        numpy.testing.assert_allclose(
            energies_ref, se.energies[:nstates], rtol=1.0e-2)
        # Eigenfunctions have arbitrary global signs, before we compare with the reference
        # we have to align the phases.
        signs = numpy.sign(numpy.einsum('xi,xi->i',
                                        wavefunctions_ref,
                                        se.wavefunctions[:,:nstates]))
        # numerical wavefunctions with aligned global signs
        wavefunctions = numpy.einsum('xi,i->xi', se.wavefunctions[:,:nstates], signs)

        # compare eigenfunctions
        numpy.testing.assert_allclose(
            wavefunctions_ref,
            wavefunctions,
            atol=1.0e-2 * numpy.max(numpy.abs(wavefunctions_ref))
            )

    def test_harmonic_oscillators(self):
        """
        Test that the correct eigenvalues and eigenfunctions are computed
        numerically for harmonic oscillators with different masses and frequencies.
        """
        for omega in [1.0, 1.56]:
            for hbar in [1.0, 1.7]:
                for mass in [1.0, 0.8]:
                    with self.subTest(omega=omega, hbar=hbar, mass=mass):
                        self.check_harmonic_oscillator(
                            omega=omega,
                            hbar=hbar,
                            mass=mass)


if __name__ == "__main__":
    unittest.main()
