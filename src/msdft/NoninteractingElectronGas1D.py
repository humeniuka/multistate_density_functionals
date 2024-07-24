#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy

from msdft.SchroedingerEquation1D import SchroedingerEquation1D


class NoninteractingElectronGas1D:
    def __init__(self, se : SchroedingerEquation1D, nelec : int):
        """
        Non-interacting spinless Fermions in a one-dimensional
        confinement potential V(x).

        :param se: solution of Schroedinger equation with single-particle
            wavefunctions for a particle in the potential V(x).
        :type se: :class:`~.SchroedingerEquation1D`

        :param nelec: number of Fermions
        :type nelec: int >= 1
        """
        assert nelec >= 1
        # The solutions of the single-particle Schroedinger equation are used
        # to construct the many-electron density of the non-interacting electron gas.
        self.se = se
        self.nelec = nelec

    def electron_density(self):
        """
        compute the total electron density on the grid

            ρ(x) = ∑ᵢ |ψᵢ(x)|²

        :return density: electron density ρ(x)
        :rtype density: numpy.ndarray of shape (nx,)
        """
        density = numpy.zeros_like(self.se.grid_x, dtype=complex)
        # Sum over lowest nelec orbitals.
        # Each orbital can be occupied by a single spinless Fermion.
        for i in range(0, self.nelec):
            density += abs(self.se.wavefunctions[:,i])**2

        return density

    def kinetic_energy_density(self):
        """
        compute the total kinetic energy density on the grid

            t(x) = ∑ᵢ -ħ/(2 m) ψᵢ*(x) d²/dx² ψᵢ(x)

        :return ked: kinetic energy density t(x)
        :rtype ked: numpy.ndarray of shape (nx,)
        """
        ked = numpy.zeros_like(self.se.grid_x, dtype=complex)
        # Sum over lowest nelec orbitals.
        # Each orbital can be occupied by a single spinless Fermion.
        for i in range(0, self.nelec):
            wfn = self.se.wavefunctions[:,i]
            # ψᵢ*(x) T ψᵢ(x)
            ked += wfn.conjugate() * self.se.T(wfn)

        return ked
