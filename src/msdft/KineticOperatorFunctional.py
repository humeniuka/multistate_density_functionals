#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy
import pyscf.dft
import scipy.linalg

from msdft.MultistateMatrixDensity import MultistateMatrixDensity


class KineticOperatorFunctional(ABC):
    def __init__(self, mol, level=8):
        """
        The abstract base class for kinetic operator functionals.

        :param mol: The molecule defines the integration grid.
        :type mol: pyscf.gto.Mole

        :param level: The level (3-8) controls the number of grid points
           in the integration grid.
        :type level: int
        """
        # generate a multicenter integration grid
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = level
        self.grids.build()

    @abstractmethod
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        pass

    def __call__(
            self,
            msmd : MultistateMatrixDensity):
        """
        compute the matrix of the kinetic energy operator in the subspace
        of electronic states by evaluating the kinetic energy functional T[D(r)]
        on the matrix density D(r):

          Tᵢⱼ = <Ψᵢ|-1/2 ∑ₙ∇ₙ²|Ψⱼ> = T[D(r)]ᵢⱼ ,

        where Dᵢⱼ(r) is the electronic density of the state Ψᵢ, Dᵢᵢ(r) = ρᵢ(r),
        or the transition density between the states Ψᵢ and Ψⱼ, Dᵢⱼ(r).

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy functional should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :return kinetic_matrix: The kinetic energy matrix Tᵢⱼ in the subspace
           of the electronic states i,j=1,...,nstate
        :rtype kinetic_matrix: numpy.ndarray of shape (nstate,nstate)
        """
        # number of grid points
        ncoord = self.grids.coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # matrix element of the kinetic energy operator <i|Top|j>
        kinetic_matrix = numpy.zeros((nstate,nstate))

        # Evaluate the kinetic energy density on the grid.
        KED = self.kinetic_energy_density(msmd, self.grids.coords)

        # The matrix of the kinetic energy operator in the subspace is obtained
        # by integration T_{i,j}(r) over space and spin
        #
        #   Tᵢⱼ = ∫ KEDᵢⱼ(r) dr
        #
        kinetic_matrix = numpy.einsum('r,sijr->ij', self.grids.weights, KED)

        return kinetic_matrix


class VonWeizsaeckerFunctional(KineticOperatorFunctional):
    """
    A von-Weizsäcker-like functional that maps the matrix density D(r)
    to the matrix of the kinetic energy in the subspace.

    This functional should give the exact kinetic energy matrix for
    1-electron systems.
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the kinetic energy density

           KEDᵢⱼ(r) = -1/2 ϕᵢ*(r) ∇²ϕⱼ(r)

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           KED[s,i,j,r] is the kinetic energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(coords)

        # Trace over electronic states to get tr(D)(r) and ∇tr(D)(r) = tr(∇D(r))
        # `trace_D` has shape (2,Ncoord,), trace_D[s,:] = sum_i D[spin,i,i,:]
        trace_D = numpy.einsum('siir->sr', D)
        # `grad_trace_D` has shape (2,3,Ncoord) and is the gradient of `trace_D`.
        grad_trace_D = numpy.einsum('siiar->sar', grad_D)

        # Loop over spins. The kinetic energy is computed separately for each spin
        # projection and added.
        for s in range(0, nspin):
            if numpy.all(trace_D[s,...] == 0.0):
                # There are no electrons with spin projection s
                # that could contribute to the kinetic energy.
                continue
            #
            # R_{i,j}(r) = D_{i,j}(r) / tr(D(r))
            #
            R = D[s,...] / numpy.expand_dims(trace_D[s,...], axis=(0,1))
            #                  ∑ₖ∇D_{i,k}·∇D_{k,j}
            # C_{i,j}(r) = 1/2 -------------------
            #                        tr(D)
            #
            C_numerator = numpy.einsum('ikar,kjar->ijr', grad_D[s,...], grad_D[s,...])
            C_denominator = numpy.expand_dims(trace_D[s,...], axis=(0,1))
            C = 0.5 * C_numerator / C_denominator

            # For each grid point we have to solve the linear equation
            #  (1 - K)·T = C
            # where T_{i,j}(r) and C_{i,j}(r) are interpreted as vectors in ℂ^{Mstate x Mstate}
            dim2 = nstate*nstate
            # identity matrix
            delta = numpy.eye(nstate)
            # K_{i,j;m,n} =  - R_{i,j} delta_{m,n}  - delta_{i,m} R_{n,j}  - R_{i,m} delta_{n,j}
            K = numpy.zeros((dim2,dim2,ncoord))
            # ij is a multiindex that runs over all combinations of (i,j) (rows of K)
            ij = 0
            for i in range(0, nstate):
                for j in range(0, nstate):
                    # mn is a multiindex that runs over all combinations of (m,n) (columns of K)
                    mn = 0
                    for m in range(0, nstate):
                        for n in range(0, nstate):
                            K[ij,mn,:] = (
                                -R[i,j,:]*delta[m,n]) -delta[i,m]*R[n,j,:] -R[i,m,:]*delta[n,j]
                            # increase column counter
                            mn += 1
                    # increase row counter
                    ij += 1

            # The kinetic energy density
            #
            #  KEDᵢⱼ(r) = 1/2 ∇ϕᵢ*(r) ∇ϕⱼ(r)
            #
            # is obtained by solving (1 - K).T = C for each grid point
            Id = numpy.eye(dim2)
            for r in range(0, ncoord):
                Cvec = C[:,:,r].flatten()
                Tvec = numpy.linalg.solve(Id - K[:,:,r], Cvec)
                # Reinterpret the vector Tvec as a square matrix.
                KED[s,:,:,r] = numpy.reshape(Tvec, (nstate,nstate))

        return KED


class VonWeizsaeckerAdHocFunctional(KineticOperatorFunctional):
    """
    A von-Weizsäcker-like functional that maps the matrix density D(r)
    to the matrix of the kinetic energy in the subspace.

    The von-Weizsäcker functional for the electronic ground state

                        (∇ρ)²
           T[ρ] = ∫ 1/8 ----
                          ρ

    is turned into a matrix-density functional by replacing the density
    with the matrix density, ρ(r) -> D(r),

           T[D]ᵢⱼ = ∫ 1/8 ∑ₖ∑ₗ ∇Dᵢₖ D⁻¹ₖₗ ∇Dₗⱼ

    The matrix-inverse of D is placed symmetrically between the gradients.

    It is not clear how this ad-hoc functional can be derived, since it
    does not give the exact kinetic energy matrix for 1-electron systems.
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the kinetic energy density

           KEDᵢⱼ(r) = <Ψᵢ|-1/2 ∑ₙ δ(r-rₙ) ∇ₙ²|Ψⱼ>

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           KED[s,i,j,r] is the kinetic energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(coords)

        # Trace over electronic states to get tr(D)(r)
        # `trace_D` has shape (2,Ncoord,), trace_D[s,:] = sum_i D[spin,i,i,:]
        trace_D = numpy.einsum('siir->sr', D)

        # Loop over spins. The kinetic energy is computed separately for each spin
        # projection and added.
        for s in range(0, nspin):
            if numpy.all(trace_D[s,...] == 0.0):
                # There are no electrons with spin projection s
                # that could contribute to the kinetic energy.
                continue
            # inverse of matrix density, D⁻¹ₖₗ(r) at each grid point
            invD = numpy.zeros_like(D[s,...])
            for r in range(0, ncoord):
                invD[:,:,r] = scipy.linalg.pinv(D[s,:,:,r])
            #
            # KED_{i,j}(r) = 1/8 ∑ₖ∑ₗ ∇D_{i,k} D⁻¹_{k,l} ·∇D_{l,j}
            #
            KED[s,...] = 1.0/8.0 * numpy.einsum(
                'ikar,klr,ljar->ijr',
                grad_D[s,...], invD, grad_D[s,...])

        return KED


class ThomasFermiFunctional(KineticOperatorFunctional):
    """
    A Thomas-Fermi-like functional that maps the matrix density D(r)
    to the matrix of the kinetic energy in the subspace.

      Tᵢⱼ = 3/10 (6π²)²ᐟ³ ∫ (D(r)⁵ᐟ³)ᵢⱼ dr

    Note that the power of 5/3 is not taken element-wise. D(r)⁵ᐟ³ is a matrix
    power that mixes the different elements of D(r).

    The kinetic energy is calculated separately for the spin-up and spin-down
    parts of the density matrix and summed:

      Tᵢⱼ[D(up)] + Tᵢⱼ[D(down)]

    Therefore the prefactor of the Thomas-Fermi energy contains (6π²) instead
    of (3π²), which is for the kinetic energy of the total density, Tᵢⱼ[D(up)+D(down)].
    """
    def kinetic_energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the kinetic energy density of the free electron gas

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: KEDᵢⱼ(r), kinetic energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           KED[s,i,j,r] is the kinetic energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # kinetic energy density KEDᵢⱼ(r)
        KED = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Evaluate D(r) on the integration grid.
        D, _, _ = msmd.evaluate(coords)

        # Trace over electronic states to get tr(D)(r).
        # `trace_D` has shape (2,Ncoord,), trace_D[s,:] = sum_i D[spin,i,i,:]
        trace_D = numpy.einsum('siir->sr', D)

        # Loop over spins. The kinetic energy is computed separately for each spin
        # projection and added.
        for s in range(0, nspin):
            if numpy.all(trace_D[s,...] == 0.0):
                # There are no electrons with spin projection s
                # that could contribute to the kinetic energy.
                continue

            # The kinetic energy density
            #
            #  KEDᵢⱼ(r) = 3/10 (6π²)²ᐟ³ (D(r)⁵ᐟ³)ᵢⱼ
            #
            # For a closed-shell molecule, Dtot = 2*D(up), so that
            #
            #  t = 3/10 (3π²)²ᐟ³ (2 Dtot) = 2 3/10 (6π²)²ᐟ³ D(up)
            #    = 2 tₛₚᵢₙ
            prefactor = 3.0/10.0 * pow(6.0*numpy.pi**2, 2.0/3.0)
            for r in range(0, ncoord):
                ked_r = prefactor * scipy.linalg.fractional_matrix_power(
                    D[s,:,:,r], 5.0/3.0)
                # Check that the kinetic energy density is real.
                assert numpy.sum(abs(ked_r.imag)) < 1.0e-10

                KED[s,:,:,r] = ked_r.real

        return KED
