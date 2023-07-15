#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy

import pyscf.dft

from msdft.MultistateMatrixDensity import MultistateMatrixDensity

class KineticOperatorFunctional(object):
    def __init__(self, mol, level=8):
        """
        A von-Weizsäcker-like functional that maps the matrix density D(r)
        to the matrix of the kinetic energy in the subspace.

        This functional should give the exact kinetic energy matrix for
        1-electron systems.

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

    def __call__(
            self,
            msmd : MultistateMatrixDensity):
        """
        compute the matrix of the kinetic energy operator in the subspace
        of excited states by evaluating the kinetic energy functional T[D(r)]
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
        # up or down spin
        nspin = 2
        # matrix element of the kinetic energy operator <i|Top|j>
        kinetic_matrix = numpy.zeros((nstate,nstate))

        # Evaluate D(r), ∇D(r), tr(D)(r) and ∇tr(D)(r) on the integration grid.
        D, grad_D, trace_D, grad_trace_D = msmd.evaluate(self.grids.coords)

        # Loop over spins. For kinetic energy is computed separately for each spin
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
            #  T_{i,j}(r) = 1/2 ∇ϕᵢ*(r) ∇ϕⱼ(r)
            #
            # is obtained by solving (1 - K).T = C for each grid point
            Id = numpy.eye(dim2)
            T = numpy.zeros((nstate,nstate,ncoord))
            for r in range(0, ncoord):
                Cvec = C[:,:,r].flatten()
                Tvec = numpy.linalg.solve(Id - K[:,:,r], Cvec)
                # Reinterpret the vector Tvec as a square matrix.
                T[:,:,r] = numpy.reshape(Tvec, (nstate,nstate))

            # The matrix of the kinetic energy operator in the subspace is obtained
            # by integration T_{i,j}(r) over space
            #
            #  ∫ -1/2 ϕᵢ*(r) ∇²ϕⱼ(r) = ∫ 1/2 ∇ϕᵢ*(r) ∇ϕⱼ(r)
            #
            kinetic_matrix += numpy.einsum('r,ijr->ij', self.grids.weights, T)

        return kinetic_matrix
