#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The multistate matrix density D(r) for a subspace of N-electronic states an N x N
matrix with the state densities on the diagonal and the transition densities on the
off-diagonal.
"""
import numpy
from pyscf.dft import numint

class MultistateMatrixDensity(object):
    def __init__(
            self,
            mol,
            rhf,
            cisolver,
            fcivecs):
        """
        This class holds the multistate matrix density and can evaluate
        D(r), ∇D(r), tr(D)(r) and ∇tr(D)(r) on a grid.
        The state densities and transition densities are constructed from
        a full configuration interaction calculation with pyscf.

        :param mol: molecule with atomic coordinates, basis set and spin
        :type mol: pyscf.gto.Mole

        :param rhf: restricted self-consistent field solution with molecular orbitals
        :type rhf: pyscf.scf.RHF

        :param cisolver: full configuration interaction solved
        :type cisolver: pyscf.fci.FCI

        :param fcivecs: list of solutions vectors of the full CI problem for
          each electronic state in the subspace
        :type fcivecs: list of numpy.ndarrays
        """
        # Save molecule with AO basis.
        self.mol = mol
        # number of atomic orbitals and molecular orbitals
        nao, nmo = rhf.mo_coeff.shape

        def density_matrix_mo2ao(dm_mo):
            """
            transform a density matrix in the MO basis in the AO basis

              P^AO_{a,b}   = sum_{m,n} C*_{a,m} P^MO_{m,n} C_{b,n}

            a,b enumerate atomic orbitals, m,n enumerate molecular orbitals
            and C_{a,m} are the self-consistent field MO coefficients.

            :param dm_mo: density matrix in MO basis
            :type dm_mo: numpy.ndarray of shape (nmo,nmo)

            :return dm_ao: density matrix in AO basis
            :rtype dm_ao: numpy.ndarray of shape (nao,nao)
            """
            assert dm_mo.shape == (nmo,nmo)
            dm_ao = numpy.einsum(
                'am,mn,bn->ab',
                rhf.mo_coeff.conjugate(), dm_mo, rhf.mo_coeff)
            return dm_ao

        # number of electronic states
        nstate = len(fcivecs)
        self.number_of_states = nstate
        # Compute the (transition) density matrices in the AO basis.
        nspin = 2
        self.density_matrices = numpy.zeros(
            (nspin,nstate,nstate,nao,nao), dtype=complex)
        for i in range(0, nstate):
            for j in range(0, nstate):
                if i == j:
                    # 1-particle density matrix of state i in MO basis
                    dm1a, dm1b = cisolver.make_rdm1s(fcivecs[i], nmo, mol.nelec)
                    # for spin-up
                    self.density_matrices[0,i,i,:,:] = density_matrix_mo2ao(dm1a)
                    # for spin-down
                    self.density_matrices[1,i,i,:,:] = density_matrix_mo2ao(dm1b)
                else:
                    # 1-particle transition density matrix
                    # between electronic states i and j.
                    tdm1a, tdm1b = cisolver.trans_rdm1s(fcivecs[i], fcivecs[j], nmo, mol.nelec)               # for spin-up
                    self.density_matrices[0,i,j,:,:] = density_matrix_mo2ao(tdm1a)
                    # for spin-down
                    self.density_matrices[1,i,j,:,:] = density_matrix_mo2ao(tdm1b)

    def evaluate(self, coords):
        """
        evaluate the multistate matrix density D(r), its gradient ∇D(r),
        its trace over states tr(D) and the gradient of the trace ∇tr(D)
        on a grid.

        Mstate is the number of electronic states
        Ncoord is the number of grid points.

        :param coords: The Cartesian coordinates of the grid r
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: D, grad_D, trace_D, grad_trace_D
        :rtype: tuple of numpy.ndarray
          `D` has shape (2,Mstate,Mstate,Ncoord), D[s,i,j,c] is the (transition) density matrix
          for electrons with spin projection s=0 (up) or s=1 (down) evaluated at the grid point coords[c,:]
          `grad_D` has shape (2,Mstate,Mstate,3,Ncoord), grad_D[s,i,j,xyz,c] is the first-order
          derivative dD_ij(r)/dq (q=0(x), 1(y), 2(z)) evaluated at the grid point coords[c,:]
          `trace_D` has shape (2,Ncoord,), trace_D[s,:] = sum_i D[spin,i,i,:]
          `grad_trace_D` has shape (2,3,Ncoord) and is the gradient of `trace_D`.
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states
        nstate = self.number_of_states
        # number of spins (up and down)
        nspin = 2

        # Create empty arrays for return values.
        D = numpy.zeros((nspin,nstate,nstate,ncoord), dtype=complex)
        grad_D = numpy.zeros((nspin,nstate,nstate,3,ncoord), dtype=complex)
        trace_D = numpy.zeros((nspin,ncoord), dtype=complex)
        grad_trace_D = numpy.zeros((nspin,3,ncoord), dtype=complex)

        # Evaluate atomic orbitals on the grid.
        # The orbital values and their gradients are returned in a single
        # array of shape (4,ncoord,norb).
        ao_value_all = numint.eval_ao(self.mol, coords, deriv=1)
        # value AO(r)
        ao_value = ao_value_all[0,:,:]
        # gradient d(AO)/dx, d(AO)/dy, d(AO)/dz
        grad_ao_value = ao_value_all[1:4,:,:]

        # Evaluate the matrix density functions on the grid.
        for spin in range(0, nspin):
            for i in range(0, nstate):
                for j in range(0, nstate):
                    # (transition) density in AO basis.
                    dao_ij = self.density_matrices[spin,i,j,:,:]
                    D[spin,i,j,:] = numpy.einsum('ab,ra,rb->r', dao_ij, ao_value.conjugate(), ao_value)
                    grad_D[spin,i,j,:,:] = (
                        numpy.einsum('ab,gra,rb->gr', dao_ij, grad_ao_value.conjugate(), ao_value)
                        +numpy.einsum('ab,ra,grb->gr', dao_ij, ao_value.conjugate(), grad_ao_value))
                    if i == j:
                        # trace over electronic states.
                        trace_D[spin,:] += D[spin,i,i,:]
                        # grad tr(D) = tr(grad D)
                        grad_trace_D[spin,:,:] += grad_D[spin,i,i,:,:]

        return D, grad_D, trace_D, grad_trace_D
