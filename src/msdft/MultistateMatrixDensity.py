#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The multistate matrix density D(r) for a subspace of N-electronic states an N x N
matrix with the state densities on the diagonal and the transition densities on the
off-diagonal.
"""
from abc import ABC

import numpy
import scipy.linalg
import scipy.special

from pyscf.dft import numint


class MultistateMatrixDensity(ABC):
    def __init__(
            self,
            mol,
            density_matrices):
        """
        This class holds the multistate matrix density and can evaluate
        D(r), ∇D(r) and ∇²D(r) on a grid.

        This is the base class, derived classes have to implement their own __init__
        functions to compute the (transition) density matrices and then call
        super().__init__(mol, density_matrices).

        :param mol: molecule with atomic coordinates, basis set and spin
        :type mol: pyscf.gto.Mole

        :param density_matrices:
           density_matrices[spin,i,j,:,:] is the (transition) density matrix between
           the electronic states i and j in the AO basis.
        :type density_matrices: numpy.ndarray of shape (2,nstate,nstate,nao,nao)
           nstate - number of electronic states
           nao - number of atomic orbitals
        """
        # Save molecule with AO basis.
        self.mol = mol
        # Check the dimensions of the (transition) density matrix.
        nspin, nstate1, nstate2, nao1, nao2 = density_matrices.shape
        assert nspin == 2, "Density matrix needs components for spin-up and spin-down."
        assert nstate1 == nstate2, "Matrix density has to be square"
        assert nao1 == nao2, "AO density matrix has to be square"

        # Number of electronic states.
        self.number_of_states = nstate1
        # Save (transition) density matrices.
        self.density_matrices = density_matrices

    def exact_1e_operator(self, intor='int1e_kin'):
        """
        For testing purposes the matrix of one-electron operators in the
        basis of the electronic states is calculated by contracting the
        (transition) density matrices in the AO basis with the AO integrals
        of the operator:

          Oᵢⱼ = <Ψᵢ|∑ₙ oₙ|Ψⱼ>

              = sum_{a,b} P^{i,j}_{a,b} <a|o|b>

        where i,j enumerate many-electron states, a,b are AOs and P^{i,j}_{a,b}
        is the (transition) density between the states i and j in the AO basis.

        :param intor: Name of the 1-electron integrals, e.g. 'int1e_kin' for
           the kinetic energy.
        :type intor: str

        :return matrix_elements: The matrix elements of the operator in the
           basis of the many-electron states in the subspace.
        :rtype matrix_elements: numpy.ndarray of shape (nstate,nstate)
        """
        integrals_1e_ao = self.mol.intor_symmetric(intor)
        matrix_elements = numpy.einsum(
            'ab,sijab->ij',
            integrals_1e_ao,
            self.density_matrices)

        return matrix_elements

    def exact_kinetic_energy(self):
        """
        compute the exact matrix elements of the kinetic energy operator

          Tᵢⱼ = <Ψᵢ|∑ₙ ∇²ₙ|Ψⱼ>

        :return kinetic_matrix: The matrix elements of the kinetic energy
           in the basis of the many-electron states in the subspace.
        :rtype kinetic_matrix: numpy.ndarray of shape (nstate,nstate)
        """
        return self.exact_1e_operator(intor='int1e_kin')

    def exact_coulomb_energy(self):
        """
        Compute the Coulomb integrals for all possible combinations of
        (transition) densities using the exact integrals between the Gaussian
        atomic orbitals.

          C[i,j,k,l] = ∫∫' Dᵢⱼ(r) Dₖₗ(r') /|r-r'|

                     = sum_{a,b,c,d} P^{i,j}_{a,b} (ab|cd) P^{k,l}_{c,d}

        where the (transition) density is expanded in the AO basis.

          Dᵢⱼ(r) = sum_{a,b} P^{i,j}_{a,b} χ_a(r) χ_b(r)

        :return coulomb_integrals:
           Coulomb integrals between (transition) densities
        :rtype coulomb_integrals:
           numpy.ndarray of shape (nstate,nstate,nstate,nstate)
        """
        nstate = self.number_of_states
        # Electron repulsion integrals (ab|cd)
        integrals_eri = self.mol.intor('int2e')
        # sum over spin
        dm_spin_trace = self.density_matrices[0,...] + self.density_matrices[1,...]

        # All combinations of Coulomb interactions between (transition densities)
        # D_{i,j}(r) and D_{k,l}(r)
        coulomb_integrals = numpy.einsum(
            'ijab,abcd,klcd->ijkl',
            dm_spin_trace,
            integrals_eri,
            dm_spin_trace)

        return coulomb_integrals

    def evaluate(self, coords):
        """
        evaluate the multistate matrix density D(r), its gradient ∇D(r)
        and its Laplacian ∇²D(r) on a grid.

        Mstate is the number of electronic states
        Ncoord is the number of grid points.

        :param coords: The Cartesian coordinates of the grid r
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: D, grad_D, lapl_D
        :rtype: tuple of numpy.ndarray
          `D` has shape (2,Mstate,Mstate,Ncoord), D[s,i,j,c] is the (transition) density matrix
            for electrons with spin projection s=0 (up) or s=1 (down) evaluated at the grid point coords[c,:]
          `grad_D` has shape (2,Mstate,Mstate,3,Ncoord), grad_D[s,i,j,xyz,c] is the first-order
            derivative dD_ij(r)/dq (q=0(x), 1(y), 2(z)) evaluated at the grid point coords[c,:]
          `lapl_D` has shape (2,Mstate,Mstate,Ncoord), D[s,i,j,c] is the Laplacian of the
            (transition) density matrix for electrons with spin projection s=0 (up) or s=1 (down)
            evaluated at the grid point coords[c,:]
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states
        nstate = self.number_of_states
        # number of spins (up and down)
        nspin = 2

        # Create empty arrays for return values.
        D = numpy.zeros((nspin,nstate,nstate,ncoord))
        grad_D = numpy.zeros((nspin,nstate,nstate,3,ncoord))
        lapl_D = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Evaluate atomic orbitals 𝛘ₐ(r) on the grid.
        # The orbital values and their gradients and Laplacian are returned in a single
        # array of shape (10,ncoord,norb).
        ao_value_all = numint.eval_ao(self.mol, coords, deriv=2)
        # value AO(r)
        ao_value = ao_value_all[0,:,:]
        # gradient d(AO)/dx, d(AO)/dy, d(AO)/dz
        grad_ao_value = ao_value_all[1:4,:,:]
        # Laplacian ∇²(AO)(r) = d^2(AO)/dx^2 + d^2(AO)/dy^2 + d^2(AO)/dz^2
        lapl_ao_value = ao_value_all[4,:,:] + ao_value_all[7,:,:] + ao_value_all[9,:,:]

        # Evaluate the matrix density functions on the grid.
        for spin in range(0, nspin):
            for i in range(0, nstate):
                for j in range(0, nstate):
                    # (transition) density in AO basis.
                    dao_ij = self.density_matrices[spin,i,j,:,:]
                    D[spin,i,j,:] = numpy.einsum('ab,ra,rb->r', dao_ij, ao_value, ao_value)
                    grad_D[spin,i,j,:,:] = (
                        numpy.einsum('ab,gra,rb->gr', dao_ij, grad_ao_value, ao_value) +
                        numpy.einsum('ab,ra,grb->gr', dao_ij, ao_value, grad_ao_value))

                    # ∇²D(r) = sum_{a,b} P_{a,b} [ (∇²𝛘*_a)(𝛘_b) + 2 (∇𝛘_a)·(∇𝛘_b) + (𝛘_a)(∇²𝛘*_b) ]
                    lapl_D[spin,i,j,:] = (
                        numpy.einsum('ab,ra,rb->r', dao_ij, lapl_ao_value, ao_value) +
                        2*numpy.einsum('ab,gra,grb->r', dao_ij, grad_ao_value, grad_ao_value) +
                        numpy.einsum('ab,ra,rb->r', dao_ij, ao_value, lapl_ao_value)
                        )

        return D, grad_D, lapl_D

    def evaluate_derivatives(self, coords, deriv=2):
        """
        evaluate the derivatives of the multistate matrix density D(r) on a grid.

        The partial derivatives of order n=0,...,`deriv` are calculated along
        the x-, y- and z-axes:

          ∂ⁿ/∂xⁿ D(x,y,z), ∂ⁿ/∂yⁿ D(x,y,z) and ∂ⁿ/∂zⁿ D(x,y,z)

        Mixed derivatives such as ∂²/∂x∂y D(x,y,z) are left out.

        Mstate is the number of electronic states
        Ncoord is the number of grid points.

        :param coords: The Cartesian coordinates of the grid r
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :param deriv: maximum order of derivatives
        :type deriv: int >= 0

        :return: D_deriv
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,3,deriv+1,Ncoord)
          D_deriv[s,i,j,:,n,c] are the order n partial derivatives of the the (i,j)
          element of the (transition) matrix density Dˢᵢⱼ(x,y,z) with spin projection s (0=up, 1=down),
          [∂ⁿ/∂xⁿ Dˢᵢⱼ, ∂ⁿ/∂yⁿ Dˢᵢⱼ, ∂ⁿ/∂zⁿ Dˢᵢⱼ], evaluated at the grid point (x,y,z) = coords[c,:].

          Note that D_deriv[s,i,j,:,0,c] (the 0-order derivatives) is just the matrix density Dˢᵢⱼ(r)
          repeated for x, y and z.
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states
        nstate = self.number_of_states
        # number of spins (up and down)
        nspin = 2

        # Evaluate atomic orbitals 𝛘ₐ(r) on the grid.
        # The orbital values and their derivatives are returned in a single array.
        ao_derivs = numint.eval_ao(self.mol, coords, deriv=deriv)
        # The partial derivatives are ordered as follows:
        #
        # deriv=1:   ['x', 'y', 'z']
        # deriv=2:   ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
        # deriv=3:   ['xxx', 'xxy', 'xxz', 'xyy', 'xyz', 'xzz', 'yyy', 'yyz', 'yzz', 'zzz']
        #
        # The ordering for the general case deriv=n can be found with
        # >>> list(map(lambda s: ''.join(s), list(itertools.combinations_with_replacement('xyz', n))))
        #

        # We are only interested in the partial derivatives along the same axis,
        # i.e. x, xx, xxx, ...
        #      y, yy, yyy, ...
        #      z, zz, zzz, ...
        # `index_into_derivatives[k,:]` gives the indeces at with the derivatives
        # ∂ᵏ/∂xᵏ, ∂ᵏ/∂yᵏ and ∂ᵏ/∂zᵏ can be found in the array of partial derivatives returned by pyscf.
        index_into_derivatives = numpy.zeros((deriv+1, 3), dtype=int)
        offset = 0
        for k in range(0, deriv+1):
            # The derivatives of degree k contribute k*(k+1)/2 components in `ao_derivs`
            offset += (k*(k+1))//2
            # x-derivatives ∂ᵏ/∂xᵏ 𝛘 are stored at the index offset+0 of `ao_derivs`
            index_into_derivatives[k, 0] = offset + 0
            # y-derivatives ∂ᵏ/∂yᵏ 𝛘 are stored at the index offset+k*(k+1)/2 of `ao_derivs`
            index_into_derivatives[k, 1] = offset + (k*(k+1))//2
            # z-derivatives ∂ᵏ/∂zᵏ 𝛘 are stored at the index offset+(k+1)*(k+2)/2-1 of `ao_derivs`
            index_into_derivatives[k, 2] = offset + ((k+1)*(k+2))//2-1

        # Allocate memory for the derivatives
        D_derivs = numpy.zeros((nspin,nstate,nstate,3,deriv+1,ncoord))

        # Evaluate the derivatives of the matrix density Dˢᵢⱼ(x,y,z) on the grid.
        # Loop over spins
        for spin in range(0, nspin):
            # Loop over electronic states (bra)
            for i in range(0, nstate):
                # Loop over electronic state (ket)
                for j in range(0, nstate):
                    # (transition) density Pⁱʲ_{a,b} in AO basis.
                    dao_ij = self.density_matrices[spin,i,j,:,:]
                    # Since Dᵢⱼ(r) = ∑_{a,b} Pⁱʲ_{a,b} 𝛘a(r) 𝛘b(r),
                    # the derivatives of D(r) are obtained by applying the product rule repeatedly,
                    # which gives rise to Leibniz' rule:
                    #   ∂ⁿ/∂xⁿ Dᵢⱼ = ∑_{k=0}^n binom(n,k)
                    #                    ∑_{a,b} Pⁱʲ_{a,b} [∂ⁿ⁻ᵏ/∂xⁿ⁻ᵏ 𝛘a] [∂ᵏ/∂xᵏ 𝛘b]
                    for n in range(0, deriv+1):
                        # ∑_{k=0}^n
                        for k in range(0, n+1):
                            # Loop over x,y and z axis.
                            for xyz in [0,1,2]:
                                D_derivs[spin,i,j,xyz,n,:] += (
                                    scipy.special.binom(n,k) *
                                    # ∑_{a,b}
                                    numpy.einsum('ab,ra,rb->r',
                                        # Pⁱʲ_{a,b}
                                        dao_ij,
                                        # ∂ⁿ⁻ᵏ/∂xⁿ⁻ᵏ 𝛘a(r)
                                        ao_derivs[index_into_derivatives[n-k, xyz],:,:],
                                        # ∂ᵏ/∂xᵏ 𝛘b(r)
                                        ao_derivs[index_into_derivatives[k,   xyz],:,:])
                                    )

        return D_derivs

    def kinetic_energy_density(
            self,
            coords : numpy.ndarray):
        """
        The kinetic energy density

           KEDᵢⱼ(r) = <Ψᵢ|-1/2 ∑ₙ δ(r-rₙ) ∇ₙ²|Ψⱼ>

        can be computed in two ways:

          KEDᵢⱼ(r) = -1/2 ∑_a ∑_b Dᵢⱼ(a,b) 𝛘_a(r) ∇²𝛘_b(r)   (Laplacian)

        or as

          KEDᵢⱼ(r) = 1/2 ∑_a ∑_b Dᵢⱼ(a,b) ∇𝛘_a(r) · ∇𝛘_b(r)   (scalar product of gradients)

        Both kinetic energy densities integrate to the same kinetic energy matrix
        (see DOI:10.1063/1.1565316) as they only differ by a term of the form
        1/4 ∇²D(r) that vanishes after integrating over all space.

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: (KEDlapᵢⱼ(r), KEDggᵢⱼ(r))
           kinetic energy densities computed in the two ways
        :rtype: two numpy.ndarray's of shape (2,Mstate,Mstate,Ncoord) each
           KED[s,i,j,r] is the kinetic energy density
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states
        nstate = self.number_of_states
        # number of spins (up and down)
        nspin = 2

        # Create empty arrays for the kinetic energy densities.
        KED_laplacian = numpy.zeros((nspin,nstate,nstate,ncoord))
        KED_gradgrad = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Evaluate atomic orbitals 𝛘ₐ(r) on the grid.
        # The orbital values and their gradients are returned in a single
        # array of shape (4,ncoord,norb).
        ao_value_all = numint.eval_ao(self.mol, coords, deriv=2)
        # value AO(r)
        ao_value = ao_value_all[0,:,:]
        # gradient d(AO)/dx, d(AO)/dy, d(AO)/dz
        grad_ao_value = ao_value_all[1:4,:,:]
        # Laplacian ∇²(AO)(r) = d^2(AO)/dx^2 + d^2(AO)/dy^2 + d^2(AO)/dz^2
        lapl_ao_value = ao_value_all[4,:,:] + ao_value_all[7,:,:] + ao_value_all[9,:,:]

        # Evaluate the kinetic energy density
        for spin in range(0, nspin):
            for i in range(0, nstate):
                for j in range(0, nstate):
                    # (transition) density in AO basis.
                    dao_ij = self.density_matrices[spin,i,j,:,:]

                    # using the Laplacian of the orbitals
                    KED_laplacian[spin,i,j,:] = -0.5 * numpy.einsum(
                        'ab,ra,rb->r',
                        dao_ij, ao_value, lapl_ao_value)
                    # or using the gradients of the orbitals.
                    KED_gradgrad[spin,i,j,:] = 0.5 * numpy.einsum(
                        'ab,dra,drb->r',
                        dao_ij, grad_ao_value, grad_ao_value)

        return KED_laplacian, KED_gradgrad


class MultistateMatrixDensityFCI(MultistateMatrixDensity):
    def __init__(
            self,
            mol,
            rhf,
            cisolver,
            fcivecs):
        """
        This class holds the multistate matrix density and can evaluate
        D(r), ∇D(r) and ∇²D(r) on a grid.
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
        :type fcivecs: list of numpy.ndarray
        """
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
                rhf.mo_coeff, dm_mo, rhf.mo_coeff)
            return dm_ao

        # number of electronic states
        nstate = len(fcivecs)
        # Compute the (transition) density matrices in the AO basis.
        nspin = 2
        density_matrices = numpy.zeros((nspin,nstate,nstate,nao,nao))
        for i in range(0, nstate):
            for j in range(0, nstate):
                if i == j:
                    # 1-particle density matrix of state i in MO basis
                    dm1a, dm1b = cisolver.make_rdm1s(fcivecs[i], nmo, mol.nelec)
                    # for spin-up
                    density_matrices[0,i,i,:,:] = density_matrix_mo2ao(dm1a)
                    # for spin-down
                    density_matrices[1,i,i,:,:] = density_matrix_mo2ao(dm1b)
                else:
                    # 1-particle transition density matrix
                    # between electronic states i and j.
                    tdm1a, tdm1b = cisolver.trans_rdm1s(fcivecs[i], fcivecs[j], nmo, mol.nelec)
                    # for spin-up
                    density_matrices[0,i,j,:,:] = density_matrix_mo2ao(tdm1a)
                    # for spin-down
                    density_matrices[1,i,j,:,:] = density_matrix_mo2ao(tdm1b)

        # Initialize base class.
        super().__init__(mol, density_matrices)


class MultistateMatrixDensityTDDFT(MultistateMatrixDensity):
    def __init__(
            self,
            mol,
            rks,
            tddft):
        """
        This class holds the multistate matrix density and can evaluate
        D(r), ∇D(r) and ∇²D(r) on a grid.
        The state densities and transition densities are constructed from
        a TD-DFT calculation with pyscf.

        :param mol: molecule with atomic coordinates, basis set and spin
        :type mol: pyscf.gto.Mole

        :param rks: restricted solution of Kohn-Sham equations with molecular orbitals
        :type rks: pyscf.dft.RKS

        :param tddft: converged solution of the TD-DFT problem
        :type tddf: pyscf.tddft.TDDFT
        """
        # Check input TD-DFT calculation.
        assert numpy.all(tddft.converged), "TD-DFT calculation is not converged."
        assert tddft.singlet == True, "Only singlet excited states are supported."

        # number of atomic orbitals and molecular orbitals
        nao, nmo = rks.mo_coeff.shape

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
                rks.mo_coeff, dm_mo, rks.mo_coeff)
            return dm_ao

        # number of electronic states (ground + excited states)
        nstate = tddft.nstates+1
        # Compute the (transition) density matrices in the AO basis.
        nspin = 2

        # A and B matrices from Casida's equation
        #  [A  B] [X]     [0  1] [X]
        #  [    ] [ ] = w [    ] [ ]
        #  [B  A] [Y]     [-1 0] [Y]
        A, B = tddft.get_ab()
        nocc, nvir, _, _ = A.shape
        assert nocc+nvir == nmo

        # Compute (A-B)⁻¹ᐟ²
        AminusB = A-B
        # convert 4D tensor into matrix, (i,a,j,b) -> (ia,jb)
        AminusB = numpy.reshape(AminusB, (nocc*nvir, nocc*nvir))
        # perform linear operations on matrix
        invsqrtAminusB = scipy.linalg.inv(scipy.linalg.sqrtm(AminusB))
        # convert matrix back to 4D tensor (ia,jb) -> (i,a,j,b)
        invsqrtAminusB = numpy.reshape(invsqrtAminusB, (nocc,nvir,nocc,nvir))

        # Although the concept of a wavefunction is alien to density functional theory,
        # the Casida ansatz (Casida 1995), assigns a CIS-like wavefunction to an excited
        # state.
        cis_coefficients = numpy.zeros((nstate-1,nocc,nvir))
        # Loop over excited states and convert excitation (X) and deexcitation (Y)
        # coefficients into coefficients in the basis of singly-excited, spin-adapted
        # configuration functions.
        for istate in range(1, nstate):
            Xi,Yi = tddft.xy[istate-1]
            XplusY = Xi+Yi
            # CIS[istate,o,v] = sqrt(w) (A-B)⁻¹ᐟ² (X+Y)
            cis_coefficients[istate-1,:,:] = (
                numpy.sqrt(tddft.e[istate-1]) *
                numpy.einsum('iajb,jb->ia', invsqrtAminusB, XplusY))
            # Normalize CIS coefficients,
            # this is needed because X and Y are normalized to 1.
            cis_coefficients[istate-1,:,:] /= scipy.linalg.norm(
                cis_coefficients[istate-1,:,:])

            # NOTE: pyscf calculates the transition matrix elements of operators
            #       directly from X+Y (Hermitian operator such as the dipole operator)
            #       or X-Y (non-Hermitian) without an intermediate "wavefunction",
            #       see eqn. (33) in https://doi.org/10.1063/1.4937410 .
            #       Apparently the transition dipoles from the CIS-like "wavefunction"
            #       differ slightly from the correct TD-DFT transition dipoles.
            #
            #       Using the CIS coefficients below leads to the same transition dipoles
            #       that are output by `tddft.transition_dipole()`, but the corresponding
            #       CIS "wavefunction" are not orthonormal.
            #Xi,Yi = tddft.xy[istate-1]
            #cis_coefficients[istate-1,:,:] = 2*(Xi+Yi)

        # Store the CIS coefficients (only needed for unittests)
        self.cis_coefficients = cis_coefficients
        # Store TD-DFT object (only needed for unittests)
        self.tddft = tddft

        # Indices of occupied and virtual orbitals
        occ_indices = numpy.arange(0, nocc)
        vir_indices = numpy.arange(nocc, nocc+nvir)

        # (transition) density matrices
        density_matrices = numpy.zeros((nspin,nstate,nstate,nao,nao))
        for i in range(0, nstate):
            for j in range(0, nstate):
                if i == j:
                    # occupied-occupied, occupied-virtual, virtual-occupied
                    # and virtual-virtual blocks in the (transition) density.
                    oo_block = numpy.ix_(occ_indices, occ_indices)
                    ov_block = numpy.ix_(occ_indices, vir_indices)
                    vo_block = numpy.ix_(vir_indices, occ_indices)
                    vv_block = numpy.ix_(vir_indices, vir_indices)

                    # 1-particle density matrix
                    dm1 = numpy.zeros((nmo,nmo))
                    if i == 0:
                        # of DFT ground state (in MO basis)
                        # Doubly occupied orbitals in the ground state.
                        dm1[occ_indices, occ_indices] = 2.0
                    else:
                        # electron density
                        dm1_electron = numpy.einsum(
                            # sum over occupied orbitals o
                            'ou,ov->uv',
                            cis_coefficients[i-1,:,:],
                            cis_coefficients[i-1,:,:])
                        # hole density
                        dm1_hole = numpy.einsum(
                            # sum over virtual orbitals v
                            'kv,lv->kl',
                            cis_coefficients[i-1,:,:],
                            cis_coefficients[i-1,:,:])
                        # total electron density of excited states
                        #  ρ(i) = ρ0 + ρ(electron) - ρ(hole)
                        # ρ0 - ground state density
                        dm1[occ_indices, occ_indices] = 2.0
                        # ρ(electron)
                        dm1[vv_block] += dm1_electron
                        # ρ(hole)
                        dm1[oo_block] -= dm1_hole

                    dm1_ao = density_matrix_mo2ao(dm1)
                    # for spin-up
                    density_matrices[0,i,i,:,:] = 0.5 * dm1_ao
                    # for spin-down
                    density_matrices[1,i,i,:,:] = 0.5 * dm1_ao
                else:
                    # 1-particle transition density matrix
                    tdm1 = numpy.zeros((nmo,nmo))

                    # between electronic states i and j.
                    if i == 0:
                        # transition density between ground and excited state j
                        tdm1[ov_block] = 0.5 * cis_coefficients[j-1,:,:]
                        tdm1[vo_block] = 0.5 * cis_coefficients[j-1,:,:].transpose()
                    elif j == 0:
                        # transition density between ground and excited state i
                        tdm1[ov_block] = 0.5 * cis_coefficients[i-1,:,:]
                        tdm1[vo_block] = 0.5 * cis_coefficients[i-1,:,:].transpose()
                    else:
                        # transition density between excited states i, j
                        tdm1[vv_block] = numpy.einsum(
                            'ou,ov->uv',
                            cis_coefficients[i-1,:,:],
                            cis_coefficients[j-1,:,:])
                        tdm1[oo_block] = numpy.einsum(
                            'kv,lv->kl',
                            cis_coefficients[i-1,:,:],
                            cis_coefficients[j-1,:,:])
                        for o in occ_indices:
                            tdm1[o,o] -= 2.0 * numpy.einsum(
                                'v,v->',
                                cis_coefficients[i-1,o,:],
                                cis_coefficients[j-1,o,:])

                    tdm1_ao = density_matrix_mo2ao(tdm1)
                    density_matrices[0,i,j,:,:] = 0.5*tdm1_ao
                    # for spin-down
                    density_matrices[1,i,j,:,:] = 0.5*tdm1_ao

        # Initialize base class.
        super().__init__(mol, density_matrices)
