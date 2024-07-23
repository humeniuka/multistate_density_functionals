#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import numpy.fft
import scipy.linalg

class SchroedingerEquation1D:
    def __init__(
            self,
            grid_x : numpy.ndarray,
            potential_x : numpy.ndarray,
            hbar=1.0,
            mass=1.0):
        """
        Numerical solutions to one-dimensional Schrödinger equation

          -ħ/(2 m) d²/dx² ψᵢ(x) + V(x) ψᵢ(x) = εᵢ ψᵢ(x)

        on an equidistance grid.

        :param grid_x: equidistant grid for x
        :type grid_x: numpy.ndarray of size (nx,)

        :parm potential_x: values of potential V(x) on the grid
        :type potential_x: numpy.ndarray of size (nx,)

        :param hbar: value of Planck's constant
        :type hbar: float

        :param mass: electron mass
        :type mass: float
        """
        nx = grid_x.shape[0]
        dx = numpy.ediff1d(grid_x)
        assert grid_x.shape == (nx,), "`grid_x` argument must be one-dimensional."
        assert len(numpy.unique(numpy.ediff1d(grid_x)) == 1), "`grid_x` must be equidistant."
        assert potential_x.shape == (nx,), "`potential_x` argument must have same shape as `grid`."
        # store potential
        # x: equidistant grid in real space
        self.grid_x = grid_x
        # V(x)
        self.potential_x = potential_x
        # k: grid in reciprocal space
        self.dx = self.grid_x[1] - self.grid_x[0]
        self.grid_k = 2.0 * numpy.pi * numpy.fft.fftfreq(nx, d=self.dx)
        # units
        self.hbar = hbar
        self.mass = mass
        # Eigenvalues and eigenfunctions become available only after the SE has been solved.
        self.__solved = False

    def T(self, psi):
        """
        The action of the kinetic energy operator on a wavefunction

            T ψ(x) = -ħ/(2 m) d²/dx² ψ(x)

        is computed by a Fourier transform.

        :param psi: values of wavefunction ψ(x) on real space grid
        :type psi: numpy.ndarray of shape (nx,)

        :return Tpsi: T ψ(x) on real space grid
        :rtype Tpsi: numpy.ndarray of shape (nx,)
        """
        assert psi.shape == self.grid_x.shape, "`psi` argument must have same shape as `grid_x`."
        psi_fourier = numpy.fft.fft(psi)
        # d²/dx² becomse (-i k)² in reciprocal space
        Tpsi = -pow(self.hbar,2.0)/(2.0*self.mass) * (
            numpy.fft.ifft( -pow(self.grid_k,2) * psi_fourier ))

        return Tpsi

    def V(self, psi):
        """
        The action of the potential is diagonal in real space

            V(x) ψ(x)

        :param psi: values of wavefunction ψ(x) on real space grid
        :type psi: numpy.ndarray of shape (nx,)

        :return Vpsi: V(x)ψ(x)
        :rtype Vpsi: numpy.ndarray of shape (nx,)
        """
        assert psi.shape == self.grid_x.shape, "`psi` argument must have same shape as `grid_x`."
        Vpsi = self.potential_x * psi

        return Vpsi

    def H(self, psi):
        """
        Action of Hamiltonian on a wavefunction

           H ψ(x) = [T + V(x)] ψ(x)

        :param psi: values of wavefunction ψ(x) on real space grid
        :type psi: numpy.ndarray of shape (nx,)

        :return Hpsi: [T + V(x)] ψ(x)
        :rtype Hpsi: numpy.ndarray of shape (nx,)
        """
        Hpsi = self.T(psi) + self.V(psi)

        return Hpsi

    def hamiltonian_matrix(self):
        """
        Matrix representation of the Hamiltonian in the basis of δ-functions.

        :return H_matrix: matrix of Hamiltonian Hᵢⱼ = <δᵢ|T+V(x)|δⱼ>
        :rtype H_matrix: numpy.ndarray of shape (nx,nx)
        """
        # number of grid points
        nx = self.grid_x.shape[0]
        # Build the Hamiltonian matrix in the basis of δ-functions
        H_matrix = numpy.zeros((nx,nx), dtype=complex)
        # Loop over grid points for δ-functions as bra vectors.
        for i in range(0, nx):
            # <δᵢ| = δ(x-x[i])
            delta_i = numpy.zeros(nx)
            delta_i[i] = 1.0
            # Loop over grid points for δ-functions as ket vectors.
            for j in range(i, nx):
                # |δⱼ> = δ(x-x[j])
                delta_j = numpy.zeros(nx)
                delta_j[j] = 1.0
                # Hamiltonian matrix
                # Hᵢⱼ = <δᵢ|T+V(x)|δⱼ>
                H_matrix[i,j] = numpy.sum(delta_i.conjugate() * self.H(delta_j))
                # Hamiltonian matrix is Hermitian
                H_matrix[j,i] = H_matrix[i,j].conjugate()

        return H_matrix

    def solve(self):
        """
        Build and diagonalize the Hamiltonian matrix to obtain the single-particle
        eigenenergies and eigenfunctions on the grid.

        The eigenenergies and eigenfunctions are accessible as the properties
        `energies` and `wavefunctions`.
        """
        H_matrix = self.hamiltonian_matrix()
        # Diagonalize the Hamiltonian
        self.eigvals, self.eigvecs = scipy.linalg.eigh(H_matrix)
        # The eigenvectors are normalized such that ∑_k |ψᵢ(x[k])| = 1,
        # however the correct normalization should include the differential dx,
        # ∫ |ψᵢ(x)|² dx = ∑_k |ψᵢ(x[k])|² dx = 1
        self.eigvecs /= numpy.sqrt(self.dx)
        # The sign of the eigenfunctions is arbitrary.

        self.__solved = True

    @property
    def energies(self):
        """
        Eigenenergies εᵢ

        :return eigvals: eigvals[i] contains εᵢ
        :rtype eigvals: numpy.ndarray of shape (nx,)
        """
        if not self.__solved:
            raise RuntimeError("Call solve() before accessing eigenenergies.")
        return self.eigvals

    @property
    def wavefunctions(self):
        """
        Eigenfunctions ψᵢ

        :return eigvecs: eigvecs[:,i] contains ψᵢ(x) on the grid
        :rtype eigvecs: numpy.ndarray of shape (nx,nx)
        """
        if not self.__solved:
            raise RuntimeError("Call solve() before accessing wavefunctions.")
        return self.eigvecs
