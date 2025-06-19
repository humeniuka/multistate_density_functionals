#!/usr/bin/env python
# coding: utf-8
"""
There are at least two different (non-equivalent) ways to turn the
reduced density gradient squared

    x² = (∇ρ)² / ρ⁸ᐟ³,

which is used in Becke's 88 GGA, into a matrix functional. These are
    (1)  X² = D⁻⁴ᐟ³ ∇D·∇D D⁻⁴ᐟ³
and
    (2) from the gradient of the Wigner-Seitz radius as
    R  = (4π/3 D)⁻¹ᐟ³
    X² = (36π)²ᐟ³ ∇R(r)·∇R(r)

This script shows that the enhancement factor explodes whenever the
matrix density has eigenvalues that are zero.
"""
import functools
import matplotlib.pyplot as plt
import numpy

import pyscf.gto
from msdft.LinearAlgebra import matrix_function_batch, matrix_function_derivatives_batch
from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

# Always use greedy optimization
einsum = functools.partial(numpy.einsum, optimize='greedy')


def reduced_density_gradient_eqn1(msmd, coords):
    """
    compute reduced gradient of matrix density X² from the Wigner-Seitz radius as

        X² = D⁻⁴ᐟ³ ∇D·∇D D⁻⁴ᐟ³
    """
    epsilon_zero = 1.0e-12
    # number of grid points
    ncoord = coords.shape[0]
    # number of electronic states in the subspace
    nstate = msmd.number_of_states
    # up or down spin
    nspin = 2

    # Evaluate D(r) and ∇D(r) on the integration grid.
    D, grad_D, _ = msmd.evaluate(coords)

    def matrix_power(density):
        # Avoid dividing by zero for ρ=0.
        # Non-zero eigenvalues, for which division is not problematic.
        good = abs(density) > epsilon_zero
        # When ρ=0, the electron radius should be r=inf. However, since the
        # xc-energy is 0 if there are no electrons, any value can be chosen
        # for r(ρ=0). Here we set r(ρ=0) to 0.
        f = numpy.zeros_like(density)
        # Compute f(ρ) = ρ⁻⁴ᐟ³ for grid point where ρ > 0.
        f[good] = pow(abs(density[good]), -4.0/3.0)
        return f

    # The fractional matrix power is obtained from the eigenvalue decomposition.
    # as D⁻⁴ᐟ³(r) = U(r) Λ⁻⁴ᐟ³(r) Uᵀ(r)
    D_matrix_power = matrix_function_batch(matrix_power, D)

    # ∑ₐ ∇Dᵢₐ·∇Dₐⱼ
    grad_D_squared = einsum('siadr,sajdr->sijr', grad_D, grad_D)
    # D⁻⁴ᐟ³ ∇D·∇D D⁻⁴ᐟ³
    X2 = numpy.einsum('siar,sabr,sbjr->sijr', D_matrix_power, grad_D_squared, D_matrix_power)

    return grad_D_squared, X2

def reduced_density_gradient_eqn2(msmd, coords):
    """
    compute reduced gradient of matrix density X² from the Wigner-Seitz radius as

        R  = (4π/3 D)⁻¹ᐟ³
        X² = (36π)²ᐟ³ ∇R(r)·∇R(r)
    """
    epsilon_zero = 1.0e-12
    epsilon_degeneracy = 1.0e-12
    # number of grid points
    ncoord = coords.shape[0]
    # number of electronic states in the subspace
    nstate = msmd.number_of_states
    # up or down spin
    nspin = 2

    # Evaluate D(r) and ∇D(r) on the integration grid.
    D, grad_D, _ = msmd.evaluate(coords)

    # The fractional matrix power is obtained from the eigenvalue decomposition.
    # as D²ᐟ³(r) = U(r) Λ²ᐟ³(r) Uᵀ(r)
    D_matrix_power = matrix_function_batch(lambda L: pow(abs(L), 2.0/3.0), D)

    # The matrix version of the Wigner-Seitz radius is also calculated from the eigenvalue
    # decomposition as R(r) = U(r) (4π/3 Λ(r))⁻¹ᐟ³ Uᵀ(r).
    def wigner_seitz_radius(density):
        # Avoid dividing by zero for ρ=0.
        # Non-zero eigenvalues, for which division is not problematic.
        good = abs(density) > epsilon_zero
        # When ρ=0, the electron radius should be r=inf. However, since the
        # xc-energy is 0 if there are no electrons, any value can be chosen
        # for r(ρ=0). Here we set r(ρ=0) to 0.
        radius = numpy.zeros_like(density)
        # Compute Wigner-Seitz radius for grid point where ρ > 0.
        radius[good] = pow((4.0*numpy.pi/3.0) * abs(density[good]), -1.0/3.0)
        return radius

    def wigner_seitz_radius_deriv1(density):
        # Derivative of the Wigner-Seitz radius w/r/t the density
        #   ∇rₐ(r) = (-1/3) (4π/3)⁻¹ᐟ³  (ρ(r))⁻⁴ᐟ³ ∇ρ(r)
        #          = (-1/3) (4π/3) rₐ⁴ ∇ρ(r)
        #          = rₐ'(ρ) ∇ρ(r)
        radius = wigner_seitz_radius(density)
        # rₐ'(ρ) = (-1/3) (4π/3) rₐ⁴
        radius_deriv1 = (-1.0/3.0) * (4.0*numpy.pi/3.0) * pow(radius, 4.0)
        return radius_deriv1

    # Wigner-Seitz radius matrix, Rᵢⱼ, and its gradient, ∇Rᵢⱼ.
    R, grad_R = matrix_function_derivatives_batch(
        # f(ρ)
        wigner_seitz_radius,
        # f'(ρ)
        wigner_seitz_radius_deriv1,
        # matrix density Dᵢⱼ
        D,
        # derivatives of matrix density ∇Dᵢⱼ
        grad_D,
        # threshold for treating eigenvalues as degenerate
        epsilon_degeneracy=epsilon_degeneracy
    )

    # Compute X²(r) = (36π)²ᐟ³ ∇R(r)·∇R(r)
    #   X²ᵢⱼ = (36π)²ᐟ³ ∑ₐ ∇Rᵢₐ·∇Rₐⱼ
    X2 = pow(36.0 * numpy.pi, 2.0/3.0) * einsum(
        'siadr,sajdr->sijr', grad_R, grad_R)

    return X2

def compare_reduce_gradients_squared(mol, nstate=2):
    """

    :param nstate: Number of electronic states in the subspace.
       The full CI problem is solved for the lowest nstate states.
    :type nstate: int > 0
    """
    msmd = MultistateMatrixDensityFCI.create_matrix_density(
        mol, nstate=nstate,
        # Spin symmetry is turned off, since we just want the lowest
        # electronic states no matter what spin state.
        spin_symmetry=False
    )

    # Plot exchange-correlation energy density along z-axis
    ncoord = 2000
    r = numpy.linspace(-rHH, rHH, ncoord)
    coords = numpy.zeros((ncoord, 3))
    coords[:,2] = r

    # Evaluate D(r) and ∇D(r) on the integration grid.
    D, _, _ = msmd.evaluate(coords)

    # eigenvalues of density
    # numpy.linalg.eigh(...) can operate on multiple matrices in parallel,
    # Since the calculation is parallelized over the first axis, we have to
    # move the coordinate axis to the first position. For each grid point r
    # and spin orientation s, the (N x N)-matrix  X(r) is diagonalized.
    # (nspin,nstate,nstate,ncoord) -> (ncoord,nspin,nstate,nstate)
    D_reordered = numpy.moveaxis(D, 3, 0)
    D_eigenvalues, U = numpy.linalg.eigh(D_reordered)
    # Restore original order of axes
    #   (ncoord,nspin,nstate) -> (npin, nstate, ncoord)
    D_eigenvalues = numpy.moveaxis(D_eigenvalues, 0, 2)
    # sum over sping
    D_eigenvalues = numpy.sum(D_eigenvalues, axis=0)

    # sum over spins
    D = numpy.sum(D, axis=0)

    # Compute X² in two different ways.
    grad_D_squared, X2_eqn1 = reduced_density_gradient_eqn1(msmd, coords)
    X2_eqn2 = reduced_density_gradient_eqn2(msmd, coords)

    def enhancement_factor(x2):
        x2 = abs(x2)
        """ The scalar function f(x²) for the enhancement factor """
        # Cₓ = (3/4) (3/pi)¹ᐟ³ = 0.7386 from Dirac's exchange-energy, Eqn. (6.1.20) in [Parr&Yang]
        Cx = 3.0/4.0 * pow(3.0/numpy.pi, 1.0/3.0)
        # The empirical value of β (see table II in [Becke88]) was determined from a least square fit.
        beta = 0.0042
        # gamma should be 6 to get the right asymptotics of Ex.
        gamma = 6.0
        x = numpy.sqrt(x2)
        f = 1.0 + beta / (pow(2.0, 1.0/3.0) * Cx) * (
            x2 / (1 + gamma*beta * x * numpy.arcsinh(x))
        )
        return f
    F_eqn1 = matrix_function_batch(enhancement_factor, X2_eqn1)
    F_eqn2 = matrix_function_batch(enhancement_factor, X2_eqn2)

    # sum over spin
    grad_D_squared = numpy.sum(grad_D_squared, axis=0)
    X2_eqn1 = numpy.sum(X2_eqn1, axis=0)
    X2_eqn2 = numpy.sum(X2_eqn2, axis=0)
    F_eqn1 = numpy.sum(F_eqn1, axis=0)
    F_eqn2 = numpy.sum(F_eqn2, axis=0)

    # Plot X^2
    fig, axes = plt.subplots(5,2, sharex=True) #, sharey=True)

    for row in [0,1]:
        axes[row,0].set_xlabel(r"r / $a_0$")
        axes[row,0].set_ylabel(r"state $X^2$(r) / Hartree")

        axes[row,1].set_xlabel(r"r / $a_0$")
        axes[row,1].set_ylabel(r"transition $X^2$(r) / Hartree")

    for column in [0,1]:
        axes[0,column].set_title(r"$X^2 = D^{-4/3} \nabla D \cdot \nabla D D^{-4/3}$")
        axes[1,column].set_title(r"$X^2 = (36 \pi)^{2/3} \nabla R \cdot \nabla R$")
        axes[2,column].set_title(r"matrix density $\boldsymbol{D}$")
        axes[3,column].set_title(r"$\nabla D \cdot \nabla D$")
        axes[4,column].set_title(r"enhancement factor")

    # Plot reduced density gradients for state and transition densities.
    for istate in range(0, nstate):
        for jstate in range(istate, nstate):
            # Diagonal (state) and off-diagonal (transition) parts of X^2 are plotted separately.
            if istate == jstate:
                column = 0
            else:
                column = 1

            # X^2 according to equation (1)
            line, = axes[0,column].plot(
                r, X2_eqn1[istate,jstate,:],
                lw=1, 
                ls='-', label=r"$X^2_{%d,%d}(r)$" % (istate, jstate))
            # X^2 according to equation (2)
            axes[1,column].plot(
                r, X2_eqn2[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls="-.", label=r"$X^2_{%d,%d}(r)$" % (istate, jstate))
            # matrix density
            axes[2,column].plot(
                r, D[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls="-.", label=r"$D_{%d,%d}(r)$" % (istate, jstate))
            if istate == jstate:
                # Eigenvalues of matrix density
                axes[2,column].plot(
                    r, D_eigenvalues[jstate,:],
                    lw=2, color=line.get_color(),
                    alpha=0.5,
                    ls="-", label=r"$d_{%d}(r)$" % (istate))
            # ∇D·∇D
            axes[3,column].plot(
                r, grad_D_squared[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls="-.", label=r"$(\nabla D \cdot \nabla D)_{%d,%d}(r)$" % (istate, jstate))
            # enhancement factor
            axes[4,column].plot(
                r, F_eqn1[istate,jstate,:],
                lw=1, color=line.get_color()
            )
            axes[4,column].plot(
                r, F_eqn2[istate,jstate,:],
                lw=1, ls="--", color=line.get_color()
            )

    for row in [0,1,2,3]:
        for column in [0,1]:
            axes[row,column].legend()

    plt.show()

if __name__ == "__main__":
    # hydrogen molecule
    rHH = 20.0
    mol = pyscf.gto.M(
        atom = f'H 0 0 {-rHH/2.0}; H 0 0 {rHH/2.0}',
        unit = 'Bohr',
        basis = 'cc-pvdz'
    )

    compare_reduce_gradients_squared(mol, nstate=10)
