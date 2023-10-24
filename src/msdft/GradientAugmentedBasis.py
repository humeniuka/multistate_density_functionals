#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Typical basis sets are very far from being complete. A basis set that contains the
functions χ1(r), χ2(r), ..., χn(r) cannot necessarily represent the gradients
∇χ1(r), ∇χ2(r), ..., ∇χn(r) of those functions, which are needed for the kinetic
energy density.

In a gradient-augmented basis set, additional basis functions are added so that the
orbital gradients can be represented exactly.
"""
import numpy

import pyscf.dft
import pyscf.gto

import scipy.linalg
import scipy.special


def augment_basis_with_gradients(mol: pyscf.gto.Mole) -> pyscf.gto.Mole:
    """
    Enlarge a basis set, such that for every basis function χ(r)
    its gradients w/r/t the electronic coordinate r, ∇χ(r), can also
    be represented in the basis set.

    If the original basis set spans the vector space
      V = span{χ1(r), χ2(r), ..., χn(r)}
    then the gradient-augmented basis contains the space
      V ∪ span{∇χ1(r), ∇χ2(r), ..., ∇χn(r)}.
    All gradients of the atomic orbitals can be represented exactly
    in the augmented basis set.

    NOTE: This only works with Cartesian basis functions (6d, 10f, 15g).

    :param mol: a molecule with a Cartesian basis set
    :type basis: pyscf.gto.Mole

    :return: copy of the molecule with the augmented basis set
        consisting of the original basis and additional functions that span
        the space of the orbital gradients as well.
    :rtype: pyscf.gto.Mole
    """
    assert mol.cart == True, "Gradient-augmentation only works with Cartesian basis functions."
    # Generate the augmented basis for orbitals and their gradients.
    mol_augmented = mol.copy()
    # Operate on the basis data in the internal format.
    mol_augmented.basis = augment_basis_dict_with_gradients(mol._basis)
    mol_augmented.build()

    return mol_augmented


def augment_basis_dict_with_gradients(basis : dict) -> dict:
    """
    Enlarge a basis set, such that for every basis function χ(r)
    its gradients w/r/t the electronic coordinate r, ∇χ(r), can also
    be represented in the basis set.

    If the original basis set spans the vector space
      V = span{χ1(r), χ2(r), ..., χn(r)}
    then the gradient-augmented basis contains the space
      V ∪ span{∇χ1(r), ∇χ2(r), ..., ∇χn(r)}.
    All gradients of the atomic orbitals can be represented exactly
    in the augmented basis set.

    NOTE: This only works with Cartesian basis functions (6d, 10f, 15g).

    :param basis: a basis set in the internal data format
        as stored in the attribute `pyscf.gto.Mole._basis`.
    :type basis: dict

    :return: the augmented basis set consisting of the
        original basis and additional functions that span
        the space of the gradients of the original functions.
    :rtype: dict
    """
    augmented_basis = {}
    # Loop over atoms with their basis sets.
    for symbol, atomic_basis in basis.items():
        # Make a copy of the existing basis functions.
        augmented_atomic_basis = atomic_basis[:]
        # Add basis functions for representing the space spanned by the gradients
        # of the existing basis functions.
        for contraction in atomic_basis:
            # A contraction has the format
            #  [angular-momentum, [exp1, coeff1], [exp2, coeff2], ...]
            L = contraction[0]

            # The derivative of a Gaussian basis function with angular momentum L,
            # consists of basis functions with angular momentum L-1 and L+1.
            # So each contraction in the original basis set adds two contractions with
            # lower and higher angular momenta to the augmented basis set:
            #
            #   (L, [α, coef], ...) --> (L-1, [α, sqrt(α)*coef]) + (L+1, [α, sqrt(α)*coef])
            #
            # It is assumed that each contraction
            #
            #   CGF(r;nx,ny,nz) = ∑ᵢ cᵢ PGF(r;αᵢ,nx,ny,nz)
            #
            # is a linear combination of *normalized* primitive Cartesian Gaussians,
            #
            #   PGF(r;α,nx,ny,nz) = N(α,nx) N(α,ny) N(α,nz) x^nx y^ny z^nz exp(-α r^2)
            #
            # with the normalization constants
            #
            #   N(α,nx) = [(2 α)/π]^(1/4) [(4α)ⁿ/(2n-1)!!]¹ᐟ²
            #
            # see Obara & Saika, J. Chem. Phys. 84, 3963 (1986); doi: 10.1063/1.450106
            #
            contraction_for_gradient = []
            # Loop over contractions.
            for exponent, coefficient in contraction[1:]:
                # The contraction coefficients in the derivative orbital become
                #  coef --> sqrt(α)*coef
                contraction_for_gradient.append([exponent, numpy.sqrt(exponent) * coefficient])
            # L+1
            contraction_Lplus1 = [L+1] + contraction_for_gradient
            augmented_atomic_basis.append(contraction_Lplus1)
            # L-1
            if L > 0:
                contraction_Lminus1 = [L-1] + contraction_for_gradient
                augmented_atomic_basis.append(contraction_Lminus1)

        # Sort the basis functions by angular momentum.
        augmented_atomic_basis.sort(key=lambda contraction: contraction[0])
        augmented_basis[symbol] = augmented_atomic_basis

    return augmented_basis


def orbital_gradient_projection(
        mol : pyscf.gto.Mole,
        coords: numpy.ndarray,
        level=8):
    """
    Compute the orbitals gradients ∇a(r) and project them onto the gradient-augmented
    basis set,

      ∇a_projected(r) = ∑ₘ∑ₙ <r|m> S⁻¹ₘₙ <n|∇a>.

    The basis functions m,n come from the gradient-augmented basis set, which
    in addition to the original basis also contains functions that span
    the gradients of the orbitals.

    The projection error should be zero,

      ||∇a - ∇a_projected|| = 0

    if the orbital gradients can be represented as a linear combination of
    basis functions from the gradient-augmented basis set.


    :param mol: Molecule with basis set.
    :type mol: pyscf.gto.Mole

    :param coords: The Cartesian positions r at which ∇𝛘(r) and ∇𝛘_projected(r)
        are compared.
    :type coords: numpy.ndarray of shape (Ncoord,3)

    :return: orbitals gradients ∇a(r) and their projections ∇a_projected(r)
        onto the gradient-augmented basis set.
    :rtype: tuple of numpy.ndarray of shape (3,Ncoord,Norb)
    """
    assert mol.cart == True, "Gradient-augmentation only works with Cartesian basis functions."
    # Generate a multicenter integration grid.
    grids = pyscf.dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build()

    # Evaluate atomic orbitals on the grid.
    # The orbital values 𝛘ₐ(r) and their gradients ∇𝛘ₐ(r) are returned
    # in a single array of shape (4,ncoord,norb).
    ao_value_all = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=1)
    # value 𝛘ₐ(r)
    ao_value = ao_value_all[0,:,:]
    # x-,y- and z-component of gradient ∇𝛘ₐ(r)
    grad_ao_value = ao_value_all[1:4,:,:]

    # Evaluate ∇𝛘ₐ(r) at the coordinates used for comparison.
    ao_value_all_r = pyscf.dft.numint.eval_ao(mol, coords, deriv=1)
    grad_ao_value_r = ao_value_all_r[1:4,:,:]

    # Generate the augmented basis for orbitals and their gradients.
    mol_augmented = augment_basis_with_gradients(mol)

    # overlap matrix in the big, augmented basis
    overlap_augmented = mol_augmented.intor('int1e_ovlp')

    # evaluate basis functions of gradient-augmented basis
    # ... on the integration grid
    ao_value_augmented = pyscf.dft.numint.eval_ao(mol_augmented, grids.coords, deriv=0)
    # ... and at the coordinates used for comparison
    ao_value_augmented_r = pyscf.dft.numint.eval_ao(mol_augmented, coords, deriv=0)

    # Project the orbital gradient onto the augmented basis,
    #
    #   ∇a_projected(r) = ∑ₘ∑ₙ <r|m> S⁻¹ₘₙ <n|∇a>
    #
    # where the basis functions m,n come from the gradient-augmented basis set
    # and S is their overlap matrix.

    # The overlap between the basis function from the augmented basis set
    # and the gradients of basis functions in the original basis set
    # is calculated numerically on a grid.
    #   <n|∇𝛘> = ∫ <n|r><r|∇𝛘> dr
    overlap_aug_orig = numpy.einsum('r,rn,dra->dna',
        grids.weights, ao_value_augmented, grad_ao_value)

    # (pseudo-)inverse of overlap matrix for resolution of identity
    inv_overlap_augmented = scipy.linalg.pinv(overlap_augmented)

    # ∇a_projected(r) = ∑ₘ∑ₙ <r|m> S⁻¹ₘₙ <n|∇a>
    projected_grad_ao_value_r = numpy.einsum('rm,mn,dna->dra',
        ao_value_augmented_r, inv_overlap_augmented, overlap_aug_orig)

    return (grad_ao_value_r, projected_grad_ao_value_r)
