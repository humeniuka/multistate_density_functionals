#!/usr/bin/env python
# coding: utf-8
"""
The kinetic energy of an (anti)symmetric wavefunction with density 𝜌(r)
is bounded from below by the von-Weizsäcker kinetic energy,

    T[𝜌] ≥ 1/8 ∫ |∇𝜌|²/𝜌.

This is theorem 1.1 in [Lieb]. Applying it to each electronic state i in
the subspace separately a lower bound on the trace of the kinetic energy
in that subspace emerges,

    ∑ᵢᵢ T[D]ᵢᵢ ≥ ∑ᵢ 1/8 ∫ |∇Dᵢᵢ|²/Dᵢᵢ.        (lower bound 1)

However, using similar tricks as in [Lieb] another lower bound of tr(T)
can be derived,

    ∑ᵢᵢ T[D]ᵢᵢ ≥ 1/8 ∫ |∑ᵢ∇Dᵢᵢ|² / (∑ⱼDⱼⱼ)    (lower bound 2)

               = 1/8 ∫ |∇tr(D)|² / tr(D)

Here we compare the two bounds to find out which one is the higher upper bound.
Except for one-electron systems, the second lower bound seems to be higher.

[Lieb] Lieb, Elliott H. "Density functionals for Coulomb systems."
    Inequalities: Selecta of Elliott H. Lieb (2002): 269-303.
"""
import numpy
import pyscf.gto

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI


# test molecules
molecules = {
    # 1-electron systems
    'hydrogen atom': pyscf.gto.M(
        atom = 'H 0 0 0',
        basis = '6-31g',
        # doublet
        spin = 1),
    'hydrogen atom (large basis set)': pyscf.gto.M(
        atom = 'H 0 0 0',
        basis = 'aug-cc-pvtz',
        # doublet
        spin = 1),
    'hydrogen molecular ion': pyscf.gto.M(
        atom = 'H 0 0 0; H 0 0 0.74',
        basis = '6-31g',
        charge = 1,
        spin = 1),
    # 2-electron systems, paired spins
    'hydrogen molecule': pyscf.gto.M(
        atom = 'H 0 0 0; H 0 0 0.74',
        basis = '6-31g',
        charge = 0,
        spin = 0),
    # 3-electron systems, one unpaired spin
    'lithium atom': pyscf.gto.M(
        atom = 'Li 0 0 0',
        basis = '6-31g',
        # doublet
        spin = 1),
    # 4-electron system, closed shell
    'lithium hydride': pyscf.gto.M(
        atom = 'Li 0 0 0; H 0 0 1.60',
        basis = '6-31g',
        # singlet
        spin = 0),
    # many electrons
    'water': pyscf.gto.M(
        atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
        basis = 'sto-3g',
        # singlet
        spin = 0),
    # noble gas atom
    'neon': pyscf.gto.M(
        atom = 'Ne  0 0 0',
        basis = '6-31g',
        # singlet
        spin = 0)
    }


for name, mol in molecules.items():
    # Compute the matrix density for the lowest 4 states (if available)
    msmd = MultistateMatrixDensityFCI.create_matrix_density(
        mol, nstate=4, raise_error=False)

    # generate a multicenter integration grid
    grids = pyscf.dft.gen_grid.Grids(mol)
    grids.level = 8
    grids.build()

    # Compute the kinetic energy matrix exactly
    T_exact = msmd.exact_kinetic_energy()
    # ∑ᵢᵢ Tᵢᵢ
    trace_T_exact = numpy.trace(T_exact)

    # evaluate D(r) and ∇D(r) on the grid
    D, grad_D, _ = msmd.evaluate(grids.coords)

    # dimensions of arrays
    nspin, nstate, nstate, ncoord = D.shape

    # (1) compute the first lower bound due to Lieb
    # ∑ᵢ 1/8 ∫ |∇Dᵢᵢ|²/Dᵢᵢ(r)
    # The lower bound is in fact valid for the kinetic energy density (KED)
    # at each point r.
    lower_bound_ked = numpy.zeros(ncoord)
    # Lower over electronic states i in subspace.
    for i in range(0, nstate):
        # |∇Dᵢᵢ|², sum over spins and do the scalar product of the gradient vectors.
        numerator = numpy.einsum('sdr,sdr->r', grad_D[:,i,i,:,:], grad_D[:,i,i,:,:])
        # Dᵢᵢ(r), sum over spins
        denominator = numpy.einsum('sr->r', D[:,i,i,:]) + 1.0e-20
        lower_bound_ked += 1.0/8.0 * numerator / denominator
    # Integrate the lower bound for the kinetic energy density over space.
    trace_T_lower_bound_1 = numpy.einsum('r,r->', grids.weights, lower_bound_ked)

    # (2) compute the second lower bound
    # ∑ᵢᵢ T[D]ᵢᵢ ≥ 1/8 ∫ |∑ᵢ∇Dᵢᵢ|² / (∑ⱼDⱼⱼ)
    # The lower bound is in fact valid for the kinetic energy density (KED)
    # at each point r.
    lower_bound_ked = numpy.zeros(ncoord)
    # ∑ᵢ∇Dᵢᵢ, sum over spins and states.
    trace_grad_D = numpy.einsum('siidr->dr', grad_D)
    # |∑ᵢ∇Dᵢᵢ|², scalar product of trace of gradient vector
    numerator = numpy.einsum('dr,dr->r', trace_grad_D, trace_grad_D)
    # denominator, sum over spin and states
    denominator = numpy.einsum('siir->r', D) + 1.0e-20
    # 1/8 |∑ᵢ∇Dᵢᵢ|² / (∑ⱼDⱼⱼ)
    lower_bound_ked = 1.0/8.0 * numerator / denominator
    # Integrate the lower bound for the kinetic energy density over space.
    trace_T_lower_bound_2 = numpy.einsum('r,r->', grids.weights, lower_bound_ked)

    # Compare exact tr(T) with lower bounds
    print(f"Molecule: {name}")
    print(f"                                   ∑ᵢᵢ Tᵢᵢ = {trace_T_exact:8.4f}")
    print(f"  lower bound 1: ∑ᵢ 1/8 ∫ |∇Dᵢᵢ|²/Dᵢᵢ      = {trace_T_lower_bound_1:8.4f}")
    print(f"  lower bound 2: 1/8 ∫ |∑ᵢ∇Dᵢᵢ|² / (∑ⱼDⱼⱼ) = {trace_T_lower_bound_2:8.4f}")

    # Check that the bounds are not violated
    epsilon = 1.0e-8
    assert trace_T_exact >= trace_T_lower_bound_1 - epsilon
    assert trace_T_exact >= trace_T_lower_bound_2 - epsilon
