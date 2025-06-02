#!/usr/bin/env python
"""
Compute the exchange-correlation energy density matrix xcᵢⱼ(r)
from the 1-particle matrix density

    Dᵢⱼ(r) = n ∫ dr_2 ... ∫ dr_n Ψᵢ(r,r_2,...,r_n) Ψⱼ(r,r_2,...,r_n)

and the 2-particle matrix density

    D2ᵢⱼ(r,r') = n*(n-1) ∫ dr_3 ... ∫ dr_n Ψᵢ(r,r',r_3,...,r_n) Ψⱼ(r,r',r_3,...,r_n)

as

                    D2ᵢⱼ(r,r') - ∑ₖ  Dᵢₖ(r) Dₖⱼ(r')
    xcᵢⱼ(r) = 1/2 ∫ --------------------------------- dr'
                            |r-r'|
"""
import numpy
import scipy.linalg

import pyscf.ao2mo
import pyscf.df
from pyscf.dft import numint
import pyscf.gto
import pyscf.scf
import pyscf.fci

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

# hydrogen molecule, closed shell
mol = pyscf.gto.M(
    atom = 'H 0 0 -0.35; H 0 0 0.35',
    basis = '6-31g',
    # singlet
    spin = 0)
nstate = 3

rhf = pyscf.scf.RHF(mol)
# supress printing of SCF energy
rhf.verbose = 0
# compute self-consistent field
rhf.kernel()

# number of atomic orbitals and molecular orbitals
nao, nmo = rhf.mo_coeff.shape
number_of_electrons = sum(mol.nelec)

# singlet=True enables the use of spin symmetry in the CI calculation.
fci = pyscf.fci.FCI(mol, rhf.mo_coeff, singlet=True)
# Solve for the lower few electronic states.
fci.nroots = nstate
fci_energies, fcivecs = fci.kernel()

if hasattr(fci.e_tot, '__len__'):
    nstate_available = len(fci.e_tot)
else:
    nstate_available = 1

# ∫ χp(r) χq(r) 1/|R(a)-r| dr
coulomb_potential = mol.intor('int1e_grids', grids=mol.atom_coords())
nuclear_attraction = numpy.einsum('a,apq->pq', -mol.atom_charges(), coulomb_potential)

# Compare with the exact nuclear attraction integrals
nuclear_attraction_ref = mol.intor('int1e_nuc')

numpy.testing.assert_allclose(nuclear_attraction, nuclear_attraction_ref, atol=1.0e-10)

# If there is only a single state, the energies and FCI vectors
# are not stored as a list.
if hasattr(fci.e_tot, '__len__'):
    eigenenergies = fci.e_tot[:len(fcivecs)]
else:
    eigenenergies = numpy.array([fci.e_tot])
    fcivecs = [fcivecs]

def _density_matrix_mo2ao(dm_mo, mo_coeff):
    """
    transform a density matrix in the MO basis in the AO basis

        P^AO_{a,b}   = sum_{m,n} C*_{a,m} P^MO_{m,n} C_{b,n}

    a,b enumerate atomic orbitals, m,n enumerate molecular orbitals
    and C_{a,m} are the self-consistent field MO coefficients.

    :param dm_mo: density matrix in MO basis
    :type dm_mo: numpy.ndarray of shape (nmo,nmo)

    :param mo_coeff: molecular orbital coefficients
    :type mo_coeff: numpy.ndarray of shape (nao,nmo)

    :return dm_ao: density matrix in AO basis
    :rtype dm_ao: numpy.ndarray of shape (nao,nao)
    """
    nao, nmo = mo_coeff.shape
    assert dm_mo.shape == (nmo,nmo)
    dm_ao = numpy.einsum(
        'am,mn,bn->ab',
        mo_coeff, dm_mo, mo_coeff)
    return dm_ao

# number of electronic states
nstate = len(fcivecs)
# Compute the spin-traced, 1-particle (transition) density matrices in the AO basis.
density_matrices_1e = numpy.zeros((nstate,nstate,nao,nao))
for i in range(0, nstate):
    for j in range(0, nstate):
        if i == j:
            # spin-traced 1-particle density matrix of state i in MO basis
            dm1 = fci.make_rdm1(fcivecs[i], nmo, mol.nelec)
        else:
            # spin-traced 1-particle transition density matrix
            # between electronic states i and j.
            dm1 = fci.trans_rdm1(fcivecs[i], fcivecs[j], nmo, mol.nelec)
    # Why do we have to take the transpose of dm1?
    # see https://pyscf.org/pyscf_api_docs/pyscf.fci.html#pyscf.fci.direct_spin0.FCISolver.make_rdm1
    # "The convention is based on McWeeney’s book, Eq (5.4.20).
    # The contraction between 1-particle Hamiltonian and rdm1 is E = einsum(‘pq,qp’, h1, rdm1)"
    dm1 = dm1.T
    # Transform from the MO basis to the AO basis
    density_matrices_1e[i,j,:,:] = _density_matrix_mo2ao(dm1, rhf.mo_coeff)

# spin-traced, 1-particle (transition) density matrices in the AO basis
density_matrices_1e = numpy.zeros((nstate,nstate,nao,nao))
# spin-traced pair (or 2-particle) density matrices in the AO basis
density_matrices_2e = numpy.zeros((nstate,nstate,nao,nao,nao,nao))
for i in range(0, nstate):
    for j in range(0, nstate):
        if i == j:
            # spin-traced 1- and 2-particle density matrices of state i in MO basis
            dm1, dm2 = fci.make_rdm12(fcivecs[i], nmo, mol.nelec)
        else:
            # spin-traced 1- and 2-particle transition density matrices
            # between electronic states i and j.
            dm1, dm2 = fci.trans_rdm12(fcivecs[i], fcivecs[j], nmo, mol.nelec)
        # Why do we have to take the transpose of dm1?
        # see https://pyscf.org/pyscf_api_docs/pyscf.fci.html#pyscf.fci.direct_spin0.FCISolver.make_rdm1
        # "The convention is based on McWeeney’s book, Eq (5.4.20).
        # The contraction between 1-particle Hamiltonian and rdm1 is E = einsum(‘pq,qp’, h1, rdm1)"
        dm1 = dm1.T
        # Transform from the MO basis to the AO basis
        density_matrices_1e[i,j,:,:] = _density_matrix_mo2ao(dm1, rhf.mo_coeff)
        # Although the module is called ao2mo, we use it to transform the 2-particle
        # density matrix from the MO to the AO basis by transforming with the transpose
        # of the orbital coefficients.
        # D2aoᵢⱼ[a,b,c,d] = ∑_{k,l,m,n} C[a,k] C[b,l] C[c,m] C[d,n] D2moᵢⱼ[k,l,m,n]
        density_matrices_2e[i,j,:,:,:,:] = pyscf.ao2mo.kernel(
            dm2,
            rhf.mo_coeff.T,
            # no symmetry
            aosym='s1')

# Check that the 1-particle matrix density can be obtained from the 2-particle
# matrix density by integrating over one of the two electron coordinates,
#   Dᵢⱼ(r) = 1/(n-1) ∫ D2ᵢⱼ(r,r') dr'
# If the matrix densities are expressed in the AO basis, this means
#   Dᵢⱼ[a,b] = 1/(n-1) ∑_{c,d} D2aoᵢⱼ[a,b,c,d] S[a,b]
# where S[a,b] = <a|b> = ∫ χa(r) χb(r) dr is the overlap between the atomic orbitals.

# overlap S[a,b]
overlap = mol.intor('int1e_ovlp')

density_matrices_1e_check = (
    1.0/(number_of_electrons-1.0) *
    numpy.einsum('ijabcd,cd->ijab', density_matrices_2e, overlap))

# compare state densities
for i in range(0, nstate):
    numpy.testing.assert_allclose(
        density_matrices_1e_check[i,i,:,:],
        density_matrices_1e[i,i,:,:],
        atol=1.0e-10
    )

# compare transition densities
for i in range(0, nstate):
    for j in range(0, nstate):
        if i == j:
            continue
        numpy.testing.assert_allclose(
            density_matrices_1e_check[i,j,:,:],
            density_matrices_1e[i,j,:,:],
            atol=1.0e-10
        )

def exchange_correlation_energy_density(
    coords : numpy.ndarray):
    """
Compute the exchange-correlation energy density matrix xcᵢⱼ(r)
from the 1-particle matrix density

    Dᵢⱼ(r) = n ∫ dr_2 ... ∫ dr_n Ψᵢ(r,r_2,...,r_n) Ψⱼ(r,r_2,...,r_n)

and the 2-particle matrix density

    D2ᵢⱼ(r,r') = n*(n-1) ∫ dr_3 ... ∫ dr_n Ψᵢ(r,r',r_3,...,r_n) Ψⱼ(r,r',r_3,...,r_n)

as

                    D2ᵢⱼ(r,r') - ∑ₖ  Dᵢₖ(r) Dₖⱼ(r')
    xcᵢⱼ(r) = 1/2 ∫ --------------------------------- dr'
                            |r-r'|

    :param coords: The Cartesian positions at which the kinetic energy
        density is calculated.
    :type coords: numpy.ndarray of shape (Ncoord,3)

    :return: xcᵢⱼ(r)
        spin-traced exchange-correlation energy density
    :rtype: numpy.ndarray of shape (Mstate,Mstate,Ncoord)
        XCED[i,j,r] is the exchange-correlation energy density between
        states i and j at the grid point coords[r,:]
    """
    # Evaluate atomic orbitals 𝛘ₐ(r) on the grid.
    ao_value = numint.eval_ao(mol, coords)

    # Coulomb potential of overlap charge densities
    # V_pq(r) = ∫ χp(r') χq(r') 1/|r-r'| dr'
    ao_coulomb_potential = mol.intor('int1e_grids', grids=coords)

    # exact electron-electron repulsion matrix computed from the 2-electron matrix density
    #                  D2ᵢⱼ(r,r')
    # Vᵢⱼ(r) = 1/2 ∫ ----------- dr'
    #                    |r-r'|
    full_electron_repulsion_2e = 0.5 * numpy.einsum('ijabcd,ra,rb,rcd->ijr',
        # D2ᵢⱼ[a,b,c,d]
        density_matrices_2e,
        # χa(r) χb(r)
        ao_value, ao_value,
        # V_cd(r)
        ao_coulomb_potential
    )
    # Hartree potential matrix computed from the 1-electron matrix density
    #                ∑ₖ  Dᵢₖ(r) Dₖⱼ(r')
    # Jᵢⱼ(r) = 1/2 ∫ ------------------ dr'
    #                      |r-r'|
    hartree_1e = 0.5 * numpy.einsum('ikab,kjcd,ra,rb,rcd->ijr',
        density_matrices_1e,
        density_matrices_1e,
        # χa(r) χb(r)
        ao_value, ao_value,
        # V_cd(r)
        ao_coulomb_potential
    )

    # Exchange-correlation energy density (full - mean field)
    xced = full_electron_repulsion_2e - hartree_1e

    return xced

# Integrate the exchange-correlation energy density on a grid and
# compare the the exchange-correlation energy calculated as
# XCᵢⱼ = (Eᵢ - N) δᵢⱼ - Tᵢⱼ - Vᵢⱼ - Jᵢⱼ

# integration grid
grids = pyscf.dft.gen_grid.Grids(mol)
grids.level = 8
grids.build()

# evaluate xcᵢⱼ(r) on the grid
xced = exchange_correlation_energy_density(grids.coords)

# Integrate XCᵢⱼ = ∫ xcᵢⱼ(r) dr on the Becke grid
XC = numpy.einsum('r,ijr->ij', grids.weights, xced)

# Compute XCᵢⱼ = (Eᵢ - N) δᵢⱼ - Tᵢⱼ - Vᵢⱼ - Jᵢⱼ
msmd = MultistateMatrixDensityFCI(mol, rhf, fci, fcivecs)
XC_ref = msmd.exact_electron_repulsion() - msmd.hartree_matrix_product()

print("XCᵢⱼ = ∫ xcᵢⱼ(r) dr")
print(XC)
print("XCᵢⱼ = (Eᵢ - N) δᵢⱼ - Tᵢⱼ - Vᵢⱼ - Jᵢⱼ")
print(XC_ref)

# Compare the two ways of calculating the XC matrix
numpy.testing.assert_allclose(XC, XC_ref, atol=1.0e-10)
