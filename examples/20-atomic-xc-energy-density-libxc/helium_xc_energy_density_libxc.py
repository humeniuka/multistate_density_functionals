#!/usr/bin/env python
"""
The exchange-correlation energy density obtained with different xc-functionals
from libxc is compared with the exact xc-energy density obtained from the
pair-density matrix.

The energy densities are compared for the ground state of helium and are plotted
on a radial grid.

The exact ground state wavefunction of helium is obtained by full CI.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy.testing

from pyscf.data.nist import HARTREE2EV
import pyscf.fci
import pyscf.gto
import pyscf.scf

from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI
#from msdft.ElectronRepulsionOperators import LDAExchangeLikeFunctional
#from msdft.ElectronRepulsionOperators import LSDAExchangeLikeFunctional
from msdft.ElectronRepulsionOperators import GGABecke88ExchangeLikeFunctional
from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional

def eval_libxc_functional(msmd: MultistateMatrixDensityFCI, xc_code: str, coords, spin=0):
    """
    Compute xc-energy density for ground state using libxc
    """
    D, grad_D, lapl_D = msmd.evaluate(coords)
    if "GGA" not in xc_code:
        # LDA
        if spin == 0:
            # sum over spins
            D = numpy.sum(D, axis=0)
            # compute exhange energy density using libxc
            # for a LDA functional, rho (*,N) is ordered as (den,)
            rho = numpy.zeros((1, ncoord))
            rho[0,:] = D[0,0,:]
            # evaluate exchange-correlation energy density with libxc
            exc, _, _, _ = pyscf.dft.libxc.eval_xc(xc_code, rho, spin=0)
            # exchange-correlation energy density, ε_{xc}(r) ρ(r)
            xced = exc * rho[0,:]
        else:
            assert spin == 1
            # compute exhange energy density using libxc
            # for a LSDA functional, rho (*,N) are ordered as (den_u,den_d)
            rho = numpy.zeros((2,1, ncoord))
            rho[:,0,:] = D[:,0,0,:]
            # evaluate exchange-correlation energy density with libxc
            exc, _, _, _ = pyscf.dft.libxc.eval_xc(xc_code, rho, spin=1)
            # exchange-correlation energy density, ε_{xc}(r) ρ(r)
            xced = exc * numpy.sum(rho[:,0,:], axis=0)
    else:
        # GGA
        if spin == 0:
            # sum over spins
            D = numpy.sum(D, axis=0)
            grad_D = numpy.sum(grad_D, axis=0)
            lapl_D = numpy.sum(lapl_D, axis=0)
            # compute exhange energy density using libxc
            # for a GGA functional, rho (*,N) are ordered as (den,grad_x,grad_y,grad_z)
            rho = numpy.zeros((4, ncoord))
            rho[0,:] = D[0,0,:]
            # components of gradient ∇D
            for xyz in [0,1,2]:
                rho[1+xyz,:] = grad_D[0,0,xyz,:]
            # evaluate exchange-correlation energy density with libxc
            exc, _, _, _ = pyscf.dft.libxc.eval_xc(xc_code, rho, spin=0)
            # exchange-correlation energy density, ε_{xc}(r) ρ(r)
            xced = exc * rho[0,:]
        else:
            assert spin == 1
            # compute exhange energy density using libxc
            # for a GGA functional, rho (*,N) are ordered as (den,grad_x,grad_y,grad_z)
            rho = numpy.zeros((2,4, ncoord))
            rho[:,0,:] = D[:,0,0,:]
            # components of gradient ∇D
            for xyz in [0,1,2]:
                rho[:,1+xyz,:] = grad_D[:,0,0,xyz,:]
            # evaluate exchange-correlation energy density with libxc
            exc, _, _, _ = pyscf.dft.libxc.eval_xc(xc_code, rho, spin=1)
            # exchange-correlation energy density, ε_{xc}(r) ρ(r)
            xced = exc * numpy.sum(rho[:,0,:], axis=0)

    return xced

def laplacian_term(rho, grad_rho, lapl_rho):
    """
    This function computes ∇²ρ²ᐟ³.

    The exchange-correlation energy density is not uniquely defined, since
    any function that integrates to zero can be added without changing the
    total energy, which is the integral over the xc energy density.
    
    In particular the Laplacian of any function of the density can be added,
    since

        ∫ ∇²f(ρ(r)) d³r = 0

    The functional form of f(ρ) is determined by coordinate scaling relations.
    If the pair density (the diagonal second order reduced density matrix) is
    transformed as

        P(r,r') -> Pλ =  λ⁶ P(λr,λr')
    
    such that its integral remains unchanged,
    
        ∫∫ Pλ d³r d³r' = ∫∫P(λr,λr') d³(λr) d³(λr') = ∫∫P(r,r') d³r d³r'
    
    it follows that the density matrix undergoes the coordinate scaling

        ρ(r) -> ρλ = λ³ρ(λr)
    
    The exchange correlation energy density, which is a functional of P and ρ,

        εxc[P,ρ](r) = 1/2 ∫ [P(r,r') - ρ(r)ρ(r')]/|r-r'| d³r'
    
    satisfies the following coordinate scaling relation,

        εxc[Pλ,ρλ](r) = 1/2 ∫ [λ⁶P(λr,λr') - λ³ρ(λr)λ³ρ(λr')]/|r-r'| d³r'

            = 1/2 λ⁴ ∫ [P(λr,λr') - ρ(λr)ρ(λr')]/|λr-λr'| d³(λr')

    Renaming the integration variable into r'' = λr', we get

        εxc[Pλ,ρλ](r) = 1/2 λ⁴ ∫ [P(λr,r'') - ρ(λr)ρ(r'')]/|λr-r''| d³r''
    
    so that

        εxc[Pλ,ρλ](r) = λ⁴ εxc[P,ρ](λr)

    If the Laplacian term ∇²f(ρ(r)) in the density functional for the exchange
    correlation energy density is to transform under coordinate scaling in the
    same way as εxc[Pλ,ρλ](r), then f has to be proportinal to ρ²ᐟ³,

        εxc[ρ] = ∇²ρ²ᐟ³ = -2/9 ρ⁻⁴ᐟ³ |∇ρ|² + 2/3 ρ⁻¹ᐟ³∇²ρ
    
    This has the correct coordinate scaling

        εxc[ρλ](r) = ∇²(λ³ρ(λr))²ᐟ³ = λ⁴( d²/d(λx)²+d²/d(λy)²+d²/d(λz)² ) ρ(λr)²ᐟ³

            = λ⁴ ∇'²ρ(r')²ᐟ³|
                            r'=λr
            = λ⁴ εxc[ρ](λr)
    """
    lapl_rho_2o3 = (
        # -2/9 ρ⁻⁴ᐟ³ |∇ρ|²
        -2.0/9.0 * pow(rho, -4.0/3.0) * numpy.sum(grad_rho**2, axis=0) +
        # 2/3 ρ⁻¹ᐟ³∇²ρ
        2.0/3.0 * pow(rho, -1.0/3.0) * lapl_rho
    )
    return lapl_rho_2o3

mol = pyscf.gto.Mole()
mol.verbose = 4
mol.atom = 'He'
mol.basis = 'aug-cc-pvqz' #'cc-pv5z' #'aug-cc-pvqz'
# Do not use spatial symmetry.
mol.symmetry = False
# The electronic configuration of helium is 1s² which is spherically symmetric
mol.spin = 0
mol.charge = 0
mol.build()

rohf = pyscf.scf.ROHF(mol)
rohf.kernel()
rohf.analyze()

# full CI for lowest two singlet states, 1¹S and 2¹S, and triplet ³S
norb = rohf.mo_energy.size
nelec = (1,1)
fci = pyscf.fci.FCI(mol, rohf.mo_coeff, singlet=False)
# Multiplicity of ¹S is 1, of ³S (Sz=0) is also 1.
# 
fci.nroots = 2*1+1
fci_energies, fcivecs = fci.kernel(nelec=nelec)
e = fci_energies
# spin multiplicities 2*S+1
multiplicities = numpy.array([fci.spin_square(x, norb, nelec)[1] for x in fcivecs])
multiplicities = numpy.round(multiplicities, decimals=2)
for i in range(0, len(fcivecs)):
    print(
        'state %d, E = %.12f ( %.12f eV) 2S+1 = %.7f' %
        (i, e[i], (e[i]-e[0])*HARTREE2EV, multiplicities[i])
    )
# Check that we got the expected states.
numpy.testing.assert_allclose(multiplicities, [1.0, 3.0, 1.0])
state_labels = [r"1¹S", r"1³S", r"2¹S"]

# Compute xc-energy density
msmd = MultistateMatrixDensityFCI(
    mol, rohf, fci, fcivecs,
    compute_pair_density=True)

ncoord = 2000
r = numpy.linspace(0.0, 10.0, ncoord)
dr = numpy.ediff1d(r, to_end=r[-1]-r[-2])
coords = numpy.zeros((ncoord,3))
coords[:,0] = r

# exact xc-energy density from pair density matrix
xced_exact = msmd.exchange_correlation_energy_density(coords)

# GGA xc-energy approximations from libxc
xced_libxc = {}
for xc_code in [
    "LDA_X,LDA_C_CHACHIYO",
    "GGA_X_PBE,LDA_C_CHACHIYO",
    "GGA_X_B88,LDA_C_CHACHIYO",
    "GGA_X_B88,GGA_C_LYP",
    #"GGA_X_SG4,LDA_C_CHACHIYO",
    #"GGA_X_SOGGA,LDA_C_CHACHIYO",
    #"GGA_X_SSB,LDA_C_CHACHIYO"
]:
    xced_libxc[xc_code] = eval_libxc_functional(msmd, xc_code, coords, spin=1)


plt.ylabel(r"XC energy density $4 \pi r^2 xc(r)$ / $E_h a_0^{-1}$")
plt.xlabel(r"r / $a_0$")
#plt.gca().set_yscale("symlog")

plt.plot(
    r, 4.0*numpy.pi*r**2 * xced_exact[0,0,:],
    lw=2, alpha=0.5,
    label="exact"
)

# Show the Laplacian of the density
D, grad_D, lapl_D = msmd.evaluate(coords)
# sum over spins
D = numpy.sum(D, axis=0)
grad_D = numpy.sum(grad_D, axis=0)
lapl_D = numpy.sum(lapl_D, axis=0)
# take only the ground state
rho = D[0,0,:]
grad_rho = grad_D[0,0,:,:]
lapl_rho = lapl_D[0,0,:]

# Compute a Laplacian term ∇²ρ²ᐟ³ that transforms in the same
# way under coordinate scaling as the exchange-correlation energy density.
lapl_rho_2o3 =laplacian_term(rho, grad_rho, lapl_rho)
# Plot Laplacians ∇²ρ² and ∇²ρ²ᐟ³
scale = 0.01
plt.plot(
    r, scale * 4.0 * numpy.pi * r**2 * lapl_rho, label=r"$%s \times$ $4 \pi r^2 \nabla^2 \rho(\vec{r})$" % scale
)
plt.plot(
    r, scale * 4.0 * numpy.pi * r**2 * lapl_rho_2o3, label=r"$%s \times$ $4 \pi r^2 \nabla^2 \rho(\vec{r})^{2/3}$" % scale
)
# Integral of Laplacians over space should vanish
lapl_rho_integral = numpy.sum(4.0 * numpy.pi * r**2 * lapl_rho * dr)
print(f"integral of Laplacian ∫∇²ρ(r) d³r = {lapl_rho_integral}")
lapl_rho_2o3_integral = numpy.sum(4.0 * numpy.pi * r**2 * lapl_rho_2o3 * dr)
print(f"integral of Laplacian ∫∇²ρ²ᐟ³(r) d³r = {lapl_rho_2o3_integral}")

for xc_code, xced_approximate in xced_libxc.items():
    line, = plt.plot(
        r, 4.0*numpy.pi*r**2 * xced_approximate,
        ls="--",
        label=xc_code
    )

    error_density = 4.0*numpy.pi*r**2 * (xced_approximate - xced_exact[0,0,:])
    error = numpy.sum(error_density * dr)
    print(f"error ({xc_code}) = {error}")
    plt.plot(
        r, error_density, label=f"{xc_code} error",
        ls="-.", color=line.get_color())

plt.legend()

#plt.savefig("helium_xc_energy_density_libxc.svg")
#plt.savefig("helium_xc_energy_density_libxc.png", dpi=300)

plt.show()
