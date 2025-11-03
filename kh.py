import matplotlib.pyplot as plt
import sys
import numpy as np
import jax.numpy as jnp
import pandas as pd
import h5py
from molcas_suite.extractor import make_extractor as make_molcas_extractor
np.set_printoptions(precision=17)

### Unitary transformation for energies
au2ev = 2.7211386021e1
au2cm = 2.1947463e5
ensv2au = (1.0 / au2ev)
c_au = 137.035999084


##Change file name here######
h5name_file = h5py.File('uo2_10_1_0_ci.rassi.h5', 'r')
soc_energies_au = h5name_file['SOS_ENERGIES'][:]
soc_energies = (soc_energies_au - soc_energies_au[0]) * au2ev

#Change for each system, range is exclusionary to second number:
N_i = range(1, 2) #initial state 3d^10 4f^14 5f^0 [ground state only, SO State 1]
N_n = range(198, 338) #intermediate states 3d^9 4f^14 5f^1 [SO State 198 - 337]
N_f = range(2, 198) #number of final states 3d^10 4f^13 5f^1 [SO State 170 - 197]

# Convert to zero-indexing
N_i = range(N_i.start - 1, N_i.stop - 1)
N_n = range(N_n.start - 1, N_n.stop - 1)
N_f = range(N_f.start - 1, N_f.stop - 1)

Ei = soc_energies[N_i] # ground state energy
En = soc_energies[N_n] #intermediate state energies
Ef = soc_energies[N_f] # final state energies

#range of E_em and E_em for RIXS map - change based on the experimental data
#M5edge: E_em_grid= 3100 - 3200; E_ex = 3500-3600
#E_em_grid = np.linspace(3140, 3220, 1000)
#E_ex_grid = np.linspace(3540, 3620, 1000)

#M4edge: E_em_grid= 3300 - 3400; E_ex = 3700-3800
E_em_grid = np.linspace(3340, 3370, 1000)#[::-1]
E_ex_grid = np.linspace(3740, 3760, 1000)#[::-1]

gamma_n = 4 #broadening of the intermediate state, eV
gamma_f = 2 #broadening of the final state, eV; should be smaller than gamma_n

edipmom_real = h5name_file['SOS_EDIPMOM_REAL']
edipmom_real_x = edipmom_real[0, :, :]
edipmom_real_y = edipmom_real[1, :, :]
edipmom_real_z = edipmom_real[2, :, :]

edipmom_imag = h5name_file['SOS_EDIPMOM_IMAG']
edipmom_imag_x = edipmom_imag[0, :, :]
edipmom_imag_y = edipmom_imag[1, :, :]
edipmom_imag_z = edipmom_imag[2, :, :]

edipmom_complex_x = edipmom_real_x + 1j * edipmom_imag_x
edipmom_complex_y = edipmom_real_y + 1j * edipmom_imag_y
edipmom_complex_z = edipmom_real_z + 1j * edipmom_imag_z

# ⟨n|Dx|i⟩ --> rows = N_n, cols = N_i
edipmom_complex_ni_x = edipmom_complex_x[np.ix_(N_n, N_i)]
edipmom_complex_ni_y = edipmom_complex_y[np.ix_(N_n, N_i)]
edipmom_complex_ni_z = edipmom_complex_z[np.ix_(N_n, N_i)]

# ⟨f|Dx|n⟩ --> rows = N_f, cols = N_n
edipmom_complex_fn_x = edipmom_complex_x[np.ix_(N_f, N_n)]
edipmom_complex_fn_y = edipmom_complex_y[np.ix_(N_f, N_n)]
edipmom_complex_fn_z = edipmom_complex_z[np.ix_(N_f, N_n)]

mu_ni = np.stack([edipmom_complex_ni_x, edipmom_complex_ni_y, edipmom_complex_ni_z], axis=0)
mu_fn = np.stack([edipmom_complex_fn_x, edipmom_complex_fn_y, edipmom_complex_fn_z], axis=0)
#print("edipmom_complex_x shape:", edipmom_complex_x.shape)
#print(mu_fn.shape)

#print('mu_ni norm:', np.linalg.norm(mu_ni))
#print('mu_fn norm:', np.linalg.norm(mu_fn))

#mu_ni_i = mu_ni[:,:,N_i].squeeze()

# Robust selection of initial-state indices (works for single or multiple initial states)
N_i_list = list(N_i)   # convert range to explicit list of indices

if len(N_i_list) == 1:
    # pick single initial state -> produce shape (3, N_n)
    mu_ni_i = mu_ni[:, :, N_i_list[0]]    # (3, N_n)
else:
    # multiple initial states -> keep shape (3, N_n, N_i_count) then choose what you need
    # here we average across the chosen initial states (change if you want a different treatment)
    mu_ni_i = np.mean(mu_ni[:, :, N_i_list], axis=2)   # (3, N_n)

# Now add explicit check and reshape for the weighted mu array below:
mu_ni_i = np.asarray(mu_ni_i)
if mu_ni_i.shape != (3, len(En)):
    raise RuntimeError(f"Unexpected mu_ni_i shape {mu_ni_i.shape}; expected (3, N_n={len(En)})")

#print("⟨n|D|i⟩ max abs:", np.abs(mu_ni).max())
#print("⟨f|D|n⟩ max abs:", np.abs(mu_fn).max())
#print("E_ex_grid range:", E_ex_grid[0], E_ex_grid[-1])
#print("E_em_grid range:", E_em_grid[0], E_em_grid[-1])


denominator = 1.0 / ((En[None, :] - Ei - E_ex_grid[:, None]) - 1j * gamma_n / 2) #reshape En from (N_n,) to (1, N_n), same for E_ex

#print(denominator.shape)#(E_ex, En)


# Expand for broadcasting:
mu_ni_i_exp = mu_ni_i[None, :, :]            # (1, 3, N_n)
denominator_exp = denominator[:, None, :]    # (M, 1, N_n)

# Apply denominator to ⟨n|D|i⟩
mu_ni_weighted = mu_ni_i_exp * denominator_exp  # (M, 3, N_n)
#print(mu_ni_weighted.shape)

##Printing for contributions##
M = E_ex_grid.size
N_f = len(Ef)
N_n = len(En)
L = E_em_grid.size

E_ex_ = E_ex_grid.reshape(M, 1, 1)  
Ef_   = Ef.reshape(1, N_f, 1)       
E_em_ = E_em_grid.reshape(1, 1, L)  
Ei_ = Ei                            

#a_n = amplitude of <f|D|n><n|D|i> / energy denominator
a_n = np.einsum('rfn,mln->mfrln', mu_fn, mu_ni_weighted)

# the total amplitude summed over intermediate states with interference
A_total = np.sum(a_n, axis=4)      # shape (M, N_f, 3, 3)
A_total_per_f = np.sum(a_n, axis=4)

#Per state no-interference intensity: |a^{(n)}|^2 summed over polarizations and final states (not yet over emission energies)
I_no_int_mfn = np.abs(a_n)**2     # (M, N_f, 3, 3, N_n)
I_no_int_mfn_sumpol = np.sum(I_no_int_mfn, axis=(2,3))  # (M, N_f, N_n)

#Per state signed contribution including interference between intermediate states:
C_mfn = np.real(np.einsum('mfrln,mfrl->mfn', a_n, np.conj(A_total)))  # (M, N_f, N_n)

#Per FINAL state contributions, which is a sum over intermediate states for each final state:
C_full_per_f = np.real(np.einsum('mfrln,mfrl->mf', a_n, np.conj(A_total_per_f)))  # shape (M, N_f)

#Compute emission weighting
second_term = gamma_f / (((Ef_ - Ei_ - E_ex_ + E_em_)**2) + 0.25*gamma_f**2)
# Sum over emission energies (axis=2)
second_term_sum_em = np.sum(second_term, axis=2)  # shape: (M, N_f)

#Calcualte fractional contributions per intermediate state for emission:
I_no_int_per_n = np.einsum('mfn,mf->mn', I_no_int_mfn_sumpol, second_term_sum_em)  
C_full_per_n = np.einsum('mfn,mf->mn', C_mfn, second_term_sum_em)  # (M, N_n)
raw_total_from_C = np.sum(C_full_per_n, axis=1)  # (M,)

#Convert to fractional contribution of total (per E_ex), for intermediate states:
total_by_m = np.sum(C_full_per_n, axis=1)  
fractions = (C_full_per_n.T / (total_by_m + 1e-30)).T  

#Convert to fractional contribution of total (per E_ex), for final states:
C_full_per_f *= second_term_sum_em
total_by_m_f = np.sum(C_full_per_f, axis=1)  # sum over final states
fractions_f = (C_full_per_f.T / (total_by_m_f + 1e-30)).T

#contributions done#

#mu_fn_T = mu_fn.transpose(0, 2, 1)
A = np.einsum('afn,mbn->mfab', mu_fn, mu_ni_weighted)  # sum over intermediate states only -> result (E_ex_grid, N_n, cart,cart)

#amplitude of the term over the intermediate states:
I = np.abs(A)**2
#I_sum = np.sum(I, axis=(2, 3))
#print(I.shape)

#second term with broadening of final state:
E_ex_ = E_ex_grid[:, None, None]   # (M,1,1)
E_em_ = E_em_grid[None, None, :]   # (1,1,L)
Ef_   = Ef[None, :, None]           # (1,N_f,1)
Ei_   = Ei[None, None, None]        # scalar broadcasted (1,1,1)

second_term = gamma_f / (((Ef_ - Ei_ - E_ex_ + E_em_)**2) + 0.25*gamma_f**2)
second_term = second_term.squeeze()
#print(second_term.shape)
second_term_sum_em = np.sum(second_term, axis=2) 
#print(second_term_sum_em.shape)

both_terms = np.einsum('ifxy,ifz->ixyz', I, second_term)  # final state summation, dims: (E_ex, cart, cart, E_em)
#both_terms = np.einsum('if,ifl->il', I_sum, second_term)  # final state summation, dims: (E_ex, E_em)
#both_terms = both_terms[::-1, ..., ::-1]
#setting up for total cross section:
### THINK ABOUT CONVERSION LATER#####
au_to_ev = 27.21138602
c_au = 137.035999084
a0_cm = 0.529177e-8
au_area_to_cm2 = a0_cm**2


prefactor_au = (8 * np.pi) / (9 * c_au**4)
sigma_total = prefactor_au * np.einsum('ixyz,i,z->iz', both_terms, E_ex_grid, E_em_grid**3)
#sigma_total = prefactor_au * both_terms * E_ex_grid[:, None] * (E_em_grid[None, :] ** 3)
#sigma_total_normalized = sigma_total / sigma_total.max()

# 9) Save results alongside SIGMA_TOTAL
with h5py.File("rixs_map_with_decomp_allstates.h5", "w") as f:
    f.create_dataset("E_EX", data=E_ex_grid)
    f.create_dataset("E_EM", data=E_em_grid)
    f.create_dataset("SIGMA_TOTAL", data=sigma_total)
    f.create_dataset("I_NO_INT_PER_N", data=I_no_int_per_n)   # (M, N_n)
    f.create_dataset("C_FULL_PER_N", data=C_full_per_n)       # (M, N_n)
    f.create_dataset("FRACTION_PER_N", data=fractions)       # (M, N_n)
    f.create_dataset("INTERMEDIATE_ENERGIES", data=En)
    f.create_dataset("C_FULL_PER_F", data=C_full_per_f)       # final states
    f.create_dataset("FRACTION_PER_F", data=fractions_f)      # final states
    f.create_dataset("FINAL_ENERGIES", data=Ef)

with h5py.File("rixs_map.h5", "w") as f:
    f.create_dataset("E_EX", data=E_ex_grid)
    f.create_dataset("E_EM", data=E_em_grid)
    f.create_dataset("SIGMA_TOTAL", data=sigma_total)

def plot_rixs_map_from_h5(h5_filename):
    
    with h5py.File(h5_filename, 'r') as f:
        E_em = f['E_EM'][:]
        E_ex = f['E_EX'][:]
        rixs_map = f['SIGMA_TOTAL'][:]
       
    #print(f"Intensity range: min={rixs_map.min()}, max={rixs_map.max()}")

    plt.figure(figsize=(8, 6))
    vmin=0 
    vmax = np.max(rixs_map)

    pcm = plt.pcolormesh(E_ex, E_em, rixs_map.T,  # transpose so emission on y-axis
                         shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)

    plt.xlabel('Incident Energy (eV)')
    plt.ylabel('Emission Energy (eV)')
   
    cbar = plt.colorbar(pcm)
    cbar.set_label('Intensity (arb.)')

    plt.tight_layout()
    plt.savefig('_test_M5_RIXS_UO2.png')
    plt.show()

plot_rixs_map_from_h5('rixs_map.h5')

