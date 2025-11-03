import h5py
import matplotlib.pyplot as plt
import numpy as np

h5_filename = "rixs_map_with_decomp_allstates.h5"
N_top_intermediate = 20   # top intermediate states to plot
N_top_final = 20          # top final states to plot

with h5py.File(h5_filename, "r") as f:
    E_ex = f['E_EX'][:]
    I_no_int_per_n = f['I_NO_INT_PER_N'][:]
    C_full_per_n   = f['C_FULL_PER_N'][:]
    fractions_n    = f['FRACTION_PER_N'][:]
    En             = f['INTERMEDIATE_ENERGIES'][:]

    C_full_per_f   = f['C_FULL_PER_F'][:]
    fractions_f    = f['FRACTION_PER_F'][:]
    Ef             = f['FINAL_ENERGIES'][:]



# Top intermediate states
total_fraction_per_n = fractions_n.sum(axis=0)
top_indices_n = np.argsort(total_fraction_per_n)[-N_top_intermediate:]
fractions_top_n = fractions_n[:, top_indices_n]
total_fraction_top_n = fractions_top_n.sum(axis=0)
sorted_idx_n = np.argsort(total_fraction_top_n)
colors_n = plt.cm.rainbow(np.linspace(0, 1, len(top_indices_n)))
labels_n = [f"n={i+1}, E_n={En[i]:.1f} eV" for i in top_indices_n]

# Fractional stackplot for intermediate states
plt.figure(figsize=(10,6))
plt.stackplot(E_ex, fractions_top_n[:, sorted_idx_n].T, labels=np.array(labels_n)[sorted_idx_n], colors=colors_n)
plt.xlabel("Incident energy (eV)")
plt.ylabel("Fractional contribution")
plt.title("Top intermediate-state contributions per incident energy")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('fractional_contributions_intermediate.png', bbox_inches='tight')

# No-interference intensity for intermediate states
plt.figure(figsize=(10,6))
for idx, n_idx in enumerate(top_indices_n[sorted_idx_n]):
    plt.plot(E_ex, I_no_int_per_n[:, n_idx], label=f"n={n_idx+1}, E_n={En[n_idx]:.1f} eV", color=colors_n[idx])
plt.xlabel("Incident energy (eV)")
plt.ylabel("Intensity (arb. units)")
plt.title("Per-intermediate-state intensity (No interference) - Top contributors")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('no_interference_contributions_intermediate.png', bbox_inches='tight')

# With-interference intensity for intermediate states
plt.figure(figsize=(10,6))
for idx, n_idx in enumerate(top_indices_n[sorted_idx_n]):
    plt.plot(E_ex, C_full_per_n[:, n_idx], label=f"n={n_idx+1}, E_n={En[n_idx]:.1f} eV", color=colors_n[idx])
plt.xlabel("Incident energy (eV)")
plt.ylabel("Intensity (arb. units)")
plt.title("Per-intermediate-state intensity (With interference) - Top contributors")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('with_interference_contributions_intermediate.png', bbox_inches='tight')

# Difference plot (interference effect) for intermediate states
plt.figure(figsize=(10,6))
for idx, n_idx in enumerate(top_indices_n[sorted_idx_n]):
    delta = C_full_per_n[:, n_idx] - I_no_int_per_n[:, n_idx]
    plt.plot(E_ex, delta, label=f"n={n_idx+1}, E_n={En[n_idx]:.1f} eV", color=colors_n[idx])
plt.xlabel("Incident energy (eV)")
plt.ylabel("ΔI (arb. units)")
plt.title("Interference effect per intermediate state")
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('difference_interference_contributions_intermediate.png', bbox_inches='tight')


# Top final states
total_fraction_per_f = fractions_f.sum(axis=0)
top_indices_f = np.argsort(total_fraction_per_f)[-N_top_final:]
fractions_top_f = fractions_f[:, top_indices_f]
total_fraction_top_f = fractions_top_f.sum(axis=0)
sorted_idx_f = np.argsort(total_fraction_top_f)
colors_f = plt.cm.rainbow(np.linspace(0, 1, len(top_indices_f)))
labels_f = [f"f={i+1}, E_f={Ef[i]:.1f} eV" for i in top_indices_f]

# Fractional contributions for final states
plt.figure(figsize=(10,6))
plt.stackplot(E_ex, fractions_top_f[:, sorted_idx_f].T, labels=np.array(labels_f)[sorted_idx_f], colors=colors_f)
plt.xlabel("Incident energy (eV)")
plt.ylabel("Fractional contribution")
plt.title("Top final-state contributions per incident energy")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('fractional_contributions_final.png', bbox_inches='tight')

# Per-final-state intensity for final states
plt.figure(figsize=(10,6))
for idx, f_idx in enumerate(top_indices_f[sorted_idx_f]):
    plt.plot(E_ex, C_full_per_f[:, f_idx], label=f"f={f_idx+1}, E_f={Ef[f_idx]:.1f} eV", color=colors_f[idx])
plt.xlabel("Incident energy (eV)")
plt.ylabel("Intensity (arb. units)")
plt.title("Per-final-state intensity - Top contributors")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('intensity_contributions_final.png', bbox_inches='tight')
