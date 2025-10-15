import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File("rixs_map_with_decomp.h5", "r") as f:
    E_ex = f['E_EX'][:]
    I_no_int = f['I_NO_INT_PER_N'][:]
    C_full   = f['C_FULL_PER_N'][:]
    En       = f['INTERMEDIATE_ENERGIES'][:]
    fractions = f['FRACTION_PER_N'][:]

#Change this for more state contributions, i.e. N_top = 10 is the top ten contributions only
N_top = 20
total_fraction_per_state = fractions.sum(axis=0)  # sum over incident energies
top_indices = np.argsort(total_fraction_per_state)[-N_top:]  # indices of top N contributors

# Compute contribution-based colors
fractions_top = fractions[:, top_indices]
total_fraction_top = fractions_top.sum(axis=0)
sorted_idx = np.argsort(total_fraction_top)  # sort lowest → highest contribution
colors = plt.cm.rainbow(np.linspace(0, 1, len(top_indices)))

# Optional: labels including intermediate state energy
labels = [f"n={i+1}, E_n={En[i]:.1f} eV" for i in top_indices]

plt.figure(figsize=(10,6))
plt.stackplot(E_ex, fractions_top[:, sorted_idx].T, labels=np.array(labels)[sorted_idx], colors=colors)
plt.xlabel("Incident energy (eV)")
plt.ylabel("Fractional contribution")
plt.title("Top intermediate-state contributions per incident energy")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('fractional_contributions.png', bbox_inches='tight')


# No-interference contributions
plt.figure(figsize=(10,6))
for idx, n_idx in enumerate(top_indices[sorted_idx]):
    plt.plot(E_ex, I_no_int[:, n_idx], label=f"n={n_idx+1}, E_n={En[n_idx]:.1f} eV", color=colors[idx])
plt.xlabel("Incident energy (eV)")
plt.ylabel("Intensity (arb. units)")
plt.title("Per-intermediate-state intensity (No interference) - Top contributors")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('no_interference_contributions.png', bbox_inches='tight')

# With-interference contributions
plt.figure(figsize=(10,6))
for idx, n_idx in enumerate(top_indices[sorted_idx]):
    plt.plot(E_ex, C_full[:, n_idx], label=f"n={n_idx+1}, E_n={En[n_idx]:.1f} eV", color=colors[idx])
plt.xlabel("Incident energy (eV)")
plt.ylabel("Intensity (arb. units)")
plt.title("Per-intermediate-state intensity (With interference) - Top contributors")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('with_interference_contributions.png', bbox_inches='tight')

# Interference effect (difference)
plt.figure(figsize=(10,6))
for idx, n_idx in enumerate(top_indices[sorted_idx]):
    delta = C_full[:, n_idx] - I_no_int[:, n_idx]
    plt.plot(E_ex, delta, label=f"n={n_idx+1}, E_n={En[n_idx]:.1f} eV", color=colors[idx])
plt.xlabel("Incident energy (eV)")
plt.ylabel("ΔI (arb. units)")
plt.title("Interference effect per intermediate state")
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('difference_interference_contributions.png', bbox_inches='tight')

