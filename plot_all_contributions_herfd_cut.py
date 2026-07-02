import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14
})


h5_filename = "rixs_map_with_decomp_allstates.h5"
N_top_intermediate = 20
N_top_final = 20

with h5py.File(h5_filename,"r") as f:

    E_ex = f['E_EX'][:]
    E_em = f['E_EM'][:]
    sigma = f['SIGMA_TOTAL'][:]

    I_no_int_per_n = f['I_NO_INT_PER_N'][:]
    C_full_per_n = f['C_FULL_PER_N'][:]
    fractions_n = f['FRACTION_PER_N'][:]
    En = f['INTERMEDIATE_ENERGIES'][:]

    C_full_per_f = f['C_FULL_PER_F'][:]
    fractions_f = f['FRACTION_PER_F'][:]
    Ef = f['FINAL_ENERGIES'][:]

    SO_n = f["INTERMEDIATE_SO_STATES"][:]
    SO_f = f["FINAL_SO_STATES"][:]


# Find HERFD emission energy
max_idx = np.unravel_index(np.argmax(sigma), sigma.shape)
ex_idx, em_idx = max_idx

E_em_max = E_em[em_idx]

print(f"HERFD emission energy: {E_em_max:.3f} eV")


# HERFD spectrum
herfd = sigma[:, em_idx]

plt.figure(figsize=(8,5),dpi=300)
plt.plot(E_ex,herfd,color='k')
plt.xlabel("Incident energy (eV)")
plt.ylabel("HERFD intensity (arb.)")
plt.title(f"HERFD cut at E_em = {E_em_max:.2f} eV")
plt.savefig("herfd_cut.png",bbox_inches='tight')


# INTERMEDIATE STATES
total_fraction_per_n = fractions_n.sum(axis=0)
top_indices_n = np.argsort(total_fraction_per_n)[-N_top_intermediate:]

fractions_top_n = fractions_n[:,top_indices_n]

total_fraction_top_n = fractions_top_n.sum(axis=0)
sorted_idx_n = np.argsort(total_fraction_top_n)

colors_n = plt.cm.rainbow(np.linspace(0,1,len(top_indices_n)))

labels_n = [
    f"SO {SO_n[i]}, E_n={En[i]:.1f} eV"
    for i in top_indices_n
]

plt.figure(figsize=(10,6),dpi=300)

plt.stackplot(
    E_ex,
    fractions_top_n[:,sorted_idx_n].T,
    labels=np.array(labels_n)[sorted_idx_n],
    colors=colors_n
)

herfd_norm = herfd / np.max(herfd)

plt.plot(E_ex, herfd_norm, color='black', linewidth=5, zorder=10)
plt.plot(E_ex, herfd_norm, color='#fcfbf4', linewidth=4, zorder=11)

plt.xlabel("Incident energy (eV)")
plt.ylabel("Fractional contribution")
plt.title(f"Intermediate states contributing to HERFD\n(E_em={E_em_max:.2f} eV)")
plt.legend(loc='upper left',bbox_to_anchor=(1,1))

plt.savefig("herfd_fractional_intermediate.png",bbox_inches='tight')


# Intermediate intensity contributions
plt.figure(figsize=(10,6),dpi=300)



for idx,n_idx in enumerate(top_indices_n[sorted_idx_n]):

    plt.plot(
        E_ex,
        C_full_per_n[:,n_idx],
        label=f"SO {SO_n[n_idx]}, E_n={En[n_idx]:.1f} eV",
        color=colors_n[idx], alpha=0.85
    )


plt.xlabel("Incident energy (eV)")
plt.ylabel("Intensity (arb.)")
plt.title(f"Intermediate contributions to HERFD\n(E_em={E_em_max:.2f} eV)")
plt.legend(loc='upper left',bbox_to_anchor=(1,1))

plt.savefig("herfd_intermediate_contributions.png",bbox_inches='tight')


# Intermediate difference plot: interference effect
plt.figure(figsize=(10,6), dpi=300)

for idx, n_idx in enumerate(top_indices_n[sorted_idx_n]):

    delta = C_full_per_n[:, n_idx] - I_no_int_per_n[:, n_idx]

    plt.plot(
        E_ex,
        delta,
        label=f"SO {SO_n[n_idx]}, E_n={En[n_idx]:.1f} eV",
        color=colors_n[idx],
        alpha=0.85
    )

plt.xlabel("Incident energy (eV)")
plt.ylabel("ΔI (arb.)")
plt.title(f"Intermediate interference effect\n(E_em={E_em_max:.2f} eV)")
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.legend(loc='upper left', bbox_to_anchor=(1,1))

plt.savefig("herfd_intermediate_difference.png", bbox_inches='tight')

# FINAL STATES
total_fraction_per_f = fractions_f.sum(axis=0)
top_indices_f = np.argsort(total_fraction_per_f)[-N_top_final:]

fractions_top_f = fractions_f[:,top_indices_f]

total_fraction_top_f = fractions_top_f.sum(axis=0)
sorted_idx_f = np.argsort(total_fraction_top_f)

colors_f = plt.cm.rainbow(np.linspace(0,1,len(top_indices_f)))

labels_f = [
    f"SO {SO_f[i]}, E$_f$={Ef[i]:.1f} eV"
    for i in top_indices_f
]

plt.figure(figsize=(10,6),dpi=300)

plt.stackplot(
    E_ex,
    fractions_top_f[:,sorted_idx_f].T,
    labels=np.array(labels_f)[sorted_idx_f],
    colors=colors_f, alpha=0.85
)


herfd_norm = herfd / np.max(herfd)

plt.plot(E_ex, herfd_norm, color='black', linewidth=5, zorder=10)
plt.plot(E_ex, herfd_norm, color='#fcfbf4', linewidth=4, zorder=11)

plt.xlabel("Incident energy (eV)")
plt.ylabel("Fractional contribution")
plt.title(f"Final states contributing to HERFD\n(E_em={E_em_max:.2f} eV)")
plt.legend(loc='upper left',bbox_to_anchor=(1,1))

plt.savefig("herfd_fractional_final.png",bbox_inches='tight')

# FINAL STATES (absolute normalized stackplot)

# normalize using ALL final states
norm = np.max(np.sum(C_full_per_f, axis=1))

C_full_norm = C_full_per_f / norm

# select top states
C_full_top_f = C_full_norm[:, top_indices_f]
C_full_top_f = C_full_top_f[:, sorted_idx_f]

# normalized HERFD
herfd_norm = herfd / np.max(herfd)

# remainder from states not plotted
remainder = herfd_norm - np.sum(C_full_top_f, axis=1)
remainder = np.clip(remainder, 0, None)

stack_data = np.vstack([C_full_top_f.T, remainder])

plt.figure(figsize=(10,6), dpi=300)

plt.stackplot(
    E_ex,
    stack_data,
    colors=[*colors_f, "lightgray"],
    labels=[*np.array(labels_f)[sorted_idx_f], "Other states"],
    alpha=0.9
)

plt.plot(E_ex, herfd_norm, color='black', linewidth=5, zorder=10)
plt.plot(E_ex, herfd_norm, color='#fcfbf4', linewidth=4, zorder=11)

plt.xlabel("Incident energy (eV)")
plt.ylabel("Normalized intensity")
plt.title(f"Absolute final-state contributions\n(E_em={E_em_max:.2f} eV)")

plt.legend(loc='upper left', bbox_to_anchor=(1,1))

plt.savefig("herfd_absolute_final.png", bbox_inches='tight')

# HERFD-CUT ORBITAL GROUPED STACKPLOT
# Orbital contributions (HERFD cut)

orbitals = {
    "δ": (r"5f$_\delta$", "#ffdbc7"),
    "φ": (r"5f$_\phi$", "#f7a482"),
    "π*": (r"5f$_{\pi^*}$", "#d85f4c"),
    "σ*": (r"5f$_{\sigma^*}$", "#b41529"),
}

state_key_f = [
    (1, 3, "δ"),
    (4, 25, "φ"),
    (26, 27, "δ"),
    (28, 29, "φ"),
    (29, 30, "π*"),
    (35, 60, "φ"),
    (60, 170, "σ*"),
]

# map SO numbers -> column index
f_index = {state: i for i, state in enumerate(SO_f)}

# normalize exactly as for the absolute final-state plot
norm = np.max(np.sum(C_full_per_f, axis=1))
C_full_norm = C_full_per_f / norm

orbital_contrib = {
    orb: np.zeros_like(C_full_norm[:, 0])
    for orb in orbitals
}

for start, stop, orb in state_key_f:

    cols = [f_index[s] for s in range(start, stop) if s in f_index]

    if cols:
        orbital_contrib[orb] += np.sum(
            C_full_norm[:, cols],
            axis=1
        )

stack_data = []
labels = []
colors = []

for orb, (label, color) in orbitals.items():
    stack_data.append(orbital_contrib[orb])
    labels.append(label)
    colors.append(color)

stack_data = np.array(stack_data)

plt.figure(figsize=(10,6), dpi=300)

plt.stackplot(
    E_ex,
    stack_data,
    colors=colors,
    labels=labels,
    alpha=0.9,
)

plt.plot(E_ex, herfd_norm, color="black", lw=5, zorder=10)
plt.plot(E_ex, herfd_norm, color="#fcfbf4", lw=4, zorder=11)

plt.xlabel("Incident energy (eV)")
plt.ylabel("Normalized intensity")
plt.title("HERFD orbital contributions")

plt.legend(loc="upper left", bbox_to_anchor=(1,1))

plt.savefig(
    "herfd_orbital_contributions.png",
    dpi=300,
    bbox_inches="tight",
)
