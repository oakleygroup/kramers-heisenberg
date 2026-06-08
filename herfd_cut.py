import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def print_and_plot_rixs_cuts(h5_filename, ex_cuts=None, em_cuts=None, max_int=False):
    with h5py.File(h5_filename, 'r') as f:
        E_ex = f['E_EX'][:]
        E_em = f['E_EM'][:]
        rixs_map = f['SIGMA_TOTAL'][:]

    if max_int:

        # find global maximum index
        max_idx = np.unravel_index(np.argmax(rixs_map), rixs_map.shape)
        ex_idx, em_idx = max_idx

        print(f"Maximum intensity at:")
        print(f"  Incident energy = {E_ex[ex_idx]:.2f} eV")
        print(f"  Emission energy = {E_em[em_idx]:.2f} eV")

        profile = rixs_map[:, em_idx]

        plt.figure()
        plt.plot(E_ex, profile)
        plt.xlabel('Incident Energy (eV)')
        plt.ylabel('Intensity (arb.)')
        plt.title(f'Incident profile at max intensity emission = {E_em[em_idx]:.2f} eV')
        plt.grid(True)
        plt.show()

        filename = f'incident_cut_max_em_{E_em[em_idx]:.2f}eV.txt'
        data_to_save = np.column_stack((E_ex, profile))
        np.savetxt(filename, data_to_save,
                   header='Incident Energy (eV)    Intensity (arb.)')

        print(f'Saved max-intensity cut to {filename}')

        # Save new H5 file for plotting_all_contributions.py
        h5_out = f'rixs_map_emcut_{E_em[em_idx]:.2f}eV.h5'

        with h5py.File(h5_out, 'w') as fout:
            with h5py.File(h5_filename, 'r') as fin:
                for key in fin.keys():
                    if key == 'SIGMA_TOTAL':
                        fout.create_dataset('SIGMA_TOTAL', data=profile)
                    elif key == 'E_EM':
                        fout.create_dataset('E_EM', data=np.array([E_em[em_idx]]))
                    else:
                        fout.create_dataset(key, data=fin[key][:])

            fout.create_dataset('SELECTED_EMISSION_ENERGY', data=E_em[em_idx])
            fout.create_dataset('SELECTED_EMISSION_INDEX', data=em_idx)

        print(f'Saved emission-cut H5 to {h5_out}')


    if ex_cuts is not None:
        for E in ex_cuts:
            idx = np.abs(E_ex - E).argmin()
            profile = rixs_map[idx, :]  # emission profile at fixed incident energy

            plt.figure()
            plt.plot(E_em, profile)
            plt.xlabel('Emission Energy (eV)')
            plt.ylabel('Intensity (arb.)')
            plt.title(f'Emission profile at Incident Energy = {E_ex[idx]:.2f} eV')
            plt.grid(True)
            plt.show()

            # Save to text file: columns -> Emission Energy, Intensity
            filename = f'emission_cut_{E_ex[idx]:.2f}eV.txt'
            data_to_save = np.column_stack((E_em, profile))
            np.savetxt(filename, data_to_save,
                       header='Emission Energy (eV)    Intensity (arb.)')
            print(f'Saved emission cut data to {filename}')


    if em_cuts is not None:
        for E in em_cuts:
            idx = np.abs(E_em - E).argmin()
            profile = rixs_map[:, idx]  # incident profile at fixed emission energy

            plt.figure()
            plt.plot(E_ex, profile)
            plt.xlabel('Incident Energy (eV)')
            plt.ylabel('Intensity (arb.)')
            plt.title(f'Incident profile at Emission Energy = {E_em[idx]:.2f} eV')
            plt.grid(True)
            plt.show()

            # Save to text file: columns -> Incident Energy, Intensity
            filename = f'incident_cut_em_{E_em[idx]:.2f}eV.txt'
            data_to_save = np.column_stack((E_ex, profile))
            np.savetxt(filename, data_to_save,
                       header='Incident Energy (eV)    Intensity (arb.)')
            print(f'Saved emission cut data to {filename}')

            # Save new H5 file for plotting_all_contributions.py
            h5_out = f'rixs_map_emcut_{E_em[idx]:.2f}eV.h5'

            with h5py.File(h5_out, 'w') as fout:
                with h5py.File(h5_filename, 'r') as fin:
                    for key in fin.keys():
                        if key == 'SIGMA_TOTAL':
                            fout.create_dataset('SIGMA_TOTAL', data=profile)
                        elif key == 'E_EM':
                            fout.create_dataset('E_EM', data=np.array([E_em[idx]]))
                        else:
                            fout.create_dataset(key, data=fin[key][:])

                fout.create_dataset('SELECTED_EMISSION_ENERGY', data=E_em[idx])
                fout.create_dataset('SELECTED_EMISSION_INDEX', data=idx)

            print(f'Saved emission-cut H5 to {h5_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot RIXS cuts from rixs_map.h5 file')
    parser.add_argument('--h5', type=str, required=True, help='Path to rixs_map.h5 file')
    parser.add_argument('--ex-cuts', type=float, nargs='*', help='List of excitation energy cuts (eV)')
    parser.add_argument('--em-cuts', type=float, nargs='*', help='List of emission energy cuts (eV)')
    parser.add_argument(
        '--max-int',
        action='store_true',
        help='Make an incident-energy cut at the emission energy with maximum intensity'
    )

    args = parser.parse_args()

    print_and_plot_rixs_cuts(
        args.h5,
        ex_cuts=args.ex_cuts,
        em_cuts=args.em_cuts,
        max_int=args.max_int
    )
