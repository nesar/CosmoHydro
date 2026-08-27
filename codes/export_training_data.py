"""One-time export of processed emulator training data for self-contained deployment.

Reproduces the preprocessing in notebooks 02 (GSMF/HMF/fGas/Pk) and 03
(cluster profiles) and writes, per observable, a single ``training_data.npz``
next to the trained pickles in ``models/<OBS>_multiz/``:

    p_train        (100, 7)           scaled design params, rows 0-99 of FinalDesign.txt
    y_vals         (100, n_snap, n_y) processed observable per training sim per snapshot
    y_ind          (n_y,)             x-axis grid (mass / radius / k)
    z_index_range  (n_trained,)       snapshot indices that have a pickle
    snapshot_ids   (n_snap,)          HACC snapshot numbers
    redshifts      (n_snap,)          z for each snapshot
    param_names    (7,)               plain-text parameter names

Reconstruct the SepiaData for snapshot ``zi`` with
``sepia_data_format(p_train, y_vals[:, zi, :], y_ind)`` and load
``multivariate_model_z_index{zi}.pkl``.

Run from the ``codes/`` directory on a machine that has the raw extracts:
    python export_training_data.py
"""
import os
import numpy as np

from cosmo_hydro_emu.load_hacc import (
    read_gsmf_all_snaps, read_hmf_all_snaps, read_gasfr_all_snaps,
    read_profile_all_snaps, read_pk_new, mass_conds, fill_nan_with_interpolation,
)
from cosmo_hydro_emu.snapshot_utils import SNAPSHOT_IDS, get_snapshot_redshifts

# --- settings identical to notebooks 02 / 03 --------------------------------
DirIn = '../data/scidac-400MPC_RUNS_5SG_2COSMO_PARAM-extracts_20260323/'
DirIn_pk = '../data/scidac-olcf-pk_3/'
start_sim_idx = 0
num_sims = 110
z_initial = 200
seed_mass_scale, vkin_scale, eps_scale = 1e6, 1e4, 1e1
PARAM_NAMES = np.array(['kappa_w', 'e_w', 'M_seed/1e6', 'v_kin/1e4',
                        'eps_kin/1e1', 'omega_m', 'sigma_8'])

# --- design ------------------------------------------------------------------
params_all = np.loadtxt('../data/FinalDesign.txt', delimiter=',', skiprows=1)
params32 = params_all[start_sim_idx:start_sim_idx + num_sims].copy()
params32[:, 2] /= seed_mass_scale
params32[:, 3] /= vkin_scale
params32[:, 4] /= eps_scale
test_sim_indices = np.arange(100, 110)
train_sim_indices = np.array([i for i in range(num_sims) if i not in test_sim_indices])
params_train = params32[train_sim_indices]

z_all, a_all = get_snapshot_redshifts(SNAPSHOT_IDS, z_initial=z_initial)
snapshot_ids = np.array(SNAPSHOT_IDS)


def save(obs, y_vals, y_ind, z_index_range, extra=None):
    out_dir = f'../models/{obs}_multiz/'
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'training_data.npz')
    payload = dict(p_train=params_train, y_vals=y_vals[train_sim_indices], y_ind=y_ind,
                   z_index_range=np.asarray(z_index_range), snapshot_ids=snapshot_ids,
                   redshifts=z_all, param_names=PARAM_NAMES)
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)
    print(f'{obs:5s} -> {path}  p_train {params_train.shape}  y_vals {payload["y_vals"].shape}  '
          f'y_ind {y_ind.shape}  z_idx {list(z_index_range)}  '
          f'{os.path.getsize(path) / 1e3:.0f} kB')


# --- GSMF (nb02 cell 11) -----------------------------------------------------
stellar_mass, gsmf_arr = read_gsmf_all_snaps(DirIn, num_sims, SNAPSHOT_IDS, start_sim_idx=start_sim_idx)
mlim1, mlim2 = mass_conds('GSMF')
mass_cond = np.where((stellar_mass > mlim1) & (stellar_mass < mlim2))[0]
for s in range(gsmf_arr.shape[1]):
    gsmf_arr[:, s, :] = fill_nan_with_interpolation(gsmf_arr[:, s, :], 'linear')
save('GSMF', 10**gsmf_arr[:, :, mass_cond], stellar_mass[mass_cond], np.arange(len(SNAPSHOT_IDS)))

# --- HMF (nb02 cell 16) ------------------------------------------------------
halo_mass, hmf_arr = read_hmf_all_snaps(DirIn, num_sims, SNAPSHOT_IDS, start_sim_idx=start_sim_idx)
mlim1_hmf, mlim2_hmf = mass_conds('HMF')
mass_cond_hmf = np.where((halo_mass > mlim1_hmf) & (halo_mass < mlim2_hmf))[0]
for s in range(hmf_arr.shape[1]):
    hmf_arr[:, s, :] = fill_nan_with_interpolation(hmf_arr[:, s, :], 'linear')
save('HMF', 10**hmf_arr[:, :, mass_cond_hmf], halo_mass[mass_cond_hmf], np.arange(len(SNAPSHOT_IDS)))

# --- fGas (nb02 cell 20; models only for snapshots 4-10) --------------------
log_halo_mass, fgas_arr = read_gasfr_all_snaps(DirIn, num_sims, SNAPSHOT_IDS, start_sim_idx=start_sim_idx)
mlim1_fg, mlim2_fg = mass_conds('fGas')
mass_cond_fg = np.where((10**log_halo_mass > mlim1_fg) & (10**log_halo_mass < mlim2_fg))[0]
for s in range(fgas_arr.shape[1]):
    fgas_arr[:, s, :] = fill_nan_with_interpolation(fgas_arr[:, s, :], 'cubic')
save('fGas', fgas_arr[:, :, mass_cond_fg], 10**log_halo_mass[mass_cond_fg], np.arange(4, 11))

# --- cluster profiles (nb03 cells 9-13; models only for snapshots 6-10) -----
PROFILE_PREFIX = {
    'CGD':  'ClusterGasDensityProfile',
    'CGED': 'ClusterGasElectronDensityProfile',
    'CPP':  'ClusterGasPressureProfile',
    'CTP':  'ClusterGasTemperatureProfile',
    'CEP':  'ClusterGasEntropyProfile',
    'CEEP': 'ClusterElectronEntropyProfile',
    'CMP':  'ClusterGasMetallicityProfile',
    'CYP':  'ClusterGasYProfile',
}
rlim1, rlim2 = mass_conds('CGD')
for obs, prefix in PROFILE_PREFIX.items():
    radius, arr = read_profile_all_snaps(DirIn, num_sims, SNAPSHOT_IDS, prefix, start_sim_idx=start_sim_idx)
    rad_cond = np.where((radius > rlim1) & (radius < rlim2))[0]
    save(obs, arr[:, :, rad_cond], radius[rad_cond], np.arange(6, len(SNAPSHOT_IDS)))

# --- Pk ratio, z=0 only (nb02 cell 24; pickle is models/Pk_multivariate_model_z_index0.pkl)
k, pk_arr, pk_go_arr, pk_ratio = read_pk_new(DirIn_pk, num_sims, redshift='0.0',
                                             pk_type='hydro.full', start_sim_idx=start_sim_idx)
mlim1_pk, mlim2_pk = mass_conds('Pk')
k_cond = np.where((k > mlim1_pk) & (k < mlim2_pk))[0]
pk_path = '../models/Pk_training_data.npz'
np.savez_compressed(pk_path, p_train=params_train, y_vals=pk_ratio[train_sim_indices][:, k_cond],
                    y_ind=k[k_cond], z_index_range=np.array([0]), redshifts=np.array([0.0]),
                    param_names=PARAM_NAMES)
print(f'Pk    -> {pk_path}  y_vals {pk_ratio[train_sim_indices][:, k_cond].shape}  '
      f'y_ind {k[k_cond].shape}  {os.path.getsize(pk_path) / 1e3:.0f} kB')
