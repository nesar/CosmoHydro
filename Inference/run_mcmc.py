#!/usr/bin/env python
"""
Unified MCMC inference driver.

Usage:
    python run_mcmc.py configs/GSMF_subgrid.yaml
    python run_mcmc.py configs/GSMF_CGD_bias.yaml
    python run_mcmc.py configs/sigma8_vkin_custom.yaml
"""

import sys
import os
import argparse
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codes'))

from cosmo_hydro_emu.pca import *
from cosmo_hydro_emu.load_hacc import (
    PARAM_NAME, PARAM_NAME_SG, PARAM_NAME_COSMO,
    seed_mass_scale, vkin_scale, eps_scale,
    sepia_data_format, mass_conds, fill_nan_with_interpolation,
    read_gsmf, read_gasfr, read_cgd, read_cged, read_hmf, read_pk, read_csfr,
    read_gsmf_all_snaps, read_hmf_all_snaps, read_gasfr_all_snaps,
    read_cgd_all_snaps, read_cged_all_snaps,
    load_gsmf_obs, load_fgas_obs, load_cgd_obs, load_cged_obs,
)
from cosmo_hydro_emu.emu import emulate, load_model_multiple
from cosmo_hydro_emu.gp import gp_load
from cosmo_hydro_emu.snapshot_utils import SNAPSHOT_IDS, get_snapshot_redshifts
from cosmo_hydro_emu.mcmc import (
    log_likelihood, ln_prob, ln_prior,
    chain_init, define_sampler, do_mcmc, mcmc_results,
)

# ---------------------------------------------------------------------------
# Short key -> PARAM_NAME label mapping
# ---------------------------------------------------------------------------
SHORT_KEY_TO_LABEL = {
    'kappa_w':  PARAM_NAME[0],
    'e_w':      PARAM_NAME[1],
    'M_seed':   PARAM_NAME[2],
    'v_kin':    PARAM_NAME[3],
    'eps_kin':  PARAM_NAME[4],
    'omega_m':  PARAM_NAME[5],
    'sigma_8':  PARAM_NAME[6],
}

# Short key -> column index in design matrix
SHORT_KEY_TO_IDX = {
    'kappa_w': 0, 'e_w': 1, 'M_seed': 2, 'v_kin': 3, 'eps_kin': 4,
    'omega_m': 5, 'sigma_8': 6,
}

# Scaling factors per column
SCALE_FACTORS = {2: seed_mass_scale, 3: vkin_scale, 4: eps_scale}


# ---------------------------------------------------------------------------
# Data readers per observable
# ---------------------------------------------------------------------------
READER_MAP = {
    'GSMF': read_gsmf,
    'fGas': read_gasfr,
    'CGD':  read_cgd,
    'CGED': read_cged,
    'HMF':  read_hmf,
    'Pk':   read_pk,
    'CSFR': read_csfr,
}

# Multi-snapshot readers (return shape: num_sims x num_snaps x num_bins)
READER_MULTIZ_MAP = {
    'GSMF': read_gsmf_all_snaps,
    'fGas': read_gasfr_all_snaps,
    'CGD':  read_cgd_all_snaps,
    'CGED': read_cged_all_snaps,
    'HMF':  read_hmf_all_snaps,
}

# Model file prefixes
MODEL_PREFIX = {
    'GSMF': 'GSMF',
    'fGas': 'fGas',
    'CGD':  'CGD',
    'CGED': 'CGED',
    'HMF':  'HMF',
    'Pk':   'Pk',
    'CSFR': 'CSFR',
}


def parse_observables(cfg):
    """Parse the observables list from config.

    Supports both formats:
        observables:                    # simple (z=0)
          - GSMF
          - CGD

        observables:                    # per-observable redshift
          - name: GSMF
            redshift: 0.0
          - name: GSMF
            redshift: 1.0
          - name: CGD
            redshift: 0.4

    Returns list of dicts: [{'name': str, 'redshift': float}, ...]
    """
    raw = cfg['observables']
    parsed = []
    for entry in raw:
        if isinstance(entry, str):
            parsed.append({'name': entry, 'redshift': 0.0})
        elif isinstance(entry, dict):
            parsed.append({
                'name': entry['name'],
                'redshift': float(entry.get('redshift', 0.0)),
            })
        else:
            raise ValueError(f"Invalid observable entry: {entry}")
    return parsed


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_design(design_file, start_sim_idx=1, num_sims=None):
    """Load parameter design matrix from CSV, apply scaling.

    Parameters
    ----------
    start_sim_idx : int
        First simulation index (1-based, matching RUN directory numbering).
    num_sims : int or None
        Number of simulations to use. If None, uses all from start_sim_idx.
    """
    import pandas as pd
    df = pd.read_csv(design_file)
    params = df.values.astype(float)
    # Slice to match the simulations we're using
    # Row 0 of the CSV data corresponds to RUN001 (i.e., the design is 0-indexed after header)
    start_row = start_sim_idx - 1
    end_row = start_row + num_sims if num_sims else params.shape[0]
    params = params[start_row:end_row]
    # Apply scaling
    for col, scale in SCALE_FACTORS.items():
        params[:, col] = params[:, col] / scale
    return params


def prepare_observable(obs_name, params, cfg):
    """Load simulation data for a single observable, apply mass/radius cuts, return y_vals, y_ind."""
    data_cfg = cfg['data']
    dir_in = os.path.join(os.path.dirname(__file__), data_cfg['DirIn'])
    num_sims = data_cfg['num_sims']
    start_sim_idx = data_cfg.get('start_sim_idx', 1)

    reader = READER_MAP[obs_name]

    if obs_name == 'Pk':
        k, pk_arr, pk_ratio = reader(dir_in, num_sims, params, start_sim_idx=start_sim_idx)
        mlim1, mlim2 = mass_conds('Pk')
        cond = np.where((k > mlim1) & (k < mlim2))
        y_vals = pk_ratio[:, cond[0]]
        y_ind = k[cond]
        return y_vals, y_ind

    x_all, y_all = reader(dir_in, num_sims, params, start_sim_idx=start_sim_idx)

    # For GSMF/fGas: x_all might be log or linear
    if obs_name == 'GSMF':
        mlim1, mlim2 = mass_conds('GSMF')
        # Fill NaN
        y_all = fill_nan_with_interpolation(y_all, 'linear')
        cond = np.where((x_all > mlim1) & (x_all < mlim2))
        y_vals = 10**y_all[:, cond[0]]
        y_ind = x_all[cond]

    elif obs_name == 'fGas':
        mlim1, mlim2 = mass_conds('fGas')
        y_all = fill_nan_with_interpolation(y_all, 'cubic')
        cond = np.where((10**x_all > mlim1) & (10**x_all < mlim2))
        y_vals = y_all[:, cond[0]]
        y_ind = 10**x_all[cond]

    elif obs_name == 'HMF':
        mlim1, mlim2 = mass_conds('HMF')
        cond = np.where((x_all > mlim1) & (x_all < mlim2))
        y_vals = y_all[:, cond[0]]
        y_ind = x_all[cond]

    elif obs_name in ('CGD', 'CGED'):
        mlim1, mlim2 = mass_conds(obs_name)
        cond = np.where((x_all > mlim1) & (x_all < mlim2))
        y_vals = y_all[:, cond[0]]
        y_ind = x_all[cond]

    elif obs_name == 'CSFR':
        mlim1, mlim2 = mass_conds('CSFR')
        cond = np.where((x_all > mlim1) & (x_all < mlim2))
        y_vals = y_all[:, cond[0]]
        y_ind = x_all[cond]

    else:
        raise ValueError(f"Unknown observable: {obs_name}")

    return y_vals, y_ind


def prepare_observable_multiz(obs_name, params, cfg, snapshot_ids):
    """Load multi-snapshot simulation data for an observable.

    Returns y_vals_all (num_sims, num_snaps, num_bins) and y_ind (num_bins,).
    """
    data_cfg = cfg['data']
    dir_in = os.path.join(os.path.dirname(__file__), data_cfg['DirIn'])
    num_sims = data_cfg['num_sims']

    reader = READER_MULTIZ_MAP.get(obs_name)
    if reader is None:
        raise ValueError(f"Multi-z reader not available for '{obs_name}'")

    x_all, y_all = reader(dir_in, num_sims, snapshot_ids)
    # y_all shape: (num_sims, num_snaps, num_bins)

    if obs_name == 'GSMF':
        mlim1, mlim2 = mass_conds('GSMF')
        cond = np.where((x_all > mlim1) & (x_all < mlim2))[0]
        # Apply NaN filling per snapshot
        for s in range(y_all.shape[1]):
            y_all[:, s, :] = fill_nan_with_interpolation(y_all[:, s, :], 'linear')
        y_vals_all = 10**y_all[:, :, cond]
        y_ind = x_all[cond]

    elif obs_name == 'fGas':
        mlim1, mlim2 = mass_conds('fGas')
        cond = np.where((10**x_all > mlim1) & (10**x_all < mlim2))[0]
        for s in range(y_all.shape[1]):
            y_all[:, s, :] = fill_nan_with_interpolation(y_all[:, s, :], 'cubic')
        y_vals_all = y_all[:, :, cond]
        y_ind = 10**x_all[cond]

    elif obs_name in ('CGD', 'CGED'):
        mlim1, mlim2 = mass_conds(obs_name)
        cond = np.where((x_all > mlim1) & (x_all < mlim2))[0]
        y_vals_all = y_all[:, :, cond]
        y_ind = x_all[cond]

    elif obs_name == 'HMF':
        mlim1, mlim2 = mass_conds('HMF')
        cond = np.where((x_all > mlim1) & (x_all < mlim2))[0]
        y_vals_all = y_all[:, :, cond]
        y_ind = x_all[cond]

    else:
        raise ValueError(f"Unknown observable for multi-z: {obs_name}")

    return y_vals_all, y_ind


def load_model(model_filename, p_all, y_vals, y_ind, exp_variance):
    """Load a trained GP model."""
    sepia_data = sepia_data_format(p_all, y_vals, y_ind)
    sepia_model_i = do_pca(sepia_data, exp_variance=exp_variance)
    sepia_model = gp_load(sepia_model_i, model_filename)
    return sepia_model


def load_obs_data(obs_name, cfg):
    """Load observational data for a given observable. Returns dict with x, y, yerr."""
    obs_dirs = cfg.get('obs_dirs', {})

    if obs_name == 'GSMF':
        gsmf_dir = obs_dirs.get('gsmf')
        if gsmf_dir is None:
            raise ValueError("obs_dirs.gsmf not set in config")
        mlim1, mlim2 = mass_conds('GSMF')
        x_raw, y_raw, yerr_raw = load_gsmf_obs(gsmf_dir)
        m = (x_raw > mlim1) & (x_raw < mlim2)
        return {'x': x_raw[m], 'y': 10**y_raw[m], 'yerr': yerr_raw[:, m]}

    elif obs_name == 'fGas':
        mlim1, mlim2 = mass_conds('fGas')
        x1_raw, y1_raw, yerr1_raw = load_fgas_obs()
        m = (x1_raw > mlim1) & (x1_raw < mlim2)
        return {'x': x1_raw[m], 'y': y1_raw[m], 'yerr': yerr1_raw[m]}

    elif obs_name == 'CGD':
        cgd_dir = obs_dirs.get('cgd')
        if cgd_dir is None:
            raise ValueError("obs_dirs.cgd not set in config")
        data_cgd = load_cgd_obs(cgd_dir)
        rlim1, rlim2 = mass_conds('CGD')
        x2_raw = data_cgd['mcdonald2017_avg'][0]
        y2_raw = data_cgd['mcdonald2017_avg'][1][:, 0]
        r = (x2_raw > rlim1) & (x2_raw < rlim2)
        y_cgd = y2_raw[r]
        return {'x': x2_raw[r], 'y': y_cgd, 'yerr': 0.05 * y_cgd}

    elif obs_name == 'CGED':
        cged_dir = obs_dirs.get('cged')
        if cged_dir is None:
            raise ValueError("obs_dirs.cged not set in config")
        data_cged = load_cged_obs(cged_dir)
        rlim1, rlim2 = mass_conds('CGED')
        # Use first available dataset key
        key = list(data_cged.keys())[0]
        x_raw = data_cged[key][0]
        y_raw = data_cged[key][1][:, 0]
        r = (x_raw > rlim1) & (x_raw < rlim2)
        y_cged = y_raw[r]
        return {'x': x_raw[r], 'y': y_cged, 'yerr': 0.05 * y_cged}

    else:
        raise ValueError(f"No observational data loader for '{obs_name}'. "
                         f"Add target data or implement load_obs_data for it.")


def build_param_space(cfg, design_params):
    """
    Build the parameter list and fixed_params dict from config.

    Returns
    -------
    params_list : list of [name, init, min, max]
    fixed_params : dict {PARAM_NAME_label: value}
    param_names : list of PARAM_NAME labels used by the emulator
    with_underestimation_bias : bool
    """
    param_mode = cfg.get('param_mode', 'subgrid+cosmo')
    bias_cfg = cfg.get('bias_params', [])
    with_bias = len(bias_cfg) > 0

    # Determine which parameter names the emulator uses
    if param_mode == 'subgrid':
        param_names = list(PARAM_NAME_SG) + list(PARAM_NAME_COSMO)  # emulator always needs all 7
    elif param_mode == 'cosmo':
        param_names = list(PARAM_NAME_SG) + list(PARAM_NAME_COSMO)
    elif param_mode in ('subgrid+cosmo', 'all'):
        param_names = list(PARAM_NAME)
    elif param_mode == 'custom':
        param_names = list(PARAM_NAME)
    else:
        raise ValueError(f"Unknown param_mode: {param_mode}")

    # Get parameter ranges from design matrix
    allMin = np.min(design_params, axis=0)
    allMax = np.max(design_params, axis=0)

    # Build fixed_params dict (latex labels -> values)
    fixed_params = {}
    cfg_fixed = cfg.get('fixed_params', {})
    for short_key, val in cfg_fixed.items():
        label = SHORT_KEY_TO_LABEL[short_key]
        fixed_params[label] = val

    # Determine which params are free vs fixed based on param_mode
    if param_mode == 'subgrid':
        # Fix cosmo params at midpoints
        for i, label in enumerate(PARAM_NAME_COSMO):
            if label not in fixed_params:
                idx = 5 + i  # cosmo params are columns 5,6
                fixed_params[label] = 0.5 * (allMin[idx] + allMax[idx])

    elif param_mode == 'cosmo':
        # Fix subgrid params at midpoints
        for i, label in enumerate(PARAM_NAME_SG):
            if label not in fixed_params:
                fixed_params[label] = 0.5 * (allMin[i] + allMax[i])

    elif param_mode == 'custom':
        free_keys = cfg.get('free_params', [])
        free_labels = set(SHORT_KEY_TO_LABEL[k] for k in free_keys)
        for i, label in enumerate(PARAM_NAME):
            if label not in free_labels and label not in fixed_params:
                fixed_params[label] = 0.5 * (allMin[i] + allMax[i])

    # Build params_list for free parameters
    params_list = []
    for i, label in enumerate(param_names):
        if label not in fixed_params:
            init_val = 0.5 * (allMin[i] + allMax[i])
            params_list.append([label, init_val, allMin[i], allMax[i]])

    # Add bias params
    for bp in bias_cfg:
        params_list.append([bp['name'], bp['initial'], bp['min'], bp['max']])

    return params_list, fixed_params, param_names, with_bias


def main():
    parser = argparse.ArgumentParser(description='Run MCMC inference trial')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Set up everything but skip MCMC sampling')
    args = parser.parse_args()

    cfg = load_config(args.config)
    trial_name = cfg['trial_name']
    data_cfg = cfg['data']

    # Parse observables (supports both simple list and per-z dicts)
    obs_entries = parse_observables(cfg)

    print(f"=== Trial: {trial_name} ===")
    for entry in obs_entries:
        print(f"  {entry['name']}  z={entry['redshift']}")

    # Load design matrix
    design_file = os.path.join(os.path.dirname(__file__), data_cfg['design_file'])
    start_sim_idx = data_cfg.get('start_sim_idx', 1)
    num_sims = data_cfg['num_sims']
    design_params = load_design(design_file, start_sim_idx=start_sim_idx, num_sims=num_sims)
    print(f"Design matrix: {design_params.shape[0]} simulations, {design_params.shape[1]} parameters")

    # Build parameter space
    params_list, fixed_params, param_names, with_bias = build_param_space(cfg, design_params)
    flat_indices = cfg.get('flat_prior_indices', [])

    print(f"Free parameters ({len(params_list)}):")
    for p in params_list:
        print(f"  {p[0]:30s}  init={p[1]:.4f}  range=[{p[2]:.4f}, {p[3]:.4f}]")
    if fixed_params:
        print(f"Fixed parameters: {fixed_params}")

    # Check if any observable needs multi-z
    any_nonzero_z = any(e['redshift'] > 0 for e in obs_entries)

    # Compute snapshot redshifts if needed
    snapshot_ids = SNAPSHOT_IDS
    z_all = None
    if any_nonzero_z:
        z_all_arr, _ = get_snapshot_redshifts(snapshot_ids)
        z_all = z_all_arr
        print(f"Multi-z mode: {len(snapshot_ids)} snapshots, z range [{z_all.min():.2f}, {z_all.max():.2f}]")

    # Load emulator models and observational data
    model_dir = os.path.join(os.path.dirname(__file__), data_cfg['model_dir'])
    exp_variance = data_cfg['exp_variance']
    z_index = data_cfg.get('z_index', 0)

    x_grids = []
    sepia_models = []
    data_list = []
    redshifts = []     # per-observable target redshift (0.0 = use single model)
    obs_names = []     # for case_labels

    for entry in obs_entries:
        obs = entry['name']
        z_target = entry['redshift']
        obs_names.append(obs)
        redshifts.append(z_target)

        print(f"\nLoading {obs} (z={z_target})...")

        if z_target > 0 and z_all is not None:
            # --- Multi-z path: load all snapshots, build model list ---
            y_vals_all, y_ind = prepare_observable_multiz(obs, design_params, cfg, snapshot_ids)
            print(f"  y_vals_all shape: {y_vals_all.shape}, y_ind shape: {y_ind.shape}")

            model_subdir = os.path.join(model_dir, f'{MODEL_PREFIX[obs]}_multiz/')
            z_index_range = np.arange(len(snapshot_ids))
            model_list, _ = load_model_multiple(
                model_dir=model_subdir,
                p_train_all=design_params,
                y_vals_all=y_vals_all,
                y_ind_all=y_ind,
                z_index_range=z_index_range,
            )
            sepia_models.append(model_list)  # list of models across snapshots
            x_grids.append(y_ind)
        else:
            # --- Single-z path (z=0): use z_index model ---
            y_vals, y_ind = prepare_observable(obs, design_params, cfg)
            print(f"  y_vals shape: {y_vals.shape}, y_ind shape: {y_ind.shape}")

            model_file = os.path.join(model_dir,
                                      f'{MODEL_PREFIX[obs]}_multivariate_model_z_index{z_index}')
            model = load_model(model_file, design_params, y_vals, y_ind, exp_variance)
            sepia_models.append(model)
            x_grids.append(y_ind)

        # Load observational data
        obs_data = load_obs_data(obs, cfg)
        data_list.append(obs_data)
        print(f"  Obs data: {len(obs_data['x'])} points")

    # Build case_labels string (underscore-joined observable names)
    case_labels = '_'.join(obs_names)

    # Test likelihood at initial point
    theta0 = np.array([p[1] for p in params_list])
    print(f"\nTest theta: {theta0}")
    print(f"ln_prior = {ln_prior(theta0, params_list, flat_indices=flat_indices):.4f}")

    for i, entry in enumerate(obs_entries):
        ll = log_likelihood(theta0, x_grids[i], sepia_models[i],
                            data_list[i]['x'], data_list[i]['y'], data_list[i]['yerr'],
                            fixed_params=fixed_params,
                            with_underestimation_bias=with_bias,
                            case_label=entry['name'],
                            param_names=param_names,
                            redshift=redshifts[i],
                            z_all=z_all)
        print(f"  LL({entry['name']}, z={entry['redshift']}) = {ll:.4f}")

    if args.dry_run:
        print("\n--- DRY RUN: skipping MCMC ---")
        return

    # Run MCMC
    mcmc_cfg = cfg['mcmc']
    ndim = len(params_list)
    nwalkers = mcmc_cfg['nwalkers']
    nburn = mcmc_cfg['nburn']
    nrun = mcmc_cfg['nrun']
    use_pool = mcmc_cfg.get('parallel', True)

    print(f"\nMCMC: ndim={ndim}, nwalkers={nwalkers}, nburn={nburn}, nrun={nrun}")

    pos0 = chain_init(params_list, ndim, nwalkers)

    sampler_kwargs = dict(
        ndim=ndim, nwalkers=nwalkers, params_list=params_list,
        x_grids=x_grids, sepia_models=sepia_models, data=data_list,
        fixed_params=fixed_params, with_underestimation_bias=with_bias,
        case_labels=case_labels, flat_indices=flat_indices,
        param_names=param_names, redshifts=redshifts, z_all=z_all,
    )

    if use_pool:
        from multiprocessing import Pool
        with Pool() as pool:
            sampler = define_sampler(**sampler_kwargs, pool=pool)
            pos, prob, state, samples, sampler = do_mcmc(sampler, pos0, nburn, ndim, if_burn=True)
            pos, prob, state, samples, sampler = do_mcmc(sampler, pos, nrun, ndim, if_burn=False)
    else:
        sampler = define_sampler(**sampler_kwargs)
        pos, prob, state, samples, sampler = do_mcmc(sampler, pos0, nburn, ndim, if_burn=True)
        pos, prob, state, samples, sampler = do_mcmc(sampler, pos, nrun, ndim, if_burn=False)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), cfg.get('output_dir', 'results/'))
    os.makedirs(output_dir, exist_ok=True)

    samples_file = os.path.join(output_dir, f'samples_{trial_name}.npy')
    params_file = os.path.join(output_dir, f'params_list_{trial_name}.npy')
    config_copy = os.path.join(output_dir, f'config_{trial_name}.yaml')

    np.save(samples_file, samples)
    np.save(params_file, np.array(params_list, dtype=object), allow_pickle=True)

    # Also save a copy of the config for reproducibility
    import shutil
    shutil.copy2(args.config, config_copy)

    print(f"\n=== Results saved ===")
    print(f"  Samples:     {samples_file}")
    print(f"  Params list: {params_file}")
    print(f"  Config copy: {config_copy}")

    # Print summary
    p_mcmc = mcmc_results(samples)
    print(f"\nMCMC median results:")
    for i, p in enumerate(params_list):
        print(f"  {p[0]:30s}  = {p_mcmc[i]:.4f}")


if __name__ == '__main__':
    main()
