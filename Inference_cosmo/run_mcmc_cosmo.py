#!/usr/bin/env python
"""
MCMC driver for cosmology-target (power spectrum) inference.

Mirrors Inference/run_mcmc.py: YAML trial configs (deep-merged over
configs/_defaults.yaml), emcee ensemble sampling, results saved as
    results/samples_{trial}.npy
    results/params_list_{trial}.npy
    results/config_{trial}.yaml
(same format as Inference/, so the existing plot tooling applies).

Usage:
    python run_mcmc_cosmo.py configs/Pk_amod_5subgrid_fidcosmo.yaml
    python run_mcmc_cosmo.py configs/Pk_kids_amod_7p.yaml --dry-run

Config schema (see configs/*.yaml for working examples):

    trial_name: Pk_kids_amod_7p
    targets:
      - kind: kids            # direct Pm(k, z_fid) likelihood
        nz: nz3               # 'nz1' or 'nz3'
        k_min: 0.03           # h/Mpc; must stay within emulated [0.0157, 8.04]
        k_max: 7.0
        z_bins: [0.15, 0.45]  # optional subset of fiducial redshifts
      - kind: amod            # scalar A_mod projection likelihood
        constraint: DES_Y3_Planck
        k_fit_min: 0.1
        k_fit_max: 8.0
      - kind: boss            # Kaiser+AP multipoles (methods-level: no window)
        patch: NGC
        zbin: z1
        k_min: 0.03
        k_max: 0.15
        use_quad: true
        bias:       {initial: 2.0, min: 0.5, max: 4.0}
        shot_noise: {initial: 0.0, min: -5000.0, max: 5000.0}   # optional

    param_mode: cosmo | subgrid | subgrid+cosmo | custom
    free_params: [omega_m, sigma_8]      # for param_mode: custom
    fixed_params: {kappa_w: 3.0, ...}    # short keys; overrides defaults
    param_ranges: {...}                  # optional hard truncation
    gaussian_priors: {omega_m: {mu: 0.14176, sigma: 0.001}}
    flat_prior_indices: []
    mcmc: {nwalkers: 32, nburn: 500, nrun: 2000, parallel: true}
"""

import argparse
import os
import shutil
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'codes'))
sys.path.insert(0, os.path.join(_HERE, '..', 'Inference'))

from cosmo_hydro_emu.load_hacc import PARAM_NAME               # noqa: E402
from cosmo_hydro_emu.mcmc import (                             # noqa: E402
    ln_prior, chain_init, do_mcmc, mcmc_results,
)
from run_mcmc import SHORT_KEY_TO_LABEL, load_config           # noqa: E402

from pk_data import load_design                                # noqa: E402
from targets import load_kids, load_amod, load_boss            # noqa: E402
from pk_likelihood import (                                    # noqa: E402
    PkEmulator, PmLikelihood, AmodLikelihood, BossLikelihood,
    setup_global_posterior, ln_prob_global,
)
from linear_theory import LinearPk                             # noqa: E402

# Fiducial subgrid parameters in scaled design units
# (kappa_w, e_w, M_seed/1e6, v_kin/1e4, eps_kin/1e1) — the project fiducial
# hydro model, NOT design midpoints.
FIDUCIAL_SUBGRID = {
    'kappa_w': 3.0, 'e_w': 0.5, 'M_seed': 0.8, 'v_kin': 0.51, 'eps_kin': 0.13,
}
FIDUCIAL_COSMO = {'omega_m': 0.14176, 'sigma_8': 0.8102}

SHORT_KEYS_ORDERED = ['kappa_w', 'e_w', 'M_seed', 'v_kin', 'eps_kin',
                      'omega_m', 'sigma_8']


def build_param_space(cfg, design_params):
    """params_list for free params + fixed_params dict (PARAM_NAME labels).

    Unlike Inference/run_mcmc.py, parameters fixed here default to the
    PROJECT FIDUCIAL values (subgrid and cosmology), not design midpoints.
    """
    param_mode = cfg.get('param_mode', 'subgrid+cosmo')
    allMin = np.min(design_params, axis=0)
    allMax = np.max(design_params, axis=0)

    if param_mode == 'subgrid+cosmo' or param_mode == 'all':
        free_keys = SHORT_KEYS_ORDERED
    elif param_mode == 'cosmo':
        free_keys = ['omega_m', 'sigma_8']
    elif param_mode == 'subgrid':
        free_keys = SHORT_KEYS_ORDERED[:5]
    elif param_mode == 'custom':
        free_keys = list(cfg.get('free_params', []))
    else:
        raise ValueError(f'unknown param_mode: {param_mode}')

    fixed_params = {}
    cfg_fixed = cfg.get('fixed_params', {}) or {}
    fiducial = {**FIDUCIAL_SUBGRID, **FIDUCIAL_COSMO}
    for i, key in enumerate(SHORT_KEYS_ORDERED):
        label = SHORT_KEY_TO_LABEL[key]
        if key in free_keys:
            continue
        val = cfg_fixed.get(key, fiducial[key])
        fixed_params[label] = float(val)

    params_list = []
    for i, key in enumerate(SHORT_KEYS_ORDERED):
        if key not in free_keys:
            continue
        label = SHORT_KEY_TO_LABEL[key]
        init = 0.5 * (allMin[i] + allMax[i])
        params_list.append([label, init, allMin[i], allMax[i]])

    # optional hard truncation of ranges (same schema as Inference)
    overrides = cfg.get('param_ranges', {}) or {}
    by_label = {SHORT_KEY_TO_LABEL[k]: v for k, v in overrides.items()}
    for entry in params_list:
        if entry[0] in by_label:
            lo, hi = by_label[entry[0]]
            entry[2], entry[3] = float(lo), float(hi)
            entry[1] = 0.5 * (entry[2] + entry[3])
            print(f'  [param_ranges] {entry[0]}: truncated to [{lo}, {hi}]')

    return params_list, fixed_params


def build_components(cfg, params_list):
    """Construct likelihood objects; appends nuisance params to params_list.

    Returns (components, nuisance_map).
    """
    # Which emulator redshifts are needed decides nothing here — all 5
    # snapshots of both quantities are loaded once and shared.
    emu = PkEmulator()
    lin = None
    components = []
    nuisance_map = {}

    for i, spec in enumerate(cfg['targets']):
        kind = spec['kind']
        if kind == 'kids':
            target = load_kids(nz=spec.get('nz', 'nz3'),
                               k_min=spec.get('k_min', 0.03),
                               k_max=spec.get('k_max', 7.0),
                               z_bins=spec.get('z_bins'))
            like = PmLikelihood(target, emu,
                                interp_sys_frac=spec.get('interp_sys_frac', 0.01))
            name = f'kids_{spec.get("nz", "nz3")}'
        elif kind == 'amod':
            if lin is None:
                lin = LinearPk()
            target = load_amod(spec.get('constraint', 'DES_Y3_Planck'))
            like = AmodLikelihood(target, emu, linear_pk=lin,
                                  k_fit_min=spec.get('k_fit_min', 0.1),
                                  k_fit_max=spec.get('k_fit_max', 8.0))
            name = f'amod_{spec.get("constraint", "DES_Y3_Planck")}'
        elif kind == 'boss':
            target = load_boss(patch=spec.get('patch', 'NGC'),
                               zbin=spec.get('zbin', 'z1'),
                               k_min=spec.get('k_min', 0.03),
                               k_max=spec.get('k_max', 0.15),
                               use_quad=spec.get('use_quad', True))
            like = BossLikelihood(target, emu)
            name = f'boss_{spec.get("patch", "NGC")}_{spec.get("zbin", "z1")}'
            # nuisance: linear bias (required), shot noise (optional)
            bias = spec.get('bias', {'initial': 2.0, 'min': 0.5, 'max': 4.0})
            params_list.append([f'$b_1$ {name}', float(bias['initial']),
                                float(bias['min']), float(bias['max'])])
            nuisance_map[name] = {'b1': len(params_list) - 1}
            if 'shot_noise' in spec:
                sn = spec['shot_noise']
                params_list.append([f'$P_{{sn}}$ {name}', float(sn['initial']),
                                    float(sn['min']), float(sn['max'])])
                nuisance_map[name]['P_sn'] = len(params_list) - 1
        else:
            raise ValueError(f'unknown target kind: {kind}')
        components.append((name, like))
        print(f'  target[{i}]: {target["name"] if "name" in target else name} '
              f'-> {like.__class__.__name__}')

    return components, nuisance_map


def main():
    ap = argparse.ArgumentParser(description='Cosmology-target MCMC trial')
    ap.add_argument('config')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    cfg = load_config(args.config)
    trial_name = cfg['trial_name']
    print(f'=== Trial: {trial_name} ===')

    design_params = load_design()
    params_list, fixed_params = build_param_space(cfg, design_params)

    print('Building targets and likelihoods...')
    components, nuisance_map = build_components(cfg, params_list)

    # Gaussian priors on free params (short keys, same schema as Inference)
    gaussian_priors = {}
    free_labels = [p[0] for p in params_list]
    for short_key, spec in (cfg.get('gaussian_priors') or {}).items():
        label = SHORT_KEY_TO_LABEL.get(short_key, short_key)
        if label not in free_labels:
            print(f'  [gaussian_priors] {short_key} fixed/absent — skipping')
            continue
        idx = free_labels.index(label)
        gaussian_priors[idx] = (float(spec['mu']), float(spec['sigma']))
        print(f'  [gaussian_priors] {short_key}: N({spec["mu"]}, {spec["sigma"]})')

    flat_indices = cfg.get('flat_prior_indices', [])

    print(f'Free parameters ({len(params_list)}):')
    for p in params_list:
        print(f'  {p[0]:30s} init={p[1]:.4f} range=[{p[2]:.4f}, {p[3]:.4f}]')
    print(f'Fixed parameters: { {k: round(v, 5) for k, v in fixed_params.items()} }')

    setup_global_posterior(components, params_list, fixed_params,
                           param_names=list(PARAM_NAME),
                           flat_indices=flat_indices,
                           gaussian_priors=gaussian_priors,
                           nuisance_map=nuisance_map)

    theta0 = np.array([p[1] for p in params_list])
    lp0 = ln_prob_global(theta0)
    print(f'\nln_prob at init point: {lp0:.4f}')
    if not np.isfinite(lp0):
        raise RuntimeError('non-finite posterior at the initial point')

    if args.dry_run:
        print('--- DRY RUN: skipping MCMC ---')
        return

    import emcee
    mcmc_cfg = cfg['mcmc']
    ndim = len(params_list)
    nwalkers = mcmc_cfg['nwalkers']
    nburn, nrun = mcmc_cfg['nburn'], mcmc_cfg['nrun']
    print(f'\nMCMC: ndim={ndim}, nwalkers={nwalkers}, nburn={nburn}, nrun={nrun}')

    pos0 = chain_init(params_list, ndim, nwalkers)

    if mcmc_cfg.get('parallel', True):
        # fork-based Pool inherits the module-level _GLOBAL state; only theta
        # is pickled per call.
        from multiprocessing import Pool
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob_global,
                                            pool=pool)
            pos, _, _, _, sampler = do_mcmc(sampler, pos0, nburn, ndim, if_burn=True)
            pos, _, _, samples, sampler = do_mcmc(sampler, pos, nrun, ndim, if_burn=False)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob_global)
        pos, _, _, _, sampler = do_mcmc(sampler, pos0, nburn, ndim, if_burn=True)
        pos, _, _, samples, sampler = do_mcmc(sampler, pos, nrun, ndim, if_burn=False)

    output_dir = os.path.join(_HERE, cfg.get('output_dir', 'results/'))
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'samples_{trial_name}.npy'), samples)
    np.save(os.path.join(output_dir, f'params_list_{trial_name}.npy'),
            np.array(params_list, dtype=object), allow_pickle=True)
    shutil.copy2(args.config, os.path.join(output_dir, f'config_{trial_name}.yaml'))

    print('\n=== Results saved ===')
    print(f'  {output_dir}samples_{trial_name}.npy')
    try:
        acc = float(np.mean(sampler.acceptance_fraction))
        print(f'  mean acceptance fraction: {acc:.3f}')
    except Exception:
        pass

    p_mcmc = mcmc_results(samples)
    print('\nMCMC medians:')
    for i, p in enumerate(params_list):
        print(f'  {p[0]:30s} = {p_mcmc[i]:.4f}')


if __name__ == '__main__':
    main()
