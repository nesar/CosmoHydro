#!/usr/bin/env python
"""
Train ONLY the missing GP (SEPIA) emulators for cosmology-target inference,
reusing everything already trained by codes/02_train_emulators_multiz.ipynb.

Already trained and REUSED (never retrained here):
  models/Pk_multivariate_model_z_index0.pkl
      P_hydro.full/P_go suppression ratio at z=0, exp_variance=0.95,
      trained on runs 000-099 (notebook 02, cells 24-25).
      Loaded via ``load_pk_model('ratio', '0.0')`` below.

Missing, trained here into models/Pk_cosmo/:
  logP_go_z{0.0,0.1,0.5,1.0,2.0} : log10 P_go(k) — gravity-only spectra,
      cosmology-only inputs (design columns [omega_m, sigma_8]; GO runs do
      not depend on subgrid physics). Needed for absolute-P(k) targets
      (KiDS-Legacy Pm, BOSS) and the A_mod template (P_L/P_GO).
  ratio_z{0.1,0.5,1.0,2.0}       : suppression ratio at the remaining
      snapshots (z=0 already exists), same recipe as the notebook z=0 model
      (exp_variance=0.95, 7 input params). Needed because
      P_hydro(k,z) = ratio(k,z) * P_go(k,z) at the KiDS/BOSS redshifts.

Conventions follow notebook 02 exactly: design rows 0-109 = run000-109,
train on runs 0-99, hold out runs 100-109; k cut = mass_conds('Pk');
SEPIA training = tune_step_sizes(50, 20) + 1000 MCMC steps (project default,
via cosmo_hydro_emu.gp.do_gp_train — existing code, not modified).

Usage:
  python train_pk_emulators.py                       # train all missing
  python train_pk_emulators.py --quantity logP_go    # subset
  python train_pk_emulators.py --ztags 0.1 0.5
  python train_pk_emulators.py --validate-only       # incl. existing z=0 model
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'codes'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cosmo_hydro_emu.load_hacc import sepia_data_format
from cosmo_hydro_emu.pca import do_pca
from cosmo_hydro_emu.gp import do_gp_train, gp_load
from cosmo_hydro_emu.emu import emulate, load_model_autosync

from pk_data import (
    load_design, load_pk_suite, emulation_targets,
    PK_DIR_DEFAULT, PK_REDSHIFT_TAGS, COSMO_COLS,
    NUM_SIMS_DEFAULT, START_SIM_IDX_DEFAULT, TRAIN_INDICES, TEST_INDICES,
)

MODEL_DIR = os.path.join(_HERE, '..', 'models', 'Pk_cosmo')
DIAG_DIR = os.path.join(_HERE, 'diagnostics')

# Path of the pre-existing notebook-trained z=0 ratio model (reused, read-only)
EXISTING_RATIO_Z0 = os.path.join(_HERE, '..', 'models',
                                 'Pk_multivariate_model_z_index0')

QUANTITIES = {
    # name: (exp_variance, param_cols or None for all 7, ztags to train)
    'logP_go': (0.999, COSMO_COLS, ['0.0', '0.1', '0.5', '1.0', '2.0']),
    'ratio':   (0.95,  None,       ['0.1', '0.5', '1.0', '2.0']),
}


def model_paths(quantity, ztag):
    base = os.path.join(MODEL_DIR, f'{quantity}_z{ztag}')
    return base, base + '.pkl', base + '_meta.json'


def build_training_data(quantity, ztag, params_all):
    suite = load_pk_suite(PK_DIR_DEFAULT, ztag=ztag, pk_type='hydro.full',
                          num_sims=NUM_SIMS_DEFAULT,
                          start_sim_idx=START_SIM_IDX_DEFAULT)
    k_cut, y_all = emulation_targets(suite, quantity)
    param_cols = QUANTITIES[quantity][1] if quantity in QUANTITIES else None
    p_all = params_all if param_cols is None else params_all[:, param_cols]
    return k_cut, y_all, p_all


def train_one(quantity, ztag, params_all, retrain=False):
    exp_variance = QUANTITIES[quantity][0]
    base, pkl_path, meta_path = model_paths(quantity, ztag)

    if os.path.exists(pkl_path) and os.path.exists(meta_path) and not retrain:
        print(f'  [skip] {quantity} z={ztag} already trained')
        return

    k_cut, y_all, p_all = build_training_data(quantity, ztag, params_all)
    train_idx = np.array(TRAIN_INDICES)

    sepia_data = sepia_data_format(p_all[train_idx], y_all[train_idx], k_cut)
    with contextlib.redirect_stdout(io.StringIO()):
        sepia_model = do_pca(sepia_data, exp_variance=exp_variance)
    n_pc = sepia_data.sim_data.K.shape[0]
    print(f'  [train] {quantity} z={ztag}: {train_idx.size} sims, '
          f'{k_cut.size} k bins, {n_pc} PCs, {p_all.shape[1]} params ... ',
          end='', flush=True)

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        sepia_model = do_gp_train(sepia_model, base)
    print(f'done in {(time.time() - t0) / 60:.1f} min')

    meta = {
        'quantity': quantity,
        'ztag': ztag,
        'exp_variance': exp_variance,
        'n_pc': int(n_pc),
        'param_cols': QUANTITIES[quantity][1],
        'num_sims': NUM_SIMS_DEFAULT,
        'start_sim_idx': START_SIM_IDX_DEFAULT,
        'train_indices': train_idx.tolist(),
        'test_indices': TEST_INDICES,
        'k': k_cut.tolist(),
        'sepia_mcmc_steps': 1000,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f)


def load_pk_model(quantity, ztag, params_all=None):
    """Load a trained model for (quantity, ztag), reusing the notebook-trained
    z=0 ratio model where it exists.

    Returns (sepia_model, info) where info holds k grid, param_cols, and the
    full-suite (y_all, p_all) arrays for validation use.
    """
    if params_all is None:
        params_all = load_design()

    if quantity == 'ratio' and ztag == '0.0':
        # Pre-existing notebook model: rebuild its exact training SepiaData
        # (runs 0-99, all 7 params, ratio on the k cut) and restore.
        k_cut, y_all, p_all = build_training_data('ratio', '0.0', params_all)
        train_idx = np.array(TRAIN_INDICES)
        sepia_data = sepia_data_format(p_all[train_idx], y_all[train_idx], k_cut)
        with contextlib.redirect_stdout(io.StringIO()):
            model = load_model_autosync(EXISTING_RATIO_Z0, sepia_data,
                                        exp_variance=0.95)
        info = {'k': k_cut, 'param_cols': None, 'y_all': y_all, 'p_all': p_all,
                'source': os.path.basename(EXISTING_RATIO_Z0)}
        return model, info

    base, pkl_path, meta_path = model_paths(quantity, ztag)
    with open(meta_path) as f:
        meta = json.load(f)
    k_cut, y_all, p_all = build_training_data(quantity, ztag, params_all)
    if not np.allclose(k_cut, np.array(meta['k'])):
        raise RuntimeError(f'k grid changed since training of {base}')
    train_idx = np.array(meta['train_indices'])
    sepia_data = sepia_data_format(p_all[train_idx], y_all[train_idx], k_cut)
    with contextlib.redirect_stdout(io.StringIO()):
        model = do_pca(sepia_data, exp_variance=meta['n_pc'])
        model = gp_load(model, base)
    info = {'k': k_cut, 'param_cols': meta['param_cols'],
            'y_all': y_all, 'p_all': p_all, 'source': os.path.basename(base)}
    return model, info


def validate_one(quantity, ztag, params_all, results):
    model, info = load_pk_model(quantity, ztag, params_all)
    test_idx = np.array(TEST_INDICES)
    pred_mean, pred_std = emulate(model, info['p_all'][test_idx])
    pred_mean, pred_std = pred_mean.T, pred_std.T

    truth = info['y_all'][test_idx]
    if quantity.startswith('logP'):
        frac = np.abs(10 ** pred_mean / 10 ** truth - 1.0)
        unit = 'frac dP/P'
    else:
        frac = np.abs(pred_mean - truth)
        unit = 'abs dS'

    metrics = {
        'quantity': quantity, 'ztag': ztag, 'source': info['source'],
        'median_abs_err': float(np.median(frac)),
        'p95_abs_err': float(np.percentile(frac, 95)),
        'max_abs_err': float(frac.max()),
    }
    results.append(metrics)
    print(f'  [valid] {quantity} z={ztag} ({info["source"]}): {unit} median '
          f'{metrics["median_abs_err"]:.4f}, 95% {metrics["p95_abs_err"]:.4f}, '
          f'max {metrics["max_abs_err"]:.4f}')
    return info['k'], truth, pred_mean, pred_std


def validation_figure(quantity, per_z):
    ztags = sorted(per_z.keys(), key=float)
    fig, axes = plt.subplots(2, len(ztags), figsize=(4.2 * len(ztags), 7),
                             sharex=True, squeeze=False)
    for j, zt in enumerate(ztags):
        k, truth, pred, std = per_z[zt]
        for i in range(truth.shape[0]):
            if quantity.startswith('logP'):
                axes[0, j].loglog(k, 10 ** truth[i], 'k-', lw=0.8, alpha=0.6)
                axes[0, j].loglog(k, 10 ** pred[i], 'r--', lw=0.8, alpha=0.8)
                res = 10 ** pred[i] / 10 ** truth[i] - 1
            else:
                axes[0, j].semilogx(k, truth[i], 'k-', lw=0.8, alpha=0.6)
                axes[0, j].semilogx(k, pred[i], 'r--', lw=0.8, alpha=0.8)
                res = pred[i] - truth[i]
            axes[1, j].semilogx(k, res, lw=0.8)
        axes[0, j].set_title(f'z = {zt}')
        axes[1, j].axhline(0, color='k', ls=':', lw=0.8)
        axes[1, j].set_xlabel(r'$k$ [h/Mpc]')
        axes[1, j].set_ylim(-0.05, 0.05)
    axes[0, 0].set_ylabel(f'{quantity} (black truth, red emu)')
    axes[1, 0].set_ylabel('residual')
    fig.suptitle(f'Held-out validation (runs 100-109): {quantity}')
    fig.tight_layout()
    out = os.path.join(DIAG_DIR, f'emu_validation_{quantity}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quantity', choices=list(QUANTITIES) + ['all'],
                    default='all')
    ap.add_argument('--ztags', nargs='+', default=None,
                    help='subset of redshift tags (default: all per quantity)')
    ap.add_argument('--retrain', action='store_true',
                    help='retrain models in models/Pk_cosmo/ (never touches '
                         'the pre-existing notebook-trained models)')
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DIAG_DIR, exist_ok=True)

    quantities = list(QUANTITIES) if args.quantity == 'all' else [args.quantity]
    params_all = load_design()

    if not args.validate_only:
        for q in quantities:
            ztags = args.ztags or QUANTITIES[q][2]
            for zt in ztags:
                train_one(q, zt, params_all, retrain=args.retrain)

    results = []
    for q in quantities:
        # validate every z that has a model, including the reused z=0 ratio
        all_z = PK_REDSHIFT_TAGS if q == 'ratio' else QUANTITIES[q][2]
        ztags = args.ztags or all_z
        per_z = {}
        for zt in ztags:
            is_existing = (q == 'ratio' and zt == '0.0')
            if not is_existing and not os.path.exists(model_paths(q, zt)[1]):
                print(f'  [valid] {q} z={zt}: no model on disk, skipping')
                continue
            per_z[zt] = validate_one(q, zt, params_all, results)
        if per_z:
            validation_figure(q, per_z)

    out_json = os.path.join(DIAG_DIR, 'emu_validation_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'metrics -> {out_json}')


if __name__ == '__main__':
    main()
