#!/usr/bin/env python
"""
HMF analogs of the P(k) diagnostics:

  diagnostics/hmf_suite_overview.png   suite phi(M) at 5 snapshots + outlier
                                       scan (MAD flagging, like the Pk one)
  diagnostics/emu_validation_HMF.png   held-out (runs 100-109) validation of
                                       the PRE-EXISTING HMF_multiz emulators
                                       at z_index 9 (z=0.0998, the GAMA
                                       likelihood snapshot) and 10 (z=0)
  diagnostics/emu_validation_HMF.json  metrics

Models are the notebook-trained ones (models/HMF_multiz/) — nothing is
retrained here; SepiaData is rebuilt exactly as in notebook 02 (y = 10**phi,
mass cut mass_conds('HMF'), runs 0-99 train).
"""

import contextlib
import io
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'codes'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cosmo_hydro_emu.load_hacc import mass_conds, sepia_data_format
from cosmo_hydro_emu.emu import load_model_autosync, emulate
from cosmo_hydro_emu.snapshot_utils import SNAPSHOT_IDS, get_snapshot_redshifts

from pk_data import (load_design, load_hmf_snapshot,
                     TRAIN_INDICES, TEST_INDICES)

DIAG = os.path.join(_HERE, 'diagnostics')
MODEL_DIR = os.path.join(_HERE, '..', 'models', 'HMF_multiz')

Z_ALL, _ = get_snapshot_redshifts(SNAPSHOT_IDS)

# snapshots mirroring the Pk overview redshifts (z = 2, 1, 0.5, 0.1, 0)
OVERVIEW_IDX = [0, 4, 6, 9, 10]
VALIDATE_IDX = [9, 10]          # z=0.0998 (GAMA snapshot) and z=0


def suite_overview():
    m1, m2 = mass_conds('HMF')
    fig, axes = plt.subplots(2, len(OVERVIEW_IDX), figsize=(22, 8), sharex=True)
    flagged_all = {}
    for j, zi in enumerate(OVERVIEW_IDX):
        hmf = load_hmf_snapshot(snap_id=SNAPSHOT_IDS[zi])
        cond = (hmf['M'] > m1) & (hmf['M'] < m2)
        logM = np.log10(hmf['M'][cond])
        phi = hmf['phi'][:, cond]
        logphi = np.log10(np.where(phi > 0, phi, np.nan))

        med = np.nanmedian(logphi, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(logphi - med), axis=0)
        mad = np.maximum(mad, 5e-3)
        dev = np.abs(logphi - med) / mad
        # sustained outliers: >30% of populated bins beyond 5 MAD
        frac = np.nanmean(np.where(np.isfinite(logphi), dev > 5.0, np.nan), axis=1)
        flagged = np.where(frac > 0.3)[0]
        flagged_all[zi] = flagged

        for i in range(phi.shape[0]):
            c = 'crimson' if i in flagged else 'gray'
            zo = 3 if i in flagged else 1
            axes[0, j].plot(logM, logphi[i], color=c, alpha=0.35, lw=0.6, zorder=zo)
            axes[1, j].plot(logM, logphi[i] - med, color=c, alpha=0.35, lw=0.6,
                            zorder=zo)
        axes[0, j].set_title(f'z = {Z_ALL[zi]:.2f}  (snap {SNAPSHOT_IDS[zi]})')
        axes[1, j].axhline(0, ls=':', color='k', lw=0.8)
        axes[1, j].set_ylim(-0.6, 0.6)
        axes[1, j].set_xlabel(r'$\log_{10} M_{\rm 200c}$ [$M_\odot/h$]')
        print(f'  z={Z_ALL[zi]:.2f}: sustained >5 MAD outlier runs: {list(flagged)}')
    axes[0, 0].set_ylabel(r'$\log_{10}\phi$ [(Mpc/$h$)$^{-3}$ dex$^{-1}$]')
    axes[1, 0].set_ylabel(r'$\Delta\log_{10}\phi$ vs suite median')
    fig.suptitle('SOD halo mass function suite overview — 110 runs '
                 '(red = flagged sustained outliers)')
    fig.tight_layout()
    out = os.path.join(DIAG, 'hmf_suite_overview.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


def load_hmf_model(z_index, params_all):
    hmf = load_hmf_snapshot(snap_id=SNAPSHOT_IDS[z_index])
    m1, m2 = mass_conds('HMF')
    cond = (hmf['M'] > m1) & (hmf['M'] < m2)
    y_all = 10 ** hmf['phi'][:, cond]         # notebook 02 convention
    logM = np.log10(hmf['M'][cond])
    train = np.array(TRAIN_INDICES)
    sd = sepia_data_format(params_all[train], y_all[train], logM)
    base = os.path.join(MODEL_DIR, f'multivariate_model_z_index{z_index}')
    with contextlib.redirect_stdout(io.StringIO()):
        model = load_model_autosync(base, sd, exp_variance=0.999)
    counts_cut = hmf['counts'][:, cond]
    return model, logM, y_all, counts_cut


def emu_validation():
    params_all = load_design()
    metrics = []
    fig, axes = plt.subplots(2, len(VALIDATE_IDX), figsize=(6.5 * len(VALIDATE_IDX), 8),
                             sharex=True)
    for j, zi in enumerate(VALIDATE_IDX):
        model, logM, y_all, counts_cut = load_hmf_model(zi, params_all)
        test = np.array(TEST_INDICES)
        pred, std = emulate(model, params_all[test])
        phi_pred = np.log10(np.maximum(pred.T, 1e-12))     # linear phi
        phi_true = np.log10(y_all[test])                   # linear phi
        with np.errstate(divide='ignore', invalid='ignore'):
            l10_pred = np.log10(phi_pred)
            l10_true = np.log10(np.where(phi_true > 0, phi_true, np.nan))
        for i in range(test.size):
            axes[0, j].plot(logM, l10_true[i], 'k-', lw=0.8, alpha=0.6)
            axes[0, j].plot(logM, l10_pred[i], 'r--', lw=0.8, alpha=0.8)
            axes[1, j].plot(logM, phi_pred[i] / phi_true[i] - 1, lw=0.8)
        axes[0, j].set_title(f'z = {Z_ALL[zi]:.2f}')
        axes[1, j].axhline(0, color='k', ls=':', lw=0.8)
        axes[1, j].set_ylim(-0.6, 0.6)
        axes[1, j].set_xlabel(r'$\log_{10} M_{\rm 200c}$ [$M_\odot/h$]')

        # metrics in the GAMA-relevant range, vs the Poisson expectation
        hi = (logM > 12.6)
        ok = phi_true[:, hi] > 1e-9
        rel = np.abs(phi_pred[:, hi][ok] / phi_true[:, hi][ok] - 1)
        Nbar = counts_cut[test][:, hi][ok]
        poiss = 1 / np.sqrt(np.maximum(Nbar, 1))
        metrics.append({
            'z_index': zi, 'z': round(float(Z_ALL[zi]), 4),
            'median_frac_err_logM>12.6': float(np.median(rel)),
            'p95_frac_err': float(np.percentile(rel, 95)),
            'median_resid_over_poisson': float(np.median(rel / poiss)),
        })
        print(f'  z={Z_ALL[zi]:.2f}: phi frac err median {np.median(rel):.3f}, '
              f'95% {np.percentile(rel, 95):.3f}; '
              f'median resid/Poisson = {np.median(rel/poiss):.2f}')
    axes[0, 0].set_ylabel(r'$\log_{10}\phi$ (black truth, red emu)')
    axes[1, 0].set_ylabel(r'$\phi_{\rm emu}/\phi_{\rm true} - 1$')
    fig.suptitle(f'HMF emulator held-out validation (runs 100-109) — '
                 'pre-existing models/HMF_multiz')
    fig.tight_layout()
    out = os.path.join(DIAG, 'emu_validation_HMF.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')

    with open(os.path.join(DIAG, 'emu_validation_HMF.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    os.makedirs(DIAG, exist_ok=True)
    print('== HMF suite overview ==')
    suite_overview()
    print('== HMF emulator validation ==')
    emu_validation()
