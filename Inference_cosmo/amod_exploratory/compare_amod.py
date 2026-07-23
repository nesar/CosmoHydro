#!/usr/bin/env python
"""EXPLORATORY: emulated baryonic suppression vs A_mod template bands.

Moved out of compare_pk_targets.py (2026-07-23). The template bands are a
Planck-conditioned reinterpretation of the published nonlinear-boost
modulation — a visual reference, NOT a fitted dataset. See README.md.

Output: amod_exploratory/plots/compare_amod.png
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _BASE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_BASE, '..', 'codes'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pk_likelihood import PkEmulator, IDX_OMEGA_M, IDX_SIGMA_8
from linear_theory import LinearPk
from amod_likelihood import load_amod, AmodLikelihood, AMOD_CONSTRAINTS

FIDUCIAL7 = np.array([3.0, 0.5, 0.8, 0.51, 0.13, 0.14176, 0.8102])
OUT = os.path.join(_HERE, 'plots', 'compare_amod.png')


def main():
    emu = PkEmulator()
    lin = LinearPk()
    target = load_amod('DES_Y3_Planck')
    like = AmodLikelihood(target, emu, linear_pk=lin)
    A_fid = like.model_amod(FIDUCIAL7)
    print(f'model-implied A_mod at fiducial = {A_fid:.3f} '
          f'(DES Y3+Planck: {target["Amod"]} +/- {target["sigma"]})')

    k = emu.k_grid
    S_fid, S_err = emu.ratio(0.0, FIDUCIAL7)
    P_go, _ = emu.P_go(0.0, FIDUCIAL7)
    P_L = lin(k, 0.0, FIDUCIAL7[IDX_OMEGA_M], FIDUCIAL7[IDX_SIGMA_8])
    t = 1.0 - P_L / P_go

    from pk_data import load_pk_suite, k_trust_mask, PK_DIR_DEFAULT
    suite = load_pk_suite(PK_DIR_DEFAULT, ztag='0.0')
    mm = k_trust_mask(suite['k'])
    S_all = suite['ratio'][:, mm]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.fill_between(k, S_all.min(axis=0) - 1, S_all.max(axis=0) - 1,
                    color='gray', alpha=0.25, label='design envelope (110 sims)')
    ax.plot(k, S_fid - 1, 'r-', lw=2, label='emulated fiducial suppression')
    ax.fill_between(k, S_fid - 1 - S_err, S_fid - 1 + S_err, color='r', alpha=0.2)
    A, sA = target['Amod'], target['sigma']
    ax.plot(k, (A - 1) * t, 'b-', lw=1.5,
            label=f'A_mod = {A} (DES Y3+Planck)')
    ax.fill_between(k, (A - sA - 1) * t, (A + sA - 1) * t, color='b', alpha=0.15)
    A_k = AMOD_CONSTRAINTS['KiDS1000']['Amod']
    ax.plot(k, (A_k - 1) * t, 'g--', lw=1.2,
            label=f'A_mod = {A_k} (KiDS-1000, central only)')
    ax.set_xscale('log')
    ax.set_xlim(0.02, 9)
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.set_xlabel(r'$k$ [$h$/Mpc]')
    ax.set_ylabel(r'$P_{\rm hydro}/P_{\rm GO} - 1$')
    ax.set_title(r'EXPLORATORY: suppression vs $A_{\rm mod}$ templates '
                 r'(Planck-conditioned reinterpretation; $z=0$)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
