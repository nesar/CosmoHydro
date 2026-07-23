#!/usr/bin/env python
"""EXPLORATORY: summary figures for the archived A_mod-based MCMC runs.

Reuses the standard plot_summary_cosmo machinery; registers the suppression
panel (with the A_mod reference band) and points the results lookup at
amod_exploratory/results/. See README.md for why these are quarantined.

    python plot_summary_amod.py
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _BASE)
sys.path.insert(0, _HERE)

import plot_summary_cosmo as PS                     # noqa: E402
from pk_likelihood import IDX_OMEGA_M, IDX_SIGMA_8  # noqa: E402
from linear_theory import LinearPk                  # noqa: E402
from amod_likelihood import load_amod               # noqa: E402

AMOD_RESULTS = os.path.join(_HERE, 'results')


def panel_suppression(ax, chains, ctx):
    """Emulated suppression at each chain's best fit + the A_mod reference
    band (EXPLORATORY: Planck-conditioned nonlinear-boost modulation, not a
    direct measurement of P_hydro/P_GO)."""
    emu, lin = ctx['emu'], ctx['lin']
    amod = load_amod('DES_Y3_Planck')
    k = emu.k_grid
    for c in chains:
        Sk, _ = emu.ratio(0.0, c['full_theta'])
        ls = '-' if c is chains[0] else '--'
        ax.semilogx(k, Sk - 1, ls, color=c['color'], lw=2,
                    label=f'MCMC best fit: {c["label"]}')
    P_go, _ = emu.P_go(0.0, PS.FIDUCIAL7)
    P_L = lin(k, 0.0, PS.FIDUCIAL7[IDX_OMEGA_M], PS.FIDUCIAL7[IDX_SIGMA_8])
    t = 1.0 - P_L / P_go
    A, sA = amod['Amod'], amod['sigma']
    ax.fill_between(k, (A - sA - 1) * t, (A + sA - 1) * t, color='gray',
                    alpha=0.35,
                    label=f'$A_{{\\rm mod}}={A}\\pm{sA}$ (DES Y3; exploratory)')
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.set_xlim(0.02, 9)
    ax.set_xlabel(r'$k$ [$h\,{\rm Mpc}^{-1}$]', fontsize=13)
    ax.set_ylabel(r'$P_{\rm hydro}/P_{\rm GO} - 1$', fontsize=13)
    ax.legend(fontsize=8, loc='lower left')


def main():
    # register suppression panel + route results lookups to this folder for
    # the amod trials (standard trials stay in ../results via fallback)
    PS.PANEL_FUNCS['suppression'] = panel_suppression

    amod_trials = {'Pk_amod_5subgrid_fidcosmo', 'Pk_kids_amod_7p',
                   'Pk_kids_amod_hmf_7p'}
    std_results = PS.RESULTS
    orig_chain = PS._chain

    def chain_router(trial, label, color, shared_labels):
        saved = PS.RESULTS
        PS.RESULTS = AMOD_RESULTS if trial in amod_trials else std_results
        try:
            return orig_chain(trial, label, color, shared_labels)
        finally:
            PS.RESULTS = saved

    PS._chain = chain_router
    saved_results = std_results

    ctx = PS.build_ctx(need_hmf=False)
    ctx['lin'] = LinearPk()

    PS.RESULTS = AMOD_RESULTS       # figures land here
    try:
        PS.make_figure(
            'Pk_kids_amod_7p', 'Pk_kids_2cosmo',
            'subgrid marginalized (7p)', 'subgrid fixed (2p)',
            PS.COSMO_SHARED, PS.COSMO_LIMS,
            r'EXPLORATORY — KiDS $P_m$+$A_{\rm mod}$ cosmology: subgrid '
            'marginalized (red) vs fixed (blue)',
            'plot_summary_cosmo_marg_vs_fixed_AMOD.png',
            ['kids', 'suppression'], ctx)

        PS.make_figure(
            'Pk_kids_amod_7p', 'Pk_amod_5subgrid_fidcosmo',
            'cosmo marginalized (7p)', 'cosmo fixed (5p)',
            PS.SUBGRID_SHARED, PS.SUBGRID_LIMS,
            r'EXPLORATORY — $A_{\rm mod}$(+KiDS) subgrid: cosmology '
            'marginalized (red) vs fixed (blue)',
            'plot_summary_subgrid_marg_vs_fixed_AMOD.png',
            ['suppression', 'kids'], ctx)
    finally:
        PS.RESULTS = saved_results


if __name__ == '__main__':
    main()
