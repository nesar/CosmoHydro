#!/usr/bin/env python
"""
Compare emulated power spectra against every observational target, at the
project fiducial parameters. Produces the figures and a chi^2 summary that
should be inspected BEFORE trusting any MCMC run.

Figures (written to diagnostics/):
  compare_kids.png       emulated P_hydro(k, z_fid) vs KiDS-Legacy Pm bands
  compare_boss.png       Kaiser-model multipoles (best-fit b1) vs BOSS data
  compare_go_halofit.png emulated GO P(k) vs CAMB halofit (absolute check)

Text summary: diagnostics/compare_targets_summary.txt

Uses the exact same likelihood objects as run_mcmc_cosmo.py, so a chi^2
printed here is the chi^2 the MCMC sees.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'codes'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pk_data import load_design, TEST_INDICES
from targets import load_kids, load_boss
from pk_likelihood import (
    PkEmulator, PmLikelihood, BossLikelihood,
    IDX_OMEGA_M, IDX_SIGMA_8,
)
from linear_theory import LinearPk, FIXED_COSMO

DIAG = os.path.join(_HERE, 'diagnostics')

# project fiducial in scaled design units
FIDUCIAL7 = np.array([3.0, 0.5, 0.8, 0.51, 0.13, 0.14176, 0.8102])

lines = []


def log(msg=''):
    print(msg)
    lines.append(msg)


def fig_kids(emu):
    target = load_kids(nz='nz3', k_min=0.03, k_max=7.0)
    like = PmLikelihood(target, emu)
    y_mod, var_emu = like.model_vector(FIDUCIAL7)
    chi2 = -2.0 * like(FIDUCIAL7)
    nd = target['y'].size
    log(f'KiDS nz3 (k=0.03-7): chi2 = {chi2:.1f} for {nd} points '
        f'(chi2/dof = {chi2 / nd:.2f}) at fiducial')

    zs = np.unique(target['z'])
    fig, axes = plt.subplots(2, len(zs), figsize=(5.2 * len(zs), 7.5),
                             sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    for j, z in enumerate(zs):
        m = target['z'] == z
        k = target['k'][m]
        axes[0, j].errorbar(k, target['y'][m], yerr=target['sigma'][m],
                            fmt='o', ms=4, color='k', capsize=2,
                            label='KiDS-Legacy $P_m$ (68%)')
        P_hyd, _ = emu.P_hydro(z, FIDUCIAL7)
        P_go, _ = emu.P_go(z, FIDUCIAL7)
        axes[0, j].loglog(emu.k_grid, P_hyd, 'r-', lw=1.5,
                          label='emulated hydro (fiducial)')
        axes[0, j].loglog(emu.k_grid, P_go, 'b--', lw=1.0,
                          label='emulated gravity-only')
        axes[0, j].set_title(f'$z_{{\\rm fid}} = {z}$')
        axes[0, j].set_xlim(0.02, 9)
        r = target['y'][m] / y_mod[m]
        axes[1, j].errorbar(k, r, yerr=target['sigma'][m] / y_mod[m],
                            fmt='o', ms=4, color='k', capsize=2)
        axes[1, j].axhline(1, color='r', ls='-', lw=1)
        axes[1, j].set_xscale('log')
        axes[1, j].set_ylim(0, 2.2)
        axes[1, j].set_xlabel(r'$k$ [$h$/Mpc]')
    axes[0, 0].set_ylabel(r'$P_m(k)$ [(Mpc/$h$)$^3$]')
    axes[1, 0].set_ylabel('data / model')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('KiDS-Legacy deprojected $P_m(k,z)$ vs emulator at fiducial')
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG, 'compare_kids.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


# fig_amod: MOVED to amod_exploratory/compare_amod.py (2026-07-23).


def fig_boss(emu):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    col = 0
    for patch in ['NGC', 'SGC']:
        for zbin in ['z1', 'z3']:
            target = load_boss(patch=patch, zbin=zbin, k_min=0.03, k_max=0.15)
            like = BossLikelihood(target, emu)
            # best-fit (b1, P_sn) at fiducial cosmology by 2-D minimization
            from scipy.optimize import minimize
            nll = lambda x: -like(FIDUCIAL7, x[0], x[1])
            res = minimize(nll, x0=[2.0, 0.0], method='Nelder-Mead')
            b1, psn = res.x
            chi2 = 2.0 * res.fun
            nd = target['y'].size
            log(f'BOSS {patch} {zbin}: best-fit b1 = {b1:.3f}, '
                f'P_sn = {psn:+.0f}; chi2 = {chi2:.1f} for {nd} pts - 2 fit '
                f'params (chi2/dof = {chi2 / (nd - 2):.2f}) '
                f'[Kaiser+AP, no window]')
            y_mod = like.model_vector(FIDUCIAL7, b1, psn)
            k = target['k']
            nk = k.size
            sig = np.sqrt(np.diag(target['cov']))
            ax = axes[0, col]
            ax.errorbar(k, k * target['y'][:nk], yerr=k * sig[:nk], fmt='o',
                        ms=4, color='k', capsize=2, label=r'$P_0$ data')
            ax.plot(k, k * y_mod[:nk], 'r-', lw=1.5, label=r'$P_0$ model')
            ax.errorbar(k, k * target['y'][nk:], yerr=k * sig[nk:], fmt='s',
                        ms=4, color='gray', capsize=2, label=r'$P_2$ data')
            ax.plot(k, k * y_mod[nk:], 'b--', lw=1.5, label=r'$P_2$ model')
            ax.set_title(f'{patch} {zbin} (z_eff={target["z_eff"]}), '
                         f'$b_1$={b1:.2f}')
            axr = axes[1, col]
            axr.errorbar(k, target['y'][:nk] / y_mod[:nk],
                         yerr=sig[:nk] / y_mod[:nk], fmt='o', ms=4,
                         color='k', capsize=2)
            axr.axhline(1, color='r', lw=1)
            axr.set_ylim(0.85, 1.15)
            axr.set_xlabel(r'$k$ [$h$/Mpc]')
            if col == 0:
                ax.set_ylabel(r'$k\,P_\ell(k)$ [(Mpc/$h$)$^2$]')
                axr.set_ylabel(r'$P_0$ data/model')
                ax.legend(fontsize=8)
            col += 1
    fig.suptitle('BOSS DR12 multipoles vs Kaiser+AP model at fiducial '
                 '(best-fit bias/shot noise; NO window convolution — '
                 'methods-level comparison)')
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG, 'compare_boss.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


def fig_go_halofit(emu, lin):
    """Absolute cross-check of conventions: emulated GO P(k) vs CAMB
    halofit (mead2020) at the fiducial cosmology, plus the raw held-out sims.
    Agreement to ~5% over 0.03 < k < 5 validates units, h-convention, and
    the sigma_8 normalization end-to-end against an external code.
    """
    import camb
    h = FIXED_COSMO['h']
    ombh2 = FIXED_COSMO['omega_b_h2']
    om, s8 = FIDUCIAL7[IDX_OMEGA_M], FIDUCIAL7[IDX_SIGMA_8]
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=om - ombh2,
                       mnu=FIXED_COSMO['m_nu'], omk=0.0)
    pars.InitPower.set_params(As=2.1e-9, ns=FIXED_COSMO['n_s'])
    pars.set_matter_power(redshifts=[0.0], kmax=30.0, nonlinear=True)
    pars.NonLinearModel.set_params(halofit_version='mead2020')
    res = camb.get_results(pars)
    s8_camb = res.get_sigma8_0()
    kh, _, pk_nl = res.get_matter_power_spectrum(minkh=1e-3, maxkh=25.0,
                                                 npoints=400)
    # rescale amplitude to the target sigma_8 (approximate for the nonlinear
    # spectrum; exact only in linear theory — fine for a ~% level check)
    amp = (s8 / s8_camb) ** 2
    P_halofit = np.interp(emu.k_grid, kh, pk_nl[0]) * amp

    P_go, P_go_err = emu.P_go(0.0, FIDUCIAL7)
    ratio = P_go / P_halofit

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(emu.k_grid, ratio, 'r-', lw=1.5,
                label='emulated GO / CAMB halofit (mead2020)')
    ax.fill_between(emu.k_grid, ratio * (1 - P_go_err / P_go),
                    ratio * (1 + P_go_err / P_go), color='r', alpha=0.2)
    ax.axhline(1, color='k', ls=':', lw=0.8)
    ax.axhspan(0.95, 1.05, color='gray', alpha=0.15, label=r'$\pm$5%')
    ax.set_xlim(0.02, 9)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel(r'$k$ [$h$/Mpc]')
    ax.set_ylabel('ratio')
    ax.set_title('Absolute convention check at fiducial cosmology, z=0')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG, 'compare_go_halofit.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)

    m = (emu.k_grid > 0.03) & (emu.k_grid < 5.0)
    log(f'GO vs halofit(mead2020) at fiducial, k=0.03-5: '
        f'median ratio {np.median(ratio[m]):.3f}, '
        f'range [{ratio[m].min():.3f}, {ratio[m].max():.3f}] '
        f'(halofit itself is only ~5% accurate)')


def fig_hmf():
    """GAMA DR4 HMF vs the (pre-existing) HMF emulator at z=0.0998.

    Everything in simulation units: Msun/h, (Mpc/h)^-3 dex^-1 — the Driver
    tables are already in h-units per the paper (Sec. 2), identity conversion.
    """
    from targets import load_gama_hmf, load_mrp_fits, mrp_phi
    from pk_likelihood import HmfLikelihood
    from pk_data import load_hmf_snapshot

    target = load_gama_hmf(logM_max=14.9)
    like = HmfLikelihood(target)
    phi_fid, phi_std = like.model_phi(FIDUCIAL7)
    chi2 = -2.0 * like(FIDUCIAL7)
    log(f'GAMA HMF (logM=12.8-14.9): chi2 = {chi2:.1f} for {target["y"].size} '
        f'points at fiducial (data excess at logM~13.6-14.2 is the '
        f'intermediate-mass excess Driver et al. discuss)')

    hmf = load_hmf_snapshot()
    grid_logM = like.logM_grid
    yg, _ = __import__('cosmo_hydro_emu.emu', fromlist=['emulate']).emulate(
        like.model, FIDUCIAL7)
    # invert the notebook's 10**phi training transform: phi_lin is linear phi
    phi_lin = np.log10(np.maximum(yg[:, 0], 1e-12))

    mrp = load_mrp_fits()['GAMA5+SDSS5+REFLEXII']
    logM_mrp = np.linspace(12.7, 15.2, 100)     # Msun/h (paper units)
    phi_mrp = mrp_phi(logM_mrp, mrp)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [2, 1]})
    m = hmf['M'] > 10 ** 12.3
    for i in range(hmf['phi'].shape[0]):
        nz = hmf['phi'][i, m] > 0
        ax.plot(np.log10(hmf['M'][m][nz]), np.log10(hmf['phi'][i, m][nz]),
                '-', color='gray', alpha=0.15, lw=0.5)
    ax.plot(grid_logM, np.log10(np.maximum(phi_lin, 1e-12)), 'r-', lw=2,
            label='emulated fiducial (z=0.0998)')
    ax.plot(logM_mrp, np.log10(phi_mrp), 'g--', lw=1.5,
            label='MRP joint fit (GAMA+SDSS+REFLEX II)')
    ax.errorbar(target['logM'], np.log10(target['y']),
                yerr=[np.log10(target['y']) - np.log10(np.maximum(target['y'] - target['sigma'], 1e-12)),
                      np.log10(target['y'] + target['sigma']) - np.log10(target['y'])],
                fmt='o', ms=5, color='k', capsize=2,
                label='GAMA DR4 (Driver+22, Eddington-corr.)')
    ax.set_ylabel(r'$\log_{10}\,\phi$ [(Mpc/$h$)$^{-3}$ dex$^{-1}$]')
    ax.set_ylim(-7, -1.5)
    ax.legend(fontsize=9)
    ax.set_title('Halo mass function: sims (gray: 110 runs) vs GAMA DR4 at '
                 r'$z_{\rm eff}\simeq0.1$')
    phi_at_data = np.interp(target['logM'], grid_logM, phi_lin)
    ratio = target['y'] / phi_at_data
    axr.errorbar(target['logM'], ratio, yerr=target['sigma'] / phi_at_data,
                 fmt='o', ms=5, color='k', capsize=2)
    axr.axhline(1, color='r', lw=1)
    axr.set_yscale('log')
    axr.set_ylim(0.2, 5)
    axr.set_xlabel(r'$\log_{10}\,M_{\rm 200c}$ [$M_\odot/h$]')
    axr.set_ylabel('data / model')
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG, 'compare_hmf.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='+',
                    choices=['halofit', 'kids', 'boss', 'hmf'],
                    default=None, help='run only these comparison figures')
    args = ap.parse_args()
    todo = set(args.only or ['halofit', 'kids', 'boss', 'hmf'])

    os.makedirs(DIAG, exist_ok=True)
    log('=== Emulator vs targets at project fiducial '
        f'(omega_m={FIDUCIAL7[IDX_OMEGA_M]}, sigma_8={FIDUCIAL7[IDX_SIGMA_8]}, '
        'fiducial subgrid) ===\n')
    need_pk_emu = todo & {'halofit', 'kids', 'boss'}
    emu = PkEmulator() if need_pk_emu else None
    lin = LinearPk() if 'halofit' in todo else None

    if 'halofit' in todo:
        fig_go_halofit(emu, lin)
    if 'kids' in todo:
        fig_kids(emu)
    if 'boss' in todo:
        fig_boss(emu)
    if 'hmf' in todo:
        fig_hmf()

    out = os.path.join(DIAG, 'compare_targets_summary.txt')
    mode = 'a' if args.only else 'w'
    with open(out, mode) as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSummary -> {out}')


if __name__ == '__main__':
    main()
