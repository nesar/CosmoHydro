#!/usr/bin/env python
"""
Compare emulated power spectra against every observational target, at the
project fiducial parameters. Produces the figures and a chi^2 summary that
should be inspected BEFORE trusting any MCMC run.

Figures (written to diagnostics/):
  compare_kids.png       emulated P_hydro(k, z_fid) vs KiDS-Legacy Pm bands
  compare_amod.png       emulated suppression vs A_mod template bands
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
from targets import load_kids, load_amod, load_boss, AMOD_CONSTRAINTS
from pk_likelihood import (
    PkEmulator, PmLikelihood, AmodLikelihood, BossLikelihood,
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


def fig_amod(emu, lin):
    target = load_amod('DES_Y3_Planck')
    like = AmodLikelihood(target, emu, linear_pk=lin)
    A_fid = like.model_amod(FIDUCIAL7)
    chi2 = -2.0 * like(FIDUCIAL7)
    log(f'A_mod: model-implied A_mod at fiducial = {A_fid:.3f} '
        f'(DES Y3+Planck: {target["Amod"]} +/- {target["sigma"]}; '
        f'chi2 = {chi2:.2f})')

    # suppression plane figure
    k = emu.k_grid
    S_fid, S_err = emu.ratio(0.0, FIDUCIAL7)
    P_go, _ = emu.P_go(0.0, FIDUCIAL7)
    P_L = lin(k, 0.0, FIDUCIAL7[IDX_OMEGA_M], FIDUCIAL7[IDX_SIGMA_8])
    t = 1.0 - P_L / P_go

    # design envelope from the raw suite
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
    ax.set_title(r'Baryonic suppression vs weak-lensing $A_{\rm mod}$ '
                 r'templates ($z=0$)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG, 'compare_amod.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


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


def main():
    os.makedirs(DIAG, exist_ok=True)
    log('=== Emulator vs targets at project fiducial '
        f'(omega_m={FIDUCIAL7[IDX_OMEGA_M]}, sigma_8={FIDUCIAL7[IDX_SIGMA_8]}, '
        'fiducial subgrid) ===\n')
    emu = PkEmulator()
    lin = LinearPk()

    fig_go_halofit(emu, lin)
    fig_kids(emu)
    fig_amod(emu, lin)
    fig_boss(emu)

    out = os.path.join(DIAG, 'compare_targets_summary.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSummary -> {out}')


if __name__ == '__main__':
    main()
