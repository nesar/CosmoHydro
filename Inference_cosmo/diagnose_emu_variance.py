"""Thorough check of adding emulator variance to the GSMF/CGD likelihood.

Concern: adding sigma_emu^2 to the chi^2 denominator WITHOUT the matching
Gaussian normalisation (logdet) term rewards high-variance regions (design
edges) -> the chain flees there. This script quantifies:

  A. magnitude of sigma_emu vs the observational error, per observable.
  B. chi^2 three ways at the 3 chain best-fits + fiducial:
       (obs)      = r^2 / sigma_obs^2                       [current likelihood]
       (obs+emu)  = r^2 / (sigma_obs^2 + sigma_emu^2)       [NO logdet -> risky]
       (-2lnL)    = r^2/(so^2+se^2) + log(2pi(so^2+se^2))   [proper Gaussian]
     and the DISCRIMINATION (spread among the 3 real chains) for each.
  C. pathology test: over random points in the design box, is the NO-logdet
     score anti-correlated with sigma_emu (i.e. does high emulator variance buy
     a spuriously good 'fit')? Does the proper (-2lnL) version remove that?
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'Inference'))
sys.path.insert(0, os.path.join(_HERE, '..', 'codes'))
import gsmf_cgd_target as GC
from cosmo_hydro_emu.emu import emulate

FID7 = np.array([3.0, 0.5, 0.8, 0.51, 0.13, 0.14176, 0.8102])
BESTFIT = {
    'Pk_7p (marg)':      np.array([3.2765, 0.5515, 1.1859, 0.6965, 0.2154, 0.1549, 0.8190]),
    'Pk_5p (cosmo fix)': np.array([3.2889, 0.4734, 1.1958, 1.1430, 0.2272, 0.1418, 0.8102]),
    'Pk_2cosmo@A':       np.array([3.2909, 0.4769, 1.1914, 1.1395, 0.2325, 0.1424, 0.8108]),
    'fiducial':          FID7,
}
BOX = np.array([[2.0, 4.0], [0.2, 1.0], [0.6, 2.0], [0.1, 1.2],
                [0.02, 1.2], [0.12, 0.155], [0.70, 0.90]])


def _pred(like, theta):
    mg, sd = emulate(like.model, np.asarray(theta, float))
    model = np.interp(like.x, like.x_grid, mg[:, 0])
    semu = np.interp(like.x, like.x_grid, sd[:, 0])
    return like.y - model, like.yerr, np.abs(semu)


def variants(like, theta):
    r, so, se = _pred(like, theta)
    v = so ** 2 + se ** 2
    return dict(chi2_obs=np.sum(r ** 2 / so ** 2),
                chi2_emu=np.sum(r ** 2 / v),
                m2lnL=np.sum(r ** 2 / v + np.log(2 * np.pi * v)),
                ratio=np.median(se / so), n=len(r))


def main():
    gsmf = GC._build_component('gsmf')
    cgd = GC._build_component('cgd')
    comps = [('GSMF', gsmf), ('CGD', cgd)]

    print('=' * 78)
    print('A. sigma_emu / sigma_obs (per-point median) at each best fit')
    print('=' * 78)
    for name, th in BESTFIT.items():
        s = '  '.join(f'{c}: {variants(l, th)["ratio"]:.2f}' for c, l in comps)
        print(f'  {name:18s}  {s}')

    print('\n' + '=' * 78)
    print('B. chi^2 variants (GSMF+CGD combined) and discrimination')
    print('=' * 78)
    real = ['Pk_7p (marg)', 'Pk_5p (cosmo fix)', 'Pk_2cosmo@A']
    tot = {}
    print(f'  {"point":18s} | {"chi2(obs)":>10s} | {"chi2(obs+emu)":>13s} | {"-2lnL(emu)":>11s}')
    print('  ' + '-' * 62)
    for name, th in BESTFIT.items():
        a = variants(gsmf, th); b = variants(cgd, th)
        co = a['chi2_obs'] + b['chi2_obs']
        ce = a['chi2_emu'] + b['chi2_emu']
        ml = a['m2lnL'] + b['m2lnL']
        tot[name] = (co, ce, ml)
        print(f'  {name:18s} | {co:10.1f} | {ce:13.1f} | {ml:11.1f}')
    print(f'\n  dof (GSMF+CGD) = {variants(gsmf, FID7)["n"] + variants(cgd, FID7)["n"]}')
    print('\n  DISCRIMINATION among the 3 real chains (max-min):')
    for k, lab in [(0, 'chi2(obs)   '), (1, 'chi2(obs+emu)'), (2, '-2lnL(emu)  ')]:
        vals = [tot[r][k] for r in real]
        print(f'    {lab}: spread = {max(vals) - min(vals):7.1f}')

    print('\n' + '=' * 78)
    print('C. PATHOLOGY TEST: 300 random design-box points')
    print('=' * 78)
    rng = np.random.default_rng(0)
    pts = rng.uniform(BOX[:, 0], BOX[:, 1], size=(300, 7))
    se_mean, chi2_emu_nolog, m2lnL = [], [], []
    for th in pts:
        a = variants(gsmf, th); b = variants(cgd, th)
        se_mean.append(0.5 * (a['ratio'] + b['ratio']))
        chi2_emu_nolog.append(a['chi2_emu'] + b['chi2_emu'])
        m2lnL.append(a['m2lnL'] + b['m2lnL'])
    se_mean = np.array(se_mean); chi2_emu_nolog = np.array(chi2_emu_nolog)
    m2lnL = np.array(m2lnL)
    print(f'  sigma_emu/sigma_obs over the box: '
          f'min {se_mean.min():.2f}, median {np.median(se_mean):.2f}, max {se_mean.max():.2f}')
    print(f'  correlation( sigma_emu ratio , chi2(obs+emu, NO logdet) ) = '
          f'{np.corrcoef(se_mean, chi2_emu_nolog)[0,1]:+.3f}')
    print('    (strong NEGATIVE => high emulator variance buys a low chi2 = PATHOLOGY)')
    print(f'  correlation( sigma_emu ratio , -2lnL WITH logdet )        = '
          f'{np.corrcoef(se_mean, m2lnL)[0,1]:+.3f}')
    print('    (logdet term should cancel the reward; expect ~0 or positive)')
    # where does each score prefer? (lowest = best fit)
    i_nolog = int(np.argmin(chi2_emu_nolog)); i_log = int(np.argmin(m2lnL))
    print(f'\n  best NO-logdet point: sigma_emu ratio = {se_mean[i_nolog]:.2f} '
          f'(rank {int((se_mean > se_mean[i_nolog]).sum())}/300 by variance)')
    print(f'  best WITH-logdet point: sigma_emu ratio = {se_mean[i_log]:.2f}')


if __name__ == '__main__':
    main()
