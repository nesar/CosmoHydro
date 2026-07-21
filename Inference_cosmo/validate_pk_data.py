#!/usr/bin/env python
"""
Data-quality validation of the simulation power spectra and the observational
P(k) targets. Run this before training emulators or running MCMC.

Checks performed
----------------
Simulation side (scidac-olcf-pk_2 and _3):
  1. completeness: all 110 runs x 5 z x 4 types present in both directories
  2. pk_2 vs pk_3 byte-identity (MD5)
  3. every file loads, is finite, has strictly increasing k, P0 > 0
  4. identical k grid across all files
  5. ErrorBar column consistent with Gaussian mode counting  P*sqrt(2/nModes)
  6. suppression-ratio outlier scan (ratio-space cancels cosmic variance):
     per (z, k-bin) median/MAD across runs, flag runs with sustained deviation
  7. z-interpolation accuracy: reconstruct P(z=0.1) from log-P linear
     interpolation in scale factor between z=0.0 and z=0.5 (the same scheme
     the likelihood uses between snapshots) and report the error
  8. hydro.full internal consistency: P_full vs (f_c sqrt(P_cdm) + f_b sqrt(P_bar))^2
     at low k (perfectly correlated components) as a physical sanity check

Observational side:
  9. KiDS-Legacy files present, correlation matrices symmetric with unit
     diagonal and positive-definite; 68% band ordering sane
 10. BOSS DR12 monoquad tables load; Patchy covariance block structure
     identified by matching sqrt(diag) against the sig_P0/sig_P2 columns

Outputs
-------
  diagnostics/pk_data_validation.txt   full report
  diagnostics/pk_suite_overview.png    suite P(k), ratios, outliers per z
"""

import hashlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pk_data import (
    PK_DIR_DEFAULT, PK_REDSHIFT_TAGS, PK_REDSHIFTS, PK_TYPES,
    load_pk_suite, k_trust_mask,
)

PK2_DIR = os.path.join(_HERE, '..', 'data', 'scidac-olcf-pk_2')
PK3_DIR = os.path.join(_HERE, '..', 'data', 'scidac-olcf-pk_3')
TARGET_DIR = os.path.join(_HERE, '..', 'data', 'Power_spec_targets',
                          'nonlinear_pk_targets')
OUT_TXT = os.path.join(_HERE, 'diagnostics', 'pk_data_validation.txt')
OUT_PNG = os.path.join(_HERE, 'diagnostics', 'pk_suite_overview.png')

N_RUNS = 110

report = []


def log(msg=''):
    print(msg)
    report.append(msg)


def md5(path, blocksize=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(blocksize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def expected_files():
    for run in range(N_RUNS):
        for z in PK_REDSHIFT_TAGS:
            for t in PK_TYPES:
                yield f'run{run:03d}_z{z}.{t}.pk.txt'


def check_completeness_and_identity():
    log('== 1/2. Completeness and pk_2 vs pk_3 identity ==')
    missing2, missing3, differing = [], [], []
    for fname in expected_files():
        p2 = os.path.join(PK2_DIR, fname)
        p3 = os.path.join(PK3_DIR, fname)
        e2, e3 = os.path.exists(p2), os.path.exists(p3)
        if not e2:
            missing2.append(fname)
        if not e3:
            missing3.append(fname)
        if e2 and e3 and md5(p2) != md5(p3):
            differing.append(fname)
    log(f'  expected files per dir: {N_RUNS * len(PK_REDSHIFT_TAGS) * len(PK_TYPES)}')
    log(f'  missing in pk_2: {len(missing2)}  {missing2[:5]}')
    log(f'  missing in pk_3: {len(missing3)}  {missing3[:5]}')
    log(f'  files differing between pk_2 and pk_3: {len(differing)}  {differing[:5]}')
    if not missing2 and not missing3 and not differing:
        log('  -> pk_2 and pk_3 are complete and byte-identical; using pk_3.')
    return len(missing3) == 0


def check_file_contents():
    log('')
    log('== 3/4/5. File contents, common k grid, ErrorBar consistency ==')
    k_ref = None
    bad = []
    err_dev_max = 0.0
    for run in range(N_RUNS):
        for z in PK_REDSHIFT_TAGS:
            for t in PK_TYPES:
                fname = f'run{run:03d}_z{z}.{t}.pk.txt'
                d = np.loadtxt(os.path.join(PK3_DIR, fname))
                k, p, e, nm = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
                if not np.all(np.isfinite(d)):
                    bad.append((fname, 'non-finite values'))
                    continue
                if np.any(np.diff(k) <= 0):
                    bad.append((fname, 'k not strictly increasing'))
                if np.any(p <= 0):
                    bad.append((fname, f'{np.sum(p <= 0)} bins with P<=0'))
                if np.any(d[:, 4] != 0):
                    bad.append((fname, 'nonzero P_2 (unexpected for real space)'))
                if k_ref is None:
                    k_ref = k
                elif k.shape != k_ref.shape or not np.allclose(k, k_ref, rtol=1e-10):
                    bad.append((fname, 'k grid differs from reference'))
                # ErrorBar vs Gaussian mode counting
                expec = p * np.sqrt(2.0 / nm)
                dev = np.max(np.abs(e / expec - 1.0))
                err_dev_max = max(err_dev_max, dev)
    log(f'  files scanned: {N_RUNS * len(PK_REDSHIFT_TAGS) * len(PK_TYPES)}')
    log(f'  k grid: {k_ref.size} bins, k = [{k_ref[0]:.5f}, {k_ref[-1]:.4f}] h/Mpc, '
        f'dk ~ {np.median(np.diff(k_ref)):.5f}')
    log(f'  max |ErrorBar / (P*sqrt(2/nModes)) - 1| over all files: {err_dev_max:.2e}')
    if bad:
        log(f'  PROBLEMS ({len(bad)}):')
        for fname, why in bad[:20]:
            log(f'    {fname}: {why}')
    else:
        log('  -> all files finite, positive, on one common k grid.')
    return k_ref, bad


def ratio_outlier_scan():
    log('')
    log('== 6. Suppression-ratio outlier scan (per z) ==')
    flagged = {}
    suites = {}
    for ztag in PK_REDSHIFT_TAGS:
        suite = load_pk_suite(PK3_DIR, ztag=ztag, pk_type='hydro.full',
                              num_sims=N_RUNS, start_sim_idx=0)
        suites[ztag] = suite
        m = k_trust_mask(suite['k'])
        r = suite['ratio'][:, m]
        med = np.median(r, axis=0)
        mad = 1.4826 * np.median(np.abs(r - med), axis=0)
        mad = np.maximum(mad, 1e-4)
        # a run is flagged if >30% of its k-bins sit beyond 5 sigma-MAD:
        # sustained deviation, not single-bin noise. The design intentionally
        # spans strong feedback variations, so only gross outliers are flagged.
        frac_out = np.mean(np.abs(r - med) / mad > 5.0, axis=1)
        idx = np.where(frac_out > 0.3)[0]
        flagged[ztag] = idx
        log(f'  z={ztag}: ratio range at k~1: '
            f'[{r[:, np.argmin(np.abs(suite["k"][m] - 1.0))].min():.3f}, '
            f'{r[:, np.argmin(np.abs(suite["k"][m] - 1.0))].max():.3f}]; '
            f'runs with sustained >5 MAD deviation: {list(idx)}')
    log('  NOTE: flagged runs are extreme-feedback corners of the design, not')
    log('  necessarily corrupt data — inspect them in pk_suite_overview.png.')
    return suites, flagged


def z_interp_accuracy(suites):
    log('')
    log('== 7. Redshift-interpolation accuracy (log P linear in a) ==')
    # Reconstruct z=0.1 from z=0.0 and z=0.5, compare to truth.
    a = 1.0 / (1.0 + PK_REDSHIFTS)
    a0, a1, at = a[0], a[2], a[1]     # z=0.0, z=0.5, target z=0.1
    w = (at - a1) / (a0 - a1)
    errs = {}
    for typ, key in [('hydro.full', 'P'), ('go', 'P_go')]:
        P0 = suites['0.0'][key]
        P1 = suites['0.5'][key]
        Pt = suites['0.1'][key]
        pred = 10 ** (w * np.log10(P0) + (1 - w) * np.log10(P1))
        m = k_trust_mask(suites['0.0']['k'])
        rel = np.abs(pred[:, m] / Pt[:, m] - 1.0)
        errs[typ] = rel
        log(f'  {typ}: median |dP/P| = {np.median(rel):.4f}, '
            f'95th pct = {np.percentile(rel, 95):.4f}, max = {rel.max():.4f}')
    log('  (This is the widest snapshot gap re-scaled; interpolating between')
    log('   adjacent snapshots, e.g. for BOSS z_eff=0.38 or KiDS z=0.45, is')
    log('   more accurate than the numbers above. Ratio interpolation errors')
    log('   largely cancel between hydro and GO.)')
    return errs


def hydro_component_check(suites):
    log('')
    log('== 8. hydro.full vs cdm+bar composition (z=0, low k) ==')
    # At low k the cdm and baryon fields are near-perfectly correlated, so
    # P_full ~ (f_c sqrt(P_cdm) + f_b sqrt(P_bar))^2 with f_i the mass
    # fractions. We check the correlation-coefficient proxy
    #   rho_eff = (P_full - f_c^2 P_cdm - f_b^2 P_bar) / (2 f_c f_b sqrt(P_cdm P_bar))
    # which must lie in [-1, 1] and -> 1 at low k.
    # f_b = omega_b / omega_m per design row; omega_b h^2 = 0.02242 fixed.
    import pandas as pd
    design = pd.read_csv(os.path.join(_HERE, '..', 'data', 'FinalDesign.txt')).values
    omega_b_h2 = 0.02242
    bad = 0
    rho_low_all = []
    for run in range(N_RUNS):
        fb = omega_b_h2 / design[run, 5]
        fc = 1.0 - fb
        dfull = np.loadtxt(os.path.join(PK3_DIR, f'run{run:03d}_z0.0.hydro.full.pk.txt'))
        dcdm = np.loadtxt(os.path.join(PK3_DIR, f'run{run:03d}_z0.0.hydro.cdm.pk.txt'))
        dbar = np.loadtxt(os.path.join(PK3_DIR, f'run{run:03d}_z0.0.hydro.bar.pk.txt'))
        k = dfull[:, 0]
        low = (k > 0.02) & (k < 0.1)
        rho = ((dfull[low, 1] - fc**2 * dcdm[low, 1] - fb**2 * dbar[low, 1])
               / (2 * fc * fb * np.sqrt(dcdm[low, 1] * dbar[low, 1])))
        rho_low_all.append(np.mean(rho))
        if np.any(np.abs(rho) > 1.05):
            bad += 1
    rho_low_all = np.array(rho_low_all)
    log(f'  effective cdm-baryon correlation at k=0.02-0.1: '
        f'mean {rho_low_all.mean():.4f}, min {rho_low_all.min():.4f} '
        f'(expect ~1; runs violating |rho|<=1.05: {bad})')


def check_kids():
    log('')
    log('== 9. KiDS-Legacy target files ==')
    kdir = os.path.join(TARGET_DIR, 'kids_legacy')
    ok = True
    for tag, nbin in [('nz1', 20), ('nz3', 60)]:
        f_pm = os.path.join(kdir, f'KiDSLegacy_{tag}_Pm.txt')
        f_cm = os.path.join(kdir, f'{tag}-pmcm.dat')
        if not os.path.exists(f_pm):
            log(f'  MISSING {f_pm} — run kids_legacy/get_kids_legacy_pm.py')
            ok = False
            continue
        d = np.loadtxt(f_pm)
        corr = np.loadtxt(f_cm)
        band_ok = np.all((d[:, 3] <= d[:, 2]) & (d[:, 2] <= d[:, 4]))
        sym = np.allclose(corr, corr.T, atol=1e-6)
        diag1 = np.allclose(np.diag(corr), 1.0, atol=1e-6)
        evmin = np.linalg.eigvalsh(0.5 * (corr + corr.T)).min()
        log(f'  {tag}: {d.shape[0]} rows (expect {nbin}), corr {corr.shape}, '
            f'68% band ordering ok: {band_ok}, symmetric: {sym}, '
            f'unit diag: {diag1}, min eigenvalue: {evmin:.3e}')
        ok = ok and band_ok and sym and diag1 and (d.shape[0] == nbin) and evmin > 0
    # headline check: wide-bin suppression at k = 3-20 h/Mpc should be ~30%
    f_nz1 = os.path.join(kdir, 'KiDSLegacy_nz1_Pm.txt')
    if os.path.exists(f_nz1):
        d = np.loadtxt(f_nz1)
        m = (d[:, 0] > 3) & (d[:, 0] < 20)
        log(f'  nz1 mean fdelta at k=3-20: {d[m, 5].mean():.3f} '
            f'(paper headline: 0.70 +/- 0.10)')
    return ok


def check_boss():
    log('')
    log('== 10. BOSS DR12 monoquad tables + Patchy covariance blocks ==')
    from targets import load_boss_covariance_blocks   # noqa: local import; defined below
    bdir = os.path.join(TARGET_DIR, 'boss_dr12')
    ok = True
    for patch in ['NGC', 'SGC']:
        for zbin in ['z1', 'z3']:
            tab = np.loadtxt(os.path.join(
                bdir, f'BOSS_DR12_{patch}_{zbin}_pk_monoquad_dk0p01.txt'))
            res = load_boss_covariance_blocks(patch, zbin, boss_dir=bdir)
            d0 = np.abs(np.sqrt(np.diag(res['C00'])) / tab[:, 2] - 1).max()
            d2 = np.abs(np.sqrt(np.diag(res['C22'])) / tab[:, 4] - 1).max()
            log(f'  {patch} {zbin}: {tab.shape[0]} k bins; multipole block order '
                f'{res["block_order"]}; sqrt(diag C00) vs sig_P0 max rel dev '
                f'{d0:.2e}; C22 vs sig_P2: {d2:.2e}')
            ok = ok and d0 < 1e-3 and d2 < 1e-3
    return ok


def overview_plot(suites, flagged):
    fig, axes = plt.subplots(2, len(PK_REDSHIFT_TAGS), figsize=(22, 8),
                             sharex=True)
    for j, ztag in enumerate(PK_REDSHIFT_TAGS):
        suite = suites[ztag]
        k = suite['k']
        m = k_trust_mask(k)
        for i in range(suite['P'].shape[0]):
            c = 'crimson' if i in flagged[ztag] else 'gray'
            zo = 3 if i in flagged[ztag] else 1
            axes[0, j].loglog(k[m], suite['P'][i, m], color=c, alpha=0.4,
                              lw=0.6, zorder=zo)
            axes[1, j].semilogx(k[m], suite['ratio'][i, m], color=c, alpha=0.4,
                                lw=0.6, zorder=zo)
        axes[0, j].set_title(f'z = {ztag}  (red = flagged)')
        axes[1, j].axhline(1.0, ls=':', color='k', lw=0.8)
        axes[1, j].set_xlabel(r'$k$ [h/Mpc]')
        axes[1, j].set_ylim(0.55, 1.25)
    axes[0, 0].set_ylabel(r'$P^{\rm hydro}(k)$ [(Mpc/h)$^3$]')
    axes[1, 0].set_ylabel(r'$P^{\rm hydro}/P^{\rm GO}$')
    fig.suptitle('Simulation P(k) suite overview — 110 runs, scidac-olcf-pk_3')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches='tight')
    log('')
    log(f'  wrote {OUT_PNG}')


def main():
    ok = check_completeness_and_identity()
    if not ok:
        log('FATAL: missing simulation files; aborting further checks.')
    else:
        k_ref, bad = check_file_contents()
        suites, flagged = ratio_outlier_scan()
        z_interp_accuracy(suites)
        hydro_component_check(suites)
        overview_plot(suites, flagged)
    check_kids()
    try:
        check_boss()
    except Exception as e:
        log(f'  BOSS check failed: {e!r}')
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, 'w') as f:
        f.write('\n'.join(report) + '\n')
    print(f'\nReport written to {OUT_TXT}')


if __name__ == '__main__':
    main()
