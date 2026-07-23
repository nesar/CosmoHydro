"""
Observational power-spectrum target loaders.

All loaders return plain dicts with numpy arrays; nothing is invented — every
number traces back to the files under data/Power_spec_targets/nonlinear_pk_targets/
whose provenance is documented in that directory's README.md.

Targets
-------
KiDS-Legacy (Broxterman et al. 2025, A&A 703, L3):
    deprojected matter power spectrum Pm(k, z) = fdelta * Pdmo in 1 or 3
    tomographic z bins.  This is a *direct* matter-P(k) target: the natural
    comparison is the emulated hydro (total matter) P(k, z_fid).
    Caveats baked into the measurement (see paper): lensing kernel assumes
    Omega_m = 0.305 +/- 0.012 flat LCDM; NLA-M intrinsic alignments.

A_mod: quarantined to amod_exploratory/ (2026-07-23) — it modulates the
    nonlinear boost at fixed Planck cosmology and is NOT a measurement of
    P_hydro/P_GO; see amod_exploratory/README.md.

BOSS DR12 (Beutler & McDonald 2021):
    pre-recon galaxy power spectrum multipoles, rebinned to dk = 0.01, with
    the 2048-mock MultiDark-Patchy covariance.  This is a REDSHIFT-SPACE
    GALAXY spectrum convolved with the survey window.  The likelihood in this
    package models it with linear (Kaiser) RSD + linear bias + Alcock-
    Paczynski dilation, WITHOUT window convolution — restrict to
    k in ~[0.03, 0.15] h/Mpc and treat results as a methods-level cosmology
    constraint, not a publication-grade full-shape analysis (window matrices
    would need to be added for that; see README).
"""

import gzip
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_BASE = os.path.join(_HERE, '..', 'data', 'Power_spec_targets',
                           'nonlinear_pk_targets')

BOSS_ZEFF = {'z1': 0.38, 'z3': 0.61}
# Fiducial cosmology of the BOSS DR12 catalogue analysis (Beutler & McDonald
# 2021, Sec. 2; standard BOSS DR12 fiducial), needed for Alcock-Paczynski.
BOSS_FIDUCIAL = {'Omega_m': 0.31, 'h': 0.676}

# ---------------------------------------------------------------------------
# KiDS-Legacy
# ---------------------------------------------------------------------------
def load_kids(nz='nz3', k_min=None, k_max=None, z_bins=None,
              kids_dir=None):
    """Load the KiDS-Legacy deprojected Pm(k, z).

    Parameters
    ----------
    nz : 'nz1' (one wide bin, z_fid=0.3) or 'nz3' (z_fid = 0.15, 0.45, 1.3)
    k_min, k_max : optional cut in h/Mpc (applied AFTER building the full
        covariance, so correlations with removed bins are dropped correctly)
    z_bins : optional list of z_fid values to keep (e.g. [0.15, 0.45])

    Returns dict:
      k, z      : (n,) wavenumber [h/Mpc] and fiducial redshift per point
      y         : (n,) measured Pm [(Mpc/h)^3]  (posterior median)
      cov       : (n, n) covariance of Pm, built as
                  corr_ij * sigma_i * sigma_j with sigma = symmetrized 68% CI
                  of fdelta times the fiducial Pdmo
      sigma     : (n,) sqrt(diag cov)
      fdelta, pdmo : raw columns for reference
    """
    kids_dir = kids_dir or os.path.join(TARGET_BASE, 'kids_legacy')
    f_pm = os.path.join(kids_dir, f'KiDSLegacy_{nz}_Pm.txt')
    if not os.path.exists(f_pm):
        raise FileNotFoundError(
            f'{f_pm} not found — run kids_legacy/get_kids_legacy_pm.py first')
    d = np.loadtxt(f_pm)
    corr = np.loadtxt(os.path.join(kids_dir, f'{nz}-pmcm.dat'))

    k, z = d[:, 0], d[:, 1]
    pm, fdelta, pdmo = d[:, 2], d[:, 5], d[:, 8]
    # symmetrized 68% credible interval on fdelta -> Pm error
    sigma_fdelta = 0.5 * (d[:, 7] - d[:, 6])
    sigma = sigma_fdelta * pdmo
    if corr.shape != (k.size, k.size):
        raise ValueError(f'correlation matrix shape {corr.shape} != {k.size}')
    corr = 0.5 * (corr + corr.T)
    cov = corr * np.outer(sigma, sigma)

    keep = np.ones(k.size, dtype=bool)
    if k_min is not None:
        keep &= k >= k_min
    if k_max is not None:
        keep &= k <= k_max
    if z_bins is not None:
        keep &= np.isin(z, np.asarray(z_bins, dtype=float))

    idx = np.where(keep)[0]
    return {
        'name': f'KiDS-Legacy {nz}',
        'kind': 'pm',
        'k': k[idx], 'z': z[idx], 'y': pm[idx],
        'cov': cov[np.ix_(idx, idx)],
        'sigma': sigma[idx],
        'fdelta': fdelta[idx], 'pdmo': pdmo[idx],
        'ref': 'Broxterman et al. 2025, A&A 703, L3',
    }


# ---------------------------------------------------------------------------
# A_mod: MOVED to amod_exploratory/amod_likelihood.py (2026-07-23).
# It is a Planck-conditioned modulation of the NONLINEAR BOOST (P_NL - P_L),
# not a measurement of P_hydro/P_GO — see amod_exploratory/README.md.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GAMA DR4 halo mass function (Driver et al. 2022)
# ---------------------------------------------------------------------------
HMF_TARGET_DIR = os.path.join(_HERE, '..', 'data', 'Halo_mass_function_targets',
                              'hmf_targets')
# Driver et al. 2022 analysis cosmology (Planck 2018): H0 = 67.37
GAMA_H = 0.6737


def load_gama_hmf(logM_min=12.8, logM_max=None, include_cosvar=True,
                  hmf_dir=None):
    """GAMA DR4 empirical HMF (Driver et al. 2022, Table 1) in simulation
    units.

    UNITS (verified against the published paper, MNRAS 515, 2138, Sec. 2:
    "our units are Msun h^-1_P18" and space densities "Mpc^-3 h^3_P18"):
    the table masses are ALREADY Msun/h and phi ALREADY (Mpc/h)^-3 dex^-1,
    so the conversion to simulation units is the IDENTITY (h-unit quantities
    are h-invariant; the 0.4% h_P18=0.6737 vs h_sim=0.6766 difference is
    absorbed by the h-unit convention). NOTE: the local data-package README
    header suggests physical-Mpc units — the paper text is authoritative.

    Rows kept: credible_flag=1 AND logM >= 12.8 (the paper's stated mass
    completeness limit; the file flags 16 rows down to logM=12.4, but the
    12.4/12.6 bins sit below the completeness turnover — visible as a break
    in the phi trend — and the paper adopts 10^12.8 as the limit).
    Uses the Eddington-bias-corrected column log10_phi_corr.

    Errors: sigma_phi = phi * sqrt(fsig_combined^2 [+ fsig_cosvar^2]),
    treated as independent Gaussian per bin (the paper publishes no
    covariance).

    z_eff ~ 0.1 — matched by evaluating the sim HMF emulator at snapshot 567
    (z = 0.0998), so NO redshift shift is applied to the data.

    Known systematic (paper Sec. 4): the absolute mass scale rides on the
    dynamical-mass calibration A = 13.9; enable the `mass_shift` nuisance in
    the MCMC config to marginalize a rigid log-mass offset.

    Returns dict with logM (Msun/h), y = phi, sigma, and bookkeeping.
    """
    hmf_dir = hmf_dir or HMF_TARGET_DIR
    d = np.loadtxt(os.path.join(hmf_dir, 'GAMA_DR4_HMF_Driver2022_Table1.txt'))
    # columns: log10M  N  log10_phi_raw  log10_phi_corr  fsig_p  fsig_MC
    #          fsig_cosvar  fsig_comb  credible
    cred = d[:, 8] == 1
    d = d[cred]
    logM = d[:, 0]                # already Msun/h (h_P18) per the paper
    phi = 10 ** d[:, 3]           # already (Mpc/h)^-3 dex^-1 per the paper
    fsig = d[:, 7]
    if include_cosvar:
        fsig = np.sqrt(fsig ** 2 + d[:, 6] ** 2)
    sigma = phi * fsig

    keep = np.ones(logM.size, dtype=bool)
    if logM_min is not None:
        keep &= logM >= logM_min
    if logM_max is not None:
        keep &= logM <= logM_max
    return {
        'name': 'GAMA DR4 HMF (Driver et al. 2022)',
        'kind': 'gama_hmf',
        'logM': logM[keep], 'y': phi[keep], 'sigma': sigma[keep],
        'n_groups': d[keep, 1],
        'z_eff': 0.1,
        'ref': 'Driver et al. 2022, MNRAS 515, 2138',
    }


def load_mrp_fits(hmf_dir=None):
    """MRP fit parameters (Driver et al. 2022, Table 2) for PLOTTING only.

    phi(log10 M) = ln(10) * phistar * beta * (M/Mstar)^(alpha+1)
                   * exp(-(M/Mstar)^beta)
    Valid over 10^12.7-10^15.5 Msun at z ~ 0.1, in the paper's units
    (Msun, Mpc^-3 dex^-1) — convert like the binned data before overplotting.
    No parameter covariance is published, so these must NOT be used as a
    likelihood.
    """
    hmf_dir = hmf_dir or HMF_TARGET_DIR
    out = {}
    path = os.path.join(hmf_dir, 'MRP_fits_Driver2022_Table2.txt')
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            p = line.split()
            out[p[0]] = {'log10Mstar': float(p[1]),
                         'log10phistar': float(p[4]),
                         'alpha': float(p[7]),
                         'beta': float(p[10])}
    return out


def mrp_phi(logM_data_units, fit):
    """Evaluate an MRP fit at log10 M in the paper's (Msun) units."""
    x = 10 ** (logM_data_units - fit['log10Mstar'])
    return (np.log(10.0) * 10 ** fit['log10phistar'] * fit['beta']
            * x ** (fit['alpha'] + 1) * np.exp(-x ** fit['beta']))


# ---------------------------------------------------------------------------
# BOSS DR12
# ---------------------------------------------------------------------------
def _read_patchy_matrix(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        M = np.loadtxt(f)
    if M.shape != (200, 200):
        raise ValueError(f'unexpected Patchy matrix shape {M.shape} in {path}')
    return M


def load_boss_covariance_blocks(patch, zbin, boss_dir=None):
    """Load the 200x200 Patchy covariance and identify the P0/P2 blocks.

    The matrix covers 5 multipoles x 40 k-bins (0 < k < 0.4, dk = 0.01).
    The multipole ordering is identified empirically by matching sqrt(diag)
    of each 40x40 diagonal block against the sig_P0 / sig_P2 columns of the
    monoquad convenience table (which came from the same covariance).

    Returns dict with C00, C22, C02 (40x40 blocks) and 'block_order' info.
    """
    boss_dir = boss_dir or os.path.join(TARGET_BASE, 'boss_dr12')
    cov_file = None
    for fn in os.listdir(boss_dir):
        if fn.startswith(f'C_2048_BOSS_DR12_{patch}_{zbin}') and '.matrix' in fn:
            cov_file = os.path.join(boss_dir, fn)
            break
    if cov_file is None:
        raise FileNotFoundError(f'no Patchy covariance for {patch} {zbin}')
    C = _read_patchy_matrix(cov_file)

    tab = np.loadtxt(os.path.join(
        boss_dir, f'BOSS_DR12_{patch}_{zbin}_pk_monoquad_dk0p01.txt'))
    sig0, sig2 = tab[:, 2], tab[:, 4]

    nb = 40
    diag_blocks = [np.sqrt(np.diag(C[i * nb:(i + 1) * nb, i * nb:(i + 1) * nb]))
                   for i in range(5)]
    dev0 = [np.median(np.abs(db / sig0 - 1)) for db in diag_blocks]
    dev2 = [np.median(np.abs(db / sig2 - 1)) for db in diag_blocks]
    i0, i2 = int(np.argmin(dev0)), int(np.argmin(dev2))
    if i0 == i2 or dev0[i0] > 1e-3 or dev2[i2] > 1e-3:
        raise ValueError(
            f'could not identify P0/P2 blocks for {patch} {zbin}: '
            f'dev0={dev0}, dev2={dev2}')
    sl0 = slice(i0 * nb, (i0 + 1) * nb)
    sl2 = slice(i2 * nb, (i2 + 1) * nb)
    return {
        'C00': C[sl0, sl0], 'C22': C[sl2, sl2], 'C02': C[sl0, sl2],
        'block_order': {'P0': i0, 'P2': i2},
        'n_mocks': 2048,
        'file': os.path.basename(cov_file),
    }


def load_boss(patch='NGC', zbin='z1', k_min=0.03, k_max=0.15,
              use_quad=True, boss_dir=None):
    """Load BOSS DR12 P0 (and optionally P2) with the Patchy covariance.

    The Hartlap factor (Hartlap et al. 2007) for the inverse-covariance bias
    of a 2048-mock estimate is applied by the likelihood, not here; this
    returns the raw covariance and n_mocks.

    Returns dict:
      k       : (nk,) effective wavenumbers within [k_min, k_max]
      y       : (nd,) data vector [P0; P2] (or just P0)
      cov     : (nd, nd) Patchy covariance for the same vector
      z_eff   : effective redshift (0.38 for z1, 0.61 for z3)
      nmodes  : (nk,) modes per bin
    """
    boss_dir = boss_dir or os.path.join(TARGET_BASE, 'boss_dr12')
    tab = np.loadtxt(os.path.join(
        boss_dir, f'BOSS_DR12_{patch}_{zbin}_pk_monoquad_dk0p01.txt'))
    blocks = load_boss_covariance_blocks(patch, zbin, boss_dir=boss_dir)

    k_all = tab[:, 0]
    m = (k_all >= k_min) & (k_all <= k_max)
    idx = np.where(m)[0]
    k = k_all[idx]
    P0 = tab[idx, 1]
    P2 = tab[idx, 3]

    C00 = blocks['C00'][np.ix_(idx, idx)]
    if use_quad:
        C22 = blocks['C22'][np.ix_(idx, idx)]
        C02 = blocks['C02'][np.ix_(idx, idx)]
        y = np.concatenate([P0, P2])
        cov = np.block([[C00, C02], [C02.T, C22]])
    else:
        y = P0
        cov = C00

    return {
        'name': f'BOSS DR12 {patch} {zbin}',
        'kind': 'boss_multipoles',
        'patch': patch, 'zbin': zbin,
        'k': k, 'y': y, 'cov': cov,
        'use_quad': use_quad,
        'z_eff': BOSS_ZEFF[zbin],
        'n_mocks': blocks['n_mocks'],
        'nmodes': tab[idx, 5],
        'fiducial': dict(BOSS_FIDUCIAL),
        'ref': 'Beutler & McDonald 2021, JCAP 11 (2021) 031',
    }
