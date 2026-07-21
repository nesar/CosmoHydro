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

A_mod (Amon & Efstathiou 2022; Preston, Amon & Efstathiou 2023):
    scalar suppression parameter defined through
        P_obs = P_L + A_mod (P_NL^DMO - P_L).
    The constraint is ONE number, so the correct likelihood treats it as a
    single datum: project the emulated suppression S(k) = P_hydro/P_GO onto
    the A_mod template (see pk_likelihood.AmodLikelihood).

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

AMOD_CONSTRAINTS = {
    # Preston, Amon & Efstathiou 2023 (arXiv:2305.09827): DES Y3 + Planck prior
    'DES_Y3_Planck': {'Amod': 0.858, 'sigma': 0.052, 'ref': 'arXiv:2305.09827'},
    # Amon & Efstathiou 2022 (arXiv:2206.11794): KiDS-1000; no published sigma,
    # central value only — usable for plots, NOT for a Gaussian likelihood.
    'KiDS1000': {'Amod': 0.69, 'sigma': None, 'ref': 'arXiv:2206.11794'},
}


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
# A_mod
# ---------------------------------------------------------------------------
def load_amod(constraint='DES_Y3_Planck'):
    """Return the A_mod scalar constraint dict (Amod, sigma, ref)."""
    c = dict(AMOD_CONSTRAINTS[constraint])
    c['name'] = f'A_mod {constraint}'
    c['kind'] = 'amod'
    if c['sigma'] is None:
        raise ValueError(
            f"constraint '{constraint}' has no published uncertainty; it can "
            f"be plotted but not used in a Gaussian likelihood")
    return c


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
