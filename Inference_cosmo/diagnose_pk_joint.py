"""Diagnostics for the joint GSMF+CGD+Pk runs:

(1) Point-A consistency: point A was chosen from the GSMF+CGD-only (Planck-prior)
    7p run. Is it still a sensible fixed-hydro point for THIS run set? Compare A
    to the subgrid posteriors of GSMF_CGD_Pk_7p and GSMF_CGD_Pk_5p_fid_cosmo.

(2) Component-wise log-likelihood: evaluate the GSMF, CGD and KiDS-Pk likelihood
    terms SEPARATELY at each chain's best fit (and at fiducial), to see how much
    each observable actually constrains / whether one term dominates.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'Inference'))
sys.path.insert(0, os.path.join(_HERE, '..', 'codes'))

RES = os.path.join(_HERE, 'results')

# Point A (from GSMF_CGD_7p_planck) and the project fiducial hydro/cosmo.
A_SUBGRID = np.array([3.2909, 0.4769, 1.1914, 1.1395, 0.2325])
FID7 = np.array([3.0, 0.5, 0.8, 0.51, 0.13, 0.14176, 0.8102])

# Best-fit (GMM mode) full-theta of each chain (from plot_summary output).
BESTFIT = {
    'Pk_7p (marg)':     np.array([3.2765, 0.5515, 1.1859, 0.6965, 0.2154, 0.1549, 0.8190]),
    'Pk_5p (cosmo fix)':np.array([3.2889, 0.4734, 1.1958, 1.1430, 0.2272, 0.1418, 0.8102]),
    'Pk_2cosmo@A':      np.array([3.2909, 0.4769, 1.1914, 1.1395, 0.2325, 0.1424, 0.8108]),
    'fiducial':         FID7,
}

SGN = ['kappa_w', 'e_w', 'M_seed/1e6', 'v_kin/1e4', 'eps_kin/1e1']


def mahalanobis(pt, samples):
    med = np.median(samples, axis=0)
    cov = np.cov(samples, rowvar=False)
    inv = np.linalg.inv(cov)
    d = pt - med
    return float(np.sqrt(d @ inv @ d)), med, np.sqrt(np.diag(cov))


def part1_A_consistency():
    print('=' * 74)
    print('(1) Is point A self-consistent with the GSMF+CGD+Pk run set?')
    print('=' * 74)
    refs = {
        'GSMF_CGD_7p_planck (A came from here)': ('../Inference/results', 5),
        'GSMF_CGD_Pk_7p (this set, marginalized)': ('results', 5),
        'GSMF_CGD_Pk_5p_fid_cosmo (this set, cosmo fixed)': ('results', 5),
    }
    trials = {
        'GSMF_CGD_7p_planck (A came from here)': 'GSMF_CGD_7p_planck',
        'GSMF_CGD_Pk_7p (this set, marginalized)': 'GSMF_CGD_Pk_7p',
        'GSMF_CGD_Pk_5p_fid_cosmo (this set, cosmo fixed)': 'GSMF_CGD_Pk_5p_fid_cosmo',
    }
    base = {'GSMF_CGD_7p_planck (A came from here)': os.path.join(_HERE, '..', 'Inference', 'results')}
    for name, trial in trials.items():
        d = base.get(name, RES)
        f = os.path.join(d, f'samples_{trial}.npy')
        if not os.path.exists(f):
            print(f'  MISSING {trial}'); continue
        sg = np.load(f)[:, :5]
        dist, med, std = mahalanobis(A_SUBGRID, sg)
        print(f'\n  {name}')
        print(f'    {"param":12s} {"A":>9s} {"median":>9s} {"std":>8s} {"(A-med)/std":>12s}')
        for i, n in enumerate(SGN):
            print(f'    {n:12s} {A_SUBGRID[i]:9.4f} {med[i]:9.4f} {std[i]:8.4f} '
                  f'{(A_SUBGRID[i]-med[i])/std[i]:12.2f}')
        print(f'    => Mahalanobis distance of A from this posterior: {dist:.1f} sigma')


def part2_components():
    print('\n' + '=' * 74)
    print('(2) Component-wise log-likelihood (GSMF, CGD, KiDS-Pk)')
    print('=' * 74)
    print('  building GSMF/CGD/Pk likelihoods (loads emulators, ~1-2 min)...')
    import gsmf_cgd_target as GC
    from targets import load_kids
    from pk_likelihood import PkEmulator, PmLikelihood

    gsmf = GC._build_component('gsmf')
    cgd = GC._build_component('cgd')
    emu = PkEmulator()
    kids = load_kids(nz='nz3', k_min=0.03, k_max=7.0, z_bins=[0.15, 0.45])
    pm = PmLikelihood(kids, emu)

    n = {'GSMF': len(gsmf.x), 'CGD': len(cgd.x), 'KiDS Pk': len(kids['y'])}
    print(f"\n  data points (dof): GSMF={n['GSMF']}, CGD={n['CGD']}, "
          f"KiDS-Pk={n['KiDS Pk']} (2 z-bins, strong cross-bin covariance)")

    def kids_pure_chi2(th):
        """Pure chi2 = r^T C^-1 r (WITHOUT the logdet term in -2lnL)."""
        y_mod, var_emu = pm.model_vector(th)
        cov = pm.t['cov'] + np.diag(var_emu + (pm.interp_sys_frac * y_mod) ** 2)
        r = pm.t['y'] - y_mod
        L = np.linalg.cholesky(cov)
        x = np.linalg.solve(L, r)
        return float(x @ x)

    def chi2s(th):
        # GSMF/CGD: -2 lnL is pure chi2 (no normalisation). KiDS: pure r^T C^-1 r.
        return {'GSMF': -2 * float(gsmf(th)),
                'CGD':  -2 * float(cgd(th)),
                'KiDS Pk': kids_pure_chi2(th)}

    print(f'\n  PURE chi2 (and chi2/dof) per component:')
    print(f'  {"point":18s} | {"GSMF (7)":>16s} | {"CGD (8)":>16s} | {"KiDS Pk (28)":>16s}')
    print('  ' + '-' * 76)
    rows = {}
    for name, th in BESTFIT.items():
        c = chi2s(th); rows[name] = c
        print(f'  {name:18s} | {c["GSMF"]:8.1f} ({c["GSMF"]/n["GSMF"]:4.1f}) | '
              f'{c["CGD"]:8.1f} ({c["CGD"]/n["CGD"]:4.1f}) | '
              f'{c["KiDS Pk"]:8.1f} ({c["KiDS Pk"]/n["KiDS Pk"]:4.1f})')

    real = ['Pk_7p (marg)', 'Pk_5p (cosmo fix)', 'Pk_2cosmo@A']
    print('\n  DISCRIMINATION = chi2 spread among the 3 real chains (excl. fiducial):')
    for comp in ('GSMF', 'CGD', 'KiDS Pk'):
        vals = [rows[k][comp] for k in real]
        d = max(vals) - min(vals)
        print(f'    {comp:10s}: dchi2 = {d:7.1f}  -> '
              f'{"DRIVES the fit" if d > 10 else ("mild" if d > 2 else "nearly FLAT (no constraint)")}')
    print('\n  (large chi2 spread vs fiducial just means fiducial hydro is a bad GSMF/CGD fit;')
    print('   the spread among the 3 real chains is what shapes the posterior.)')


if __name__ == '__main__':
    part1_A_consistency()
    part2_components()
