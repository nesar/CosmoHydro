#!/usr/bin/env python
"""
Overlay plots comparing MCMC posteriors that share a forward model.

For each observable suite (GSMF, GSMF+CGD) it builds a triangle in the
7-parameter space with two reduced-dimension posteriors overlaid:
  - the 2-cosmology-parameter (hydro fixed) posterior on the cosmology panels,
  - the 5-subgrid-parameter (cosmology fixed) posterior on the subgrid panels.

It also makes a 7-parameter comparison of GSMF vs GSMF+CGD, and the same set for
the hard-truncated-prior (`*_trunc`) runs.

The 1D diagonal panels show the prior as a dashed line (peak-normalized):
cosmology = the (truncated) fiducial Gaussian, eps_kin = flat, other subgrid =
the broad default Gaussian (midpoint, sigma = half-range).

Chains are skipped if missing, so this is safe to run while runs are ongoing.

    python plot_7p_vs_2cosmo_5subgrid.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples

RESULTS = os.path.join(os.path.dirname(__file__), 'results')

# Project fiducial cosmology (crosshairs + prior centers).
FID_OM, FID_S8 = 0.14176, 0.8102

# Cosmology prior per run-family, as {param: (mu, sigma, trunc_lo, trunc_hi)}.
# Moderate = configs/_defaults.yaml (Gaussian truncated only by the design box).
MODERATE_PRIOR = {'omega_m': (FID_OM, 0.005, 0.12, 0.155),
                  'sigma_8': (FID_S8, 0.03,  0.70, 0.90)}
# Trunc = Planck-tight Gaussian hard-truncated at fiducial +/- 1 sigma
# (matches the *_trunc configs).
TRUNC_PRIOR = {'omega_m': (FID_OM, 0.0011, 0.14066, 0.14286),
               'sigma_8': (FID_S8, 0.006,  0.8042, 0.8162)}

SETTINGS = {'mult_bias_correction_order': 0.5,
            'smooth_scale_2D': 4, 'smooth_scale_1D': 4}


def load(trial):
    """Load (samples, names, ranges) for a trial_name, or None if absent."""
    s = os.path.join(RESULTS, f'samples_{trial}.npy')
    p = os.path.join(RESULTS, f'params_list_{trial}.npy')
    if not (os.path.exists(s) and os.path.exists(p)):
        print(f"  skip (missing): {trial}")
        return None
    samples = np.load(s)
    params_list = np.load(p, allow_pickle=True).tolist()
    names = [q[0] for q in params_list]
    ranges = {q[0]: (float(q[2]), float(q[3])) for q in params_list}
    print(f"  loaded {trial}: {samples.shape}")
    return samples, names, ranges


def mcsamples(loaded, label):
    s, names, ranges = loaded
    return MCSamples(samples=s, names=names, labels=[n.strip('$') for n in names],
                     label=label, ranges=ranges, settings=SETTINGS)


def _cosmo_kind(name):
    """Return 'omega_m', 'sigma_8', 'eps_kin' (flat), or None for a latex name."""
    if 'Omega' in name:
        return 'omega_m'
    if 'sigma_8' in name:
        return 'sigma_8'
    if 'epsilon' in name:
        return 'eps_kin'
    return None


def fiducial_crosshairs(g, names):
    """Dotted crosshairs at fiducial cosmology on the (omega_m, sigma_8) panels."""
    fid = {names[5]: FID_OM, names[6]: FID_S8}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if j > i:
                continue
            ax = g.subplots[i, j]
            if ax is None:
                continue
            if i == j and ni in fid:
                ax.axvline(fid[ni], color='k', lw=1.0, ls=':')
            elif ni in fid and nj in fid:
                ax.axvline(fid[nj], color='k', lw=1.0, ls=':')
                ax.axhline(fid[ni], color='k', lw=1.0, ls=':')


def overlay_priors(g, names, ranges, cosmo_prior):
    """Dashed prior curve (peak-normalized) on each 1D diagonal panel.

    cosmo_prior : dict {'omega_m': (mu, sigma, lo, hi), 'sigma_8': (...)} giving
    the cosmology Gaussian and its hard-truncation window (MODERATE_PRIOR for the
    moderate runs, TRUNC_PRIOR for the *_trunc runs).
    """
    for i, name in enumerate(names):
        ax = g.subplots[i, i]
        if ax is None:
            continue
        lo, hi = ranges[name]
        x = np.linspace(lo, hi, 500)
        kind = _cosmo_kind(name)
        if kind in ('omega_m', 'sigma_8'):
            mu, sig, tlo, thi = cosmo_prior[kind]
            y = np.exp(-0.5 * ((x - mu) / sig) ** 2)
            y[(x < tlo) | (x > thi)] = 0.0
        elif kind == 'eps_kin':            # flat prior
            y = np.ones_like(x)
        else:                              # broad default Gaussian (midpoint, half-range)
            mu = 0.5 * (lo + hi)
            sig = 0.5 * (hi - lo)
            y = np.exp(-0.5 * ((x - mu) / sig) ** 2)
        if y.max() > 0:
            y = y / y.max()
        # getdist peak-normalizes the 1D posterior to 1.0; draw the prior on the
        # same scale and add headroom so a flat prior (eps_kin) isn't hidden
        # under the top frame.
        ax.plot(x, y, color='k', ls='--', lw=1.2, alpha=0.75)
        ax.set_ylim(0, 1.15)


def _plotter():
    g = plots.get_subplot_plotter(subplot_size=2.0)
    g.settings.axes_fontsize = 12
    g.settings.axes_labelsize = 14
    g.settings.legend_fontsize = 14
    g.settings.alpha_filled_add = 0.6
    g.settings.solid_contour_palefactor = 0.6
    g.settings.num_plot_contours = 2
    return g


def make_triangle(obs_label, t7p, t2c, t5p, output, cosmo_prior=MODERATE_PRIOR):
    """7p triangle with 2cosmo (cosmo panels) and 5subgrid (subgrid panels)
    overlaid. cosmo_prior defaults to the moderate _defaults prior."""
    print(f"\n=== triangle: {obs_label} ===")
    l7 = load(t7p)
    if l7 is None:
        print(f"  no 7p chain ({t7p}) — skipping this triangle")
        return
    l2 = load(t2c) if t2c else None
    l5 = load(t5p) if t5p else None

    names_7p, ranges_7p = l7[1], l7[2]

    mcs, colors, labels = [mcsamples(l7, f'7 params ({obs_label}, cosmo + hydro free)')], ['#1f77b4'], []
    labels.append(mcs[-1].label)
    if l2 is not None:
        mcs.append(mcsamples(l2, f'2 cosmo only ({obs_label}, hydro fixed at fiducial)'))
        colors.append('#d62728'); labels.append(mcs[-1].label)
    if l5 is not None:
        mcs.append(mcsamples(l5, f'5 subgrid only ({obs_label}, cosmology fixed at fiducial)'))
        colors.append('#2ca02c'); labels.append(mcs[-1].label)

    g = _plotter()
    g.triangle_plot(mcs, params=names_7p, filled=True, legend_labels=labels,
                    param_limits=ranges_7p, contour_colors=colors)
    fiducial_crosshairs(g, names_7p)
    overlay_priors(g, names_7p, ranges_7p, cosmo_prior)
    plt.suptitle(f'{obs_label} — 7-param vs 2-cosmology vs 5-subgrid MCMC '
                 '(dashed = prior)', y=1.005, fontsize=14)
    plt.savefig(output, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  saved: {output}")


def make_7p_comparison(suites, output, cosmo_prior=MODERATE_PRIOR):
    """Overlay the 7-parameter posteriors of several suites on one triangle.
    suites = [(trial_name, label, color), ...]."""
    print("\n=== 7p comparison ===")
    mcs, colors, labels, names_ref, ranges_ref = [], [], [], None, None
    for trial, label, color in suites:
        l = load(trial)
        if l is None:
            continue
        mcs.append(mcsamples(l, label)); colors.append(color); labels.append(label)
        if names_ref is None:
            names_ref, ranges_ref = l[1], l[2]
    if len(mcs) < 2:
        print("  fewer than 2 7p chains available — skipping comparison")
        return
    g = _plotter()
    g.triangle_plot(mcs, params=names_ref, filled=True, legend_labels=labels,
                    param_limits=ranges_ref, contour_colors=colors)
    fiducial_crosshairs(g, names_ref)
    overlay_priors(g, names_ref, ranges_ref, cosmo_prior)
    plt.suptitle('7-param MCMC comparison (dashed = prior)', y=1.005, fontsize=14)
    plt.savefig(output, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  saved: {output}")


def main():
    # --- moderate-prior runs (design-box truncation) ---
    make_triangle('GSMF', 'GSMF_7p', 'GSMF_2cosmo', 'GSMF_5p_fid_cosmo',
                  os.path.join(RESULTS, 'plot_7p_vs_2cosmo_5subgrid.png'))
    make_triangle('GSMF+CGD', 'GSMF_CGD_7p', 'GSMF_CGD_2cosmo', 'GSMF_CGD_5p_fid_cosmo',
                  os.path.join(RESULTS, 'plot_7p_vs_2cosmo_5subgrid_GSMF_CGD.png'))
    make_7p_comparison(
        [('GSMF_7p', 'GSMF (7p)', '#1f77b4'),
         ('GSMF_CGD_7p', 'GSMF+CGD (7p)', '#ff7f0e')],
        os.path.join(RESULTS, 'plot_7p_GSMF_vs_GSMF_CGD.png'))

    # --- hard-truncated-prior runs (Planck-tight Gaussian, fiducial +/-1 sigma) ---
    make_triangle('GSMF, Planck prior', 'GSMF_7p_trunc', 'GSMF_2cosmo_trunc', None,
                  os.path.join(RESULTS, 'plot_7p_vs_2cosmo_5subgrid_trunc.png'),
                  cosmo_prior=TRUNC_PRIOR)
    make_triangle('GSMF+CGD, Planck prior', 'GSMF_CGD_7p_trunc', 'GSMF_CGD_2cosmo_trunc', None,
                  os.path.join(RESULTS, 'plot_7p_vs_2cosmo_5subgrid_GSMF_CGD_trunc.png'),
                  cosmo_prior=TRUNC_PRIOR)
    make_7p_comparison(
        [('GSMF_7p_trunc', 'GSMF (7p, Planck)', '#1f77b4'),
         ('GSMF_CGD_7p_trunc', 'GSMF+CGD (7p, Planck)', '#ff7f0e')],
        os.path.join(RESULTS, 'plot_7p_GSMF_vs_GSMF_CGD_trunc.png'),
        cosmo_prior=TRUNC_PRIOR)


if __name__ == '__main__':
    main()
