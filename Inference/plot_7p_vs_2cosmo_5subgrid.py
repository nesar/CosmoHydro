#!/usr/bin/env python
"""
Overlay plots comparing MCMC posteriors that share a forward model.

For each observable suite (GSMF, and GSMF+CGD) it builds a triangle in the
7-parameter space with two reduced-dimension posteriors overlaid:
  - the 2-cosmology-parameter (hydro fixed) posterior on the cosmology panels,
  - the 5-subgrid-parameter (cosmology fixed) posterior on the subgrid panels.

It also makes a 7-parameter comparison of GSMF vs GSMF+CGD (how adding CGD
shifts the joint posterior, especially cosmology).

Chains are skipped if missing, so this is safe to run while runs are ongoing.

    python plot_7p_vs_2cosmo_5subgrid.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples

RESULTS = os.path.join(os.path.dirname(__file__), 'results')

# Project fiducial cosmology (crosshairs on the cosmology panels).
FID_OM, FID_S8 = 0.14176, 0.8102

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


def make_triangle(suite, obs_label, output):
    """7p triangle with 2cosmo (cosmo panels) and 5subgrid (subgrid panels)
    overlaid, for one observable suite. suite is the trial-name prefix, e.g.
    'GSMF' or 'GSMF_CGD'."""
    print(f"\n=== triangle: {obs_label} ({suite}) ===")
    l7 = load(f'{suite}_7p')
    if l7 is None:
        print(f"  no 7p chain for {suite} — skipping this triangle")
        return
    l2 = load(f'{suite}_2cosmo')
    l5 = load(f'{suite}_5p_fid_cosmo')

    names_7p, ranges_7p = l7[1], l7[2]
    mcs, colors, labels = [], [], []
    mcs.append(mcsamples(l7, f'7 params ({obs_label}, cosmo + hydro free)'))
    colors.append('#1f77b4'); labels.append(mcs[-1].label)
    if l2 is not None:
        mcs.append(mcsamples(l2, f'2 cosmo only ({obs_label}, hydro fixed at fiducial)'))
        colors.append('#d62728'); labels.append(mcs[-1].label)
    if l5 is not None:
        mcs.append(mcsamples(l5, f'5 subgrid only ({obs_label}, cosmology fixed at fiducial)'))
        colors.append('#2ca02c'); labels.append(mcs[-1].label)

    g = plots.get_subplot_plotter(subplot_size=2.0)
    g.settings.axes_fontsize = 12
    g.settings.axes_labelsize = 14
    g.settings.legend_fontsize = 14
    g.settings.alpha_filled_add = 0.6
    g.settings.solid_contour_palefactor = 0.6
    g.settings.num_plot_contours = 2

    g.triangle_plot(mcs, params=names_7p, filled=True, legend_labels=labels,
                    param_limits=ranges_7p, contour_colors=colors)
    fiducial_crosshairs(g, names_7p)
    plt.suptitle(f'{obs_label} — 7-param vs 2-cosmology vs 5-subgrid MCMC '
                 '(same forward model)', y=1.005, fontsize=14)
    plt.savefig(output, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  saved: {output}")


def make_7p_comparison(suites, output):
    """Overlay the 7-parameter posteriors of several suites (e.g. GSMF vs
    GSMF+CGD) on one triangle. suites = [(trial_prefix, label, color), ...]."""
    print("\n=== 7p comparison: GSMF vs GSMF+CGD ===")
    mcs, colors, labels, names_ref, ranges_ref = [], [], [], None, None
    for prefix, label, color in suites:
        l = load(f'{prefix}_7p')
        if l is None:
            continue
        mcs.append(mcsamples(l, label)); colors.append(color); labels.append(label)
        if names_ref is None:
            names_ref, ranges_ref = l[1], l[2]
    if len(mcs) < 2:
        print("  fewer than 2 7p chains available — skipping comparison")
        return

    g = plots.get_subplot_plotter(subplot_size=2.0)
    g.settings.axes_fontsize = 12
    g.settings.axes_labelsize = 14
    g.settings.legend_fontsize = 14
    g.settings.alpha_filled_add = 0.6
    g.settings.solid_contour_palefactor = 0.6
    g.settings.num_plot_contours = 2

    g.triangle_plot(mcs, params=names_ref, filled=True, legend_labels=labels,
                    param_limits=ranges_ref, contour_colors=colors)
    fiducial_crosshairs(g, names_ref)
    plt.suptitle('7-param MCMC — GSMF vs GSMF+CGD (how adding CGD shifts the posterior)',
                 y=1.005, fontsize=14)
    plt.savefig(output, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  saved: {output}")


def main():
    make_triangle('GSMF', 'GSMF',
                  os.path.join(RESULTS, 'plot_7p_vs_2cosmo_5subgrid.png'))
    make_triangle('GSMF_CGD', 'GSMF+CGD',
                  os.path.join(RESULTS, 'plot_7p_vs_2cosmo_5subgrid_GSMF_CGD.png'))
    make_7p_comparison(
        [('GSMF', 'GSMF (7p)', '#1f77b4'),
         ('GSMF_CGD', 'GSMF+CGD (7p)', '#ff7f0e')],
        os.path.join(RESULTS, 'plot_7p_GSMF_vs_GSMF_CGD.png'))


if __name__ == '__main__':
    main()
