#!/usr/bin/env python
"""
Overlay plots comparing MCMC posteriors that share a forward model.

For each observable suite (GSMF, GSMF+CGD) it builds a triangle in the
7-parameter space with two reduced-dimension posteriors overlaid:
  - the 2-cosmology-parameter (hydro fixed) posterior on the cosmology panels,
  - the 5-subgrid-parameter (cosmology fixed) posterior on the subgrid panels.

It also makes a 7-parameter comparison of GSMF vs GSMF+CGD, and the same set for
the Planck-prior (`*_pk`) runs.

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
# PK = Planck-width Gaussian (no hard cut; bounded only by the design box).
# Matches the *_pk configs. (The hard-truncated *_trunc runs were retired to
# old_results/retired_trunc/ — the wall amputated the real posterior.)
PK_PRIOR = {'omega_m': (FID_OM, 0.0011, 0.12, 0.155),
            'sigma_8': (FID_S8, 0.006,  0.70, 0.90)}

# Full design-box (valid emulator) range for the cosmology params. Used to pad
# the axis a bit OUTSIDE a tight prior window so the hard wall is visible (the
# chain's KDE is still clipped at its own range, so nothing leaks).
COSMO_VALID = {'omega_m': (0.12, 0.155), 'sigma_8': (0.70, 0.90)}
AXIS_PAD_FRAC = 0.3   # pad each side by this fraction of the cosmo data span

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
    """Dotted crosshairs at fiducial cosmology on the (omega_m, sigma_8) panels.
    Identifies cosmo params by name, so it works for any param subset."""
    fidval = {'omega_m': FID_OM, 'sigma_8': FID_S8}
    for i, ni in enumerate(names):
        for j in range(i + 1):
            nj = names[j]
            ax = g.subplots[i, j]
            if ax is None:
                continue
            ki, kj = _cosmo_kind(ni), _cosmo_kind(nj)
            if i == j:
                if ki in fidval:
                    ax.axvline(fidval[ki], color='k', lw=1.0, ls=':')
            else:
                if kj in fidval:
                    ax.axvline(fidval[kj], color='k', lw=1.0, ls=':')
                if ki in fidval:
                    ax.axhline(fidval[ki], color='k', lw=1.0, ls=':')


def overlay_priors(g, names, ranges, cosmo_prior):
    """Dashed prior curve (peak-normalized) on each 1D diagonal panel.

    cosmo_prior : dict {'omega_m': (mu, sigma, lo, hi), 'sigma_8': (...)} giving
    the cosmology Gaussian and its truncation window (MODERATE_PRIOR for the
    moderate runs, PK_PRIOR for the *_pk Planck-prior runs).
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


def axis_limits(loaded, names):
    """Axis limits per parameter. Subgrid params use the chain range. Cosmology
    params are auto-zoomed to where the posteriors actually live (data 0.3-99.7
    percentile across all chains, fiducial always included, padded), clamped to
    the design box. This shows the railing moderate runs over the full box AND
    the tight Planck (_pk) runs zoomed near fiducial, without manual tuning."""
    anchor = max(loaded, key=lambda l: len(l[1]))
    ranges = anchor[2]
    lim = {}
    for nm in names:
        kind = _cosmo_kind(nm)
        if kind in COSMO_VALID:
            vals = [l[0][:, l[1].index(nm)] for l in loaded if nm in l[1]]
            allv = np.concatenate(vals)
            lo, hi = np.percentile(allv, 0.3), np.percentile(allv, 99.7)
            fid = FID_OM if kind == 'omega_m' else FID_S8
            lo, hi = min(lo, fid), max(hi, fid)
            pad = (hi - lo) * AXIS_PAD_FRAC + 1e-6
            vlo, vhi = COSMO_VALID[kind]
            lim[nm] = (max(vlo, lo - pad), min(vhi, hi + pad))
        else:
            lim[nm] = ranges[nm]
    return lim


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
    """Triangle with 7p, 2cosmo (cosmo panels) and 5subgrid (subgrid panels)
    overlaid. The param-space is anchored on the richest AVAILABLE chain, so it
    still renders if the 7p chain isn't done yet (e.g. only 2cosmo present).
    Each chain keeps its own range, so getdist clips truncated chains at their
    hard wall instead of smearing the KDE past it."""
    print(f"\n=== triangle: {obs_label} ===")
    specs = [(t7p, '#1f77b4', f'7 params ({obs_label}, cosmo + hydro free)'),
             (t5p, '#2ca02c', f'5 subgrid only ({obs_label}, cosmology fixed at fiducial)'),
             (t2c, '#d62728', f'2 cosmo only ({obs_label}, hydro fixed at fiducial)')]
    avail = []
    for trial, color, label in specs:
        if not trial:
            continue
        l = load(trial)
        if l is not None:
            avail.append((l, color, label))
    if not avail:
        print(f"  no chains for {obs_label} — skipping this triangle")
        return

    # Anchor the triangle param-space on the chain with the most parameters.
    anchor = max(avail, key=lambda t: len(t[0][1]))
    names, ranges = anchor[0][1], anchor[0][2]
    mcs    = [mcsamples(l, label) for (l, _, label) in avail]
    colors = [color for (_, color, _) in avail]
    labels = [label for (_, _, label) in avail]

    g = _plotter()
    g.triangle_plot(mcs, params=names, filled=True, legend_labels=labels,
                    param_limits=axis_limits([l for (l, _, _) in avail], names),
                    contour_colors=colors)
    fiducial_crosshairs(g, names)
    overlay_priors(g, names, ranges, cosmo_prior)
    plt.suptitle(f'{obs_label} — 7-param vs 2-cosmology vs 5-subgrid MCMC '
                 '(dashed = prior)', y=1.005, fontsize=14)
    plt.savefig(output, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  saved: {output}")


def make_7p_comparison(suites, output, cosmo_prior=MODERATE_PRIOR):
    """Overlay the 7-parameter posteriors of several suites on one triangle.
    suites = [(trial_name, label, color), ...]."""
    print("\n=== 7p comparison ===")
    mcs, colors, labels, loaded, names_ref, ranges_ref = [], [], [], [], None, None
    for trial, label, color in suites:
        l = load(trial)
        if l is None:
            continue
        mcs.append(mcsamples(l, label)); colors.append(color); labels.append(label)
        loaded.append(l)
        if names_ref is None:
            names_ref, ranges_ref = l[1], l[2]
    if len(mcs) < 2:
        print("  fewer than 2 7p chains available — skipping comparison")
        return
    g = _plotter()
    g.triangle_plot(mcs, params=names_ref, filled=True, legend_labels=labels,
                    param_limits=axis_limits(loaded, names_ref), contour_colors=colors)
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

    # --- Planck-prior runs (Planck-width Gaussian, no hard cut) ---
    make_triangle('GSMF, Planck prior', 'GSMF_7p_pk', 'GSMF_2cosmo_pk', None,
                  os.path.join(RESULTS, 'plot_7p_vs_2cosmo_5subgrid_pk.png'),
                  cosmo_prior=PK_PRIOR)
    make_triangle('GSMF+CGD, Planck prior', 'GSMF_CGD_7p_pk', 'GSMF_CGD_2cosmo_pk', None,
                  os.path.join(RESULTS, 'plot_7p_vs_2cosmo_5subgrid_GSMF_CGD_pk.png'),
                  cosmo_prior=PK_PRIOR)
    make_7p_comparison(
        [('GSMF_7p_pk', 'GSMF (7p, Planck)', '#1f77b4'),
         ('GSMF_CGD_7p_pk', 'GSMF+CGD (7p, Planck)', '#ff7f0e')],
        os.path.join(RESULTS, 'plot_7p_GSMF_vs_GSMF_CGD_pk.png'),
        cosmo_prior=PK_PRIOR)


if __name__ == '__main__':
    main()
