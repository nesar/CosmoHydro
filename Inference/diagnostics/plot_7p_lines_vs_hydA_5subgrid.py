"""Variant of plot_7p_vs_2cosmo_5subgrid_GSMF_CGD_planck.png with two changes:

  1. the 2-cosmology overlay (originally hydro fixed at the Frontier-E fiducial)
     is REPLACED by the fixed-@-A run -- hydro pinned at the 7p posterior peak
     (point A of the fixed-hydro scan);
  2. the 7p (hydro+cosmo marginalized) posterior is drawn as UNFILLED red outline
     lines instead of filled contours (same style as the *_clean scan plot), so
     it frames the two filled overlays without masking them.

Reuses the helpers in Inference/plot_7p_vs_2cosmo_5subgrid.py.

Output: diagnostics/gsmf_cgd_7p_vs_2cosmoA_5subgrid_planck.png
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INFER = os.path.dirname(HERE)
sys.path.insert(0, INFER)
import plot_7p_vs_2cosmo_5subgrid as P          # reuse loaders + prior/crosshair helpers


def main():
    # (trial, color, filled, linewidth, label)
    specs = [
        ('GSMF_CGD_7p_planck',        '#d62728', False, 2.6,
         '7 params (GSMF+CGD, Planck prior, cosmo + hydro free)'),
        ('GSMF_CGD_5p_fid_cosmo', '#2ca02c', True, 1.5,
         '5 subgrid only (GSMF+CGD, cosmology fixed at fiducial)'),
        ('GSMF_CGD_2cosmo_hydA',  '#1f77b4', True, 1.5,
         '2 cosmo only (GSMF+CGD, hydro fixed at A = 7p peak)'),
    ]
    avail = []
    for trial, color, filled, lw, label in specs:
        l = P.load(trial)
        if l is None:
            print(f'  skip (missing): {trial}')
            continue
        avail.append((l, color, filled, lw, label))
    if not avail:
        print('  no chains — nothing to plot')
        return

    # anchor the 7-param space on the richest available chain (the 7p run)
    anchor = max(avail, key=lambda t: len(t[0][1]))
    names, ranges = anchor[0][1], anchor[0][2]

    mcs    = [P.mcsamples(l, label) for (l, _, _, _, label) in avail]
    colors = [c for (_, c, _, _, _) in avail]
    filled = [f for (_, _, f, _, _) in avail]
    lws    = [w for (_, _, _, w, _) in avail]
    labels = [lab for (_, _, _, _, lab) in avail]

    g = P._plotter()
    g.triangle_plot(mcs, params=names, filled=filled, legend_labels=labels,
                    param_limits=P.axis_limits([l for (l, _, _, _, _) in avail], names),
                    contour_colors=colors,
                    line_args=[{'color': c, 'lw': w} for c, w in zip(colors, lws)])
    P.fiducial_crosshairs(g, names)
    P.overlay_priors(g, names, ranges, P.PK_PRIOR)
    plt.suptitle('GSMF+CGD, Planck prior — 7p (outline) vs 2-cosmology @ A vs '
                 '5-subgrid MCMC (dashed = prior)', y=1.005, fontsize=14)

    out = os.path.join(HERE, 'gsmf_cgd_7p_vs_2cosmoA_5subgrid_planck.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
