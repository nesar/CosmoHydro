"""Fixed-hydro scan, WIDE-PRIOR variant: same scan points A-D (and the
fiducial fix), but with a FLAT cosmology prior spanning the full experimental
design box (omega_m in [0.12, 0.155], sigma_8 in [0.70, 0.90]) instead of the
Planck-width Gaussians. Shows what the DATA alone (GSMF+CGD) say about
cosmology at each fixed subgrid point.

The hydro-MARGINALIZED 7p chain is deliberately kept as-is (its original run
and prior) as the reference — only the fixed-hydro chains are re-run wide.

This is a SEPARATE plot from check_fixed_hydro_scan.py, not a replacement.
Reuses that module's loader and conventions.

Outputs (this directory):
  fixed_hydro_scan_wide.png, fixed_hydro_scan_wide_summary.txt
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import plots as gd_plots
from getdist import MCSamples

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check_fixed_hydro_scan as base          # reuse loader + conventions

SUITE = base.SUITE
NAMES, LABELS, FID = base.NAMES, base.LABELS, base.FID
# full design box (the flat-prior support), with a hair of margin for drawing
AXIS = {'omega_m': (0.120, 0.155), 'sigma_8': (0.70, 0.90)}
DESIGN_BOX = {'omega_m': (0.12, 0.155), 'sigma_8': (0.70, 0.90)}

# (trial suffix, cols, label, color, linewidth). The marginalized reference is
# resolved in main(): prefer _7p_wide (FLAT cosmo prior, matches the fixed runs);
# fall back to _7p_pk (Planck prior) with a warning label until _7p_wide finishes.
MARG_WIDE = ('_7p_wide', (5, 6), 'hydro marginalized (7p, flat prior)', 'tab:red', 2.2)
MARG_PK   = ('_7p_pk',   (5, 6), 'hydro marginalized (7p, PLANCK prior!)', 'tab:red', 2.2)
SPECS = [
    MARG_WIDE,   # replaced with MARG_PK in main() if _7p_wide isn't on disk yet
    ('_2cosmo_wide',       (0, 1), 'fixed @ Frontier-E fiducial (~22$\\sigma$)', 'k',      2.2),
    ('_2cosmo_hydA_wide',  (0, 1), 'fixed @ A (7p peak, 0$\\sigma$)',    'tab:blue',   1.6),
    ('_2cosmo_hydB_wide',  (0, 1), 'fixed @ B (1$\\sigma$)',             'tab:green',  1.6),
    ('_2cosmo_hydC_wide',  (0, 1), 'fixed @ C (2$\\sigma$)',             'tab:purple', 1.6),
    ('_2cosmo_hydD_wide',  (0, 1), 'fixed @ D (2$\\sigma$)',             'tab:orange', 1.6),
]


def main():
    print('=== fixed-hydro scan (WIDE flat prior over design box) ===')
    # Consistent reference: use the flat-prior 7p run if it exists, else fall back
    # to the Planck-prior 7p (flagged) so the plot still renders while 7p_wide runs.
    specs = list(SPECS)
    if not os.path.exists(os.path.join(base.RES, f'samples_{SUITE}_7p_wide.npy')):
        print('  (7p_wide not on disk yet -> using 7p_pk Planck reference for now)')
        specs[0] = MARG_PK
    loaded = []
    for suf, cols, label, color, lw in specs:
        trial = f'{SUITE}{suf}'
        r = base._load(trial, cols)
        if r is None:
            print(f'  MISSING  {trial}')
            continue
        sub, ranges = r
        print(f'  LOADED   {trial:34s} median=({np.median(sub[:,0]):.4f}, '
              f'{np.median(sub[:,1]):.4f})')
        loaded.append(dict(trial=trial, label=label, color=color, lw=lw,
                           samples=sub,
                           mc=MCSamples(samples=sub, names=NAMES, labels=LABELS,
                                        label=label, ranges=ranges)))
    if not loaded:
        print('  no chains yet — nothing to plot')
        return

    g = gd_plots.get_subplot_plotter(width_inch=8)
    g.settings.alpha_filled_add = 0.35
    g.settings.legend_fontsize = 10
    g.settings.axes_fontsize = 11
    g.settings.lab_fontsize = 14
    g.settings.num_plot_contours = 2
    g.triangle_plot([c['mc'] for c in loaded], NAMES, filled=True,
                    contour_colors=[c['color'] for c in loaded],
                    line_args=[{'color': c['color'], 'lw': c['lw']} for c in loaded],
                    param_limits=AXIS, legend_loc='upper right')

    ax_om, ax_s8, ax_2d = g.subplots[0, 0], g.subplots[1, 1], g.subplots[1, 0]
    # flat prior: a horizontal dashed line across the design box (constant density)
    # plus the box edges marking its support.
    for ax, key in ((ax_om, 'omega_m'), (ax_s8, 'sigma_8')):
        ax.axvline(FID[key], color='0.4', ls=':', lw=1.0)
        lo, hi = DESIGN_BOX[key]
        top = ax.get_ylim()[1]
        ax.plot([lo, hi], [0.5 * top, 0.5 * top], color='k', ls='--', lw=1.3,
                alpha=0.7)                       # flat prior (uniform) density
        for edge in (lo, hi):
            ax.axvline(edge, color='k', ls='--', lw=1.0, alpha=0.4)
    ax_2d.axvline(FID['omega_m'], color='0.4', ls=':', lw=1.0)
    ax_2d.axhline(FID['sigma_8'], color='0.4', ls=':', lw=1.0)
    ax_2d.plot([FID['omega_m']], [FID['sigma_8']], marker='*', ms=13, mfc='gold',
               mec='k', mew=0.8, ls='', zorder=20)

    g.fig.suptitle(
        'Fixed-hydro scan, WIDE prior: 2p cosmology posterior with a FLAT prior '
        'over the full design box\n(scan points A-D as before; dashed = design-box '
        'edge = prior support, star = fiducial cosmology)', y=1.06, fontsize=11)
    png = os.path.join(HERE, 'fixed_hydro_scan_wide.png')
    g.export(png)
    print(f'  wrote {png}')

    txt = os.path.join(HERE, 'fixed_hydro_scan_wide_summary.txt')
    marg = next((c for c in loaded if c['trial'].endswith('_7p_pk')), None)
    with open(txt, 'w') as f:
        f.write('Fixed-hydro scan (WIDE flat prior over the design box)\n')
        f.write(f'suite: {SUITE}; 7p reference uses its original prior\n\n')
        f.write(f'{"chain":40s}  {"omega_m":>18s}  {"sigma_8":>18s}')
        if marg is not None:
            f.write(f'  {"d(sigma_8) vs marg":>19s}')
        f.write('\n')
        for c in loaded:
            s = c['samples']
            row = (f'{c["trial"]:40s}  {np.median(s[:,0]):.5f}+/-{s[:,0].std():.5f}  '
                   f'{np.median(s[:,1]):.5f}+/-{s[:,1].std():.5f}')
            if marg is not None:
                d = np.median(s[:, 1]) - np.median(marg['samples'][:, 1])
                row += f'  {d:+19.5f}'
            f.write(row + '\n')
    print(f'  wrote {txt}')


if __name__ == '__main__':
    main()
