"""Fixed-hydro scan, WIDE-PRIOR variant: same scan points A-D (and the
fiducial fix), but with a FLAT cosmology prior spanning the full experimental
design box (omega_m in [0.12, 0.155], sigma_8 in [0.70, 0.90]) instead of the
Planck-width Gaussians. Shows what the DATA alone (GSMF+CGD) say about
cosmology at each fixed subgrid point.

The hydro-MARGINALIZED 7p chain is deliberately kept as-is (its original run
and prior) as the reference — only the fixed-hydro chains are re-run wide.

This is a SEPARATE plot from check_fixed_hydro_scan.py, not a replacement.
Reuses that module's loader and conventions. Every chain uses the SAME flat
cosmology prior (marginalized reference = GSMF_CGD_7p_wide), drawn as a dotted
horizontal line.

Outputs (this directory):
  gsmf_cgd_fixed_hydro_scan_wide.png          (all chains)
  gsmf_cgd_fixed_hydro_scan_wide_noFE.png     (without the Frontier-E outlier)
  gsmf_cgd_fixed_hydro_scan_wide_summary.txt
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
# fall back to _7p_planck (Planck prior) with a warning label until _7p_wide finishes.
MARG_WIDE = ('_7p_wide', (5, 6), 'hydro marginalized (7p, flat prior)', 'tab:red', 2.2)
MARG_PK   = ('_7p_planck',   (5, 6), 'hydro marginalized (7p, PLANCK prior!)', 'tab:red', 2.2)
FRONTIER_E = '_2cosmo_wide'                     # the outlier a "no-FE" version drops
SPECS = [
    MARG_WIDE,   # replaced with MARG_PK in main() if _7p_wide isn't on disk yet
    (FRONTIER_E,           (0, 1), 'fixed @ Frontier-E fiducial (~22$\\sigma$)', 'k',      2.2),
    ('_2cosmo_hydA_wide',  (0, 1), 'fixed @ A (7p peak, 0$\\sigma$)',    'tab:blue',   1.6),
    ('_2cosmo_hydB_wide',  (0, 1), 'fixed @ B (1$\\sigma$)',             'tab:green',  1.6),
    ('_2cosmo_hydC_wide',  (0, 1), 'fixed @ C (2$\\sigma$)',             'tab:purple', 1.6),
    ('_2cosmo_hydD_wide',  (0, 1), 'fixed @ D (2$\\sigma$)',             'tab:orange', 1.6),
]


def _draw_flat_prior(ax, key):
    """Dotted flat (uniform) prior: horizontal line across the design box."""
    lo, hi = DESIGN_BOX[key]
    top = ax.get_ylim()[1]
    ax.plot([lo, hi], [0.5 * top, 0.5 * top], color='k', ls=':', lw=1.8,
            alpha=0.85)
    for edge in (lo, hi):
        ax.axvline(edge, color='k', ls=(0, (1, 5)), lw=0.9, alpha=0.35)


def make_plot(loaded, out_png, title, axis):
    g = gd_plots.get_subplot_plotter(width_inch=8)
    g.settings.alpha_filled_add = 0.35
    g.settings.legend_fontsize = 10
    g.settings.axes_fontsize = 11
    g.settings.lab_fontsize = 14
    g.settings.num_plot_contours = 2
    g.triangle_plot([c['mc'] for c in loaded], NAMES, filled=True,
                    contour_colors=[c['color'] for c in loaded],
                    line_args=[{'color': c['color'], 'lw': c['lw']} for c in loaded],
                    param_limits=axis, legend_loc='upper right')

    ax_om, ax_s8, ax_2d = g.subplots[0, 0], g.subplots[1, 1], g.subplots[1, 0]
    for ax, key in ((ax_om, 'omega_m'), (ax_s8, 'sigma_8')):
        ax.axvline(FID[key], color='0.5', ls=(0, (1, 3)), lw=1.0)   # fiducial
        _draw_flat_prior(ax, key)                                    # flat prior (dotted)
    ax_2d.axvline(FID['omega_m'], color='0.5', ls=(0, (1, 3)), lw=1.0)
    ax_2d.axhline(FID['sigma_8'], color='0.5', ls=(0, (1, 3)), lw=1.0)
    ax_2d.plot([FID['omega_m']], [FID['sigma_8']], marker='*', ms=13, mfc='gold',
               mec='k', mew=0.8, ls='', zorder=20)

    g.fig.suptitle(title, y=1.06, fontsize=11)
    g.export(out_png)
    print(f'  wrote {out_png}')


def write_summary(loaded, marg, out_txt):
    with open(out_txt, 'w') as f:
        f.write('Fixed-hydro scan (WIDE flat prior over the design box)\n')
        f.write(f'suite: {SUITE}; marginalized reference = {marg["trial"]}\n')
        f.write('NOTE: under a flat prior the marginalized reference itself rails '
                'toward the box edge,\nso compare d(sigma_8) to the FIDUCIAL '
                f'(sigma_8={FID["sigma_8"]:.4f}) rather than to marg.\n\n')
        f.write(f'{"chain":40s}  {"omega_m":>18s}  {"sigma_8":>18s}'
                f'  {"d(s8) vs fid":>13s}  {"d(s8) vs marg":>14s}\n')
        for c in loaded:
            s = c['samples']
            dfid = np.median(s[:, 1]) - FID['sigma_8']
            dmarg = np.median(s[:, 1]) - np.median(marg['samples'][:, 1])
            f.write(f'{c["trial"]:40s}  {np.median(s[:,0]):.5f}+/-{s[:,0].std():.5f}  '
                    f'{np.median(s[:,1]):.5f}+/-{s[:,1].std():.5f}  '
                    f'{dfid:+13.5f}  {dmarg:+14.5f}\n')
    print(f'  wrote {out_txt}')


def main():
    print('=== fixed-hydro scan (WIDE flat prior over design box) ===')
    # Consistent reference: use the flat-prior 7p run if it exists, else fall back
    # to the Planck-prior 7p (flagged) so the plot still renders while 7p_wide runs.
    specs = list(SPECS)
    if not os.path.exists(os.path.join(base.RES, f'samples_{SUITE}_7p_wide.npy')):
        print('  (7p_wide not on disk yet -> using 7p_planck Planck reference for now)')
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
        loaded.append(dict(trial=trial, suf=suf, label=label, color=color, lw=lw,
                           samples=sub,
                           mc=MCSamples(samples=sub, names=NAMES, labels=LABELS,
                                        label=label, ranges=ranges)))
    if not loaded:
        print('  no chains yet — nothing to plot')
        return
    marg = loaded[0]                              # marginalized reference is first

    base_name = 'gsmf_cgd_fixed_hydro_scan_wide'
    title_full = ('GSMF+CGD fixed-hydro scan (flat prior): 2p cosmology posterior '
                  'with a FLAT cosmology prior over the design box\n'
                  '(scan points A-D; dotted = flat prior, star = fiducial)')
    make_plot(loaded, os.path.join(HERE, f'{base_name}.png'), title_full, AXIS)
    write_summary(loaded, marg, os.path.join(HERE, f'{base_name}_summary.txt'))

    # version without the Frontier-E outlier — auto-zoomed to the remaining chains
    no_fe = [c for c in loaded if c['suf'] != FRONTIER_E]
    if len(no_fe) < len(loaded):
        title_nofe = ('GSMF+CGD fixed-hydro scan (flat prior), Frontier-E omitted: '
                      'marginalized vs. reasonable fixed points A-D\n'
                      '(dotted = flat prior, star = fiducial)')
        make_plot(no_fe, os.path.join(HERE, f'{base_name}_noFE.png'), title_nofe,
                  base._auto_axis(no_fe))


if __name__ == '__main__':
    main()
