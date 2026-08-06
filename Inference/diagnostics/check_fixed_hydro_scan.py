"""Fixed-hydro scan (PLANCK cosmology prior): how much does the 2p cosmology
posterior move when the hydro parameters are pinned at different (but all
reasonable) points?

Motivation
----------
The standard "hydro fixed" run pins the subgrid parameters at the project /
Frontier-E fiducial (3, 0.5, 0.8, 0.51, 0.13). That point is NOT the peak of the
marginalized 7p posterior -- it sits ~22 sigma away (see
diagnostics/select_fixed_hydro_points.py and fixed_hydro_scan_points.txt). If the
2p cosmology posterior is sensitive to the hydro choice, then the tight cosmology
"constraint" from a fixed-hydro run is an artefact of that choice, not a
measurement.

This scan pins hydro at 4 points drawn from the 7p posterior itself (Mahalanobis
radius ~0, 1, 2, 2 sigma) and overlays the resulting 2p cosmology posteriors on
the marginalized 7p result and on the fiducial-fixed run. Every chain here uses
the SAME Planck-width cosmology prior (drawn as a dotted curve).

Chains (skipped if absent):
  GSMF_CGD_7p_planck             hydro marginalized              (cosmo cols 5,6)
  GSMF_CGD_2cosmo_planck         hydro fixed at Frontier-E fid.  (cols 0,1)
  GSMF_CGD_2cosmo_hyd{A..D}  hydro fixed at scan points      (cols 0,1)

Outputs (this directory):
  gsmf_cgd_fixed_hydro_scan_planck.png          (all chains)
  gsmf_cgd_fixed_hydro_scan_planck_noFE.png     (without the Frontier-E outlier)
  gsmf_cgd_fixed_hydro_scan_planck_summary.txt
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import plots as gd_plots
from getdist import MCSamples

HERE = os.path.dirname(os.path.abspath(__file__))
INFER = os.path.dirname(HERE)
RES = os.path.join(INFER, 'results')
OUT = HERE

SUITE = 'GSMF_CGD'
NAMES = ['omega_m', 'sigma_8']
LABELS = [r'\omega_m \equiv \Omega_m h^2', r'\sigma_8']
FID = {'omega_m': 0.14176, 'sigma_8': 0.8102}
PLANCK_SIGMA = {'omega_m': 0.0011, 'sigma_8': 0.006}
AXIS = {'omega_m': (0.132, 0.152), 'sigma_8': (0.755, 0.845)}

# The Frontier-E fixed run is the outlier that a "no-FE" version drops.
FRONTIER_E = '_2cosmo_planck'

# (trial suffix, cols, label, color, linewidth)
SPECS = [
    ('_7p_planck',        (5, 6), 'hydro marginalized (7p)',                  'tab:red', 2.2),
    (FRONTIER_E,      (0, 1), 'fixed @ Frontier-E fiducial (~22$\\sigma$)', 'k',     2.2),
    ('_2cosmo_hydA',  (0, 1), 'fixed @ A (7p peak, 0$\\sigma$)',     'tab:blue',   1.6),
    ('_2cosmo_hydB',  (0, 1), 'fixed @ B (1$\\sigma$)',              'tab:green',  1.6),
    ('_2cosmo_hydC',  (0, 1), 'fixed @ C (2$\\sigma$)',              'tab:purple', 1.6),
    ('_2cosmo_hydD',  (0, 1), 'fixed @ D (2$\\sigma$)',              'tab:orange', 1.6),
]


def _load(trial, cols):
    sp = os.path.join(RES, f'samples_{trial}.npy')
    pp = os.path.join(RES, f'params_list_{trial}.npy')
    if not (os.path.exists(sp) and os.path.exists(pp)):
        return None
    arr = np.load(sp)
    pl = np.load(pp, allow_pickle=True).tolist()
    if arr.ndim != 2 or arr.shape[1] <= max(cols):
        return None
    sub = np.column_stack([arr[:, cols[0]], arr[:, cols[1]]])
    ranges = {'omega_m': (float(pl[cols[0]][2]), float(pl[cols[0]][3])),
              'sigma_8': (float(pl[cols[1]][2]), float(pl[cols[1]][3]))}
    return sub, ranges


def _auto_axis(loaded, pad=0.3):
    """Axis from the included chains (0.3-99.7 pct), fiducial always in view."""
    ax = {}
    for key, i in (('omega_m', 0), ('sigma_8', 1)):
        vals = np.concatenate([c['samples'][:, i] for c in loaded])
        lo, hi = np.percentile(vals, 0.3), np.percentile(vals, 99.7)
        lo, hi = min(lo, FID[key]), max(hi, FID[key])
        p = (hi - lo) * pad + 1e-6
        ax[key] = (lo - p, hi + p)
    return ax


def _draw_prior(ax, key, axis):
    """Dotted Planck-Gaussian prior on a 1D diagonal (peak-normalized)."""
    lo, hi = axis[key]
    x = np.linspace(lo, hi, 600)
    y = np.exp(-0.5 * ((x - FID[key]) / PLANCK_SIGMA[key]) ** 2)
    ax.plot(x, y / y.max() * ax.get_ylim()[1], color='k', ls=':', lw=1.8,
            alpha=0.85)


def make_plot(loaded, out_png, title, axis, alpha_filled=0.35, filled=True):
    g = gd_plots.get_subplot_plotter(width_inch=8)
    g.settings.alpha_filled_add = alpha_filled
    g.settings.legend_fontsize = 11
    g.settings.axes_fontsize = 11
    g.settings.lab_fontsize = 14
    g.settings.num_plot_contours = 2
    g.triangle_plot([c['mc'] for c in loaded], NAMES, filled=filled,
                    contour_colors=[c['color'] for c in loaded],
                    line_args=[{'color': c['color'], 'lw': c['lw']} for c in loaded],
                    param_limits=axis, legend_loc='upper right')

    ax_om, ax_s8, ax_2d = g.subplots[0, 0], g.subplots[1, 1], g.subplots[1, 0]
    for ax, key in ((ax_om, 'omega_m'), (ax_s8, 'sigma_8')):
        ax.axvline(FID[key], color='0.5', ls=(0, (1, 3)), lw=1.0)   # fiducial (fine dots)
        _draw_prior(ax, key, axis)                                   # prior (dotted)
    ax_2d.axvline(FID['omega_m'], color='0.5', ls=(0, (1, 3)), lw=1.0)
    ax_2d.axhline(FID['sigma_8'], color='0.5', ls=(0, (1, 3)), lw=1.0)
    ax_2d.plot([FID['omega_m']], [FID['sigma_8']], marker='*', ms=13, mfc='gold',
               mec='k', mew=0.8, ls='', zorder=20)

    g.fig.suptitle(title, y=1.06, fontsize=11)
    g.export(out_png)
    print(f'  wrote {out_png}')


def write_summary(loaded, out_txt):
    marg = next((c for c in loaded if c['trial'].endswith('_7p_planck')), None)
    with open(out_txt, 'w') as f:
        f.write('Fixed-hydro scan (Planck prior) — 2p cosmology vs fixed subgrid point\n')
        f.write(f'suite: {SUITE}\n\n')
        f.write(f'{"chain":34s}  {"omega_m":>18s}  {"sigma_8":>18s}')
        if marg is not None:
            f.write(f'  {"d(sigma_8) vs marg":>19s}')
        f.write('\n')
        for c in loaded:
            s = c['samples']
            row = (f'{c["trial"]:34s}  {np.median(s[:,0]):.5f}+/-{s[:,0].std():.5f}  '
                   f'{np.median(s[:,1]):.5f}+/-{s[:,1].std():.5f}')
            if marg is not None:
                d = np.median(s[:, 1]) - np.median(marg['samples'][:, 1])
                row += f'  {d:+19.5f}'
            f.write(row + '\n')
    print(f'  wrote {out_txt}')


def main():
    print('=== fixed-hydro scan (Planck prior) ===')
    loaded = []
    for suf, cols, label, color, lw in SPECS:
        trial = f'{SUITE}{suf}'
        r = _load(trial, cols)
        if r is None:
            print(f'  MISSING  {trial}')
            continue
        sub, ranges = r
        print(f'  LOADED   {trial:28s} median=({np.median(sub[:,0]):.4f}, '
              f'{np.median(sub[:,1]):.4f})')
        loaded.append(dict(trial=trial, suf=suf, label=label, color=color, lw=lw,
                           samples=sub,
                           mc=MCSamples(samples=sub, names=NAMES, labels=LABELS,
                                        label=label, ranges=ranges)))
    if not loaded:
        print('  no chains yet — nothing to plot')
        return

    base = 'gsmf_cgd_fixed_hydro_scan_planck'
    title_full = ('GSMF+CGD fixed-hydro scan (Planck prior): 2p cosmology posterior '
                  'vs. fixed subgrid point\n(scan points A-D drawn from the 7p '
                  'posterior; dotted = cosmology prior, star = fiducial)')
    make_plot(loaded, os.path.join(OUT, f'{base}.png'), title_full, AXIS)
    write_summary(loaded, os.path.join(OUT, f'{base}_summary.txt'))

    # version without the Frontier-E outlier — auto-zoomed to the remaining chains
    no_fe = [c for c in loaded if c['suf'] != FRONTIER_E]
    if len(no_fe) < len(loaded):
        title_nofe = ('GSMF+CGD fixed-hydro scan (Planck prior), Frontier-E omitted: '
                      'marginalized vs. reasonable fixed points A-D\n'
                      '(dotted = cosmology prior, star = fiducial)')
        make_plot(no_fe, os.path.join(OUT, f'{base}_noFE.png'), title_nofe,
                  _auto_axis(no_fe))

    # minimal "clean" version: only marginalized + A + D. The marginalized chain
    # (broadest) is drawn UNFILLED (thick outline) so the filled A and D contours
    # on top stay clearly visible; high-contrast red / blue / green.
    CLEAN_KEEP = {'_7p_planck': ('#d62728', 2.8),        # marginalized -> red (outline)
                  '_2cosmo_hydA': ('#1f77b4', 2.4),    # A (7p peak) -> blue (filled)
                  '_2cosmo_hydD': ('#2ca02c', 2.4)}    # D (2 sigma) -> green (filled)
    order = ['_7p_planck', '_2cosmo_hydA', '_2cosmo_hydD']   # marg first = bottom layer
    by_suf = {c['suf']: c for c in loaded}
    clean, filled_flags = [], []
    for suf in order:
        if suf in by_suf:
            col, lw = CLEAN_KEEP[suf]
            clean.append({**by_suf[suf], 'color': col, 'lw': lw})
            filled_flags.append(suf != '_7p_planck')         # marginalized unfilled
    if len(clean) >= 2:
        title_clean = ('GSMF+CGD fixed-hydro scan (Planck prior): marginalized (outline) '
                       'vs. fixed @ A (7p peak) and @ D (2$\\sigma$)\n'
                       '(dotted = cosmology prior, star = fiducial)')
        make_plot(clean, os.path.join(OUT, f'{base}_clean.png'), title_clean,
                  _auto_axis(clean), alpha_filled=0.5, filled=filled_flags)


if __name__ == '__main__':
    main()
