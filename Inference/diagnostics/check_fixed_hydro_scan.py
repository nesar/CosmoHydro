"""Fixed-hydro scan: how much does the 2p cosmology posterior move when the
hydro parameters are pinned at different (but all reasonable) points?

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
the marginalized 7p result and on the fiducial-fixed run.

Chains (skipped if absent):
  <suite>_7p_pk               hydro marginalized              (cosmo cols 5,6)
  <suite>_2cosmo_pk           hydro fixed at project fiducial (cols 0,1)
  <suite>_2cosmo_hyd{A..D}    hydro fixed at scan points      (cols 0,1)

Outputs (this directory):
  fixed_hydro_scan.png, fixed_hydro_scan_summary.txt
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
PK_SIGMA = {'omega_m': 0.0011, 'sigma_8': 0.006}
AXIS = {'omega_m': (0.132, 0.152), 'sigma_8': (0.755, 0.845)}

# (trial suffix, cols, label, color, linewidth)
SPECS = [
    ('_7p_pk',        (5, 6), 'hydro marginalized (7p)',                  'tab:red', 2.2),
    ('_2cosmo_pk',    (0, 1), 'fixed @ Frontier-E fiducial (~22$\\sigma$)', 'k',     2.2),
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


def main():
    print('=== fixed-hydro scan ===')
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
    for ax, key in ((ax_om, 'omega_m'), (ax_s8, 'sigma_8')):
        ax.axvline(FID[key], color='0.4', ls=':', lw=1.0)
        lo, hi = AXIS[key]
        x = np.linspace(lo, hi, 600)
        y = np.exp(-0.5 * ((x - FID[key]) / PK_SIGMA[key]) ** 2)
        ax.plot(x, y / y.max() * ax.get_ylim()[1], color='k', ls='--', lw=1.2,
                alpha=0.6)
    ax_2d.axvline(FID['omega_m'], color='0.4', ls=':', lw=1.0)
    ax_2d.axhline(FID['sigma_8'], color='0.4', ls=':', lw=1.0)
    ax_2d.plot([FID['omega_m']], [FID['sigma_8']], marker='*', ms=13, mfc='gold',
               mec='k', mew=0.8, ls='', zorder=20)

    g.fig.suptitle(
        'Fixed-hydro scan: 2p cosmology posterior vs. choice of fixed subgrid point\n'
        '(scan points A-D drawn from the 7p posterior; dashed = cosmology prior, '
        'star = fiducial cosmology)', y=1.06, fontsize=11)
    png = os.path.join(OUT, 'fixed_hydro_scan.png')
    g.export(png)
    print(f'  wrote {png}')

    txt = os.path.join(OUT, 'fixed_hydro_scan_summary.txt')
    marg = next((c for c in loaded if c['trial'].endswith('_7p_pk')), None)
    with open(txt, 'w') as f:
        f.write('Fixed-hydro scan — 2p cosmology posterior vs fixed subgrid point\n')
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
    print(f'  wrote {txt}')


if __name__ == '__main__':
    main()
