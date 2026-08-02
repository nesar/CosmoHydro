"""Companion to the fixed-hydro scan: WHERE do the fixed subgrid points sit in
the full 7-parameter space, relative to the posterior they were drawn from?

Shows a 7x7 corner with three layers:
  - the experimental design (109 Latin-hypercube points)  -> light grey dots
  - the GSMF+CGD 7p posterior (the "posterior used" to pick A-D)  -> filled contours
  - the 5 fixed points marked in every panel:
        A, B, C, D  (chosen from the 7p posterior, ~0/1/2/2 sigma)
        Frontier-E  (the project fiducial, ~22 sigma away)

Each point's cosmology coordinate (omega_m, sigma_8) is that run's recovered
2p-cosmology posterior median (the subgrid coords are the fixed input values).
This makes visible the whole story: A-D sit inside the posterior in every
dimension; Frontier-E is a gross outlier in M_seed and v_kin (and lands at
anomalously low sigma_8).

Output (this directory): fixed_points_design.png
"""
import os
import sys
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import plots as gd_plots
from getdist import MCSamples

HERE = os.path.dirname(os.path.abspath(__file__))
INFER = os.path.dirname(HERE)
RES = os.path.join(INFER, 'results')
CFG = os.path.join(INFER, 'configs')
sys.path.insert(0, os.path.join(INFER, '..', 'codes'))
from cosmo_hydro_emu.load_hacc import PARAM_NAME     # 7 canonical latex labels

SUITE = 'GSMF_CGD'
POST = f'{SUITE}_7p_pk'                               # the posterior A-D came from
NAMES = ['kappa_w', 'e_w', 'M_seed', 'v_kin', 'eps_kin', 'omega_m', 'sigma_8']
# scale factors to convert raw design columns -> scaled params (M/1e6, v/1e4, eps/1e1)
DESIGN_SCALE = np.array([1, 1, 1e-6, 1e-4, 1e-1, 1, 1])

# fixed points: (letter, config trial for subgrid, 2cosmo trial for cosmo median,
#                color, marker)
POINTS = [
    ('Frontier-E', f'{SUITE}_2cosmo_pk',   f'{SUITE}_2cosmo_pk',   'k',          '*'),
    ('A', f'{SUITE}_2cosmo_hydA', f'{SUITE}_2cosmo_hydA', 'tab:blue',   'o'),
    ('B', f'{SUITE}_2cosmo_hydB', f'{SUITE}_2cosmo_hydB', 'tab:green',  's'),
    ('C', f'{SUITE}_2cosmo_hydC', f'{SUITE}_2cosmo_hydC', 'tab:purple', 'D'),
    ('D', f'{SUITE}_2cosmo_hydD', f'{SUITE}_2cosmo_hydD', 'tab:orange', 'P'),
]
SHORT = ['kappa_w', 'e_w', 'M_seed', 'v_kin', 'eps_kin']   # config keys (subgrid)


def _fixed_point(cfg_trial, cosmo_trial):
    """7D coord: subgrid from the config, cosmo = 2cosmo posterior median."""
    with open(os.path.join(CFG, f'{cfg_trial}.yaml')) as f:
        fp = (yaml.safe_load(f) or {}).get('fixed_params', {})
    if fp:                                             # hydA..D / Frontier-E fixed
        sg = [float(fp[k]) for k in SHORT]
    else:                                              # fallback: project fiducial
        sg = [3.0, 0.5, 0.8, 0.51, 0.13]
    cs = np.load(os.path.join(RES, f'samples_{cosmo_trial}.npy'))
    om, s8 = np.median(cs[:, 0]), np.median(cs[:, 1])
    return np.array(sg + [om, s8])


def main():
    # posterior + design
    post = np.load(os.path.join(RES, f'samples_{POST}.npy'))[:, :7]
    pl = np.load(os.path.join(RES, f'params_list_{POST}.npy'),
                 allow_pickle=True).tolist()
    ranges = {NAMES[i]: (float(pl[i][2]), float(pl[i][3])) for i in range(7)}

    design_file = os.path.join(INFER, '..', 'data', 'FinalDesign.txt')
    design = np.loadtxt(design_file, delimiter=',', skiprows=1) * DESIGN_SCALE

    pts = [( *p, _fixed_point(p[1], p[2])) for p in POINTS]
    print('fixed points (7D, scaled):')
    for letter, *_ , coord in pts:
        print(f'  {letter:11s} ' + '  '.join(f'{n}={v:.3f}'
              for n, v in zip(NAMES, coord)))

    mc = MCSamples(samples=post, names=NAMES,
                   labels=[s.strip('$') for s in PARAM_NAME], ranges=ranges,
                   label='GSMF+CGD 7p posterior',
                   settings={'smooth_scale_2D': 3, 'smooth_scale_1D': 3})

    g = gd_plots.get_subplot_plotter(subplot_size=1.4)
    g.settings.axes_fontsize = 9
    g.settings.lab_fontsize = 13
    g.settings.alpha_filled_add = 0.55
    g.settings.num_plot_contours = 2
    g.triangle_plot([mc], NAMES, filled=True, contour_colors=['tab:red'],
                    param_limits=ranges)

    # overlay: design scatter (2D), fixed-point markers (2D) + lines (1D)
    for i in range(7):
        for j in range(i + 1):
            ax = g.subplots[i][j]
            if ax is None:
                continue
            if i == j:                                 # 1D diagonal: vertical lines
                for letter, _c, _t, color, _m, coord in pts:
                    ax.axvline(coord[i], color=color, lw=1.6,
                               ls='--' if letter == 'Frontier-E' else '-',
                               alpha=0.9)
            else:                                      # 2D panel
                ax.scatter(design[:, j], design[:, i], s=6, c='0.75',
                           edgecolors='none', zorder=0)
                for letter, _c, _t, color, marker, coord in pts:
                    ax.scatter([coord[j]], [coord[i]], s=90 if marker == '*' else 55,
                               marker=marker, facecolor=color, edgecolor='k',
                               linewidth=0.8, zorder=10)

    # legend (proxy handles)
    handles = [plt.Line2D([], [], color='tab:red', lw=6, alpha=0.55,
                          label='GSMF+CGD 7p posterior'),
               plt.Line2D([], [], marker='o', ls='', mfc='0.75', mec='none',
                          label='design (109 LHC points)')]
    for letter, _c, _t, color, marker, _coord in pts:
        handles.append(plt.Line2D([], [], marker=marker, ls='', mfc=color,
                                  mec='k', ms=10, label=f'fixed @ {letter}'))
    g.fig.legend(handles=handles, loc='upper right', fontsize=12,
                 bbox_to_anchor=(0.98, 0.98))
    g.fig.suptitle('Where the fixed subgrid points sit in the 7p parameter space\n'
                   '(design = grey; posterior = red; cosmo coord = each run\'s '
                   'recovered median)', y=1.02, fontsize=13)

    png = os.path.join(HERE, 'fixed_points_design.png')
    g.export(png)
    print(f'\nwrote {png}')


if __name__ == '__main__':
    main()
