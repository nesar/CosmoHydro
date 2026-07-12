"""Planck-prior (*_pk) ONLY: marginalized vs fixed, side by side.

This is the clean, uncluttered version of the marg-vs-fixed story using just the
Planck-width-prior chains (the moderate-prior chains are dropped here — see
check_cosmo_marg_vs_fixed.py / check_subgrid_marg_vs_fixed.py for the full 4-way
overlays). The point is to see the effect of marginalization alone, at fixed
(Planck) prior:

  COSMOLOGY (omega_m, sigma_8):
    - <suite>_7p_pk       hydro MARGINALIZED   (filled red)
    - <suite>_2cosmo_pk   hydro FIXED at fid   (filled blue)
    => how much fixing hydro biases/shrinks the cosmology posterior.

  SUBGRID (5 hydro params):
    - <suite>_7p_pk       cosmology MARGINALIZED (Planck prior)  (filled red)
    - <suite>_5p_fid_cosmo cosmology FIXED at fid                (filled blue)
    => how much fixing cosmology biases/shrinks the subgrid posterior.
    (The 5p chain is prior-independent: cosmology is pinned, so no cosmo prior
     acts. It's the correct "fixed" counterpart for the Planck-prior 7p chain.)

Each chain gets ITS OWN saved range so getdist's KDE is boundary-corrected in the
right place. Run for both observable suites (GSMF, GSMF+CGD).

Outputs (this directory):
  pk_cosmo_marg_vs_fixed{,_GSMF_CGD}.png     (+ _medians.txt)
  pk_subgrid_marg_vs_fixed{,_GSMF_CGD}.png   (+ _medians.txt)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import plots as gd_plots
from getdist import MCSamples

RES = '/home/nramachandra/Projects/Hydro_runs/CosmoHydro/Inference/results'
OUT = os.path.dirname(os.path.abspath(__file__))

FID = {'omega_m': 0.14176, 'sigma_8': 0.8102}
# Planck-width Gaussian used by the *_pk configs (dashed prior overlay).
PK_SIGMA = {'omega_m': 0.0011, 'sigma_8': 0.006}
COSMO_AXIS = {'omega_m': (0.128, 0.152), 'sigma_8': (0.75, 0.86)}

CNAMES = ['omega_m', 'sigma_8']
CLABELS = [r'\omega_m \equiv \Omega_m h^2', r'\sigma_8']

SNAMES = ['kappa_w', 'e_w', 'M_seed_e6', 'v_kin_e4', 'eps_kin_e1']
SLABELS = [r'\kappa_\mathrm{w}', r'e_\mathrm{w}', r'M_\mathrm{seed}/10^{6}',
           r'v_\mathrm{kin}/10^{4}', r'\epsilon_\mathrm{kin}/10^{1}']
SRANGES = {'kappa_w': (2.0, 4.0), 'e_w': (0.2, 1.0), 'M_seed_e6': (0.6, 2.0),
           'v_kin_e4': (0.1, 1.2), 'eps_kin_e1': (0.02, 1.2)}

MARG, FIXED = 'tab:red', 'tab:blue'
SUITES = [('GSMF', 'GSMF', ''), ('GSMF+CGD', 'GSMF_CGD', '_GSMF_CGD')]


def _load(trial, cols, names):
    sp = os.path.join(RES, f'samples_{trial}.npy')
    pp = os.path.join(RES, f'params_list_{trial}.npy')
    if not (os.path.exists(sp) and os.path.exists(pp)):
        print(f'  MISSING  {trial}')
        return None
    arr = np.load(sp)
    pl = np.load(pp, allow_pickle=True).tolist()
    if arr.ndim != 2 or arr.shape[1] <= max(cols):
        print(f'  SKIP     {trial} bad shape {arr.shape}')
        return None
    sub = arr[:, list(cols)]
    ranges = {names[k]: (float(pl[c][2]), float(pl[c][3])) for k, c in enumerate(cols)}
    print(f'  LOADED   {trial:24s} {arr.shape} -> {len(cols)}D')
    return sub, ranges


def _mc(sub, ranges, names, labels, label):
    return MCSamples(samples=sub, names=names, labels=labels, label=label,
                     ranges=ranges)


# --------------------------------------------------------------------------- #
def cosmo_check(obs_label, prefix, suffix):
    print(f'\n=== PK cosmo marg-vs-fixed: {obs_label} ===')
    specs = [(f'{prefix}_7p_pk',     (5, 6), 'hydro marginalized (7p, Planck)', MARG),
             (f'{prefix}_2cosmo_pk', (0, 1), 'hydro fixed at fiducial (2p, Planck)', FIXED)]
    loaded = []
    for trial, cols, label, color in specs:
        r = _load(trial, cols, CNAMES)
        if r is None:
            continue
        sub, ranges = r
        loaded.append(dict(label=label, color=color, samples=sub,
                           mc=_mc(sub, ranges, CNAMES, CLABELS, label)))
    if len(loaded) < 1:
        print('  no chains — skipping')
        return

    g = gd_plots.get_subplot_plotter(width_inch=7)
    g.settings.alpha_filled_add = 0.6
    g.settings.legend_fontsize = 11
    g.settings.axes_fontsize = 11
    g.settings.lab_fontsize = 14
    g.triangle_plot([c['mc'] for c in loaded], CNAMES, filled=True,
                    contour_colors=[c['color'] for c in loaded],
                    param_limits=COSMO_AXIS, legend_loc='upper right')

    ax_om, ax_s8, ax_2d = g.subplots[0, 0], g.subplots[1, 1], g.subplots[1, 0]
    ax_om.axvline(FID['omega_m'], color='k', ls=':', lw=1.0)
    ax_s8.axvline(FID['sigma_8'], color='k', ls=':', lw=1.0)
    ax_2d.axvline(FID['omega_m'], color='k', ls=':', lw=1.0)
    ax_2d.axhline(FID['sigma_8'], color='k', ls=':', lw=1.0)
    ax_2d.plot([FID['omega_m']], [FID['sigma_8']], marker='*', ms=13, mfc='gold',
               mec='k', mew=0.8, ls='', zorder=20)
    # Planck-prior curve on each 1D diagonal (dashed).
    for ax, key in ((ax_om, 'omega_m'), (ax_s8, 'sigma_8')):
        lo, hi = COSMO_AXIS[key]
        x = np.linspace(lo, hi, 600)
        y = np.exp(-0.5 * ((x - FID[key]) / PK_SIGMA[key]) ** 2)
        ax.plot(x, y / y.max() * ax.get_ylim()[1], color='k', ls='--', lw=1.3)

    g.fig.suptitle(f'{obs_label} cosmology, Planck prior: hydro marginalized (red) '
                   'vs fixed (blue)\n(dashed = Planck prior, star/dotted = fiducial)',
                   y=1.04, fontsize=11)
    png = os.path.join(OUT, f'pk_cosmo_marg_vs_fixed{suffix}.png')
    g.export(png)
    print(f'  wrote {png}')
    _write_medians(os.path.join(OUT, f'pk_cosmo_marg_vs_fixed{suffix}_medians.txt'),
                   obs_label, loaded, CNAMES, 'cosmology (omega_m, sigma_8)')


def subgrid_check(obs_label, prefix, suffix):
    print(f'\n=== PK subgrid marg-vs-fixed: {obs_label} ===')
    cols = (0, 1, 2, 3, 4)
    specs = [(f'{prefix}_7p_pk',        'cosmology marginalized (7p, Planck)', MARG),
             (f'{prefix}_5p_fid_cosmo', 'cosmology fixed at fiducial (5p)',    FIXED)]
    loaded = []
    for trial, label, color in specs:
        r = _load(trial, cols, SNAMES)
        if r is None:
            continue
        sub, ranges = r
        # all 5p/7p chains share the same subgrid design box; use the fixed box
        ranges = dict(SRANGES)
        loaded.append(dict(label=label, color=color, samples=sub,
                           mc=_mc(sub, ranges, SNAMES, SLABELS, label)))
    if len(loaded) < 1:
        print('  no chains — skipping')
        return

    g = gd_plots.get_subplot_plotter(width_inch=9)
    g.settings.alpha_filled_add = 0.6
    g.settings.legend_fontsize = 12
    g.settings.axes_fontsize = 11
    g.settings.lab_fontsize = 13
    g.triangle_plot([c['mc'] for c in loaded], SNAMES, filled=True,
                    contour_colors=[c['color'] for c in loaded],
                    legend_loc='upper right')
    # subgrid prior on each diagonal: broad default Gaussian, flat for eps_kin
    for i, nm in enumerate(SNAMES):
        ax = g.subplots[i, i]
        if ax is None:
            continue
        lo, hi = SRANGES[nm]
        x = np.linspace(lo, hi, 400)
        if nm == 'eps_kin_e1':
            y = np.ones_like(x)
        else:
            mu, sig = 0.5 * (lo + hi), 0.5 * (hi - lo)
            y = np.exp(-0.5 * ((x - mu) / sig) ** 2)
        ax.plot(x, y / y.max(), color='k', ls='--', lw=1.2, alpha=0.7)
        ax.set_ylim(0, 1.15)

    g.fig.suptitle(f'{obs_label} subgrid, Planck prior: cosmology marginalized (red) '
                   'vs fixed (blue) — dashed = prior', y=1.02, fontsize=13)
    png = os.path.join(OUT, f'pk_subgrid_marg_vs_fixed{suffix}.png')
    g.export(png)
    print(f'  wrote {png}')
    _write_medians(os.path.join(OUT, f'pk_subgrid_marg_vs_fixed{suffix}_medians.txt'),
                   obs_label, loaded, SNAMES, 'subgrid (scaled units)')


def _write_medians(path, obs_label, loaded, names, header):
    with open(path, 'w') as f:
        f.write(f'Planck-prior marg-vs-fixed on {header} — {obs_label}\n\n')
        f.write(f'{"param":12s}  ' + '  '.join(f'{c["label"]:38s}' for c in loaded) + '\n')
        for i, nm in enumerate(names):
            row = f'  {nm:10s}  '
            for c in loaded:
                s = c['samples'][:, i]
                row += f'{np.median(s):9.4f} +/- {np.std(s):7.4f}   [{s.min():.4f},{s.max():.4f}]   '
            f.write(row.rstrip() + '\n')
    print(f'  wrote {path}')


if __name__ == '__main__':
    for obs_label, prefix, suffix in SUITES:
        cosmo_check(obs_label, prefix, suffix)
        subgrid_check(obs_label, prefix, suffix)
