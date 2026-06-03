"""2D (omega_m, sigma_8) contour overlay: marginalized-hydro vs. fixed-hydro.

What this answers
-----------------
  "If we fix the hydro parameters instead of marginalizing over them,
   how much do the cosmology constraints change?"

Chains compared
---------------
  1. OLD 7p (GSMF+CGD+fGas) marginalized to (omega_m, sigma_8)
       old_results/results_pre_fid_cosmo/samples_GSMF_CGD_fGas_7p.npy
  2. NEW 7p (GSMF+CGD+fGas) marginalized to (omega_m, sigma_8)   [if present]
       results/samples_GSMF_CGD_fGas_7p.npy
  3. NEW 2p fixed-hydro (broad-prior, matches 7p)                 [if present]
       results/samples_GSMF_CGD_fGas_2cosmo_match7p.npy

For each chain that does not yet exist on disk, the script just skips it and
prints a note, so it is safe to re-run while the 7p / 2p MCMCs are still going.

Outputs (in this directory)
---------------------------
  cosmo_marg_vs_fixed.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import plots as gd_plots
from getdist import MCSamples


V2  = '/home/nramachandra/Projects/Hydro_runs/CosmoHydro'
OUT = os.path.join(V2, 'Inference/v1_v2_comparison')

# Design ranges (must match FinalDesign cosmology columns)
RANGES = {'omega_m': (0.12, 0.155), 'sigma_8': (0.7, 0.9)}
# Project fiducial cosmology
FID = {'omega_m': 0.14176, 'sigma_8': 0.8102}
# Default broad prior actually used by the 7p run (midpoint Gaussian).
# For reference only; not drawn.
PRIOR = {'omega_m': (0.5*(0.12+0.155), 0.5*(0.155-0.12)),
         'sigma_8': (0.5*(0.7+0.9),    0.5*(0.9-0.7))}

NAMES  = ['omega_m', 'sigma_8']
LABELS = [r'\omega_m \equiv \Omega_m h^2', r'\sigma_8']


CANDIDATES = [
    dict(
        label='7p, hydro marginalized (OLD chain)',
        path=f'{V2}/Inference/old_results/results_pre_fid_cosmo/samples_GSMF_CGD_fGas_7p.npy',
        cols=(5, 6), color='tab:red', ls='-',
    ),
    dict(
        label='7p, hydro marginalized (NEW chain)',
        path=f'{V2}/Inference/results/samples_GSMF_CGD_fGas_7p.npy',
        cols=(5, 6), color='tab:orange', ls='-',
    ),
    dict(
        label='2p, hydro FIXED at midpoints (match-7p priors)',
        path=f'{V2}/Inference/results/samples_GSMF_CGD_fGas_2cosmo_match7p.npy',
        cols=(0, 1), color='tab:blue', ls='-',
    ),
]


def _mcsamples(chain2d, label):
    return MCSamples(samples=chain2d, names=NAMES, labels=LABELS,
                     label=label, ranges=RANGES)


loaded = []
print('chain inventory:')
for c in CANDIDATES:
    if not os.path.exists(c['path']):
        print(f'  MISSING  {c["label"]:55s}  ({c["path"]})')
        continue
    arr = np.load(c['path'])
    if arr.ndim != 2 or arr.shape[1] <= max(c['cols']):
        print(f'  SKIP     {c["label"]:55s}  bad shape {arr.shape}')
        continue
    sub = np.column_stack([arr[:, c['cols'][0]], arr[:, c['cols'][1]]])
    print(f'  LOADED   {c["label"]:55s}  shape={arr.shape}  '
          f'-> 2D (omega_m,sigma_8) {sub.shape}')
    print(f'           median = ({np.median(sub[:,0]):.4f}, {np.median(sub[:,1]):.4f})  '
          f'std = ({np.std(sub[:,0]):.4f}, {np.std(sub[:,1]):.4f})')
    loaded.append({**c, 'samples': sub, 'mc': _mcsamples(sub, c['label'])})

if not loaded:
    raise SystemExit('No chains found — nothing to plot.')


# ----- triangle / 2D plot ---------------------------------------------------
g = gd_plots.get_subplot_plotter(width_inch=7.5)
g.settings.alpha_filled_add = 0.55
g.settings.legend_fontsize  = 11
g.settings.axes_fontsize    = 11
g.settings.lab_fontsize     = 13

g.triangle_plot(
    [c['mc'] for c in loaded], NAMES, filled=True,
    contour_colors=[c['color'] for c in loaded],
    line_args=[{'color': c['color'], 'ls': c['ls'], 'lw': 1.6} for c in loaded],
    legend_loc='upper right',
)

# overlay fiducial cosmology
ax_om   = g.subplots[0, 0]   # 1D omega_m
ax_s8   = g.subplots[1, 1]   # 1D sigma_8
ax_2d   = g.subplots[1, 0]   # 2D (omega_m, sigma_8)
for ax in (ax_om,):
    ax.axvline(FID['omega_m'], color='k', ls=':', lw=1.0, alpha=0.7)
for ax in (ax_s8,):
    ax.axvline(FID['sigma_8'], color='k', ls=':', lw=1.0, alpha=0.7)
ax_2d.axvline(FID['omega_m'], color='k', ls=':', lw=1.0, alpha=0.7)
ax_2d.axhline(FID['sigma_8'], color='k', ls=':', lw=1.0, alpha=0.7)
ax_2d.plot([FID['omega_m']], [FID['sigma_8']], marker='*', ms=12,
           mfc='gold', mec='k', mew=0.8, ls='', label='fiducial', zorder=20)

g.fig.suptitle(
    'GSMF+CGD+fGas posteriors on cosmology: marginalized vs. fixed hydro',
    y=1.02, fontsize=13,
)

out_path = os.path.join(OUT, 'cosmo_marg_vs_fixed.png')
g.export(out_path)
print(f'\nwrote {out_path}')


# ----- text summary ---------------------------------------------------------
sumpath = os.path.join(OUT, 'cosmo_marg_vs_fixed_medians.txt')
with open(sumpath, 'w') as f:
    f.write('Cosmology posterior summary on (omega_m, sigma_8)\n')
    f.write(f'fiducial : omega_m = {FID["omega_m"]:.5f},  sigma_8 = {FID["sigma_8"]:.4f}\n')
    f.write(f'prior    : omega_m ~ N({PRIOR["omega_m"][0]:.4f}, {PRIOR["omega_m"][1]:.4f}), '
            f'sigma_8 ~ N({PRIOR["sigma_8"][0]:.4f}, {PRIOR["sigma_8"][1]:.4f})\n\n')
    f.write(f'{"chain":55s}  {"omega_m (med +/- 1sig)":24s}   {"sigma_8 (med +/- 1sig)":24s}\n')
    for c in loaded:
        s = c['samples']
        f.write(f'  {c["label"]:55s}  {np.median(s[:,0]):.5f} +/- {np.std(s[:,0]):.5f}    '
                f'{np.median(s[:,1]):.4f} +/- {np.std(s[:,1]):.4f}\n')
print(f'wrote {sumpath}')
