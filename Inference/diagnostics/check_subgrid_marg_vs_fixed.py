"""Triangle overlay on the 5 subgrid params: cosmo-marginalized vs. cosmo-fixed.

What this answers
-----------------
  "If we fix the cosmology instead of marginalizing over it, how much do the
   subgrid (hydro) constraints change?"

This is the subgrid-space analog of check_cosmo_marg_vs_fixed.py (which asks the
mirror question for cosmology). Both runs are GSMF-only and use the shared
project-default priors (configs/_defaults.yaml), so the only difference is
whether cosmology is free or pinned at the fiducial.

Chains compared (GSMF only)
---------------------------
  1. 7p, cosmo marginalized -> 5 subgrid params      [if present]
       results/samples_GSMF_7p.npy            (cols 0..4)
  2. 5p, cosmo FIXED at fiducial
       results/samples_GSMF_5p_fid_cosmo.npy  (cols 0..4)

For each chain that does not yet exist on disk, the script just skips it and
prints a note, so it is safe to re-run while the MCMCs are still going.

Outputs (in this directory)
---------------------------
  subgrid_marg_vs_fixed.png
  subgrid_marg_vs_fixed_medians.txt
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import plots as gd_plots
from getdist import MCSamples


V2  = '/home/nramachandra/Projects/Hydro_runs/CosmoHydro'
OUT = os.path.join(V2, 'Inference/diagnostics')

# 5 subgrid params (scaled units, matching the design matrix / check_v1_v2.py)
NAMES  = ['kappa_w', 'e_w', 'M_seed_e6', 'v_kin_e4', 'eps_kin_e1']
LABELS = [r'\kappa_\mathrm{w}', r'e_\mathrm{w}',
          r'M_\mathrm{seed}/10^{6}', r'v_\mathrm{kin}/10^{4}',
          r'\epsilon_\mathrm{kin}/10^{1}']
RANGES = {'kappa_w': (2.0, 4.0), 'e_w': (0.2, 1.0),
          'M_seed_e6': (0.6, 2.0), 'v_kin_e4': (0.1, 1.2),
          'eps_kin_e1': (0.02, 1.2)}

# both chains carry the 5 subgrid params in columns 0..4
SUBGRID_COLS = (0, 1, 2, 3, 4)

CANDIDATES = [
    dict(
        label='7p, cosmo marginalized (GSMF)',
        path=f'{V2}/Inference/results/samples_GSMF_7p.npy',
        color='tab:red', ls='-',
    ),
    dict(
        label='5p, cosmo FIXED at fiducial (GSMF)',
        path=f'{V2}/Inference/results/samples_GSMF_5p_fid_cosmo.npy',
        color='tab:blue', ls='-',
    ),
]


def _mcsamples(chain5d, label):
    return MCSamples(samples=chain5d, names=NAMES, labels=LABELS,
                     label=label, ranges=RANGES)


loaded = []
print('chain inventory:')
for c in CANDIDATES:
    if not os.path.exists(c['path']):
        print(f'  MISSING  {c["label"]:40s}  ({c["path"]})')
        continue
    arr = np.load(c['path'])
    if arr.ndim != 2 or arr.shape[1] <= max(SUBGRID_COLS):
        print(f'  SKIP     {c["label"]:40s}  bad shape {arr.shape}')
        continue
    sub = arr[:, list(SUBGRID_COLS)]
    print(f'  LOADED   {c["label"]:40s}  shape={arr.shape}  -> 5D subgrid {sub.shape}')
    loaded.append({**c, 'samples': sub, 'mc': _mcsamples(sub, c['label'])})

if not loaded:
    raise SystemExit('No chains found — nothing to plot.')


# ----- triangle plot --------------------------------------------------------
g = gd_plots.get_subplot_plotter(width_inch=9)
g.settings.alpha_filled_add = 0.55
g.settings.legend_fontsize  = 12
g.settings.axes_fontsize    = 11
g.settings.lab_fontsize     = 13

g.triangle_plot(
    [c['mc'] for c in loaded], NAMES, filled=True,
    contour_colors=[c['color'] for c in loaded],
    line_args=[{'color': c['color'], 'ls': c['ls'], 'lw': 1.6} for c in loaded],
    legend_loc='upper right',
)

g.fig.suptitle(
    'GSMF subgrid posteriors: cosmology marginalized (7p) vs. fixed (5p)',
    y=1.02, fontsize=13,
)

out_path = os.path.join(OUT, 'subgrid_marg_vs_fixed.png')
g.export(out_path)
print(f'\nwrote {out_path}')


# ----- text summary ---------------------------------------------------------
sumpath = os.path.join(OUT, 'subgrid_marg_vs_fixed_medians.txt')
with open(sumpath, 'w') as f:
    f.write('Subgrid posterior summary (scaled units) — GSMF-only fits\n\n')
    f.write(f'{"param":14s}  ' + '  '.join(f'{c["label"]:32s}' for c in loaded) + '\n')
    for i, nm in enumerate(NAMES):
        row = f'  {nm:12s}  '
        for c in loaded:
            s = c['samples'][:, i]
            row += f'{np.median(s):8.4f} +/- {np.std(s):7.4f}            '
        f.write(row.rstrip() + '\n')
print(f'wrote {sumpath}')
