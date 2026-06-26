"""2D (omega_m, sigma_8) overlay: marginalized-hydro vs fixed-hydro, and the
moderate-prior vs hard-truncated-prior (*_trunc) runs.

What this answers
-----------------
  - "If we fix hydro instead of marginalizing, how much does cosmology change?"
  - "Does the Planck-tight hard-truncated prior actually confine cosmology?"

IMPORTANT: each chain's MCSamples is given ITS OWN range (read from the saved
params_list), so getdist's KDE is boundary-corrected at the true hard wall. If
you instead pass the wide design-box range for a *_trunc chain, getdist smears
the wall-piled density well past the wall and it *looks* like the run escaped
the truncation — it didn't. The truncation is enforced in the sampler.

Chains (GSMF + CGD); each skipped if absent:
  - GSMF_CGD_7p           moderate prior, hydro marginalized   (cols 5,6)
  - GSMF_CGD_2cosmo       moderate prior, hydro FIXED          (cols 0,1)
  - GSMF_CGD_7p_trunc     Planck-tight prior, hydro marginalized (cols 5,6)
  - GSMF_CGD_2cosmo_trunc Planck-tight prior, hydro FIXED        (cols 0,1)

Outputs: cosmo_marg_vs_fixed.png, cosmo_marg_vs_fixed_medians.txt
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import plots as gd_plots
from getdist import MCSamples

V2  = '/home/nramachandra/Projects/Hydro_runs/CosmoHydro'
RES = os.path.join(V2, 'Inference/results')
OUT = os.path.join(V2, 'Inference/diagnostics')

# Axis limits (design box) and fiducial.
AXIS = {'omega_m': (0.12, 0.155), 'sigma_8': (0.70, 0.90)}
FID = {'omega_m': 0.14176, 'sigma_8': 0.8102}
# Hard-truncation window of the *_trunc runs (drawn as a box).
TRUNC_WIN = {'omega_m': (0.14066, 0.14286), 'sigma_8': (0.8042, 0.8162)}

NAMES = ['omega_m', 'sigma_8']
LABELS = [r'\omega_m \equiv \Omega_m h^2', r'\sigma_8']

CANDIDATES = [
    dict(label='7p marg, moderate prior',  trial='GSMF_CGD_7p',          cols=(5, 6), color='tab:red'),
    dict(label='2p fixed, moderate prior',  trial='GSMF_CGD_2cosmo',      cols=(0, 1), color='tab:blue'),
    dict(label='7p marg, Planck-trunc',     trial='GSMF_CGD_7p_trunc',    cols=(5, 6), color='tab:purple'),
    dict(label='2p fixed, Planck-trunc',    trial='GSMF_CGD_2cosmo_trunc', cols=(0, 1), color='tab:green'),
]


def _load(trial, cols):
    """Return (sub2d, ranges_dict) using the chain's OWN saved param ranges."""
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


loaded = []
print('chain inventory:')
for c in CANDIDATES:
    r = _load(c['trial'], c['cols'])
    if r is None:
        print(f'  MISSING  {c["label"]:28s} ({c["trial"]})')
        continue
    sub, ranges = r
    mc = MCSamples(samples=sub, names=NAMES, labels=LABELS, label=c['label'], ranges=ranges)
    print(f'  LOADED   {c["label"]:28s} median=({np.median(sub[:,0]):.4f}, {np.median(sub[:,1]):.4f}) '
          f'range_om=[{ranges["omega_m"][0]:.4f},{ranges["omega_m"][1]:.4f}]')
    loaded.append({**c, 'samples': sub, 'ranges': ranges, 'mc': mc})

if not loaded:
    raise SystemExit('No chains found — nothing to plot.')

# ----- 2D triangle ----------------------------------------------------------
g = gd_plots.get_subplot_plotter(width_inch=8)
g.settings.alpha_filled_add = 0.55
g.settings.legend_fontsize = 10
g.settings.axes_fontsize = 11
g.settings.lab_fontsize = 13

g.triangle_plot([c['mc'] for c in loaded], NAMES, filled=True,
                contour_colors=[c['color'] for c in loaded],
                param_limits=AXIS, legend_loc='upper right')

ax_om, ax_s8, ax_2d = g.subplots[0, 0], g.subplots[1, 1], g.subplots[1, 0]
# fiducial
ax_om.axvline(FID['omega_m'], color='k', ls=':', lw=1.0)
ax_s8.axvline(FID['sigma_8'], color='k', ls=':', lw=1.0)
ax_2d.axvline(FID['omega_m'], color='k', ls=':', lw=1.0)
ax_2d.axhline(FID['sigma_8'], color='k', ls=':', lw=1.0)
ax_2d.plot([FID['omega_m']], [FID['sigma_8']], marker='*', ms=12, mfc='gold',
           mec='k', mew=0.8, ls='', zorder=20)
# hard-truncation window (the wall the *_trunc runs live inside)
(olo, ohi), (slo, shi) = TRUNC_WIN['omega_m'], TRUNC_WIN['sigma_8']
ax_2d.add_patch(plt.Rectangle((olo, slo), ohi - olo, shi - slo, fill=False,
                              ec='k', lw=1.4, ls='--', zorder=19))
for v in (olo, ohi):
    ax_om.axvline(v, color='k', ls='--', lw=1.0, alpha=0.6)
for v in (slo, shi):
    ax_s8.axvline(v, color='k', ls='--', lw=1.0, alpha=0.6)

g.fig.suptitle('GSMF+CGD cosmology: marginalized vs fixed hydro, '
               'moderate vs Planck-truncated prior\n'
               '(dashed box = truncation window; star = fiducial)',
               y=1.04, fontsize=12)
out_path = os.path.join(OUT, 'cosmo_marg_vs_fixed.png')
g.export(out_path)
print(f'\nwrote {out_path}')

# ----- text summary ---------------------------------------------------------
sumpath = os.path.join(OUT, 'cosmo_marg_vs_fixed_medians.txt')
with open(sumpath, 'w') as f:
    f.write('Cosmology posterior on (omega_m, sigma_8)\n')
    f.write(f'fiducial: omega_m={FID["omega_m"]:.5f}, sigma_8={FID["sigma_8"]:.4f}\n')
    f.write(f'trunc window: omega_m{TRUNC_WIN["omega_m"]}, sigma_8{TRUNC_WIN["sigma_8"]}\n\n')
    f.write(f'{"chain":28s}  {"omega_m med [min,max]":34s}  {"sigma_8 med [min,max]"}\n')
    for c in loaded:
        s = c['samples']
        f.write(f'  {c["label"]:28s}  {np.median(s[:,0]):.5f} [{s[:,0].min():.5f},{s[:,0].max():.5f}]  '
                f'{np.median(s[:,1]):.5f} [{s[:,1].min():.5f},{s[:,1].max():.5f}]\n')
print(f'wrote {sumpath}')