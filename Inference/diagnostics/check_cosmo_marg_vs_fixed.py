"""2D (omega_m, sigma_8) overlay: marginalized-hydro vs fixed-hydro, and the
moderate-prior vs Planck-prior (*_pk) runs. Run for each observable suite
(GSMF and GSMF+CGD).

What this answers
-----------------
  - "If we fix hydro instead of marginalizing, how much does cosmology change?"
  - "What does the cosmology posterior look like under a Planck-width prior?"

Each chain's MCSamples is given ITS OWN range (read from the saved params_list),
so getdist's KDE is boundary-corrected at the right place and nothing leaks.

(The hard-truncated *_trunc runs were retired to old_results/retired_trunc/: the
±1σ wall amputated the real posterior — see diagnostics/2p_cosmology_issue.md.
The *_pk runs keep the Planck-width Gaussian but no hard cut.)

Chains per suite (skipped if absent):
  - <suite>_7p         moderate prior, hydro marginalized   (cols 5,6)
  - <suite>_2cosmo     moderate prior, hydro FIXED          (cols 0,1)
  - <suite>_7p_pk      Planck prior,   hydro marginalized   (cols 5,6)
  - <suite>_2cosmo_pk  Planck prior,   hydro FIXED          (cols 0,1)

Outputs, one per suite:
  cosmo_marg_vs_fixed.png / _GSMF_CGD.png  (+ matching *_medians.txt)
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

AXIS = {'omega_m': (0.12, 0.155), 'sigma_8': (0.70, 0.90)}
FID = {'omega_m': 0.14176, 'sigma_8': 0.8102}
# Planck +/-1 sigma reference band (drawn as a box for context; *_pk uses this
# as the Gaussian width, NOT a hard wall).
PLANCK_1SIG = {'omega_m': (0.14066, 0.14286), 'sigma_8': (0.8042, 0.8162)}

NAMES = ['omega_m', 'sigma_8']
LABELS = [r'\omega_m \equiv \Omega_m h^2', r'\sigma_8']

# (label, trial-suffix, cols, color) — trial = <prefix><suffix>
SPECS = [
    ('7p marg, moderate prior', '_7p',        (5, 6), 'tab:red'),
    ('2p fixed, moderate prior', '_2cosmo',    (0, 1), 'tab:blue'),
    ('7p marg, Planck prior',    '_7p_pk',     (5, 6), 'tab:purple'),
    ('2p fixed, Planck prior',   '_2cosmo_pk', (0, 1), 'tab:green'),
]
SUITES = [('GSMF', 'GSMF', ''), ('GSMF+CGD', 'GSMF_CGD', '_GSMF_CGD')]


def _load(trial, cols):
    """Return (sub2d, ranges) using the chain's OWN saved param ranges."""
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


def make_check(obs_label, prefix, suffix):
    print(f'\n=== cosmo marg-vs-fixed: {obs_label} ===')
    loaded = []
    for label, tsuf, cols, color in SPECS:
        r = _load(f'{prefix}{tsuf}', cols)
        if r is None:
            print(f'  MISSING  {label:28s} ({prefix}{tsuf})')
            continue
        sub, ranges = r
        print(f'  LOADED   {label:28s} median=({np.median(sub[:,0]):.4f}, {np.median(sub[:,1]):.4f})')
        loaded.append(dict(label=label, color=color, samples=sub,
                           mc=MCSamples(samples=sub, names=NAMES, labels=LABELS,
                                        label=label, ranges=ranges)))
    if not loaded:
        print('  no chains — skipping')
        return

    g = gd_plots.get_subplot_plotter(width_inch=8)
    g.settings.alpha_filled_add = 0.55
    g.settings.legend_fontsize = 10
    g.settings.axes_fontsize = 11
    g.settings.lab_fontsize = 13
    g.triangle_plot([c['mc'] for c in loaded], NAMES, filled=True,
                    contour_colors=[c['color'] for c in loaded],
                    param_limits=AXIS, legend_loc='upper right')

    ax_om, ax_s8, ax_2d = g.subplots[0, 0], g.subplots[1, 1], g.subplots[1, 0]
    ax_om.axvline(FID['omega_m'], color='k', ls=':', lw=1.0)
    ax_s8.axvline(FID['sigma_8'], color='k', ls=':', lw=1.0)
    ax_2d.axvline(FID['omega_m'], color='k', ls=':', lw=1.0)
    ax_2d.axhline(FID['sigma_8'], color='k', ls=':', lw=1.0)
    ax_2d.plot([FID['omega_m']], [FID['sigma_8']], marker='*', ms=12, mfc='gold',
               mec='k', mew=0.8, ls='', zorder=20)
    (olo, ohi), (slo, shi) = PLANCK_1SIG['omega_m'], PLANCK_1SIG['sigma_8']
    ax_2d.add_patch(plt.Rectangle((olo, slo), ohi - olo, shi - slo, fill=False,
                                  ec='k', lw=1.4, ls='--', zorder=19))
    for v in (olo, ohi):
        ax_om.axvline(v, color='k', ls='--', lw=1.0, alpha=0.6)
    for v in (slo, shi):
        ax_s8.axvline(v, color='k', ls='--', lw=1.0, alpha=0.6)

    g.fig.suptitle(f'{obs_label} cosmology: marginalized vs fixed hydro, '
                   'moderate vs Planck prior\n'
                   '(dashed box = Planck ±1σ; star = fiducial)', y=1.04, fontsize=12)
    png = os.path.join(OUT, f'cosmo_marg_vs_fixed{suffix}.png')
    g.export(png)
    print(f'  wrote {png}')

    txt = os.path.join(OUT, f'cosmo_marg_vs_fixed{suffix}_medians.txt')
    with open(txt, 'w') as f:
        f.write(f'Cosmology posterior on (omega_m, sigma_8) — {obs_label}\n')
        f.write(f'fiducial: omega_m={FID["omega_m"]:.5f}, sigma_8={FID["sigma_8"]:.4f}\n')
        f.write(f'Planck +/-1sig: omega_m{PLANCK_1SIG["omega_m"]}, sigma_8{PLANCK_1SIG["sigma_8"]}\n\n')
        f.write(f'{"chain":28s}  {"omega_m med [min,max]":34s}  {"sigma_8 med [min,max]"}\n')
        for c in loaded:
            s = c['samples']
            f.write(f'  {c["label"]:28s}  {np.median(s[:,0]):.5f} [{s[:,0].min():.5f},{s[:,0].max():.5f}]  '
                    f'{np.median(s[:,1]):.5f} [{s[:,1].min():.5f},{s[:,1].max():.5f}]\n')
    print(f'  wrote {txt}')


if __name__ == '__main__':
    for obs_label, prefix, suffix in SUITES:
        make_check(obs_label, prefix, suffix)
