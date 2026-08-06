"""Triangle overlay on the 5 subgrid params: cosmo-marginalized vs. cosmo-fixed.

What this answers
-----------------
  "If we fix the cosmology instead of marginalizing over it, how much do the
   subgrid (hydro) constraints change?"

Mirror of check_cosmo_marg_vs_fixed.py (same question for cosmology). Run for
each observable suite (GSMF and GSMF+CGD); the only difference between the two
chains in a suite is whether cosmology is free (7p) or pinned at fiducial (5p).

Chains per suite (skipped if absent), all on the 5 subgrid params (cols 0..4):
  - <suite>_7p            cosmo marginalized, moderate cosmo prior
  - <suite>_7p_planck         cosmo marginalized, Planck-width cosmo prior
  - <suite>_5p_fid_cosmo  cosmo FIXED at fiducial

The dashed line on each diagonal is the subgrid prior (broad default Gaussian;
flat for eps_kin).

Outputs (in this directory), one per suite:
  subgrid_marg_vs_fixed.png / _GSMF_CGD.png  (+ matching *_medians.txt)
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

NAMES  = ['kappa_w', 'e_w', 'M_seed_e6', 'v_kin_e4', 'eps_kin_e1']
LABELS = [r'\kappa_\mathrm{w}', r'e_\mathrm{w}',
          r'M_\mathrm{seed}/10^{6}', r'v_\mathrm{kin}/10^{4}',
          r'\epsilon_\mathrm{kin}/10^{1}']
RANGES = {'kappa_w': (2.0, 4.0), 'e_w': (0.2, 1.0),
          'M_seed_e6': (0.6, 2.0), 'v_kin_e4': (0.1, 1.2),
          'eps_kin_e1': (0.02, 1.2)}
COLS = (0, 1, 2, 3, 4)   # both chains carry the 5 subgrid params in cols 0..4

# One entry per observable suite: (obs_label, trial_prefix, out_suffix)
SUITES = [
    ('GSMF',     'GSMF',     ''),
    ('GSMF+CGD', 'GSMF_CGD', '_GSMF_CGD'),
]


def _load(trial, label):
    p = os.path.join(RES, f'samples_{trial}.npy')
    if not os.path.exists(p):
        print(f'  MISSING  {label:34s} ({trial})')
        return None
    arr = np.load(p)
    if arr.ndim != 2 or arr.shape[1] <= max(COLS):
        print(f'  SKIP     {label:34s} bad shape {arr.shape}')
        return None
    sub = arr[:, list(COLS)]
    print(f'  LOADED   {label:34s} {arr.shape} -> 5D subgrid')
    return dict(label=label, samples=sub,
                mc=MCSamples(samples=sub, names=NAMES, labels=LABELS,
                             label=label, ranges=RANGES))


def _overlay_priors(g):
    """Dashed subgrid prior on each 1D diagonal: broad default Gaussian
    N(midpoint, half-range), flat for eps_kin (matches ln_prior)."""
    for i, nm in enumerate(NAMES):
        ax = g.subplots[i, i]
        if ax is None:
            continue
        lo, hi = RANGES[nm]
        x = np.linspace(lo, hi, 400)
        if nm == 'eps_kin_e1':                     # flat prior
            y = np.ones_like(x)
        else:                                      # broad default Gaussian
            mu, sig = 0.5 * (lo + hi), 0.5 * (hi - lo)
            y = np.exp(-0.5 * ((x - mu) / sig) ** 2)
        ax.plot(x, y / y.max(), color='k', ls='--', lw=1.2, alpha=0.7)
        ax.set_ylim(0, 1.15)


def make_check(obs_label, prefix, suffix):
    print(f'\n=== subgrid marg-vs-fixed: {obs_label} ===')
    specs = [
        (f'{prefix}_7p',           f'7p, cosmo marg (moderate) ({obs_label})', 'tab:red'),
        (f'{prefix}_7p_planck',        f'7p, cosmo marg (Planck) ({obs_label})',   'tab:purple'),
        (f'{prefix}_5p_fid_cosmo', f'5p, cosmo FIXED ({obs_label})',           'tab:blue'),
    ]
    loaded, colors = [], []
    for trial, label, color in specs:
        c = _load(trial, label)
        if c is not None:
            loaded.append(c); colors.append(color)
    if not loaded:
        print('  no chains — skipping')
        return

    g = gd_plots.get_subplot_plotter(width_inch=9)
    g.settings.alpha_filled_add = 0.55
    g.settings.legend_fontsize = 12
    g.settings.axes_fontsize = 11
    g.settings.lab_fontsize = 13
    g.triangle_plot([c['mc'] for c in loaded], NAMES, filled=True,
                    contour_colors=colors,
                    line_args=[{'color': c, 'ls': '-', 'lw': 1.6} for c in colors],
                    legend_loc='upper right')
    _overlay_priors(g)
    g.fig.suptitle(f'{obs_label} subgrid posteriors: cosmology marginalized (7p, '
                   'moderate & Planck priors) vs. fixed (5p) — dashed = prior',
                   y=1.02, fontsize=13)
    png = os.path.join(OUT, f'subgrid_marg_vs_fixed{suffix}.png')
    g.export(png)
    print(f'  wrote {png}')

    txt = os.path.join(OUT, f'subgrid_marg_vs_fixed{suffix}_medians.txt')
    with open(txt, 'w') as f:
        f.write(f'Subgrid posterior summary (scaled units) — {obs_label} fits\n\n')
        f.write(f'{"param":12s}  ' + '  '.join(f'{c["label"]:38s}' for c in loaded) + '\n')
        for i, nm in enumerate(NAMES):
            row = f'  {nm:10s}  '
            for c in loaded:
                s = c['samples'][:, i]
                row += f'{np.median(s):8.4f} +/- {np.std(s):7.4f}                '
            f.write(row.rstrip() + '\n')
    print(f'  wrote {txt}')


if __name__ == '__main__':
    print('chain inventory:')
    for obs_label, prefix, suffix in SUITES:
        make_check(obs_label, prefix, suffix)
