#!/usr/bin/env python
"""
Plots for completed Inference_cosmo MCMC trials.

For each trial with results/samples_{trial}.npy:
  results/corner_{trial}.png            getdist triangle plot

And across all trials that sample cosmology:
  results/cosmo_2d_overlay.png          (omega_m, sigma_8) 68/95% contours,
                                        one color per trial, fiducial marked

Usage:
  python plot_cosmo_results.py                       # all trials found
  python plot_cosmo_results.py Pk_kids_2cosmo ...    # subset
"""

import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, 'results')

from getdist import plots, MCSamples

FID = {r'$\Omega_\text{m}$': 0.14176, '$\\sigma_8$': 0.8102,
       r'$\kappa_\text{w}$': 3.0, r'$e_\text{w}$': 0.5,
       r'$M_\text{seed}/10^{6}$': 0.8, r'$v_\text{kin}/10^{4}$': 0.51,
       r'$\epsilon_\text{kin}/10^{1}$': 0.13}

OM_LABEL = r'$\Omega_\text{m}$'      # design column label (actually omega_m)
S8_LABEL = '$\\sigma_8$'


def find_trials():
    out = []
    for f in sorted(os.listdir(RESULTS)):
        m = re.match(r'samples_(.+)\.npy$', f)
        if m:
            out.append(m.group(1))
    return out


def load_trial(trial):
    samples = np.load(os.path.join(RESULTS, f'samples_{trial}.npy'))
    plist = np.load(os.path.join(RESULTS, f'params_list_{trial}.npy'),
                    allow_pickle=True)
    names = [str(r[0]) for r in plist]
    ranges = {str(r[0]): (float(r[2]), float(r[3])) for r in plist}
    return samples, names, ranges


def _mcsamples(trial, samples, names, ranges):
    tags = [f'p{i}' for i in range(len(names))]
    labels = [n.strip('$') for n in names]
    return MCSamples(samples=samples, names=tags, labels=labels,
                     ranges={f'p{i}': ranges[n] for i, n in enumerate(names)},
                     label=trial.replace('_', ' '))


def corner_plot(trial):
    samples, names, ranges = load_trial(trial)
    mc = _mcsamples(trial, samples, names, ranges)
    g = plots.get_subplot_plotter(width_inch=2.2 * len(names))
    g.triangle_plot([mc], filled=True)
    # fiducial markers
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if j > i:
                continue
            ax = g.subplots[i, j]
            if ax is None:
                continue
            if i == j:
                if ni in FID:
                    ax.axvline(FID[ni], color='r', ls='--', lw=1)
            else:
                if nj in FID:
                    ax.axvline(FID[nj], color='r', ls='--', lw=1)
                if ni in FID:
                    ax.axhline(FID[ni], color='r', ls='--', lw=1)
    out = os.path.join(RESULTS, f'corner_{trial}.png')
    g.export(out)
    print(f'wrote {out}')


def cosmo_overlay(trials):
    mcs, kept = [], []
    for t in trials:
        samples, names, ranges = load_trial(t)
        if OM_LABEL not in names or S8_LABEL not in names:
            continue
        iom, is8 = names.index(OM_LABEL), names.index(S8_LABEL)
        mc = MCSamples(samples=samples[:, [iom, is8]],
                       names=['om', 's8'],
                       labels=[r'\omega_m \equiv \Omega_m h^2', r'\sigma_8'],
                       ranges={'om': ranges[OM_LABEL], 's8': ranges[S8_LABEL]},
                       label=t.replace('_', ' '))
        mcs.append(mc)
        kept.append(t)
    if not mcs:
        print('no cosmology-sampling trials found for overlay')
        return
    g = plots.get_single_plotter(width_inch=7)
    g.plot_2d(mcs, 'om', 's8', filled=True)
    ax = g.subplots[0, 0]
    ax.plot(0.14176, 0.8102, '*', ms=16, mfc='gold', mec='k', zorder=10,
            label='fiducial')
    ax.axvline(0.14176, color='r', ls='--', lw=1, alpha=0.6)
    ax.axhline(0.8102, color='r', ls='--', lw=1, alpha=0.6)
    g.add_legend([t.replace('_', ' ') for t in kept] + ['fiducial'],
                 legend_loc='upper right')
    out = os.path.join(RESULTS, 'cosmo_2d_overlay.png')
    g.export(out)
    print(f'wrote {out}')


if __name__ == '__main__':
    trials = sys.argv[1:] or find_trials()
    print(f'trials: {trials}')
    for t in trials:
        corner_plot(t)
    cosmo_overlay(trials)
