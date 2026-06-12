#!/usr/bin/env python
"""
Overlay plot: 7-parameter MCMC posterior with two reduced-dimension posteriors
overlaid on their respective subspaces:
  - the 2-cosmology-parameter (fiducial-prior) posterior on the cosmology panels,
  - the 5-subgrid-parameter (fixed-cosmology) posterior on the subgrid panels.

Run after all three MCMC trials are complete:
    python plot_7p_vs_2cosmo_5subgrid.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codes'))
from cosmo_hydro_emu.load_hacc import PARAM_NAME

RESULTS = os.path.join(os.path.dirname(__file__), 'results')

SAMPLES_7P = os.path.join(RESULTS, 'samples_GSMF_7p.npy')
PARAMS_7P  = os.path.join(RESULTS, 'params_list_GSMF_7p.npy')

# Canonical 2-cosmology run. Uses the shared project-default cosmology prior
# (moderate fiducial-centered Gaussian from configs/_defaults.yaml), which keeps
# the posterior off the design-box edge. See configs/GSMF_CGD_fGas_2cosmo.yaml.
SAMPLES_2C = os.path.join(RESULTS, 'samples_GSMF_CGD_fGas_2cosmo.npy')
PARAMS_2C  = os.path.join(RESULTS, 'params_list_GSMF_CGD_fGas_2cosmo.npy')

SAMPLES_5P = os.path.join(RESULTS, 'samples_GSMF_5p_fid_cosmo.npy')
PARAMS_5P  = os.path.join(RESULTS, 'params_list_GSMF_5p_fid_cosmo.npy')

OUTPUT = os.path.join(RESULTS, 'plot_7p_vs_2cosmo_5subgrid.png')


def load(samples_path, params_path):
    samples = np.load(samples_path)
    params_list = np.load(params_path, allow_pickle=True).tolist()
    names = [p[0] for p in params_list]
    ranges = {p[0]: (float(p[2]), float(p[3])) for p in params_list}
    return samples, names, ranges


def main():
    for f in (SAMPLES_7P, PARAMS_7P, SAMPLES_2C, PARAMS_2C, SAMPLES_5P, PARAMS_5P):
        if not os.path.exists(f):
            print(f"ERROR: missing {f}")
            sys.exit(1)

    s7p, names_7p, ranges_7p = load(SAMPLES_7P, PARAMS_7P)
    s2c, names_2c, ranges_2c = load(SAMPLES_2C, PARAMS_2C)
    s5p, names_5p, ranges_5p = load(SAMPLES_5P, PARAMS_5P)

    print(f"7p: samples shape {s7p.shape}, names {names_7p}")
    print(f"2cosmo: samples shape {s2c.shape}, names {names_2c}")
    print(f"5subgrid: samples shape {s5p.shape}, names {names_5p}")

    # MCSamples: each reduced chain shares its names with the 7p chain. getdist
    # overlays the 2cosmo chain only on the omega_m / sigma_8 panels and the
    # 5subgrid chain only on the 5 subgrid panels of the 7p triangle.
    mc_7p = MCSamples(
        samples=s7p,
        names=names_7p,
        labels=[n.strip('$') for n in names_7p],
        label='7 params (GSMF+CGD+fGas, Planck prior on cosmo)',
        ranges=ranges_7p,
        settings={'mult_bias_correction_order': 0.5,
                  'smooth_scale_2D': 4, 'smooth_scale_1D': 4},
    )

    mc_2c = MCSamples(
        samples=s2c,
        names=names_2c,
        labels=[n.strip('$') for n in names_2c],
        label='2 cosmo only (GSMF+CGD+fGas, fiducial prior)',
        ranges=ranges_2c,
        settings={'mult_bias_correction_order': 0.5,
                  'smooth_scale_2D': 4, 'smooth_scale_1D': 4},
    )

    mc_5p = MCSamples(
        samples=s5p,
        names=names_5p,
        labels=[n.strip('$') for n in names_5p],
        label='5 subgrid only (GSMF, fiducial cosmology fixed)',
        ranges=ranges_5p,
        settings={'mult_bias_correction_order': 0.5,
                  'smooth_scale_2D': 4, 'smooth_scale_1D': 4},
    )

    g = plots.get_subplot_plotter(subplot_size=2.0)
    g.settings.axes_fontsize = 12
    g.settings.axes_labelsize = 14
    g.settings.legend_fontsize = 12
    g.settings.alpha_filled_add = 0.6
    g.settings.solid_contour_palefactor = 0.6
    g.settings.num_plot_contours = 2

    # Triangle plotted in 7p parameter space; getdist drops each reduced chain
    # from panels whose parameter it lacks, overlaying 2cosmo on the cosmology
    # panels and 5subgrid on the subgrid panels.
    g.triangle_plot(
        [mc_7p, mc_2c, mc_5p],
        params=names_7p,
        filled=True,
        legend_labels=[mc_7p.label, mc_2c.label, mc_5p.label],
        param_limits=ranges_7p,
        contour_colors=['#1f77b4', '#d62728', '#2ca02c'],
    )

    # Mark project fiducial cosmology as crosshairs on the cosmology panels.
    # omega_m = Omega_m*h^2 from Omega_cdm=0.26067, Omega_b*h^2=0.02242, H0=67.66.
    fiducial = {names_7p[5]: 0.14176, names_7p[6]: 0.8102}
    n = len(names_7p)
    for i, ni in enumerate(names_7p):
        for j, nj in enumerate(names_7p):
            if j > i:
                continue
            ax = g.subplots[i, j]
            if ax is None:
                continue
            if i == j and ni in fiducial:
                ax.axvline(fiducial[ni], color='k', lw=1.0, ls=':')
            elif ni in fiducial and nj in fiducial:
                ax.axvline(fiducial[nj], color='k', lw=1.0, ls=':')
                ax.axhline(fiducial[ni], color='k', lw=1.0, ls=':')

    plt.suptitle('7-param vs 2-cosmology vs 5-subgrid MCMC — fiducial prior on $\\Omega_m h^2$, $\\sigma_8$',
                 y=1.005, fontsize=14)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    plt.savefig(OUTPUT, bbox_inches='tight', dpi=150)
    print(f"Plot saved: {OUTPUT}")


if __name__ == '__main__':
    main()
