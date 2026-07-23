#!/usr/bin/env python
"""2D cosmology likelihood sweeps for the cosmology targets (Pk + HMF),
hydro subgrid held at the project fiducial — the Inference_cosmo analog of
Inference/diagnostics/likelihood_sweep_cosmo.py (same layout, scales,
contours, priors).

Cases (columns): KiDS Pm | GAMA HMF | KiDS+HMF
Rows: likelihood / x moderate prior (sigma = 0.005, 0.03) /
      x Planck prior (sigma = 0.0011, 0.006; zoomed grid).

Uses the exact MCMC likelihood objects (same k/z/mass cuts, same error
models as the ongoing runs):
  KiDS : nz3, z_bins [0.15, 0.45], k = 0.03-7    (as Pk_kids_2cosmo)
  HMF  : GAMA DR4, logM = 12.8-14.9, dlogM = 0   (as HMF_gama_2cosmo)
BOSS is omitted (free bias/shot-noise would need per-point profiling).
A_mod is quarantined to amod_exploratory/ (a 4-column version including it
is archived at amod_exploratory/plots/likelihood_sweep_2d_with_amod.png).

Output: likelihood_sweep_2d_cosmo_targets.png (+ .npz grid cache)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, '..', 'codes'))

from cosmo_hydro_emu.mcmc import ln_prior

from targets import load_kids, load_gama_hmf
from pk_likelihood import PkEmulator, PmLikelihood, HmfLikelihood

FID_HYDRO = [3.0, 0.5, 0.8, 0.51, 0.13]
FID_OM, FID_S8 = 0.14176, 0.8102
OM_RANGE = (0.12, 0.155)
S8_RANGE = (0.70, 0.90)
# zoom window matching the Inference Planck-prior row
OM_ZOOM = (0.136, 0.148)
S8_ZOOM = (0.775, 0.845)

N_GRID = 36
N_ZOOM = 30

MOD_SIG = {'om': 0.005, 's8': 0.03}
PLK_SIG = {'om': 0.0011, 's8': 0.006}
_PLIST = [['omega_m', FID_OM, *OM_RANGE], ['sigma_8', FID_S8, *S8_RANGE]]
CASES = ['KiDS $P_m$', 'GAMA HMF', 'KiDS+HMF']

CACHE = os.path.join(HERE, 'likelihood_sweep_2d_cosmo_targets.npz')
OUT = os.path.join(HERE, 'likelihood_sweep_2d_cosmo_targets.png')


def build_likes():
    print('loading emulators + targets (few minutes)...')
    emu = PkEmulator()
    kids = PmLikelihood(load_kids(nz='nz3', k_min=0.03, k_max=7.0,
                                  z_bins=[0.15, 0.45]), emu)
    hmf = HmfLikelihood(load_gama_hmf(logM_max=14.9))
    return {'kids': kids, 'hmf': hmf}


def sweep(likes, om_grid, s8_grid, tag):
    lls = {k: np.zeros((s8_grid.size, om_grid.size)) for k in likes}
    for j, s8 in enumerate(s8_grid):
        for i, om in enumerate(om_grid):
            theta = np.array(FID_HYDRO + [om, s8])
            for k, like in likes.items():
                lls[k][j, i] = like(theta)
        print(f'  [{tag}] row {j + 1}/{s8_grid.size}', flush=True)
    return lls


def prior_grid(om_grid, s8_grid, sig):
    gp = {0: (FID_OM, sig['om']), 1: (FID_S8, sig['s8'])}
    lp = np.zeros((s8_grid.size, om_grid.size))
    for j, s8 in enumerate(s8_grid):
        for i, om in enumerate(om_grid):
            lp[j, i] = ln_prior(np.array([om, s8]), _PLIST,
                                flat_indices=[], gaussian_priors=gp)
    return lp


def case_fields(lls):
    return [lls['kids'], lls['hmf'], lls['kids'] + lls['hmf']]


def panel(ax, fig, om_grid, s8_grid, field, title, cbar_label):
    OM, S8 = np.meshgrid(om_grid, s8_grid)
    d = np.clip(field - np.nanmax(field), -30, 0)
    pcm = ax.pcolormesh(OM, S8, d, cmap='viridis', shading='gouraud',
                        vmin=-30, vmax=0)
    ax.contour(OM, S8, d, levels=[-11.83, -6.17, -2.30], colors='w',
               linewidths=0.8, linestyles='--')
    jmax, imax = np.unravel_index(np.nanargmax(field), field.shape)
    ax.plot(om_grid[imax], s8_grid[jmax], '*', ms=16, mfc='gold', mec='k')
    ax.axvline(FID_OM, color='r', ls='--', lw=1.2)
    ax.axhline(FID_S8, color='r', ls='--', lw=1.2)
    ax.set_title(f'{title}\npeak ({om_grid[imax]:.4f}, {s8_grid[jmax]:.3f})',
                 fontsize=10)
    ax.set_xlabel(r'$\omega_m \equiv \Omega_m h^2$')
    ax.set_ylabel(r'$\sigma_8$')
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(cbar_label, fontsize=9)


def main():
    om_grid = np.linspace(*OM_RANGE, N_GRID)
    s8_grid = np.linspace(*S8_RANGE, N_GRID)
    om_zoom = np.linspace(*OM_ZOOM, N_ZOOM)
    s8_zoom = np.linspace(*S8_ZOOM, N_ZOOM)

    if os.path.exists(CACHE):
        print(f'using cached grids: {CACHE}')
        z = np.load(CACHE)
        lls = {k: z[f'wide_{k}'] for k in ['kids', 'hmf']}
        lls_z = {k: z[f'zoom_{k}'] for k in ['kids', 'hmf']}
    else:
        likes = build_likes()
        lls = sweep(likes, om_grid, s8_grid, 'wide')
        lls_z = sweep(likes, om_zoom, s8_zoom, 'zoom')
        np.savez_compressed(CACHE,
                            **{f'wide_{k}': v for k, v in lls.items()},
                            **{f'zoom_{k}': v for k, v in lls_z.items()})

    lp_mod = prior_grid(om_grid, s8_grid, MOD_SIG)
    lp_plk = prior_grid(om_zoom, s8_zoom, PLK_SIG)

    fig, axes = plt.subplots(3, len(CASES), figsize=(6 * len(CASES), 16.5))
    for c, (name, f_wide, f_zoom) in enumerate(zip(
            CASES, case_fields(lls), case_fields(lls_z))):
        panel(axes[0, c], fig, om_grid, s8_grid, f_wide,
              f'{name} — likelihood', r'$\Delta \ln \mathcal{L}$')
        panel(axes[1, c], fig, om_grid, s8_grid, f_wide + lp_mod,
              f'{name} — posterior x moderate prior', r'$\Delta \ln \mathcal{P}$')
        panel(axes[2, c], fig, om_zoom, s8_zoom, f_zoom + lp_plk,
              f'{name} — posterior x Planck prior (zoom)',
              r'$\Delta \ln \mathcal{P}$')
    fig.suptitle('Cosmology targets: hydro fixed at fiducial. Rows: likelihood '
                 f'/ x moderate prior (σ={MOD_SIG["om"]}, {MOD_SIG["s8"]}) '
                 f'/ x Planck prior (σ={PLK_SIG["om"]}, {PLK_SIG["s8"]}, '
                 'zoomed). Red lines = fiducial.', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
