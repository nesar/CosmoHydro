#!/usr/bin/env python
"""Combined triangle + posterior-predictive summary figures for the
Inference_cosmo trials — same style as Inference/results/plot_summary_*.png.

REUSES Inference/plot_mcmc_summary.py directly (triangle renderer, prior
overlays, best-fit textboxes, colors, getdist settings) and
Inference/plot_mcmc.py's GMM mode estimator. Only the right-hand
posterior-predictive panels are new (KiDS Pm, GAMA HMF).

Figures (written to results/):
  plot_summary_hmf_vs_kids_cosmo.png      (omega_m, sigma_8): GAMA HMF vs
      KiDS Pm cosmology
  plot_summary_cosmo_marg_vs_fixed.png    (omega_m, sigma_8): subgrid
      marginalized (Pk_kids_7p) vs fixed (Pk_kids_2cosmo) — produced once
      the amod-free Pk_kids_7p chain has run

A_mod-based summary figures live in amod_exploratory/plot_summary_amod.py.

Usage:  python plot_summary_cosmo.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
INFER = os.path.abspath(os.path.join(_HERE, '..', 'Inference'))
sys.path.insert(0, _HERE)
sys.path.insert(0, INFER)
sys.path.insert(0, os.path.join(_HERE, '..', 'codes'))

import plot_mcmc as P                       # noqa: E402  (GMM mode estimator)
import plot_mcmc_summary as S               # noqa: E402  (triangle machinery)
from cosmo_hydro_emu.load_hacc import PARAM_NAME     # noqa: E402

from run_mcmc_cosmo import (                # noqa: E402
    FIDUCIAL_SUBGRID, FIDUCIAL_COSMO, SHORT_KEYS_ORDERED,
)
from targets import load_kids, load_gama_hmf             # noqa: E402
from pk_likelihood import (                 # noqa: E402
    PkEmulator, HmfLikelihood, IDX_OMEGA_M, IDX_SIGMA_8,
)

RESULTS = os.path.join(_HERE, 'results')
MODE_SUBSAMPLE = 40000

COSMO_SHARED = [PARAM_NAME[5], PARAM_NAME[6]]
SUBGRID_SHARED = list(PARAM_NAME[:5])
COSMO_LIMS = {PARAM_NAME[5]: (0.120, 0.155), PARAM_NAME[6]: (0.77, 0.86)}
SUBGRID_LIMS = S.SUBGRID_LIMS

# Our runs use the broad default prior N(design midpoint, half-range) on every
# free design parameter (no gaussian_priors in the configs). Feeding those
# into the reused overlay_priors via a synthetic cfg makes the dashed prior
# curves exactly match run_mcmc_cosmo's ln_prior.
PRIOR_CFG = {'gaussian_priors': {
    'omega_m': {'mu': 0.1375, 'sigma': 0.0175},
    'sigma_8': {'mu': 0.8, 'sigma': 0.1},
}}

FIDUCIAL7 = np.array([FIDUCIAL_SUBGRID[k] for k in SHORT_KEYS_ORDERED[:5]]
                     + [FIDUCIAL_COSMO['omega_m'], FIDUCIAL_COSMO['sigma_8']])


def _chain(trial, label, color, shared_labels):
    """Best-fit (GMM mode) + shared-subspace samples, using the reused
    machinery's conventions. Fixed params default to the PROJECT FIDUCIAL
    (run_mcmc_cosmo convention)."""
    samples = np.load(os.path.join(RESULTS, f'samples_{trial}.npy'))
    pl = np.load(os.path.join(RESULTS, f'params_list_{trial}.npy'),
                 allow_pickle=True).tolist()
    free_labels = [str(p[0]) for p in pl]
    n_design_free = sum(1 for l in free_labels if l in list(PARAM_NAME))
    stride = max(1, samples.shape[0] // MODE_SUBSAMPLE)
    mode = P.mcmc_results(samples[::stride], peak=1)

    full_theta = []
    for i, name in enumerate(PARAM_NAME):
        if name in free_labels:
            full_theta.append(float(mode[free_labels.index(name)]))
        else:
            full_theta.append(float(FIDUCIAL7[i]))

    extras = {l: float(mode[i]) for i, l in enumerate(free_labels)
              if l not in list(PARAM_NAME)}

    own = [l for l in shared_labels if l in free_labels]
    cols = [free_labels.index(l) for l in own]
    ranges = {l: (float(pl[free_labels.index(l)][2]),
                  float(pl[free_labels.index(l)][3])) for l in own}
    shared_modes = [full_theta[list(PARAM_NAME).index(l)]
                    if l in list(PARAM_NAME) else extras[l] for l in own]
    print(f'  {trial}: mode theta7 = {tuple(round(v, 4) for v in full_theta)}'
          + (f'  extras={ {k: round(v, 3) for k, v in extras.items()} }'
             if extras else ''))
    return dict(trial=trial, label=label, color=color,
                full_theta=np.array(full_theta), own_labels=own,
                samples_shared=samples[:, cols], ranges=ranges,
                shared_modes=shared_modes, extras=extras)


# ---------------------------------------------------------------------------
# Right-hand posterior-predictive panels (target-specific, ours)
# ---------------------------------------------------------------------------
def panel_kids(ax, chains, ctx):
    emu, kids_target = ctx['emu'], ctx['kids']
    zs = np.unique(kids_target['z'])
    markers = {zs[0]: 'o', zs[-1]: 's'}
    for z in zs:
        m = kids_target['z'] == z
        ax.errorbar(kids_target['k'][m], kids_target['y'][m],
                    yerr=kids_target['sigma'][m], fmt=markers[z], ms=4,
                    color='k', mfc='k' if z == zs[0] else 'w', capsize=2,
                    lw=1, label=f'KiDS-Legacy $P_m$, $z_{{\\rm fid}}={z}$')
    for c in chains:
        for z in zs:
            Ph, _ = emu.P_hydro(z, c['full_theta'])
            ls = '-' if c is chains[0] else '--'
            ax.loglog(emu.k_grid, Ph, ls, color=c['color'], lw=2 if z == zs[0] else 1.2,
                      label=f'MCMC best fit: {c["label"]}' if z == zs[0] else None)
    ax.set_xlim(0.02, 9)
    ax.set_xlabel(r'$k$ [$h\,{\rm Mpc}^{-1}$]', fontsize=13)
    ax.set_ylabel(r'$P_m(k)\;[(h^{-1}{\rm Mpc})^3]$', fontsize=13)
    ax.legend(fontsize=8, loc='lower left')


# panel_suppression (A_mod band): MOVED to amod_exploratory/plot_summary_amod.py


def panel_hmf(ax, chains, ctx):
    hmf_like, hmf_target = ctx['hmf_like'], ctx['hmf']
    ax.errorbar(hmf_target['logM'], hmf_target['y'], yerr=hmf_target['sigma'],
                fmt='o', ms=4, color='k', capsize=2,
                label='GAMA DR4 (Driver+22)')
    for c in chains:
        dlogM = c.get('extras', {}).get(r'$\Delta\log M$ GAMA', 0.0)
        phi, _ = hmf_like.model_phi(c['full_theta'], dlogM)
        order = np.argsort(hmf_target['logM'])
        ls = '-' if c is chains[0] else '--'
        ax.plot(hmf_target['logM'][order], phi[order], ls, color=c['color'],
                lw=2, label=f'MCMC best fit: {c["label"]}')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10} M_{\rm 200c}\,[M_\odot/h]$', fontsize=13)
    ax.set_ylabel(r'$\phi$ [(Mpc/$h$)$^{-3}$ dex$^{-1}$]', fontsize=13)
    ax.legend(fontsize=8, loc='lower left')


PANEL_FUNCS = {'kids': panel_kids, 'hmf': panel_hmf}


def make_figure(marg_trial, fixed_trial, marg_label, fixed_label,
                shared_labels, limits, title, out_name,
                panels, ctx):
    print(f'\n=== summary figure: {out_name} ===')
    chains = [
        _chain(marg_trial, marg_label, S.MARG_COLOR, shared_labels),
        _chain(fixed_trial, fixed_label, S.FIXED_COLOR, shared_labels),
    ]
    tri_png = S._triangle_png(chains, shared_labels, limits, PRIOR_CFG,
                              flat_idx=set())
    tri_img = Image.open(tri_png)

    fig = plt.figure(figsize=(18, 5 * len(panels)))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.08)
    ax_tri = fig.add_subplot(gs[0, 0])
    ax_tri.imshow(tri_img)
    ax_tri.axis('off')
    gs_r = gs[0, 1].subgridspec(len(panels), 1, hspace=0.45)
    axs = [fig.add_subplot(gs_r[i]) for i in range(len(panels))]

    for ax, kind in zip(axs, panels):
        PANEL_FUNCS[kind](ax, chains, ctx)

    S._textboxes(ax_tri, chains, shared_labels)
    fig.suptitle(title, y=0.94, fontsize=15)
    out = os.path.join(RESULTS, out_name)
    plt.savefig(out, bbox_inches='tight')
    plt.close(fig)
    os.unlink(tri_png)
    print(f'  saved: {out}')




# ---------------------------------------------------------------------------
# Union triangle: chains may have DIFFERENT free-parameter sets (e.g. the
# GAMA mass-shift nuisance exists only in the HMF chain). getdist matches
# parameters by name and simply omits a chain from panels it lacks.
# ---------------------------------------------------------------------------
import tempfile
from getdist import plots as gd_plots, MCSamples

DLOGM_LABEL = r'$\Delta\log M$ GAMA'
UNION_LIMS = {**SUBGRID_LIMS, **{PARAM_NAME[5]: (0.120, 0.155),
                                 PARAM_NAME[6]: (0.70, 0.90)},
              DLOGM_LABEL: (-0.3, 0.3)}


def _overlay_priors_union(g, labels):
    """Dashed default-prior curves on the diagonals: N(midpoint, half-range)
    for every free parameter (exactly run_mcmc_cosmo's ln_prior default)."""
    canon = list(PARAM_NAME)
    for i, lab in enumerate(labels):
        ax = g.subplots[i, i]
        if ax is None:
            continue
        if lab in canon and canon.index(lab) <= 4:
            rlo, rhi = S.SG_PRIOR_RANGE[canon.index(lab)]
        elif lab == PARAM_NAME[5]:
            rlo, rhi = 0.12, 0.155
        elif lab == PARAM_NAME[6]:
            rlo, rhi = 0.70, 0.90
        else:                                  # nuisance (dlogM)
            rlo, rhi = -0.3, 0.3
        mu, sig = 0.5 * (rlo + rhi), 0.5 * (rhi - rlo)
        lo, hi = ax.get_xlim()
        x = np.linspace(lo, hi, 400)
        y = np.exp(-0.5 * ((x - mu) / sig) ** 2)
        ax.plot(x, y / y.max(), color='k', ls='--', lw=1.2, alpha=0.75)
        ax.set_ylim(0, 1.15)


def _triangle_png_union(chains, all_labels, limits):
    tagged = {l: f'p{i}' for i, l in enumerate(all_labels)}
    mcs = []
    pretty = {DLOGM_LABEL: r'\Delta\log M_{\rm GAMA}'}
    for c in chains:
        names = [tagged[l] for l in c['own_labels']]
        labels = [pretty.get(l, l.strip('$')) for l in c['own_labels']]
        mcs.append(MCSamples(samples=c['samples_shared'], names=names,
                             labels=labels, label=None,
                             ranges={tagged[l]: c['ranges'][l]
                                     for l in c['own_labels']},
                             settings=S.GD_SETTINGS))
    g = gd_plots.get_subplot_plotter(subplot_size=1.7)
    g.settings.axes_fontsize = 11
    g.settings.axes_labelsize = 13
    g.settings.alpha_filled_add = 0.7
    g.settings.solid_contour_palefactor = 0.55
    g.settings.num_plot_contours = 2
    g.triangle_plot(mcs, [tagged[l] for l in all_labels], filled=True,
                    contour_colors=[c['color'] for c in chains],
                    param_limits={tagged[l]: limits[l] for l in all_labels
                                  if l in limits})
    _overlay_priors_union(g, all_labels)
    for lg in list(g.fig.legends):
        lg.remove()
    for ax in g.fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    g.export(tmp.name)
    return tmp.name


def _textboxes_union(ax, chains):
    y0 = 0.97
    for i, c in enumerate(chains):
        props = dict(boxstyle='round4', facecolor='white', alpha=0.85,
                     edgecolor=c['color'])
        txt = f"{c['label']}:\n"
        for lab, val in zip(c['own_labels'], c['shared_modes']):
            txt += f"{lab}: {val:.4f}\n"
        n_lines = len(c['own_labels']) + 1
        ax.text(0.58, y0, txt.rstrip(), transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, color=c['color'],
                weight='bold')
        y0 -= 0.032 * (n_lines + 1)


def make_figure_union(marg_trial, fixed_trial, marg_label, fixed_label,
                      all_labels, title, out_name, panels, ctx):
    print(f'\n=== summary figure (union params): {out_name} ===')
    chains = [
        _chain(marg_trial, marg_label, S.MARG_COLOR, all_labels),
        _chain(fixed_trial, fixed_label, S.FIXED_COLOR, all_labels),
    ]
    tri_png = _triangle_png_union(chains, all_labels, UNION_LIMS)
    tri_img = Image.open(tri_png)

    fig = plt.figure(figsize=(19, 5.2 * len(panels)))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1], wspace=0.06)
    ax_tri = fig.add_subplot(gs[0, 0])
    ax_tri.imshow(tri_img)
    ax_tri.axis('off')
    gs_r = gs[0, 1].subgridspec(len(panels), 1, hspace=0.45)
    axs = [fig.add_subplot(gs_r[i]) for i in range(len(panels))]
    for ax, kind in zip(axs, panels):
        PANEL_FUNCS[kind](ax, chains, ctx)
    _textboxes_union(ax_tri, chains)
    fig.suptitle(title, y=0.95, fontsize=15)
    out = os.path.join(RESULTS, out_name)
    plt.savefig(out, bbox_inches='tight')
    plt.close(fig)
    os.unlink(tri_png)
    print(f'  saved: {out}')


def build_ctx(need_hmf=True):
    print('loading emulators + targets (few minutes)...')
    ctx = {'emu': PkEmulator(),
           'kids': load_kids(nz='nz3', k_min=0.03, k_max=7.0,
                             z_bins=[0.15, 0.45])}
    if need_hmf:
        ctx['hmf'] = load_gama_hmf(logM_max=14.9)
        ctx['hmf_like'] = HmfLikelihood(ctx['hmf'])
    return ctx


def main():
    have = {t for t in ['Pk_kids_2cosmo', 'HMF_gama_2cosmo',
                        'Pk_kids_7p', 'Pk_kids_hmf_7p']
            if os.path.exists(os.path.join(RESULTS, f'samples_{t}.npy'))}
    print(f'finished trials found: {sorted(have)}')
    ctx = build_ctx(need_hmf='HMF_gama_2cosmo' in have or 'Pk_kids_hmf_7p' in have)

    if {'HMF_gama_2cosmo', 'Pk_kids_2cosmo'} <= have:
        make_figure(
            'HMF_gama_2cosmo', 'Pk_kids_2cosmo',
            'GAMA HMF (2p)', 'KiDS $P_m$ (2p)',
            COSMO_SHARED, COSMO_LIMS,
            'Cosmology from cluster abundance (red) vs matter power spectrum '
            '(blue) — with posterior-predictive summary',
            'plot_summary_hmf_vs_kids_cosmo.png',
            ['hmf', 'kids'], ctx)

    if {'Pk_kids_7p', 'Pk_kids_2cosmo'} <= have:
        make_figure(
            'Pk_kids_7p', 'Pk_kids_2cosmo',
            'subgrid marginalized (7p)', 'subgrid fixed (2p)',
            COSMO_SHARED, COSMO_LIMS,
            'KiDS $P_m$ cosmology: subgrid marginalized (red) vs fixed (blue) '
            '— with posterior-predictive summary',
            'plot_summary_cosmo_marg_vs_fixed.png',
            ['kids', 'hmf' if 'hmf_like' in ctx else 'kids'], ctx)
    else:
        print('Pk_kids_7p (amod-free joint) not run yet — '
              'plot_summary_cosmo_marg_vs_fixed.png will be produced then.')

    if {'Pk_kids_7p', 'Pk_kids_hmf_7p'} <= have:
        all_labels = list(PARAM_NAME) + [DLOGM_LABEL]
        make_figure_union(
            'Pk_kids_hmf_7p', 'Pk_kids_7p',
            'KiDS+HMF (7p+$\\Delta\\log M$)', 'KiDS only (7p)',
            all_labels,
            'Full parameter space: KiDS+HMF (red) vs KiDS only (blue) '
            '— with posterior-predictive summary',
            'plot_summary_7p_kids_vs_kids_hmf.png',
            ['kids', 'hmf'], ctx)


if __name__ == '__main__':
    main()
