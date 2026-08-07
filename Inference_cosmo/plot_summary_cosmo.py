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

import plot_mcmc as P                       # noqa: E402  (GMM mode estimator + GSMF/CGD data)
import plot_mcmc_summary as S               # noqa: E402  (triangle machinery)
from cosmo_hydro_emu.load_hacc import PARAM_NAME, plot_strings, mass_conds  # noqa: E402
from cosmo_hydro_emu.emu import emulate     # noqa: E402  (GSMF/CGD best-fit prediction)

from run_mcmc_cosmo import (                # noqa: E402
    FIDUCIAL_SUBGRID, FIDUCIAL_COSMO, SHORT_KEYS_ORDERED,
)
from run_mcmc import SHORT_KEY_TO_LABEL      # noqa: E402  (short key -> latex label)
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
    stride = max(1, samples.shape[0] // MODE_SUBSAMPLE)
    mode = P.mcmc_results(samples[::stride], peak=1)

    # Actual fixed values from the trial's saved config (e.g. hydro pinned at
    # point A), defaulting to the PROJECT FIDUCIAL for anything not overridden.
    import yaml
    cfg_fixed = {}
    cfg_path = os.path.join(RESULTS, f'config_{trial}.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg_fixed = (yaml.safe_load(f) or {}).get('fixed_params', {}) or {}
    fixed_by_label = {SHORT_KEY_TO_LABEL[k]: float(v) for k, v in cfg_fixed.items()
                      if k in SHORT_KEY_TO_LABEL}

    full_theta = []
    for i, name in enumerate(PARAM_NAME):
        if name in free_labels:
            full_theta.append(float(mode[free_labels.index(name)]))
        elif name in fixed_by_label:
            full_theta.append(fixed_by_label[name])
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


def _panel_obs(ax, chains, ctx, obs):
    """GSMF/CGD posterior-predictive panel: emulator prediction at each chain's
    best-fit theta, overlaid on the observations (same conventions as the
    Inference/ plot_summary_* panels — GSMF plots log10(model)=number density,
    CGD plots the profile directly; both on a log y-axis)."""
    dd = ctx['gc']
    key = obs.lower()
    y_ind = dd['datasets'][key]['y_ind']
    od = dd['obs_data'][key]
    is_gsmf = (obs == 'GSMF')
    for c in chains:
        mg, _ = emulate(dd['models'][key], c['full_theta'])
        mg = np.asarray(mg).ravel()
        y = np.log10(mg) if is_gsmf else mg
        ls = '-' if c is chains[0] else '--'
        ax.plot(y_ind, y, ls, color=c['color'], lw=2 if c is chains[0] else 1.5,
                label=f'MCMC best fit: {c["label"]}')
    oy = np.log10(od['y']) if is_gsmf else od['y']
    ax.errorbar(od['x'], oy, yerr=od['yerr'], fmt='.k', capsize=2, zorder=3,
                label='observation')
    _, xlab, ylab = plot_strings(obs)
    m1, m2 = mass_conds(obs)
    ax.set_xlim(m1, m2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.legend(fontsize=8, loc='lower left')


def panel_gsmf(ax, chains, ctx):
    _panel_obs(ax, chains, ctx, 'GSMF')


def panel_cgd(ax, chains, ctx):
    _panel_obs(ax, chains, ctx, 'CGD')


PANEL_FUNCS = {'kids': panel_kids, 'hmf': panel_hmf,
               'gsmf': panel_gsmf, 'cgd': panel_cgd}


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
    """One box per chain showing ALL 7 design params (fixed ones flagged '(fix)')
    plus any nuisance extras — so a 'cosmo fixed' / 'hydro fixed' chain shows what
    it is fixed to."""
    y0 = 0.97
    canon = list(PARAM_NAME)
    for c in chains:
        free = set(c['own_labels'])
        props = dict(boxstyle='round4', facecolor='white', alpha=0.85,
                     edgecolor=c['color'])
        txt = f"{c['label']}:\n"
        for k, lab in enumerate(canon):
            tag = '' if lab in free else '  (fix)'
            txt += f"{lab}: {c['full_theta'][k]:.4f}{tag}\n"
        for lab, val in c.get('extras', {}).items():
            txt += f"{lab}: {val:.4f}\n"
        n_lines = 1 + len(canon) + len(c.get('extras', {}))
        ax.text(0.58, y0, txt.rstrip(), transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, color=c['color'],
                weight='bold')
        y0 -= 0.030 * (n_lines + 1)


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


def _infer_cfg():
    """Inference/ defaults (data paths + obs_dirs) for the GSMF/CGD emulators."""
    import yaml
    with open(os.path.join(INFER, 'configs', '_defaults.yaml')) as f:
        return yaml.safe_load(f) or {}


def _triangle_png_multi(chains, all_labels, limits, filled, lws):
    """Union triangle for >=2 chains with per-chain filled/outline + line width."""
    tagged = {l: f'p{i}' for i, l in enumerate(all_labels)}
    mcs = []
    for c in chains:
        names = [tagged[l] for l in c['own_labels']]
        labels = [l.strip('$') for l in c['own_labels']]
        mcs.append(MCSamples(samples=c['samples_shared'], names=names,
                             labels=labels, label=None,
                             ranges={tagged[l]: c['ranges'][l] for l in c['own_labels']},
                             settings=S.GD_SETTINGS))
    g = gd_plots.get_subplot_plotter(subplot_size=1.7)
    g.settings.axes_fontsize = 11
    g.settings.axes_labelsize = 13
    g.settings.alpha_filled_add = 0.6
    g.settings.solid_contour_palefactor = 0.55
    g.settings.num_plot_contours = 2
    g.triangle_plot(mcs, [tagged[l] for l in all_labels], filled=filled,
                    contour_colors=[c['color'] for c in chains],
                    line_args=[{'color': c['color'], 'lw': w}
                               for c, w in zip(chains, lws)],
                    param_limits={tagged[l]: limits[l] for l in all_labels
                                  if l in limits})
    _overlay_priors_union(g, all_labels)
    _draw_fixed_lines(g, all_labels, chains)
    for lg in list(g.fig.legends):
        lg.remove()
    for ax in g.fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    g.export(tmp.name)
    return tmp.name


def _draw_fixed_lines(g, all_labels, chains):
    """Dotted lines at each chain's FIXED parameter values, in the chain's colour
    (green where cosmology is fixed, blue where hydro is fixed), in every triangle
    panel that involves the fixed parameter."""
    canon = list(PARAM_NAME)
    n = len(all_labels)
    for c in chains:
        free = set(c['own_labels'])
        for idx, lab in enumerate(all_labels):
            if lab in free or lab not in canon:
                continue                                   # only fixed design params
            val = c['full_theta'][canon.index(lab)]
            for i in range(n):
                for j in range(i + 1):
                    ax = g.subplots[i][j]
                    if ax is None:
                        continue
                    if j == idx:
                        ax.axvline(val, color=c['color'], ls=':', lw=1.3, alpha=0.9)
                    if i == idx and i != j:
                        ax.axhline(val, color=c['color'], ls=':', lw=1.3, alpha=0.9)


def make_figure_multi(chain_specs, all_labels, title, out_name, panels, ctx,
                      limits=None):
    """Summary figure for a LIST of chains (each with its own free-param set):
    union triangle (left) + posterior-predictive panels (right).

    chain_specs : list of (trial, label, color, filled, lw).
    """
    print(f'\n=== summary figure (multi): {out_name} ===')
    limits = limits or UNION_LIMS
    chains = [_chain(t, lab, col, all_labels) for (t, lab, col, _f, _w) in chain_specs]
    filled = [f for (_t, _l, _c, f, _w) in chain_specs]
    lws = [w for (_t, _l, _c, _f, w) in chain_specs]

    tri_png = _triangle_png_multi(chains, all_labels, limits, filled, lws)
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


def build_ctx(need_hmf=True, need_kids=True, need_gc=False):
    print('loading emulators + targets (few minutes)...')
    ctx = {}
    if need_kids:
        ctx['emu'] = PkEmulator()
        ctx['kids'] = load_kids(nz='nz3', k_min=0.03, k_max=7.0,
                                z_bins=[0.15, 0.45])
    if need_hmf:
        ctx['hmf'] = load_gama_hmf(logM_max=14.9)
        ctx['hmf_like'] = HmfLikelihood(ctx['hmf'])
    if need_gc:                    # GSMF + CGD emulators + observations
        ctx['gc'] = P.load_all_data(_infer_cfg())
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
