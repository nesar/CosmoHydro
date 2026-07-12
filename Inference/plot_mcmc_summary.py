#!/usr/bin/env python
"""Combined triangle + posterior-predictive summary panels for the headline
Planck-prior (*_pk) fits, in the style of the Flamingo reference plot
(combined_mcmc_getdist_*): a getdist triangle on the left with per-chain
best-fit parameter textboxes, and best-fit emulator predictions overlaid on the
observations (GSMF, CGD) on the right.

Reuses the machinery already in plot_mcmc.py; the only reason this exists is to
(a) restrict the right-hand panels to the observables v2 actually has data for
(GSMF + CGD — there is no fGas observation in the HAvoCC set), and (b) drive the
specific *_pk headline figures with clean labels/output names.

    python plot_mcmc_summary.py            # all headline figures
    python plot_mcmc_summary.py joint      # just the GSMF+CGD joint fit
"""
import os
import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')

INFER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, INFER)
import plot_mcmc as P                                   # noqa: E402
from run_mcmc import _deep_merge, build_param_space     # noqa: E402

RESULTS = os.path.join(INFER, 'results')
OBS_PANELS = ['GSMF', 'CGD']       # v2 observations available for overlay
MODE_SUBSAMPLE = 40000             # GMM mode estimate on a stride (fast, robust)


def _load_cfg(trial):
    with open(os.path.join(INFER, 'configs', '_defaults.yaml')) as f:
        defaults = yaml.safe_load(f) or {}
    with open(os.path.join(RESULTS, f'config_{trial}.yaml')) as f:
        trialcfg = yaml.safe_load(f) or {}
    return _deep_merge(defaults, trialcfg)


def make(trials, labels, out_name, data_cache=None):
    """One combined figure for the given trial chain(s)."""
    print(f'\n=== summary figure: {out_name}  <- {trials} ===')
    cfg_ref = _load_cfg(trials[0])

    # Data/emulators are identical across trials (same design + models); load once.
    if data_cache is None or data_cache.get('dd') is None:
        print('  loading emulators + observations ...')
        dd = P.load_all_data(cfg_ref)
        if data_cache is not None:
            data_cache['dd'] = dd
    else:
        dd = data_cache['dd']

    dcfg = cfg_ref['data']
    design = P.load_design(os.path.join(INFER, dcfg['design_file']),
                           start_sim_idx=dcfg.get('start_sim_idx', 1),
                           num_sims=dcfg['num_sims'])
    _, fixed_params, param_names_used, _ = build_param_space(cfg_ref, design)

    chains_samples, p_mcmc_list, params_list = [], [], None
    for t in trials:
        s = np.load(os.path.join(RESULTS, f'samples_{t}.npy'))
        pl = np.load(os.path.join(RESULTS, f'params_list_{t}.npy'),
                     allow_pickle=True).tolist()
        if params_list is None:
            params_list = pl
        ncols = len(params_list)
        chains_samples.append(s[:, :ncols])
        # GMM mode on a stride keeps this quick on 1.6M-sample chains.
        stride = max(1, s.shape[0] // MODE_SUBSAMPLE)
        p_mcmc_list.append(P.mcmc_results(s[::stride, :ncols], peak=1))
        print(f'  {t}: {s.shape} mode={tuple(round(float(v),4) for v in p_mcmc_list[-1])}')

    flat_idx = set(cfg_ref.get('flat_prior_indices', []) or [])
    out = os.path.join(RESULTS, out_name)
    P.combined_plot(chains_samples=chains_samples, chains_labels=labels,
                    params_list=params_list, p_mcmc_list=p_mcmc_list,
                    save_path=out, fixed_params=fixed_params, data_dict=dd,
                    obs_list=OBS_PANELS, param_names=param_names_used,
                    prior_overlay=lambda g, names: overlay_priors(
                        g, names, cfg_ref, flat_idx))


# Headline figures. All 7p_pk chains share the same 7-param space, so they can be
# overlaid directly; the 2-chain comparison mirrors the Flamingo reference.
FIGURES = {
    'gsmf':  (['GSMF_7p_pk'], [r'$\mathcal{L}_\mathrm{GSMF}$ (7p)'],
              'plot_summary_GSMF_7p_pk.png'),
    'joint': (['GSMF_CGD_7p_pk'], [r'$\mathcal{L}_\mathrm{GSMF}+\mathcal{L}_\mathrm{CGD}$ (7p)'],
              'plot_summary_GSMF_CGD_7p_pk.png'),
    'compare': (['GSMF_7p_pk', 'GSMF_CGD_7p_pk'],
                [r'$\mathcal{L}_\mathrm{GSMF}$', r'$\mathcal{L}_\mathrm{GSMF}+\mathcal{L}_\mathrm{CGD}$'],
                'plot_summary_7p_GSMF_vs_GSMF_CGD_pk.png'),
}


# ===========================================================================
# Marginalized-vs-fixed summary figures. Here the two overlaid chains live in
# DIFFERENT parameter spaces (e.g. 7p vs 2-cosmo), so the triangle is drawn on
# the shared sub-space only, while each chain's FULL 7-param best fit (free
# params at their mode, the rest at that chain's fiducial fix) drives the
# emulator for the right-hand summary panels.
# ===========================================================================
from getdist import plots as gd_plots, MCSamples   # noqa: E402
from PIL import Image                               # noqa: E402
from matplotlib.gridspec import GridSpec            # noqa: E402
import matplotlib.pyplot as plt                     # noqa: E402
import tempfile                                     # noqa: E402

MARG_COLOR, FIXED_COLOR = '#E03424', '#006FED'      # red = marg, blue = fixed
GD_SETTINGS = {'mult_bias_correction_order': 0.5,
               'smooth_scale_2D': 4, 'smooth_scale_1D': 4}
# Triangle axis limits per shared sub-space, keyed by canonical label.
COSMO_LIMS = {P.PARAM_NAME[5]: (0.128, 0.152), P.PARAM_NAME[6]: (0.75, 0.86)}
SUBGRID_LIMS = {P.PARAM_NAME[0]: (2.0, 4.0), P.PARAM_NAME[1]: (0.2, 1.0),
                P.PARAM_NAME[2]: (0.6, 2.0), P.PARAM_NAME[3]: (0.1, 1.2),
                P.PARAM_NAME[4]: (0.02, 1.2)}
# Subgrid design ranges by canonical index (for the broad default Gaussian prior).
SG_PRIOR_RANGE = {0: (2.0, 4.0), 1: (0.2, 1.0), 2: (0.6, 2.0),
                  3: (0.1, 1.2), 4: (0.02, 1.2)}


def overlay_priors(g, labels, cfg, flat_idx):
    """Dashed prior curve (peak-normalized) on each 1D diagonal of a getdist
    triangle. Cosmology priors are read from cfg['gaussian_priors'] so they
    exactly match the run; subgrid params use the broad default N(midpoint,
    half-range), flat for indices in flat_idx (matches ln_prior)."""
    canon = list(P.PARAM_NAME)
    gp = cfg.get('gaussian_priors', {}) or {}
    for i, lab in enumerate(labels):
        ax = g.subplots[i, i]
        if ax is None:
            continue
        idx = canon.index(lab)
        lo, hi = ax.get_xlim()
        x = np.linspace(lo, hi, 500)
        if idx <= 4:                                   # subgrid
            if idx in flat_idx:                        # flat prior
                y = np.ones_like(x)
            else:
                rlo, rhi = SG_PRIOR_RANGE[idx]
                mu, sig = 0.5 * (rlo + rhi), 0.5 * (rhi - rlo)
                y = np.exp(-0.5 * ((x - mu) / sig) ** 2)
        else:                                          # cosmology (from config)
            key = 'omega_m' if idx == 5 else 'sigma_8'
            mu, sig = gp[key]['mu'], gp[key]['sigma']
            y = np.exp(-0.5 * ((x - mu) / sig) ** 2)
        if y.max() > 0:
            y = y / y.max()
        ax.plot(x, y, color='k', ls='--', lw=1.2, alpha=0.75)
        ax.set_ylim(0, 1.15)


def _chain_bestfit_and_triangle_data(trial, cfg, design, shared_labels):
    """Return (full_theta[7], samples_shared, ranges_shared, shared_modes)."""
    samples = np.load(os.path.join(RESULTS, f'samples_{trial}.npy'))
    pl = np.load(os.path.join(RESULTS, f'params_list_{trial}.npy'),
                 allow_pickle=True).tolist()
    free_labels = [p[0] for p in pl]
    ncols = len(free_labels)
    stride = max(1, samples.shape[0] // MODE_SUBSAMPLE)
    mode = P.mcmc_results(samples[::stride, :ncols], peak=1)   # tuple over free params

    _, fixed_params, _, _ = build_param_space(cfg, design)
    full_theta = []
    for name in P.PARAM_NAME:
        if name in free_labels:
            full_theta.append(float(mode[free_labels.index(name)]))
        else:
            full_theta.append(float(fixed_params[name]))

    cols = [free_labels.index(l) for l in shared_labels]
    samples_shared = samples[:, cols]
    ranges = {l: (float(pl[free_labels.index(l)][2]),
                  float(pl[free_labels.index(l)][3])) for l in shared_labels}
    shared_modes = [full_theta[list(P.PARAM_NAME).index(l)] for l in shared_labels]
    return np.array(full_theta), samples_shared, ranges, shared_modes


def _triangle_png(chains, shared_labels, limits, cfg, flat_idx):
    # label=None: no getdist legend — the colour-coded textboxes + title identify
    # the chains (a legend collides with the textboxes in the small 2x2 cosmology
    # triangle).
    mcs = [MCSamples(samples=c['samples_shared'], names=shared_labels,
                     label=None, ranges=c['ranges'], settings=GD_SETTINGS)
           for c in chains]
    g = gd_plots.get_subplot_plotter(subplot_size=2)
    g.settings.axes_fontsize = 13
    g.settings.axes_labelsize = 15
    g.settings.alpha_filled_add = 0.7
    g.settings.solid_contour_palefactor = 0.55
    g.settings.num_plot_contours = 2
    g.triangle_plot(mcs, shared_labels, filled=True,
                    contour_colors=[c['color'] for c in chains],
                    param_limits=limits)
    overlay_priors(g, shared_labels, cfg, flat_idx)   # dashed priors on diagonals
    for lg in list(g.fig.legends):                    # strip any auto legend
        lg.remove()
    for ax in g.fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    g.export(tmp.name)
    return tmp.name


def _textboxes(ax, chains, shared_labels):
    """Colour-coded best-fit textboxes, stacked tightly in the empty upper-right
    of the triangle. Vertical step scales with box height (#params) so the boxes
    sit close without overlapping each other or the panels."""
    n_lines = len(shared_labels) + 1                  # header + one per param
    step = 0.037 * (n_lines + 1)                      # tight but non-overlapping
    x = 0.60 if len(shared_labels) <= 2 else 0.63
    y0 = 0.97
    for i, c in enumerate(chains):
        props = dict(boxstyle='round4', facecolor='white', alpha=0.85,
                     edgecolor=c['color'])
        txt = f"{c['label']}:\n"
        for lab, val in zip(shared_labels, c['shared_modes']):
            txt += f"{lab}: {val:.4f}\n"
        ax.text(x, y0 - i * step, txt.rstrip(), transform=ax.transAxes,
                fontsize=11, verticalalignment='top', bbox=props,
                color=c['color'], weight='bold')


def make_marg_fixed(marg_trial, fixed_trial, marg_label, fixed_label,
                    shared_labels, limits, title, out_name, data_cache):
    """One combined figure: shared-subspace triangle (marg vs fixed) + summary."""
    print(f'\n=== marg-vs-fixed summary: {out_name} ===')
    cfg_marg, cfg_fixed = _load_cfg(marg_trial), _load_cfg(fixed_trial)
    if data_cache.get('dd') is None:
        print('  loading emulators + observations ...')
        data_cache['dd'] = P.load_all_data(cfg_marg)
    dd = data_cache['dd']
    dcfg = cfg_marg['data']
    design = P.load_design(os.path.join(INFER, dcfg['design_file']),
                           start_sim_idx=dcfg.get('start_sim_idx', 1),
                           num_sims=dcfg['num_sims'])

    chains = []
    for trial, cfg, label, color in (
            (marg_trial, cfg_marg, marg_label, MARG_COLOR),
            (fixed_trial, cfg_fixed, fixed_label, FIXED_COLOR)):
        theta, ss, rng, smodes = _chain_bestfit_and_triangle_data(
            trial, cfg, design, shared_labels)
        chains.append(dict(trial=trial, label=label, color=color, full_theta=theta,
                           samples_shared=ss, ranges=rng, shared_modes=smodes))
        print(f'  {trial}: full best-fit theta = '
              f'{tuple(round(v,4) for v in theta)}')

    flat_idx = set(cfg_marg.get('flat_prior_indices', []) or [])
    tri_png = _triangle_png(chains, shared_labels, limits, cfg_marg, flat_idx)
    tri_img = Image.open(tri_png)

    fig = plt.figure(figsize=(18, 5 * len(OBS_PANELS)))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.08)
    ax_tri = fig.add_subplot(gs[0, 0]); ax_tri.imshow(tri_img); ax_tri.axis('off')
    gs_r = gs[0, 1].subgridspec(len(OBS_PANELS), 1, hspace=0.45)
    axs = [fig.add_subplot(gs_r[i]) for i in range(len(OBS_PANELS))]

    # emulator predictions at each chain's full 7-param best fit
    P.plot_mcmc_bestfit([c['full_theta'] for c in chains],
                        [c['label'] for c in chains], fig, axs,
                        fixed_params={}, data_dict=dd, obs_list=OBS_PANELS,
                        param_names=list(P.PARAM_NAME))
    _textboxes(ax_tri, chains, shared_labels)
    fig.suptitle(title, y=0.94, fontsize=15)

    out = os.path.join(RESULTS, out_name)
    plt.savefig(out, bbox_inches='tight')
    plt.close(fig)
    os.unlink(tri_png)
    print(f'  saved: {out}')


COSMO_SHARED = [P.PARAM_NAME[5], P.PARAM_NAME[6]]
SUBGRID_SHARED = list(P.PARAM_NAME[:5])
ML = r'hydro marginalized (7p)'
FL_H = r'hydro fixed (2p)'
CL = r'cosmo marginalized (7p)'
FL_C = r'cosmo fixed (5p)'

MARGFIX = {
    'cosmo_gsmf': dict(marg_trial='GSMF_7p_pk', fixed_trial='GSMF_2cosmo_pk',
        marg_label=ML, fixed_label=FL_H, shared_labels=COSMO_SHARED,
        limits=COSMO_LIMS, out_name='plot_summary_pk_cosmo_marg_vs_fixed.png',
        title='GSMF cosmology: hydro marginalized (red) vs fixed (blue) '
              '— with posterior-predictive summary'),
    'cosmo_cgd': dict(marg_trial='GSMF_CGD_7p_pk', fixed_trial='GSMF_CGD_2cosmo_pk',
        marg_label=ML, fixed_label=FL_H, shared_labels=COSMO_SHARED,
        limits=COSMO_LIMS, out_name='plot_summary_pk_cosmo_marg_vs_fixed_GSMF_CGD.png',
        title='GSMF+CGD cosmology: hydro marginalized (red) vs fixed (blue) '
              '— with posterior-predictive summary'),
    'subgrid_gsmf': dict(marg_trial='GSMF_7p_pk', fixed_trial='GSMF_5p_fid_cosmo',
        marg_label=CL, fixed_label=FL_C, shared_labels=SUBGRID_SHARED,
        limits=SUBGRID_LIMS, out_name='plot_summary_pk_subgrid_marg_vs_fixed.png',
        title='GSMF subgrid: cosmology marginalized (red) vs fixed (blue) '
              '— with posterior-predictive summary'),
    'subgrid_cgd': dict(marg_trial='GSMF_CGD_7p_pk', fixed_trial='GSMF_CGD_5p_fid_cosmo',
        marg_label=CL, fixed_label=FL_C, shared_labels=SUBGRID_SHARED,
        limits=SUBGRID_LIMS, out_name='plot_summary_pk_subgrid_marg_vs_fixed_GSMF_CGD.png',
        title='GSMF+CGD subgrid: cosmology marginalized (red) vs fixed (blue) '
              '— with posterior-predictive summary'),
}


if __name__ == '__main__':
    args = sys.argv[1:]
    cache = {'dd': None}
    # same-parameter-space headline figures
    for key in (args or list(FIGURES)):
        if key in FIGURES:
            trials, labels, out = FIGURES[key]
            make(trials, labels, out, data_cache=cache)
    # marginalized-vs-fixed figures (with summary panels)
    for key in (args or list(MARGFIX)):
        if key in MARGFIX:
            make_marg_fixed(data_cache=cache, **MARGFIX[key])
    print('\nDone.')
