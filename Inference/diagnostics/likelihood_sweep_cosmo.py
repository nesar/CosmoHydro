"""Quick likelihood sweep over cosmology, hydro held at the project fiducial.

What this answers
-----------------
  "With hydro pinned at the FIDUCIAL values, does the per-observable likelihood
   actually prefer high omega_m (rail to the box edge), or does it peak near the
   fiducial?"  Decomposes the pull by observable (GSMF vs CGD).

It reuses run_mcmc's own loaders + log_likelihood, so the emulators, data, mass
cuts, and units are identical to the MCMC. No MCMC — just a grid of emulator
evaluations (fast).

Outputs (in this directory)
---------------------------
  likelihood_sweep_omega_m.png   (1D: LL vs omega_m at sigma_8 = fiducial)
  likelihood_sweep_sigma_8.png   (1D: LL vs sigma_8 at omega_m = fiducial)
  likelihood_sweep_2d.png        (2D: LL heatmaps over (omega_m, sigma_8),
                                  one panel per case: GSMF, CGD, GSMF+CGD)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INFER = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, INFER)
import run_mcmc as R           # reuse the exact loaders + log_likelihood
from cosmo_hydro_emu.snapshot_utils import SNAPSHOT_IDS

OUT = HERE
CONFIG = os.path.join(INFER, 'configs', 'GSMF_CGD_2cosmo.yaml')

# Fiducial point (hydro + cosmology), scaled units in PARAM_NAME order.
FID_HYDRO = [3.0, 0.5, 0.8, 0.51, 0.13]      # kappa_w, e_w, M_seed/1e6, v_kin/1e4, eps_kin/1e1
FID_OM, FID_S8 = 0.14176, 0.8102
OM_RANGE = (0.12, 0.155)                      # design box
S8_RANGE = (0.70, 0.90)
OBS_LIST = ['GSMF', 'CGD']

# Two live cosmology priors (both fiducial-centered Gaussians, mu == fiducial):
#   moderate = configs/_defaults.yaml ;  Planck = the *_pk configs.
PRIOR_SIGMA = {'omega_m': 0.005, 'sigma_8': 0.03}     # moderate
PK_SIGMA    = {'omega_m': 0.0011, 'sigma_8': 0.006}   # Planck-width (*_pk)
# params_list rows for R.ln_prior: [name, init, lower, upper]
_PRIOR_PLIST = [['omega_m', FID_OM, OM_RANGE[0], OM_RANGE[1]],
                ['sigma_8', FID_S8, S8_RANGE[0], S8_RANGE[1]]]
_MOD_GP = {0: (FID_OM, PRIOR_SIGMA['omega_m']), 1: (FID_S8, PRIOR_SIGMA['sigma_8'])}
_PK_GP  = {0: (FID_OM, PK_SIGMA['omega_m']),    1: (FID_S8, PK_SIGMA['sigma_8'])}
COLORS = {'GSMF': 'tab:blue', 'CGD': 'tab:green'}


def build_observable(obs, cfg, design_params):
    """Load the z=0 emulator + obs data exactly as run_mcmc does (single-z path)."""
    model_dir = os.path.join(INFER, cfg['data']['model_dir'])
    exp_variance = cfg['data']['exp_variance']
    z_index = cfg['data'].get('z_index', 0)
    y_vals, y_ind = R.prepare_observable(obs, design_params, cfg)
    last = len(SNAPSHOT_IDS) - 1
    multiz_z0 = os.path.join(model_dir, f'{R.MODEL_PREFIX[obs]}_multiz',
                             f'multivariate_model_z_index{last}')
    if os.path.exists(multiz_z0 + '.pkl'):
        model = R.load_model(multiz_z0, design_params, y_vals, y_ind, exp_variance)
    else:
        model = R.load_model(os.path.join(model_dir, f'{R.MODEL_PREFIX[obs]}_multivariate_model_z_index{z_index}'),
                             design_params, y_vals, y_ind, exp_variance)
    data = R.load_obs_data(obs, cfg)
    return dict(model=model, x_grid=y_ind, data=data)


def ll(obs_pack, om, s8):
    theta = np.array(FID_HYDRO + [om, s8])
    return R.log_likelihood(theta, obs_pack['x_grid'], obs_pack['model'],
                            obs_pack['data']['x'], obs_pack['data']['y'],
                            obs_pack['data']['yerr'],
                            fixed_params={}, param_names=R.PARAM_NAME, redshift=0)


def sweep(packs, param, grid, fixed_other):
    """Return {obs: LL array} + total, sweeping `param` over grid."""
    out = {o: np.zeros_like(grid) for o in packs}
    for k, v in enumerate(grid):
        for o, pack in packs.items():
            om, s8 = (v, fixed_other) if param == 'om' else (fixed_other, v)
            out[o][k] = ll(pack, om, s8)
    out['total'] = sum(out[o] for o in packs)
    return out


def plot_sweep(grid, res, xlabel, fid, edge_lo, edge_hi, title, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    for o in OBS_LIST:
        ax.plot(grid, res[o] - res[o].max(), color=COLORS[o], lw=2, label=f'{o} (ΔlnL)')
    ax.plot(grid, res['total'] - res['total'].max(), 'k-', lw=2.5, label='GSMF+CGD (ΔlnL)')
    ax.axvline(fid, color='gray', ls=':', lw=1.5, label='fiducial')
    ax.axvline(edge_lo, color='r', ls='--', lw=1, alpha=0.6)
    ax.axvline(edge_hi, color='r', ls='--', lw=1, alpha=0.6, label='design-box edge')
    ax.set_ylim(-30, 2)
    ax.set_xlabel(xlabel); ax.set_ylabel(r'$\Delta \ln\,\mathcal{L}$ (max-subtracted)')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc='upper right'); ax.grid(True, ls=':', alpha=0.3)
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {p}')


def _trunc_logprior(x, mu, sigma, win):
    """1D log Gaussian prior, -inf (here: very negative) outside hard window."""
    lp = -0.5 * ((x - mu) / sigma) ** 2
    if win is not None:
        lp = np.where((x < win[0]) | (x > win[1]), -np.inf, lp)
    return lp


# Priors to compare: (label, sigma_om, sigma_s8, hard-window or None=design box).
DESIGN_WIN = {'om': (0.12, 0.155), 's8': (0.70, 0.90)}
TRUNC_WIN_1S = {'om': (0.14066, 0.14286), 's8': (0.8042, 0.8162)}  # fiducial +/-1 Planck sigma
PRIOR_SET = [
    ('flat (likelihood only)',          None,   None,   None,        'tab:gray'),
    ('moderate Gaussian (0.005/0.03)',  0.005,  0.03,   DESIGN_WIN,  'tab:orange'),
    ('Planck Gaussian, no hard cut',    0.0011, 0.006,  DESIGN_WIN,  'tab:green'),
    ('Planck Gaussian + ±1σ hard cut',  0.0011, 0.006,  TRUNC_WIN_1S,'tab:red'),
]


def plot_prior_comparison(packs, fname):
    """For GSMF+CGD (hydro at fiducial), overlay the 1D posterior on omega_m and
    sigma_8 under several priors, to show which yields a COMPLETE (closed)
    posterior vs which rails to the box edge or gets cut at a hard wall."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, param, (lo, hi), mu, label in [
            (axes[0], 'om', OM_RANGE, FID_OM, r'$\omega_m \equiv \Omega_m h^2$'),
            (axes[1], 's8', S8_RANGE, FID_S8, r'$\sigma_8$')]:
        grid = np.linspace(lo, hi, 600)
        # GSMF+CGD log-likelihood along this axis (other param at fiducial)
        lnL = np.zeros_like(grid)
        for k, v in enumerate(grid):
            om, s8 = (v, FID_S8) if param == 'om' else (FID_OM, v)
            lnL[k] = sum(ll(p, om, s8) for p in packs.values())
        # likelihood (max-subtracted), shown as a filled gray curve
        ax.fill_between(grid, np.exp(np.clip(lnL - lnL.max(), -40, 0)), color='0.85',
                        label='likelihood (flat-prior)')
        for plabel, s_om, s_s8, win, color in PRIOR_SET:
            if s_om is None:                       # flat: posterior = likelihood
                post = lnL.copy()
            else:
                sig = s_om if param == 'om' else s_s8
                w = None if win is None else win[param]
                post = lnL + _trunc_logprior(grid, mu, sig, w)
            if not np.isfinite(post).any():
                continue
            y = np.exp(post - np.nanmax(post[np.isfinite(post)]))
            y[~np.isfinite(post)] = 0.0
            ax.plot(grid, y, color=color, lw=2, label=plabel)
        ax.axvline(mu, color='k', ls=':', lw=1.2)
        for v in DESIGN_WIN[param]:
            ax.axvline(v, color='k', ls='-', lw=0.8, alpha=0.4)
        for v in TRUNC_WIN_1S[param]:
            ax.axvline(v, color='tab:red', ls='--', lw=0.8, alpha=0.5)
        ax.set_xlabel(label); ax.set_ylabel('posterior (peak-normalized)')
        ax.set_ylim(0, 1.08); ax.grid(True, ls=':', alpha=0.3)
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('GSMF+CGD cosmology posterior under different priors (hydro at fiducial)\n'
                 'dotted=fiducial, thin solid=design-box edge, dashed red=±1σ hard window',
                 fontsize=11)
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {p}')


def sweep_2d(packs, om_grid, s8_grid):
    """Return {case: 2D LL array of shape (len(s8_grid), len(om_grid))}.
    Cases: each observable plus the GSMF+CGD combination."""
    ll_obs = {o: np.zeros((s8_grid.size, om_grid.size)) for o in packs}
    for j, s8 in enumerate(s8_grid):
        for i, om in enumerate(om_grid):
            for o, pack in packs.items():
                ll_obs[o][j, i] = ll(pack, om, s8)
    cases = dict(ll_obs)
    cases['GSMF+CGD'] = sum(ll_obs[o] for o in packs)
    return cases


def ln_prior_grid(om_grid, s8_grid, gp=_MOD_GP):
    """2D log-prior over (omega_m, sigma_8) using the exact MCMC ln_prior.
    gp selects the prior: _MOD_GP (moderate) or _PK_GP (Planck-width).
    Shape (len(s8), len(om))."""
    lp = np.zeros((s8_grid.size, om_grid.size))
    for j, s8 in enumerate(s8_grid):
        for i, om in enumerate(om_grid):
            lp[j, i] = R.ln_prior(np.array([om, s8]), _PRIOR_PLIST,
                                  flat_indices=[], gaussian_priors=gp)
    return lp


def _panel(ax, fig, OM, S8, field, om_grid, s8_grid, title, cbar_label):
    d = np.clip(field - np.nanmax(field), -30, 0)
    pcm = ax.pcolormesh(OM, S8, d, cmap='viridis', shading='gouraud', vmin=-30, vmax=0)
    ax.contour(OM, S8, d, levels=[-11.83, -6.17, -2.30],
               colors='w', linewidths=0.8, alpha=0.7)
    ax.axvline(FID_OM, color='red', ls='--', lw=1.2)
    ax.axhline(FID_S8, color='red', ls='--', lw=1.2)
    j, i = np.unravel_index(np.nanargmax(field), field.shape)
    ax.plot(om_grid[i], s8_grid[j], marker='*', ms=14, mfc='gold', mec='k', mew=0.8,
            label=f'peak ({om_grid[i]:.4f}, {s8_grid[j]:.3f})')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r'$\omega_m \equiv \Omega_m h^2$')
    ax.set_ylabel(r'$\sigma_8$')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    fig.colorbar(pcm, ax=ax, label=cbar_label, shrink=0.85)


def plot_2d(packs, om_grid, s8_grid, cases, fname):
    """Three rows (hydro fixed at fiducial):
      row 0: likelihood                        (full design box)
      row 1: posterior x moderate prior        (full design box)
      row 2: posterior x Planck prior (*_pk)   (ZOOMED near fiducial so the
             tight Planck posterior is actually visible)."""
    order = [c for c in ('GSMF', 'CGD', 'GSMF+CGD') if c in cases]
    OM, S8 = np.meshgrid(om_grid, s8_grid)
    lp_mod = ln_prior_grid(om_grid, s8_grid, _MOD_GP)

    # Zoomed grid for the Planck row: fiducial +/- 6 Planck sigma (clamped to box).
    def _zoom(mu, sig, lo, hi):
        return (max(lo, mu - 6 * sig), min(hi, mu + 6 * sig))
    omz = np.linspace(*_zoom(FID_OM, PK_SIGMA['omega_m'], *OM_RANGE), om_grid.size)
    s8z = np.linspace(*_zoom(FID_S8, PK_SIGMA['sigma_8'], *S8_RANGE), s8_grid.size)
    cases_z = sweep_2d(packs, omz, s8z)
    lp_pk = ln_prior_grid(omz, s8z, _PK_GP)
    OMZ, S8Z = np.meshgrid(omz, s8z)

    fig, axes = plt.subplots(3, len(order), figsize=(5.2 * len(order), 13.5),
                             constrained_layout=True, squeeze=False)
    print('\nposterior max location (omega_m, sigma_8):')
    for col, case in enumerate(order):
        _panel(axes[0, col], fig, OM, S8, cases[case], om_grid, s8_grid,
               f'{case} — likelihood', r'$\Delta\ln\mathcal{L}$')
        post_mod = cases[case] + lp_mod
        _panel(axes[1, col], fig, OM, S8, post_mod, om_grid, s8_grid,
               f'{case} — posterior x moderate prior', r'$\Delta\ln\mathcal{P}$')
        post_pk = cases_z[case] + lp_pk
        _panel(axes[2, col], fig, OMZ, S8Z, post_pk, omz, s8z,
               f'{case} — posterior x Planck prior (zoom)', r'$\Delta\ln\mathcal{P}$')
        jm, im = np.unravel_index(np.nanargmax(post_mod), post_mod.shape)
        jp, ip = np.unravel_index(np.nanargmax(post_pk), post_pk.shape)
        print(f'  {case:10s}: moderate=({om_grid[im]:.4f}, {s8_grid[jm]:.3f})  '
              f'Planck=({omz[ip]:.4f}, {s8z[jp]:.4f})')
    fig.suptitle('Hydro fixed at fiducial. Rows: likelihood / x moderate prior '
                 f'(σ={PRIOR_SIGMA["omega_m"]}, {PRIOR_SIGMA["sigma_8"]}) / x Planck prior '
                 f'(σ={PK_SIGMA["omega_m"]}, {PK_SIGMA["sigma_8"]}, zoomed). '
                 'Red lines = fiducial.', fontsize=12)
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {p}')


def main():
    cfg = R.load_config(CONFIG)
    design_params = R.load_design(os.path.join(INFER, cfg['data']['design_file']),
                                  start_sim_idx=cfg['data'].get('start_sim_idx', 1),
                                  num_sims=cfg['data']['num_sims'])
    packs = {o: build_observable(o, cfg, design_params) for o in OBS_LIST}

    # peak locations (per observable) along omega_m at fiducial sigma_8
    om_grid = np.linspace(*OM_RANGE, 60)
    res_om = sweep(packs, 'om', om_grid, FID_S8)
    print('\nomega_m of max lnL (hydro+sigma_8 at fiducial):')
    for o in OBS_LIST + ['total']:
        print(f'  {o:6s}: omega_m_peak = {om_grid[np.argmax(res_om[o])]:.4f}  '
              f'(fiducial {FID_OM:.4f}, box edge {OM_RANGE[1]:.4f})')
    plot_sweep(om_grid, res_om, r'$\omega_m \equiv \Omega_m h^2$', FID_OM,
               OM_RANGE[0], OM_RANGE[1],
               'Likelihood vs $\\omega_m$ (hydro + $\\sigma_8$ fixed at fiducial)',
               'likelihood_sweep_omega_m.png')

    s8_grid = np.linspace(*S8_RANGE, 60)
    res_s8 = sweep(packs, 's8', s8_grid, FID_OM)
    print('\nsigma_8 of max lnL (hydro+omega_m at fiducial):')
    for o in OBS_LIST + ['total']:
        print(f'  {o:6s}: sigma_8_peak = {s8_grid[np.argmax(res_s8[o])]:.4f}  '
              f'(fiducial {FID_S8:.4f})')
    plot_sweep(s8_grid, res_s8, r'$\sigma_8$', FID_S8, S8_RANGE[0], S8_RANGE[1],
               'Likelihood vs $\\sigma_8$ (hydro + $\\omega_m$ fixed at fiducial)',
               'likelihood_sweep_sigma_8.png')

    # 2D heatmaps over (omega_m, sigma_8), hydro fixed at fiducial
    om2 = np.linspace(*OM_RANGE, 70)
    s82 = np.linspace(*S8_RANGE, 70)
    cases = sweep_2d(packs, om2, s82)
    print('\n2D max-lnL location (omega_m, sigma_8):')
    for c in ('GSMF', 'CGD', 'GSMF+CGD'):
        j, i = np.unravel_index(np.argmax(cases[c]), cases[c].shape)
        print(f'  {c:10s}: ({om2[i]:.4f}, {s82[j]:.3f})')
    plot_2d(packs, om2, s82, cases, 'likelihood_sweep_2d.png')

    # posterior under different priors (the "which prior gives a complete
    # posterior" exploration)
    plot_prior_comparison(packs, 'prior_comparison.png')


if __name__ == '__main__':
    main()
