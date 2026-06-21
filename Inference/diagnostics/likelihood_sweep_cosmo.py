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

# MCMC cosmology prior (matches configs/_defaults.yaml): moderate fiducial-
# centered truncated Gaussian. mu == fiducial; sigmas below.
PRIOR_SIGMA = {'omega_m': 0.005, 'sigma_8': 0.03}
# params_list rows for R.ln_prior: [name, init, lower, upper]
_PRIOR_PLIST = [['omega_m', FID_OM, OM_RANGE[0], OM_RANGE[1]],
                ['sigma_8', FID_S8, S8_RANGE[0], S8_RANGE[1]]]
_PRIOR_GP = {0: (FID_OM, PRIOR_SIGMA['omega_m']),
             1: (FID_S8, PRIOR_SIGMA['sigma_8'])}
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
    ax.legend(fontsize=9, loc='lower center'); ax.grid(True, ls=':', alpha=0.3)
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


def ln_prior_grid(om_grid, s8_grid):
    """2D log-prior over (omega_m, sigma_8), using the exact MCMC ln_prior
    (configs/_defaults.yaml cosmology prior). Shape (len(s8), len(om))."""
    lp = np.zeros((s8_grid.size, om_grid.size))
    for j, s8 in enumerate(s8_grid):
        for i, om in enumerate(om_grid):
            lp[j, i] = R.ln_prior(np.array([om, s8]), _PRIOR_PLIST,
                                  flat_indices=[], gaussian_priors=_PRIOR_GP)
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
    ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
    fig.colorbar(pcm, ax=ax, label=cbar_label, shrink=0.85)


def plot_2d(om_grid, s8_grid, cases, fname):
    """Two rows: top = likelihood, bottom = posterior (likelihood + MCMC prior)."""
    order = [c for c in ('GSMF', 'CGD', 'GSMF+CGD') if c in cases]
    OM, S8 = np.meshgrid(om_grid, s8_grid)
    lp = ln_prior_grid(om_grid, s8_grid)
    fig, axes = plt.subplots(2, len(order), figsize=(5.2 * len(order), 9.0),
                             constrained_layout=True, squeeze=False)
    print('\nposterior (lnL + MCMC prior) max location (omega_m, sigma_8):')
    for col, case in enumerate(order):
        _panel(axes[0, col], fig, OM, S8, cases[case], om_grid, s8_grid,
               f'{case} — likelihood', r'$\Delta\ln\mathcal{L}$')
        post = cases[case] + lp
        _panel(axes[1, col], fig, OM, S8, post, om_grid, s8_grid,
               f'{case} — posterior (× MCMC prior)', r'$\Delta\ln\mathcal{P}$')
        j, i = np.unravel_index(np.nanargmax(post), post.shape)
        print(f'  {case:10s}: ({om_grid[i]:.4f}, {s8_grid[j]:.3f})')
    fig.suptitle('Hydro fixed at fiducial — top: likelihood, bottom: posterior '
                 f'(MCMC prior σ_ωm={PRIOR_SIGMA["omega_m"]}, σ_σ8={PRIOR_SIGMA["sigma_8"]}). '
                 'Red lines = fiducial cosmology.', fontsize=12)
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
    plot_2d(om2, s82, cases, 'likelihood_sweep_2d.png')


if __name__ == '__main__':
    main()
