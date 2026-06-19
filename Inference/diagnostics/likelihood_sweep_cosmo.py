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
  likelihood_sweep_omega_m.png   (LL vs omega_m at sigma_8 = fiducial)
  likelihood_sweep_sigma_8.png   (LL vs sigma_8 at omega_m = fiducial)
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


if __name__ == '__main__':
    main()
