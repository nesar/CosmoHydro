"""GSMF / CGD likelihood components for the cosmology-target (Pk) pipeline.

Registers `gsmf` and `cgd` (and `fgas`) target kinds in run_mcmc_cosmo, wrapping
the cosmo_hydro_emu emulator + observations (from the Inference/ pipeline) as
callables ``like(params7) -> log-likelihood``. This lets the KiDS power-spectrum
likelihood be fit JOINTLY with GSMF + CGD in a single 7-parameter MCMC.

Design
------
- The GSMF/CGD data, emulators and likelihood are loaded EXACTLY as
  Inference/run_mcmc.py loads them for a z=0 run (same design, same model files,
  same obs data, same cosmo_hydro_emu.mcmc.log_likelihood), so the joint fit's
  GSMF/CGD term is identical to the standalone GSMF+CGD runs.
- The Pk pipeline's ln_prob_global already reconstructs the full 7-vector
  (params7) and calls each component with it. We therefore pass params7 straight
  to log_likelihood with param_names=PARAM_NAME and empty fixed_params, so all 7
  values are used as the emulator input.

Paths resolve relative to Inference/run_mcmc.py's location (via its helpers), so
this works when driven from Inference_cosmo/.
"""
import os
import yaml
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
INFER = os.path.abspath(os.path.join(_HERE, '..', 'Inference'))

import run_mcmc as R                                  # Inference/run_mcmc.py (on sys.path)
from cosmo_hydro_emu.mcmc import log_likelihood        # noqa: E402
from cosmo_hydro_emu.load_hacc import PARAM_NAME        # noqa: E402
from cosmo_hydro_emu.snapshot_utils import SNAPSHOT_IDS  # noqa: E402

_OBS_KIND = {'gsmf': 'GSMF', 'cgd': 'CGD', 'fgas': 'fGas'}


def _infer_cfg():
    """Inference/ defaults: data paths + obs_dirs for GSMF/CGD."""
    with open(os.path.join(INFER, 'configs', '_defaults.yaml')) as f:
        return yaml.safe_load(f) or {}


class ObsLikelihood:
    """Single GSMF/CGD observable; callable on the full 7-param vector.

    with_emu_variance=True adds the GP emulator predictive variance to the error
    budget (+ the logdet normalisation term) — recommended so the GSMF/CGD chi2
    is not artificially tight (see diagnose_emu_variance.py)."""

    def __init__(self, obs, x_grid, model, x, y, yerr, with_emu_variance=False):
        self.obs, self.x_grid, self.model = obs, x_grid, model
        self.x, self.y, self.yerr = x, y, yerr
        self.with_emu_variance = with_emu_variance

    def __call__(self, params7):
        return log_likelihood(
            np.asarray(params7, dtype=float), self.x_grid, self.model,
            self.x, self.y, self.yerr,
            fixed_params=None, param_names=list(PARAM_NAME),
            case_label=self.obs, redshift=0.0, z_all=None,
            with_emu_variance=self.with_emu_variance)


def _ghirardini_frac_err(r):
    """Radius-dependent fractional error from Ghirardini+2019 (X-COP), the only
    REAL error bars among the CGD reference datasets. Interpolated in log(r/R500),
    floored at 0.15. See audit_joint_calibration.py / CALIBRATION_FIXES.md."""
    gfile = os.path.join(_infer_cfg()['obs_dirs']['cgd'], 'ghirardini2019_rho_z0.txt')
    g = np.loadtxt(gfile)
    rg, y, lo, hi = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
    frac = 0.5 * (np.abs(lo) + np.abs(hi)) / y
    return np.clip(np.interp(np.log(r), np.log(rg), frac), 0.15, None)


def _apply_err_fixes(obs, od, spec):
    """Opt-in, reversible error-model fixes (see CALIBRATION_FIXES.md).

    GSMF, `err_jacobian: true`  : transform yerr (which load_gsmf_obs returns in
        phi = linear-density space) into the 10^phi space the residual lives in:
        sigma_(10^phi) = ln(10) * 10^phi * sigma_phi.  Undoes the (ln10)^2 ~ 5.3x
        chi2 inflation verified in audit_joint_calibration.py.
    CGD, `err_model: 'ghirardini19'` : replace the invented flat 5% error with
        the radius-dependent fractional error measured by Ghirardini+2019,
        floored at 15%.  `frac_err: <float>` gives a constant fraction instead.
    Defaults (no keys) leave the legacy behaviour untouched.
    """
    if obs == 'GSMF' and spec.get('err_jacobian', False):
        od = dict(od)
        od['yerr'] = np.log(10.0) * od['y'] * od['yerr']
        print('    [gsmf_cgd_target] GSMF yerr -> 10^phi space (x ln10*10^phi)')
    if obs == 'CGD':
        if spec.get('err_model') == 'ghirardini19':
            od = dict(od)
            frac = _ghirardini_frac_err(od['x'])
            od['yerr'] = frac * od['y']
            print('    [gsmf_cgd_target] CGD yerr -> Ghirardini+19 radius-dependent '
                  f'(frac {frac.min():.2f}-{frac.max():.2f})')
        elif 'frac_err' in spec:
            od = dict(od)
            od['yerr'] = float(spec['frac_err']) * od['y']
            print(f'    [gsmf_cgd_target] CGD yerr -> flat {spec["frac_err"]:.0%}')
    return od


def _build_component(kind, with_emu_variance=False, spec=None):
    """Load the GSMF/CGD emulator + obs exactly as Inference/run_mcmc.py does."""
    spec = spec or {}
    obs = _OBS_KIND[kind]
    cfg = _infer_cfg()
    data = cfg['data']
    design = R.load_design(os.path.join(INFER, data['design_file']),
                           start_sim_idx=data.get('start_sim_idx', 1),
                           num_sims=data['num_sims'])
    y_vals, y_ind = R.prepare_observable(obs, design, cfg)

    model_dir = os.path.join(INFER, data['model_dir'])
    exp_variance = data['exp_variance']
    z_index = data.get('z_index', 0)
    # z=0 model: prefer the multiz dir's last snapshot (z=0), else standalone —
    # identical to Inference/run_mcmc.py's single-z path.
    multiz_dir = os.path.join(model_dir, f'{R.MODEL_PREFIX[obs]}_multiz')
    last = len(SNAPSHOT_IDS) - 1
    z0_file = os.path.join(multiz_dir, f'multivariate_model_z_index{last}.pkl')
    if os.path.exists(z0_file):
        model = R.load_model(
            os.path.join(multiz_dir, f'multivariate_model_z_index{last}'),
            design, y_vals, y_ind, exp_variance)
    else:
        model = R.load_model(
            os.path.join(model_dir, f'{R.MODEL_PREFIX[obs]}_multivariate_model_z_index{z_index}'),
            design, y_vals, y_ind, exp_variance)

    od = R.load_obs_data(obs, cfg)
    od = _apply_err_fixes(obs, od, spec)
    print(f'    [gsmf_cgd_target] {obs}: {len(od["x"])} obs points, '
          f'model x_grid {np.asarray(y_ind).shape}'
          + ('  [+emu variance]' if with_emu_variance else ''))
    return ObsLikelihood(obs, y_ind, model, od['x'], od['y'], od['yerr'],
                         with_emu_variance=with_emu_variance)


def register(registry):
    """Add gsmf/cgd/fgas target kinds to a run_mcmc_cosmo EXTRA_TARGET_KINDS dict.

    builder signature is (spec, get_emu) -> (name, like). Per-target config key
    `emu_variance: true` adds the GP emulator variance to that observable's error
    budget. get_emu (the Pk emulator getter) is unused here.
    """
    def _make(kind):
        def builder(spec, get_emu):
            return _OBS_KIND[kind], _build_component(
                kind, with_emu_variance=bool(spec.get('emu_variance', False)),
                spec=spec)
        return builder
    for kind in ('gsmf', 'cgd', 'fgas'):
        registry[kind] = _make(kind)
