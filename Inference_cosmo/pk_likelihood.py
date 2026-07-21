"""
Likelihoods for cosmology-target inference from the emulated power spectra.

Emulator composition
--------------------
The total-matter hydro power spectrum is composed as

    P_hydro(k, z; theta) = S(k, z; theta_7) * P_go(k, z; omega_m, sigma_8)

where S is the suppression-ratio emulator (7 params; the z=0 model is the
pre-existing notebook-trained ``models/Pk_multivariate_model_z_index0``) and
P_go the gravity-only emulator (2 cosmology params). No standalone hydro-P(k)
emulator is trained: the ratio cancels realization noise, and the GO spectrum
carries the cosmology dependence.

Redshift interpolation between the 5 snapshots (z = 0, 0.1, 0.5, 1, 2):
  * log10 P_go is interpolated GROWTH-SCALED: g = log10[P_go / D^2(z)] is
    linear in scale factor a between the bracketing snapshots, with D(z) the
    linear growth factor at the sampled cosmology. Validated at data level:
    median ~1%, max 3.4% error when reconstructing across the widest gap
    (see diagnostics/pk_data_validation.txt); target redshifts sit closer to
    snapshots than that worst case.
  * the ratio S is interpolated linearly in a (S evolves slowly and smoothly).

Likelihood components (select by 'kind' in the config):
  pm    : direct matter-P(k) data points, e.g. KiDS-Legacy Pm(k, z_fid).
          Gaussian in Pm with the published covariance + emulator variance.
  amod  : the A_mod scalar constraint. The emulated suppression S(k, z=0) is
          projected onto the Amon-Efstathiou template
              S_template(k) = 1 + (A_mod - 1) * (1 - P_L/P_go)
          by weighted least squares over a log-k grid, giving the model-
          implied A_mod(theta); the likelihood is Gaussian in that ONE number
          (0.858 +/- 0.052 for DES Y3 + Planck prior). Treating each k-bin as
          an independent datum would badly overcount a single published
          constraint.
  boss  : BOSS DR12 galaxy P0 (+P2) with the Patchy covariance (Hartlap
          corrected), modeled as linear-bias Kaiser RSD of the emulated
          matter P(k, z_eff) with Alcock-Paczynski dilation from the BOSS
          fiducial cosmology. NO survey-window convolution is applied ->
          restrict to k ~ [0.03, 0.15] h/Mpc and treat as a methods-level
          constraint (see README).

MCMC parameter conventions follow Inference/: theta contains the free subset
of the 7 design parameters (PARAM_NAME order), then any nuisance parameters
(BOSS bias/shot-noise), with fixed_params filling the rest.
"""

import os
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'codes'))

from cosmo_hydro_emu.emu import emulate                       # noqa: E402
from cosmo_hydro_emu.load_hacc import PARAM_NAME              # noqa: E402

from pk_data import PK_REDSHIFTS, PK_REDSHIFT_TAGS, COSMO_COLS  # noqa: E402
from linear_theory import (                                   # noqa: E402
    FIXED_COSMO, LinearPk, growth_factor_and_rate,
    comoving_distance, Ez, omega_to_Omega_m,
)

# design-vector indices (FinalDesign column order)
IDX_OMEGA_M = 5
IDX_SIGMA_8 = 6


# ---------------------------------------------------------------------------
# Emulator wrapper
# ---------------------------------------------------------------------------
class PkEmulator:
    """Loads the per-snapshot GP models once and predicts P_hydro / P_go / S
    at arbitrary z in [0, 2] via snapshot interpolation.

    Parameters are always the full 7-vector in design order (scaled units);
    the GO models internally use only the cosmology columns.
    """

    def __init__(self, ztags=None, need_ratio=True, need_go=True):
        from train_pk_emulators import load_pk_model
        from pk_data import load_design
        self.ztags = ztags or PK_REDSHIFT_TAGS
        self.z_snaps = np.array([float(z) for z in self.ztags])
        if not np.all(np.diff(self.z_snaps) > 0):
            raise ValueError('ztags must be sorted ascending in z')
        params_all = load_design()
        self.models = {}
        for zt in self.ztags:
            if need_ratio:
                m, info = load_pk_model('ratio', zt, params_all)
                self.models[('ratio', zt)] = (m, info)
            if need_go:
                m, info = load_pk_model('logP_go', zt, params_all)
                self.models[('logP_go', zt)] = (m, info)
        any_info = next(iter(self.models.values()))[1]
        self.k_grid = any_info['k']

    # -- snapshot-level predictions --------------------------------------
    def _predict_snap(self, quantity, ztag, params7):
        model, info = self.models[(quantity, ztag)]
        p = params7[COSMO_COLS] if info['param_cols'] == COSMO_COLS else params7
        mean, std = emulate(model, np.asarray(p))
        return mean[:, 0], std[:, 0]

    def _bracket(self, z):
        if z < self.z_snaps[0] - 1e-9 or z > self.z_snaps[-1] + 1e-9:
            raise ValueError(f'z={z} outside snapshot range')
        j = int(np.clip(np.searchsorted(self.z_snaps, z) - 1, 0,
                        len(self.z_snaps) - 2))
        return j, j + 1

    # -- public API -------------------------------------------------------
    def ratio(self, z, params7):
        """Suppression S(k, z) on self.k_grid; returns (mean, std)."""
        j1, j2 = self._bracket(z)
        z1, z2 = self.z_snaps[j1], self.z_snaps[j2]
        if abs(z - z1) < 1e-9:
            return self._predict_snap('ratio', self.ztags[j1], params7)
        if abs(z - z2) < 1e-9:
            return self._predict_snap('ratio', self.ztags[j2], params7)
        a1, a2, a = 1 / (1 + z1), 1 / (1 + z2), 1 / (1 + z)
        w = (a - a2) / (a1 - a2)
        m1, s1 = self._predict_snap('ratio', self.ztags[j1], params7)
        m2, s2 = self._predict_snap('ratio', self.ztags[j2], params7)
        return w * m1 + (1 - w) * m2, np.sqrt((w * s1) ** 2 + ((1 - w) * s2) ** 2)

    def P_go(self, z, params7):
        """Gravity-only P(k, z) on self.k_grid; growth-scaled interpolation.

        Returns (P, sigma_P).
        """
        omega_m = params7[IDX_OMEGA_M]
        Omega_m = omega_to_Omega_m(omega_m)
        j1, j2 = self._bracket(z)
        z1, z2 = self.z_snaps[j1], self.z_snaps[j2]
        if abs(z - z1) < 1e-9 or abs(z - z2) < 1e-9:
            zt = self.ztags[j1] if abs(z - z1) < 1e-9 else self.ztags[j2]
            logP, slog = self._predict_snap('logP_go', zt, params7)
            P = 10 ** logP
            return P, P * np.log(10) * slog
        D_all, _ = growth_factor_and_rate([z1, z, z2], Omega_m)
        D1, Dz, D2 = D_all
        a1, a2, a = 1 / (1 + z1), 1 / (1 + z2), 1 / (1 + z)
        w = (a - a2) / (a1 - a2)
        logP1, s1 = self._predict_snap('logP_go', self.ztags[j1], params7)
        logP2, s2 = self._predict_snap('logP_go', self.ztags[j2], params7)
        g1 = logP1 - 2 * np.log10(D1)
        g2 = logP2 - 2 * np.log10(D2)
        g = w * g1 + (1 - w) * g2
        logP = g + 2 * np.log10(Dz)
        slog = np.sqrt((w * s1) ** 2 + ((1 - w) * s2) ** 2)
        P = 10 ** logP
        return P, P * np.log(10) * slog

    def P_hydro(self, z, params7):
        """Total-matter hydro P(k, z) = S * P_go; returns (P, sigma_P)."""
        S, sS = self.ratio(z, params7)
        Pg, sPg = self.P_go(z, params7)
        P = S * Pg
        sigma = P * np.sqrt((sS / np.maximum(S, 1e-12)) ** 2
                            + (sPg / np.maximum(Pg, 1e-30)) ** 2)
        return P, sigma


# ---------------------------------------------------------------------------
# Component likelihoods
# ---------------------------------------------------------------------------
class PmLikelihood:
    """Direct matter-P(k) data (KiDS-Legacy Pm).

    chi^2 with the published covariance plus (diagonal) emulator predictive
    variance and a fractional z-interpolation systematic added to the model.
    """

    def __init__(self, target, emu, interp_sys_frac=0.01):
        self.t = target
        self.emu = emu
        self.interp_sys_frac = interp_sys_frac
        self.z_unique = np.unique(target['z'])
        for z in self.z_unique:
            if z > emu.z_snaps[-1] or z < emu.z_snaps[0]:
                raise ValueError(f'target z={z} outside emulator range')
        # data k must lie inside the emulator k grid
        kg = emu.k_grid
        if target['k'].min() < kg.min() or target['k'].max() > kg.max():
            raise ValueError(
                'target k outside emulated range — apply k_min/k_max cuts '
                f'({target["k"].min():.4f}..{target["k"].max():.2f} vs grid '
                f'{kg.min():.4f}..{kg.max():.2f})')

    def model_vector(self, params7):
        y_mod = np.empty_like(self.t['y'])
        var_emu = np.empty_like(self.t['y'])
        for z in self.z_unique:
            m = self.t['z'] == z
            P, sP = self.emu.P_hydro(z, params7)
            y_mod[m] = np.interp(self.t['k'][m], self.emu.k_grid, P)
            var_emu[m] = np.interp(self.t['k'][m], self.emu.k_grid, sP) ** 2
        return y_mod, var_emu

    def __call__(self, params7):
        y_mod, var_emu = self.model_vector(params7)
        cov = (self.t['cov']
               + np.diag(var_emu + (self.interp_sys_frac * y_mod) ** 2))
        r = self.t['y'] - y_mod
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return -np.inf
        x = np.linalg.solve(L, r)
        logdet = 2 * np.sum(np.log(np.diag(L)))
        return -0.5 * (x @ x) - 0.5 * logdet


class AmodLikelihood:
    """Scalar A_mod constraint, evaluated at z=0.

    Model-implied A_mod: weighted least squares of the emulated suppression
    S(k) against the template 1 + (A - 1) * t(k), t = 1 - P_L/P_go, over a
    log-uniform k grid in [k_fit_min, k_fit_max]:

        A_hat(theta) = 1 + sum[t (S - 1)] / sum[t^2]

    Gaussian likelihood in A_hat against the published (Amod, sigma).
    """

    def __init__(self, target, emu, linear_pk=None,
                 k_fit_min=0.1, k_fit_max=8.0, n_k_fit=40):
        self.t = target
        self.emu = emu
        self.lin = linear_pk or LinearPk()
        self.k_fit = np.logspace(np.log10(k_fit_min), np.log10(k_fit_max),
                                 n_k_fit)

    def model_amod(self, params7):
        omega_m, sigma_8 = params7[IDX_OMEGA_M], params7[IDX_SIGMA_8]
        S, _ = self.emu.ratio(0.0, params7)
        P_go, _ = self.emu.P_go(0.0, params7)
        S_f = np.interp(self.k_fit, self.emu.k_grid, S)
        Pgo_f = np.interp(self.k_fit, self.emu.k_grid, P_go)
        P_L = self.lin(self.k_fit, 0.0, omega_m, sigma_8)
        t = 1.0 - P_L / Pgo_f
        # only scales where nonlinear enhancement is meaningful contribute
        pos = t > 0.05
        if pos.sum() < 5:
            return np.nan
        return 1.0 + np.sum(t[pos] * (S_f[pos] - 1.0)) / np.sum(t[pos] ** 2)

    def __call__(self, params7):
        A_hat = self.model_amod(params7)
        if not np.isfinite(A_hat):
            return -np.inf
        return -0.5 * ((A_hat - self.t['Amod']) / self.t['sigma']) ** 2


class BossLikelihood:
    """BOSS DR12 P0(+P2) with Kaiser RSD + linear bias + AP dilation.

    APPROXIMATIONS (documented, deliberate):
      * linear Kaiser RSD on the nonlinear matter P(k) — adequate only on
        quasi-linear scales; default k range [0.03, 0.15] h/Mpc.
      * no survey-window convolution (few-% effect on these scales for the
        BOSS window; absorbed partly by the free bias).
      * no fingers-of-god damping, no loop corrections.
    Free nuisance parameters (appended to theta): linear bias b1, and
    optionally a constant shot-noise offset P_sn added to P0.

    The Patchy covariance is inverted once with the Hartlap factor
    (N_mocks - N_d - 2) / (N_mocks - 1).
    """

    def __init__(self, target, emu, n_mu=16):
        self.t = target
        self.emu = emu
        self.z_eff = target['z_eff']
        nd = target['y'].size
        hartlap = (target['n_mocks'] - nd - 2) / (target['n_mocks'] - 1)
        self.icov = hartlap * np.linalg.inv(target['cov'])
        self.nk = target['k'].size
        # Gauss-Legendre nodes on mu in [0, 1] (even integrand)
        x, w = leggauss(n_mu)
        self.mu = 0.5 * (x + 1.0)
        self.wmu = 0.5 * w
        # fiducial AP quantities (computed once)
        fid = target['fiducial']
        self.chi_fid = comoving_distance(self.z_eff, fid['Omega_m'],
                                         h=fid['h'])
        self.E_fid = float(Ez(self.z_eff, fid['Omega_m']))
        self.h_fid = fid['h']

    def model_vector(self, params7, b1, P_sn=0.0):
        omega_m = params7[IDX_OMEGA_M]
        Omega_m = omega_to_Omega_m(omega_m)
        _, f = growth_factor_and_rate(self.z_eff, Omega_m)

        # AP: distances in Mpc/h so the h-unit k of data and model agree
        chi = comoving_distance(self.z_eff, Omega_m)          # Mpc (h=0.6766)
        q_perp = (chi * FIXED_COSMO['h']) / (self.chi_fid * self.h_fid)
        E = float(Ez(self.z_eff, Omega_m))
        q_par = self.E_fid / E     # H_fid/H in h-units

        P_m, _ = self.emu.P_hydro(self.z_eff, params7)
        lnk, lnP = np.log(self.emu.k_grid), np.log(P_m)

        k_obs = self.t['k']
        # true (k, mu) corresponding to observed (k_obs, mu_obs)
        mu_o = self.mu[None, :]
        k_o = k_obs[:, None]
        F = q_par / q_perp
        k_true = (k_o / q_perp) * np.sqrt(1 + mu_o ** 2 * (1 / F ** 2 - 1))
        mu_true2 = (mu_o ** 2 / F ** 2) / (1 + mu_o ** 2 * (1 / F ** 2 - 1))

        P_interp = np.exp(np.interp(np.log(k_true), lnk, lnP))
        kaiser = (b1 + f * mu_true2) ** 2
        P2d = kaiser * P_interp / (q_perp ** 2 * q_par)

        P0 = np.sum(P2d * self.wmu[None, :], axis=1) + P_sn
        if self.t['use_quad']:
            L2 = 0.5 * (3 * self.mu ** 2 - 1)
            P2 = 5.0 * np.sum(P2d * L2[None, :] * self.wmu[None, :], axis=1)
            return np.concatenate([P0, P2])
        return P0

    def __call__(self, params7, b1, P_sn=0.0):
        if b1 <= 0:
            return -np.inf
        y_mod = self.model_vector(params7, b1, P_sn)
        r = self.t['y'] - y_mod
        return -0.5 * (r @ self.icov @ r)


# ---------------------------------------------------------------------------
# Global posterior assembly (module-level state so a forked emcee Pool can
# evaluate ln_prob without re-pickling the GP models on every call)
# ---------------------------------------------------------------------------
_GLOBAL = {}


def setup_global_posterior(components, params_list, fixed_params,
                           param_names=None, flat_indices=None,
                           gaussian_priors=None, nuisance_map=None):
    """Register the posterior state.

    components   : list of (name, likelihood_object) — likelihoods take
                   params7 (+ nuisance for BOSS)
    params_list  : [[label, init, lo, hi], ...] free params (design params
                   first, in PARAM_NAME order, then nuisance)
    fixed_params : {PARAM_NAME label: value}
    nuisance_map : {component_name: {'b1': theta_index, 'P_sn': theta_index}}
                   indices into theta for that component's nuisance params
    """
    from cosmo_hydro_emu.mcmc import ln_prior
    _GLOBAL.clear()
    _GLOBAL.update(dict(
        components=components,
        params_list=params_list,
        fixed_params=fixed_params or {},
        param_names=param_names or list(PARAM_NAME),
        flat_indices=flat_indices or [],
        gaussian_priors=gaussian_priors or {},
        nuisance_map=nuisance_map or {},
        ln_prior=ln_prior,
    ))


def _theta_to_params7(theta):
    g = _GLOBAL
    full = np.empty(len(g['param_names']))
    j = 0
    for i, name in enumerate(g['param_names']):
        if name in g['fixed_params']:
            full[i] = g['fixed_params'][name]
        else:
            full[i] = theta[j]
            j += 1
    return full


def ln_prob_global(theta):
    g = _GLOBAL
    lp = g['ln_prior'](theta, g['params_list'],
                       flat_indices=g['flat_indices'],
                       gaussian_priors=g['gaussian_priors'])
    if not np.isfinite(lp):
        return -np.inf
    params7 = _theta_to_params7(theta)
    ll = 0.0
    for name, like in g['components']:
        nmap = g['nuisance_map'].get(name, {})
        if isinstance(like, BossLikelihood):
            b1 = theta[nmap['b1']]
            P_sn = theta[nmap['P_sn']] if 'P_sn' in nmap else 0.0
            ll_i = like(params7, b1, P_sn)
        else:
            ll_i = like(params7)
        if not np.isfinite(ll_i):
            return -np.inf
        ll += ll_i
    return lp + ll
