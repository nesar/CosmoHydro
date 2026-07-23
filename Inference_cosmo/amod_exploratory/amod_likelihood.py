"""
A_mod target + likelihood — EXPLORATORY (see README.md for why this is
quarantined from the standard pipeline).

Moved verbatim from targets.py / pk_likelihood.py on 2026-07-23.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_BASE, '..', 'codes'))

from pk_likelihood import IDX_OMEGA_M, IDX_SIGMA_8      # noqa: E402
from linear_theory import LinearPk                      # noqa: E402

AMOD_CONSTRAINTS = {
    # Preston, Amon & Efstathiou 2023 (arXiv:2305.09827): DES Y3 + Planck prior
    'DES_Y3_Planck': {'Amod': 0.858, 'sigma': 0.052, 'ref': 'arXiv:2305.09827'},
    # Amon & Efstathiou 2022 (arXiv:2206.11794): KiDS-1000; no published sigma,
    # central value only — usable for plots, NOT for a Gaussian likelihood.
    'KiDS1000': {'Amod': 0.69, 'sigma': None, 'ref': 'arXiv:2206.11794'},
}


def load_amod(constraint='DES_Y3_Planck'):
    """Return the A_mod scalar constraint dict (Amod, sigma, ref)."""
    c = dict(AMOD_CONSTRAINTS[constraint])
    c['name'] = f'A_mod {constraint}'
    c['kind'] = 'amod'
    if c['sigma'] is None:
        raise ValueError(
            f"constraint '{constraint}' has no published uncertainty; it can "
            f"be plotted but not used in a Gaussian likelihood")
    return c


class AmodLikelihood:
    """Scalar A_mod constraint, evaluated at z=0.

    INTERPRETIVE MAPPING (the reason this lives in amod_exploratory/):
    the emulated baryonic suppression S(k) = P_hydro/P_GO is projected onto
    the Amon-Efstathiou template 1 + (A - 1) * t(k), t = 1 - P_L/P_go, by
    weighted least squares over a log-uniform k grid, and the likelihood is
    Gaussian in the projected scalar. This TREATS the published nonlinear-
    boost modulation as if it were baryonic suppression of the total power,
    conditional on Planck-like cosmology — an assumption, not an identity.
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
