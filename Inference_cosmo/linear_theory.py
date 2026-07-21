"""
Linear theory, growth, and background cosmology for the HACC 400 Mpc/h suite.

Fixed (simulation-time) cosmological parameters
-----------------------------------------------
Only omega_m (= Omega_m h^2) and sigma_8 vary across the design.  Everything
else is fixed at the project fiducial (chosen by the user on 2026-05-20):

    h        = 0.6766          (H0 = 67.66)
    omega_b  = Omega_b h^2 = 0.02242
    n_s      = 0.9665
    m_nu     = 0 (massless neutrinos assumed -- flagged in README; if the
                  HACC runs used massive neutrinos this module must be updated)
    flat LCDM, w = -1

NOTE: codes/cosmo_hydro_emu/load_hacc.py hardcodes hubble=0.681 in three
places for converting *observational* GSMF/fGas datasets.  That value is for
those external datasets; the simulation-time h used here is 0.6766.  If the
HACC parameter files say otherwise, change FIXED_COSMO['h'] in one place here.

What this module provides
-------------------------
* ``LinearPk`` — CAMB-based linear matter P(k, z) as a function of
  (omega_m, sigma_8), tabulated once over the design range of omega_m and
  interpolated afterwards (MCMC-fast, exact CAMB shapes).
  sigma_8 enters as a pure amplitude rescaling (exact in linear theory for
  massless neutrinos, since growth is scale-independent).
* ``growth_factor`` / ``growth_rate`` — D(z), f(z) from the exact linear
  growth ODE in flat LCDM (radiation neglected; error < 0.1% for z <= 3).
* ``comoving_distance`` / ``D_A`` / ``Hz`` — background functions for
  Alcock-Paczynski rescaling.
"""

import os

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import RegularGridInterpolator, interp1d

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, 'cache')

FIXED_COSMO = {
    'h': 0.6766,
    'omega_b_h2': 0.02242,
    'n_s': 0.9665,
    'm_nu': 0.0,
    'T_cmb': 2.7255,
}

C_LIGHT = 299792.458  # km/s


# ---------------------------------------------------------------------------
# Background (flat LCDM, matter + Lambda; radiation negligible at z <= 3)
# ---------------------------------------------------------------------------
def Ez(z, Omega_m):
    z = np.asarray(z, dtype=float)
    return np.sqrt(Omega_m * (1 + z) ** 3 + (1.0 - Omega_m))


def Hz(z, Omega_m, h=None):
    """H(z) in km/s/Mpc."""
    h = FIXED_COSMO['h'] if h is None else h
    return 100.0 * h * Ez(z, Omega_m)


def comoving_distance(z, Omega_m, h=None):
    """Comoving distance in Mpc."""
    h = FIXED_COSMO['h'] if h is None else h
    integ, _ = quad(lambda zp: 1.0 / Ez(zp, Omega_m), 0.0, z)
    return C_LIGHT / (100.0 * h) * integ


def D_A(z, Omega_m, h=None):
    """Angular-diameter distance in Mpc (flat universe)."""
    return comoving_distance(z, Omega_m, h=h) / (1.0 + z)


# ---------------------------------------------------------------------------
# Linear growth: D'' + (2 + dlnH/dlna) D' = 1.5 Omega_m(a) D   (' = d/dlna)
# ---------------------------------------------------------------------------
def _growth_solution(Omega_m, a_grid):
    def om_a(a):
        return Omega_m / (Omega_m + (1.0 - Omega_m) * a ** 3)

    def rhs(lna, y):
        a = np.exp(lna)
        om = om_a(a)
        dlnH = -1.5 * om          # dlnH/dlna for flat LCDM (matter + Lambda)
        D, Dp = y
        return [Dp, -(2.0 + dlnH) * Dp + 1.5 * om * D]

    a0 = 1e-3                     # deep in matter domination: D ~ a
    lna_span = (np.log(a0), 0.0)
    sol = solve_ivp(rhs, lna_span, [a0, a0], t_eval=np.log(a_grid),
                    rtol=1e-8, atol=1e-10, dense_output=False)
    if not sol.success:
        raise RuntimeError(f'growth ODE failed: {sol.message}')
    D = sol.y[0]
    f = sol.y[1] / sol.y[0]       # f = dlnD/dlna
    return D, f


def growth_factor_and_rate(z, Omega_m):
    """D(z) (normalized to D(z=0)=1) and f(z) = dlnD/dlna.

    Accepts scalar or array z (z <= ~999).
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    a_eval = 1.0 / (1.0 + z)
    a_grid = np.unique(np.concatenate([a_eval, [1.0]]))
    a_grid.sort()
    D_grid, f_grid = _growth_solution(Omega_m, a_grid)
    D_of_a = interp1d(a_grid, D_grid)
    f_of_a = interp1d(a_grid, f_grid)
    D0 = D_of_a(1.0)
    D = D_of_a(a_eval) / D0
    f = f_of_a(a_eval)
    if D.size == 1:
        return float(D[0]), float(f[0])
    return D, f


def growth_rate(z, Omega_m):
    """f(z) = dlnD/dlna from the exact ODE (not the gamma approximation)."""
    _, f = growth_factor_and_rate(z, Omega_m)
    return f


# ---------------------------------------------------------------------------
# Linear P(k, z; omega_m, sigma_8) via a CAMB table over the design range
# ---------------------------------------------------------------------------
class LinearPk:
    """CAMB linear matter power spectrum, tabulated over omega_m.

    The table stores, for each omega_m grid value, the CAMB linear P(k, z)
    at the requested redshifts together with the corresponding sigma_8(z=0).
    A prediction at arbitrary (omega_m, sigma_8) rescales the interpolated
    shape by (sigma_8 / sigma_8_camb)^2 — exact in linear theory because the
    only varying amplitude parameter is the primordial normalization.

    P(k) is in (Mpc/h)^3 with k in h/Mpc, matching the simulation convention.
    """

    CACHE_FILE = os.path.join(CACHE_DIR, 'linear_pk_table.npz')

    def __init__(self, omega_m_grid=None, z_list=(0.0, 0.1, 0.15, 0.38, 0.45,
                                                  0.5, 0.61, 1.0, 1.3, 2.0),
                 k_min=1e-4, k_max=25.0, n_k=400, rebuild=False):
        if omega_m_grid is None:
            # design range 0.12..0.155 with margin
            omega_m_grid = np.linspace(0.115, 0.160, 19)
        self.omega_m_grid = np.asarray(omega_m_grid)
        self.z_list = np.asarray(sorted(z_list))
        self.k = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
        if (not rebuild) and os.path.exists(self.CACHE_FILE):
            if self._load_cache():
                return
        self._build_table()
        self._save_cache()

    # -- cache ------------------------------------------------------------
    def _load_cache(self):
        d = np.load(self.CACHE_FILE)
        same = (np.array_equal(d['omega_m_grid'], self.omega_m_grid)
                and np.array_equal(d['z_list'], self.z_list)
                and np.array_equal(d['k'], self.k))
        if not same:
            return False
        self.logP_table = d['logP_table']
        self.sigma8_camb = d['sigma8_camb']
        self._make_interpolators()
        return True

    def _save_cache(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(self.CACHE_FILE,
                            omega_m_grid=self.omega_m_grid,
                            z_list=self.z_list, k=self.k,
                            logP_table=self.logP_table,
                            sigma8_camb=self.sigma8_camb)

    # -- CAMB -------------------------------------------------------------
    def _build_table(self):
        import camb
        h = FIXED_COSMO['h']
        nom, nz, nk = self.omega_m_grid.size, self.z_list.size, self.k.size
        self.logP_table = np.empty((nom, nz, nk))
        self.sigma8_camb = np.empty(nom)
        print(f'[LinearPk] building CAMB table: {nom} omega_m values, '
              f'{nz} redshifts, {nk} k points ...')
        for i, om in enumerate(self.omega_m_grid):
            ombh2 = FIXED_COSMO['omega_b_h2']
            omch2 = om - ombh2
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=omch2,
                               mnu=FIXED_COSMO['m_nu'], omk=0.0,
                               TCMB=FIXED_COSMO['T_cmb'])
            pars.InitPower.set_params(As=2.1e-9, ns=FIXED_COSMO['n_s'])
            pars.set_matter_power(redshifts=list(self.z_list[::-1]),
                                  kmax=self.k[-1] * 1.2, nonlinear=False)
            res = camb.get_results(pars)
            kh, zs, pk = res.get_matter_power_spectrum(
                minkh=self.k[0], maxkh=self.k[-1], npoints=nk)
            # camb returns redshifts sorted ascending; align to self.z_list
            zs = np.asarray(zs)
            order = np.argsort(zs)
            for j, z in enumerate(self.z_list):
                jz = order[np.argmin(np.abs(np.sort(zs) - z))]
                if abs(zs[jz] - z) > 1e-6:
                    raise RuntimeError(f'z alignment failure: {zs[jz]} vs {z}')
                pkz = np.interp(self.k, kh, pk[jz])
                self.logP_table[i, j] = np.log10(pkz)
            self.sigma8_camb[i] = res.get_sigma8_0()
        print('[LinearPk] table built.')

    def _make_interpolators(self):
        self._interp = RegularGridInterpolator(
            (self.omega_m_grid, self.z_list, np.log10(self.k)),
            self.logP_table, bounds_error=True)
        self._s8_interp = interp1d(self.omega_m_grid, self.sigma8_camb)

    def __getattr__(self, name):
        # lazily build interpolators after table construction/load
        if name in ('_interp', '_s8_interp'):
            self._make_interpolators()
            return object.__getattribute__(self, name)
        raise AttributeError(name)

    # -- prediction -------------------------------------------------------
    def __call__(self, k, z, omega_m, sigma_8):
        """Linear P(k, z) in (Mpc/h)^3 for k in h/Mpc.

        z must be one of the tabulated z_list values (all target redshifts
        are known ahead of time), or between two of them (interpolated).
        """
        k = np.atleast_1d(np.asarray(k, dtype=float))
        pts = np.column_stack([
            np.full(k.size, omega_m),
            np.full(k.size, z),
            np.log10(k),
        ])
        logP = self._interp(pts)
        s8c = self._s8_interp(omega_m)
        return 10 ** logP * (sigma_8 / s8c) ** 2

    def sigma8_at_z(self, z, omega_m, sigma_8):
        """sigma_8(z) = sigma_8 * D(z), with D from the growth ODE."""
        Omega_m = omega_m / FIXED_COSMO['h'] ** 2
        D, _ = growth_factor_and_rate(z, Omega_m)
        return sigma_8 * D


def omega_to_Omega_m(omega_m):
    """Convert omega_m = Omega_m h^2 (design convention) to Omega_m."""
    return omega_m / FIXED_COSMO['h'] ** 2
