"""
Simulation power-spectrum data handling for cosmology-target inference.

Data source
-----------
``data/scidac-olcf-pk_3/`` (identical content to ``scidac-olcf-pk_2``; verified
by ``validate_pk_data.py``). Flat files named

    run{NNN}_z{Z}.{TYPE}.pk.txt        NNN = 000..109,  Z in {0.0, 0.1, 0.5, 1.0, 2.0}
    TYPE in {go, hydro.full, hydro.cdm, hydro.bar}

Each file has 5 columns:  k [h/Mpc],  P_0(k) [(Mpc/h)^3],  ErrorBar, nModes, P_2(k).
P_2 is identically zero for these real-space measurements; ErrorBar is the
Gaussian mode-counting error  P * sqrt(2/nModes).

Conventions (matching codes/02_train_emulators_multiz.ipynb, the notebook
that trained the existing models):
  * design file: data/FinalDesign.txt, row K (0-indexed, after header) = runK
  * slice: start_sim_idx=0, num_sims=110  ->  run000..run109 (all runs)
  * train/test split: runs 0-99 train, runs 100-109 held-out test
  * subgrid scaling: M_seed/1e6, v_kin/1e4, eps_kin/1e1
  * trusted k range: mass_conds('Pk') = [2*pi/400, pi/(400/1024)]
                     = [0.0157, 8.04] h/Mpc  (fundamental mode to Nyquist)
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'codes'))

from cosmo_hydro_emu.load_hacc import (          # noqa: E402
    mass_conds, seed_mass_scale, vkin_scale, eps_scale,
)

# ---------------------------------------------------------------------------
PK_DIR_DEFAULT = os.path.join(_HERE, '..', 'data', 'scidac-olcf-pk_3')
DESIGN_FILE_DEFAULT = os.path.join(_HERE, '..', 'data', 'FinalDesign.txt')

PK_REDSHIFT_TAGS = ['0.0', '0.1', '0.5', '1.0', '2.0']
PK_REDSHIFTS = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
PK_TYPES = ['go', 'hydro.full', 'hydro.cdm', 'hydro.bar']

NUM_SIMS_DEFAULT = 110
START_SIM_IDX_DEFAULT = 0
TRAIN_INDICES = list(range(100))          # runs 000-099
TEST_INDICES = list(range(100, 110))      # runs 100-109 (held out)

# Columns of FinalDesign.txt (after header):
#   kappa_W, e_W, M_seed, v_kin, eps_kin, omega_m(=Omega_m h^2), sigma_8
SCALE_FACTORS = {2: seed_mass_scale, 3: vkin_scale, 4: eps_scale}
COSMO_COLS = [5, 6]


def load_design(design_file=DESIGN_FILE_DEFAULT,
                start_sim_idx=START_SIM_IDX_DEFAULT,
                num_sims=NUM_SIMS_DEFAULT):
    """Design matrix slice with project-standard scaling applied.

    Row K of the CSV (0-indexed, after header) corresponds to runK, so the
    slice is params[start_sim_idx : start_sim_idx + num_sims].
    """
    import pandas as pd
    df = pd.read_csv(design_file)
    params = df.values.astype(float)
    params = params[start_sim_idx:start_sim_idx + num_sims]
    if params.shape[0] != num_sims:
        raise ValueError(
            f"Design slice has {params.shape[0]} rows, expected {num_sims}")
    for col, scale in SCALE_FACTORS.items():
        params[:, col] = params[:, col] / scale
    return params


def _pk_file(pk_dir, sim_idx, ztag, pk_type):
    return os.path.join(pk_dir, f'run{sim_idx:03d}_z{ztag}.{pk_type}.pk.txt')


def load_pk_single(pk_dir, sim_idx, ztag, pk_type):
    """Load one power-spectrum file. Returns (k, P, err, nmodes)."""
    d = np.loadtxt(_pk_file(pk_dir, sim_idx, ztag, pk_type))
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]


def load_pk_suite(pk_dir=PK_DIR_DEFAULT,
                  ztag='0.0',
                  pk_type='hydro.full',
                  num_sims=NUM_SIMS_DEFAULT,
                  start_sim_idx=START_SIM_IDX_DEFAULT,
                  with_go=True):
    """Load the full suite at one redshift.

    Returns a dict with:
      k        : (nk,) common k grid [h/Mpc]
      P        : (num_sims, nk) power of `pk_type`
      err      : (num_sims, nk) mode-counting error of `pk_type`
      nmodes   : (nk,) number of modes per bin (identical across sims)
      P_go     : (num_sims, nk) gravity-only power (if with_go)
      err_go   : (num_sims, nk)
      ratio    : (num_sims, nk) P / P_go (if with_go)
    Raises if any k grid deviates from the first file's grid.
    """
    k_ref = None
    P = err = P_go = err_go = nmodes = None

    for i in range(num_sims):
        sim_idx = i + start_sim_idx
        k, p, e, nm = load_pk_single(pk_dir, sim_idx, ztag, pk_type)
        if k_ref is None:
            k_ref = k
            nk = k.size
            P = np.empty((num_sims, nk))
            err = np.empty((num_sims, nk))
            nmodes = nm
            if with_go:
                P_go = np.empty((num_sims, nk))
                err_go = np.empty((num_sims, nk))
        elif not np.allclose(k, k_ref, rtol=1e-10, atol=0):
            raise ValueError(f"k grid mismatch in run{sim_idx:03d} z{ztag} {pk_type}")
        P[i] = p
        err[i] = e
        if with_go:
            kg, pg, eg, _ = load_pk_single(pk_dir, sim_idx, ztag, 'go')
            if not np.allclose(kg, k_ref, rtol=1e-10, atol=0):
                raise ValueError(f"k grid mismatch in run{sim_idx:03d} z{ztag} go")
            P_go[i] = pg
            err_go[i] = eg

    out = {'k': k_ref, 'P': P, 'err': err, 'nmodes': nmodes}
    if with_go:
        out['P_go'] = P_go
        out['err_go'] = err_go
        out['ratio'] = P / P_go
    return out


def k_trust_mask(k):
    """Boolean mask for the trusted k range (fundamental mode .. Nyquist)."""
    kmin, kmax = mass_conds('Pk')
    return (k > kmin) & (k < kmax)


def emulation_targets(suite, quantity):
    """Build the (num_sims, nk_cut) emulation target array on the trusted k cut.

    quantity: 'logP_hydro' -> log10 P            (pk_type of the suite)
              'logP_go'    -> log10 P_go
              'ratio'      -> P / P_go
    Returns (k_cut, y_vals).
    """
    m = k_trust_mask(suite['k'])
    k_cut = suite['k'][m]
    if quantity == 'logP_hydro':
        y = np.log10(suite['P'][:, m])
    elif quantity == 'logP_go':
        y = np.log10(suite['P_go'][:, m])
    elif quantity == 'ratio':
        y = suite['ratio'][:, m]
    else:
        raise ValueError(f"unknown quantity '{quantity}'")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"non-finite values in emulation target '{quantity}'")
    return k_cut, y
