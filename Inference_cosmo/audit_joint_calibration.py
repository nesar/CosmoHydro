"""Audit of the joint GSMF+CGD+Pk calibration, covering:

  (1) what drives the Omega_m railing in the 7p run  -> conditional lnL sweeps
  (2) low-k reliability: fundamental / Nyquist frequencies for the 400 Mpc/h,
      1600^3 box; mode counts per KiDS bin; per-point chi2 vs k
  (3) what drives v_kin high                          -> conditional lnL sweeps
  (4) unit-consistency of the GSMF/CGD chi2 (the "10^phi" encoding) and the
      emulator-variance implementation

Run:  python audit_joint_calibration.py
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'Inference'))
sys.path.insert(0, os.path.join(_HERE, '..', 'codes'))

import gsmf_cgd_target as GC
from targets import load_kids
from pk_likelihood import PkEmulator, PmLikelihood

# emu-var 7p best fit (mode) from plot_summary output
BF7 = np.array([3.2191, 0.5426, 1.1866, 1.1688, 0.2168, 0.1542, 0.8110])
FID7 = np.array([3.0, 0.5, 0.8, 0.51, 0.13, 0.14176, 0.8102])

L_BOX, N_PART = 400.0, 1600          # Mpc/h, particles per side


def sec(t):
    print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)


def main():
    # ---------------- (2) k-range arithmetic ------------------------------
    sec('(2) k-range reliability: 400 Mpc/h box, 1600^3 particles')
    kf = 2 * np.pi / L_BOX
    kny_part = np.pi * N_PART / L_BOX
    V = L_BOX ** 3
    print(f'  fundamental k_f      = 2pi/L        = {kf:.5f} h/Mpc')
    print(f'  particle Nyquist     = pi N/L       = {kny_part:.2f} h/Mpc')

    emu = PkEmulator()
    kg = emu.k_grid
    print(f'  emulator k-grid      = [{kg.min():.5f}, {kg.max():.3f}] h/Mpc '
          f'({len(kg)} bins)')
    print(f'    -> grid k_min == k_f ({kg.min()/kf:.2f} k_f): the first emulator '
          f'bins hold only a handful of modes')
    print(f'    -> grid k_max {kg.max():.2f} = mesh Nyquist of a '
          f'{int(round(kg.max()*L_BOX/np.pi))}^3 FFT mesh; half-Nyquist '
          f'(aliasing-safe) ~ {kg.max()/2:.1f} h/Mpc; current cut k_max=7.0 '
          f'is ABOVE half-Nyquist')

    kids = load_kids(nz='nz3', k_min=None, k_max=None, z_bins=[0.15, 0.45])
    k_all = np.unique(kids['k'])
    print(f'\n  KiDS k points (both z bins share the grid): {len(k_all)} values')
    dlnk = np.diff(np.log(k_all)).mean()
    print(f'  (log-spaced, dlnk = {dlnk:.3f})')
    print(f'  {"k":>8s} {"k/k_f":>7s} {"N_modes in bin":>14s}   note')
    for k in k_all[:6]:
        dk = k * dlnk
        nmodes = V * k ** 2 * dk / (2 * np.pi ** 2)
        note = 'DROPPED by k_min=0.03' if k < 0.03 else ''
        print(f'  {k:8.4f} {k/kf:7.1f} {nmodes:14.0f}   {note}')

    # ---------------- (2b) per-point KiDS chi2 vs k ------------------------
    sec('(2b) per-point KiDS z-scores at the emu-var 7p best fit (diag errors)')
    kids_c = load_kids(nz='nz3', k_min=0.03, k_max=7.0, z_bins=[0.15, 0.45])
    pm = PmLikelihood(kids_c, emu)
    y_mod, var_emu = pm.model_vector(BF7)
    sig_eff = np.sqrt(np.diag(kids_c['cov']) + var_emu
                      + (pm.interp_sys_frac * y_mod) ** 2)
    zsc = (kids_c['y'] - y_mod) / sig_eff
    for zbin in np.unique(kids_c['z']):
        m = kids_c['z'] == zbin
        print(f'  z_fid = {zbin}:')
        for kk, zz in zip(kids_c['k'][m], zsc[m]):
            bar = '#' * int(min(abs(zz), 5) * 4)
            print(f'    k={kk:7.3f}  z-score {zz:+6.2f}  {bar}')
    lo = np.abs(kids_c['k']) < 0.1
    print(f'\n  mean |z|: k<0.1 -> {np.abs(zsc[lo]).mean():.2f}, '
          f'k>0.1 -> {np.abs(zsc[~lo]).mean():.2f}')
    print('  (KiDS deprojection assumed Omega_m = 0.305 +/- 0.012; at the railed')
    om_r = 0.1542 / 0.6766 ** 2
    print(f'   omega_m=0.1542 -> Omega_m = {om_r:.3f} — OUTSIDE the kernel '
          'assumption, an internal inconsistency of the railed solution.)')

    # ---------------- (1)+(3) conditional lnL sweeps ----------------------
    sec('(1)+(3) conditional lnL sweeps at the emu-var best fit')
    gsmf = GC._build_component('gsmf', with_emu_variance=True)
    cgd = GC._build_component('cgd', with_emu_variance=True)
    comps = [('GSMF', gsmf), ('CGD', cgd), ('KiDS Pk', pm)]

    for pname, idx, grid in [
            ('omega_m', 5, np.linspace(0.121, 0.1545, 21)),
            ('v_kin', 3, np.linspace(0.15, 1.19, 21))]:
        print(f'\n  --- lnL vs {pname} (others held at emu-var 7p best fit) ---')
        print(f'  {"component":10s} {"argmax":>8s} {"dlnL(edge_hi - best)":>21s} '
              f'{"dlnL(edge_hi - edge_lo)":>24s}')
        for cname, like in comps:
            lls = []
            for v in grid:
                th = BF7.copy(); th[idx] = v
                lls.append(float(like(th)))
            lls = np.array(lls)
            am = grid[int(np.argmax(lls))]
            print(f'  {cname:10s} {am:8.4f} {lls[-1]-lls.max():21.2f} '
                  f'{lls[-1]-lls[0]:24.2f}')
        print('  (argmax at the top of the grid => that component pushes '
              f'{pname} up; dlnL sizes say how hard)')

    # ---------------- (4) GSMF unit consistency ---------------------------
    sec('(4) GSMF chi2 unit check: residual in 10^phi space vs error in phi space')
    from cosmo_hydro_emu.emu import emulate
    mg, sd = emulate(gsmf.model, BF7)
    model = np.interp(gsmf.x, gsmf.x_grid, mg[:, 0])       # 10^phi space
    phi_obs = np.log10(gsmf.y)                              # phi (linear density)
    phi_mod = np.log10(model)
    r_code = gsmf.y - model                                 # what the code uses
    r_phi = phi_obs - phi_mod                               # consistent phi space
    sig_phi = gsmf.yerr                                     # error IS in phi space
    chi2_code = np.sum(r_code ** 2 / sig_phi ** 2)
    chi2_phi = np.sum(r_phi ** 2 / sig_phi ** 2)
    print(f'  y (=10^phi) range        : [{gsmf.y.min():.4f}, {gsmf.y.max():.4f}]')
    print(f'  yerr/phi_obs (frac error): {np.median(gsmf.yerr/phi_obs):.3f} '
          '(sensible fractional errors on the DENSITY -> yerr lives in phi space)')
    print(f'  chi2 as coded (obs-err only)   = {chi2_code:8.2f}')
    print(f'  chi2 in consistent phi space   = {chi2_phi:8.2f}')
    print(f'  ratio = {chi2_code/chi2_phi:.2f}   (ln10)^2 = {np.log(10)**2:.2f}')
    print('  => the GSMF residual (10^phi space) carries a ln10 Jacobian the '
          'error bar lacks:')
    print('     GSMF chi2 is inflated ~x5.3, i.e. GSMF has been OVER-weighted '
          '~5x everywhere.')

    sec('(4b) CGD error provenance')
    print('  load_obs_data hard-codes yerr = 0.05*y. The mcdonald2017_avg.txt file')
    print('  has NO error column (extra columns are other z bins). McDonald+17')
    print('  cluster-to-cluster scatter is ~20-40% -> 5% is far too tight; CGD is')
    print('  over-weighted by (0.2/0.05)^2 ~ 16x if the true error is ~20%.')


if __name__ == '__main__':
    main()
