# Calibration fixes (2026-08-07) — all opt-in, all reversible

Audit that motivated these: `audit_joint_calibration.py` (run it to reproduce
every number). Each fix is a **per-target config key**, default **off** — omit
the key and the legacy behaviour is bit-identical. The `*_cal.yaml` configs turn
all of them on; the older `*_emuvar.yaml` / plain configs are untouched.

| # | fix | config key | default (legacy) | rationale |
|---|-----|-----------|------------------|-----------|
| 1 | GSMF error Jacobian | `err_jacobian: true` on the `gsmf` target | off: yerr stays in phi space | The GP + residual live in `10^phi` space but `load_gsmf_obs` returns yerr in phi (linear density) space → chi2 inflated by (ln10)² ≈ 5.3 (verified: 31.3 → 5.7 at the best fit). Fix multiplies yerr by ln(10)·10^phi. GSMF was ~5x over-weighted in EVERY previous run (v1 legacy included). |
| 2 | CGD realistic errors | **`err_model: intersurvey`** (recommended) on the `cgd` target | off: invented flat 5% | `mcdonald2017_avg.txt` has NO error column (extra columns are z bins). See "CGD error model" below. |
| 3 | KiDS k-range | `k_min: 0.05`, `k_max: 4.0` in the `kids` target | 0.03 / 7.0 | k=0.033, 0.050 bins carry ~48/~160 modes → 10–20% box sample noise (L=400 Mpc/h ⇒ k_f=0.0157); k_max=7 exceeds the aliasing-safe half mesh-Nyquist (~4.0 for the 1024³ P(k) mesh). |
| 4 | emulator variance | `emu_variance: true` on gsmf/cgd targets | off | GP predictive variance + logdet term in the Gaussian; pathology-checked in `diagnose_emu_variance.py` (no variance-seeking: corr = +0.10). Already used by the `*_emuvar` runs. |
| 5 | eps_kin prior overlay (plot only) | — | — | union-triangle drew a Gaussian prior on eps_kin where the runs use a flat prior; fixed in `plot_summary_cosmo.py::_overlay_priors_union`. |

## CGD error model — how we got to `intersurvey` (fully data-driven, no floor)

Attempt 1, `err_model: ghirardini19` (**superseded — do not use**): took the
Ghirardini+2019 ylerr/yherr columns as the error. Those columns are the
**cluster-to-cluster scatter band** of the 12-cluster X-COP sample
(−57 %/+124 % in the core), *not* the uncertainty on the mean profile, so they
over-inflate the errors (median 27 %) and under-weight CGD.

Attempt 2, `err_model: intersurvey` (**recommended**): at each radius take the
**larger** of two independently measured quantities (max, not quadrature, since
they partly measure the same thing):
 (a) the RMS disagreement between three INDEPENDENT observational determinations
     of the z≈0 cluster gas density profile — McDonald+2017 (SPT/Chandra),
     Ghirardini+2019 (X-COP/XMM), Lehle+2023 (eROSITA) — a direct measurement of
     the systematic uncertainty;
 (b) the X-COP statistical error on the mean = scatter band / √12.
Braspenning+2023 is a FLAMINGO **simulation** and is excluded on purpose.

| r/R500 | 0.026 | 0.05 | 0.10 | 0.18 | 0.33 | 0.54 | 0.92 | 1.43 |
|---|---|---|---|---|---|---|---|---|
| frac. error | 0.26 | 0.18 | 0.10 | 0.068 | 0.050 | 0.052 | 0.063 | 0.088 |

Physically sensible shape: large in the core (cool-core/non-cool-core dichotomy,
resolution), minimum ≈5 % at mid radii, rising again in the low-S/N outskirts.
Note this **validates the old flat 5 % at mid radii** — the error was mainly the
missing *radial dependence*, not the overall normalisation.

## How to revert
Remove the keys from the config (or use the older configs). No likelihood code
path changes unless the keys are present; `git diff` touchpoints:
`gsmf_cgd_target.py` (`_apply_err_fixes`, `_ghirardini_frac_err`),
`codes/cosmo_hydro_emu/mcmc.py` (`with_emu_variance`, default False),
`plot_summary_cosmo.py` (overlay fix — cosmetic only).

## Context from the audit (what the fixes change)
- Omega_m railing (7p): driven by GSMF (conditional ΔlnL ≈ +191 across the box
  toward the 0.155 edge — would be ≈ +36 with fix 1). KiDS Pk mildly prefers
  LOW Omega_m; CGD mild (+14).
- v_kin railing: driven by CGD (ΔlnL ≈ +79; GSMF/Pk < 1) through the invented
  5% errors; fix 2 relaxes it. v_kin/1e4 ≈ 1.14 ⇒ ~11,400 km/s AGN kinetic
  velocity, ~0.5σ from the design edge — physically implausible.
- The railed omega_m=0.154 ⇒ Omega_m=0.337 also violates the KiDS deprojection
  kernel assumption (Omega_m = 0.305 ± 0.012) — see `lowk_linear_pk_plan.md`.

## Run set using all fixes
`GSMF_CGD_Pk_{7p,2cosmo_hydA,5p_fid_cosmo}_cal.yaml` → trials `*_cal`.
