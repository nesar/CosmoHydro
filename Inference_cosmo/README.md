# Inference_cosmo — cosmology-target inference (power spectra, HMF later)

MCMC inference against **cosmology-specific observational targets**, starting
with the nonlinear matter power spectrum. Complements `Inference/`, which
constrains against astrophysical targets (GSMF, CGD, ...). Same design, same
YAML-driven workflow, same results format.

## Data provenance (everything verified, nothing invented)

| Piece | Source | Status |
|---|---|---|
| Sim P(k) | `data/scidac-olcf-pk_3/` — 110 runs × z ∈ {0, 0.1, 0.5, 1, 2} × {go, hydro.full, hydro.cdm, hydro.bar} | all 2200 files validated (`diagnostics/pk_data_validation.txt`); `pk_2` is byte-identical to `pk_3` |
| KiDS-Legacy P_m(k,z) | Broxterman et al. 2025, A&A 703 L3 (CDS J/A+A/703/L3) | downloaded + parsed per the CDS byte spec; correlation matrices positive-definite; reproduces the 30% suppression headline |
| BOSS DR12 P0/P2 | Beutler & McDonald 2021 files, dk=0.01 rebin + 2048-mock Patchy covariances | covariance multipole blocks identified (P0=block 0, P2=block 2), diag matches published σ to ~4×10⁻⁷ |

## Model composition

```
P_hydro(k,z; θ) = S(k,z; θ₇) × P_go(k,z; ω_m, σ₈)
```

* `S` — suppression-ratio GP emulators. **z=0 reuses the pre-existing
  notebook-trained `models/Pk_multivariate_model_z_index0.pkl`** (not
  retrained); z = 0.1, 0.5, 1, 2 trained here with the identical recipe
  (exp_variance=0.95, runs 000–099 train / 100–109 held out).
* `P_go` — gravity-only log₁₀P GP emulators (new, `models/Pk_cosmo/`),
  **cosmology-only inputs** (GO runs don't see subgrid physics);
  exp_variance=0.999. Held-out accuracy ~0.2% median.
* Redshift interpolation between snapshots: `P_go` interpolates
  log[P/D²(z)] linearly in scale factor (growth-scaled; ≤1% typical, 3.4%
  worst-case at the widest gap — validated in the data report); `S`
  interpolates linearly in a.
* Linear theory (RSD growth rates, AP distances, halofit cross-check): CAMB tables over
  the design ω_m range (`linear_theory.py`), σ₈-integral validated to 4
  decimals. Fixed at simulation values: h=0.6766, ω_b=0.02242, n_s=0.9665.
  **Assumptions to confirm: massless neutrinos; h=0.6766 (a legacy
  `hubble=0.681` exists in load_hacc.py for converting external GSMF/fGas
  datasets only).**

## Likelihoods (`pk_likelihood.py`)

* **KiDS (`kind: kids`)** — direct Gaussian likelihood on P_m(k, z_fid)
  with the published correlation matrix × symmetrized 68% intervals, plus
  emulator variance and a 1% z-interpolation systematic. The cleanest
  cosmology target: it is what the sims actually predict. (Their
  deprojection assumed Ω_m=0.305 in the lensing kernel — inherited model
  dependence, see the paper.)
  **Default z-bin choice:** the nz3 z_fid=1.3 bin is excluded. Its fdelta
  = 1.24–1.68 (more power than the DMO reference, baryonically impossible)
  and its strong cross-bin correlations dominate χ² (405 → 15 at fiducial
  when dropped, with per-bin fits all good). The wide nz1 bin (z_fid=1.0)
  is available as an alternative (`nz: nz1`).
* **A_mod — QUARANTINED to `amod_exploratory/` (2026-07-23).** It is a
  Planck-conditioned modulation of the nonlinear boost (P_NL − P_L), not a
  measurement of P_hydro/P_GO (which is unobservable; P_GO itself contains
  nonlinear growth). See `amod_exploratory/README.md` for the full argument
  and the archived runs/plots.
* **BOSS (`kind: boss`)** — Kaiser linear RSD × linear bias on the emulated
  matter P(k, z_eff), Alcock-Paczynski dilation from the BOSS fiducial
  (Ω_m=0.31), exact-ODE growth rates, Patchy covariance with Hartlap
  factor, free bias + optional shot noise per patch/z-bin.
  **Deliberately methods-level:** no survey-window convolution, no FoG, no
  loop corrections → keep k ≤ 0.15 h/Mpc and do not quote these numbers as
  a full-shape analysis. Adding the window matrices (available on the
  Beutler hub) is the upgrade path.

## Workflow

```bash
python validate_pk_data.py                  # data QA (report in diagnostics/)
python train_pk_emulators.py                # trains ONLY missing models
python train_pk_emulators.py --validate-only
python compare_pk_targets.py                # emulator vs targets at fiducial
python run_mcmc_cosmo.py configs/Pk_kids_2cosmo.yaml --dry-run
python run_mcmc_cosmo.py configs/Pk_kids_2cosmo.yaml
```

Configs (deep-merged over `configs/_defaults.yaml`, short-key conventions as
in `Inference/`):

| Config | Free params | Targets |
|---|---|---|
| `Pk_kids_2cosmo.yaml` | ω_m, σ₈ | KiDS P_m |
| `Pk_kids_7p.yaml` | all 7 | KiDS P_m |
| `Pk_kids_hmf_7p.yaml` | all 7 (+ΔlogM) | KiDS + GAMA HMF |
| `HMF_gama_2cosmo.yaml` | ω_m, σ₈ (+ΔlogM) | GAMA HMF |
| `Pk_boss_z1z3_2cosmo.yaml` | ω_m, σ₈ (+b₁, P_sn ×2) | BOSS NGC z1+z3 |

A_mod-based configs/runs are archived in `amod_exploratory/`.

Parameters not freed are fixed at the **project fiducial** (subgrid
3, 0.5, 0.8, 0.51, 0.13 scaled; cosmology ω_m=0.14176, σ₈=0.8102), never
design midpoints.

Results land in `results/` as `samples_*.npy` / `params_list_*.npy` /
`config_*.yaml` — identical format to `Inference/results/`.

## HMF target (GAMA DR4)

`kind: gama_hmf` — Driver et al. 2022 empirical HMF (Table 1, Eddington-
corrected, credible range logM ≥ 12.8 Msun/h) vs the **pre-existing**
`models/HMF_multiz/multivariate_model_z_index9` emulator (snapshot 567,
z=0.0998 = GAMA z_eff; SOD ≈ M200c masses, compatible with Driver's
dynamical-mass calibration).

Verified conventions (against the published paper — the local data-package
header was misleading and has been corrected in its README):
* Driver's tables are **already in h-units** (Msun h⁻¹_P18, Mpc⁻³ h³_P18) —
  identity conversion to sim units.
* Sim φ = dn/dlog₁₀M verified bin-by-bin against raw halo counts;
  emulator ≈ colossus Tinker08 × 0.93 at fiducial (SOD completeness level).
* Error model: published fractional errors (+cosmic variance) ⊕ GP variance
  ⊕ box halo shot noise φ/(V·dlog₁₀M) — holdout residuals are consistent
  with the shot-noise term.
* Optional `mass_shift` nuisance (±0.3 dex) marginalizes the dynamical-mass
  calibration systematic (A=13.9 adopted; the paper's own Coma check hints
  A~6±3).
* Known data features at fiducial (χ²=41/11): the intermediate-mass excess
  at logM 13.6–14.2 that Driver et al. discuss, and low points near the
  completeness limit.

Configs: `HMF_gama_2cosmo.yaml` (abundance-only cosmology),
`Pk_kids_hmf_7p.yaml` (joint Pk+HMF, 7 params + mass shift).

## Roadmap

1. **Cluster-count forward models** — eROSITA/SPT/ACT counts (see the data
   README) would be the tighter, more model-dependent HMF route.
2. **Joint cosmology+astrophysics** — combine these components with the
   GSMF/CGD likelihoods from `Inference/` in one posterior.
3. **Sequential fixing** — cosmology from Pk (this directory) → freeze →
   subgrid from astrophysics (`Inference/`), via `fixed_params`/
   `gaussian_priors` handoff of the posterior medians.
4. **BOSS upgrade** — window convolution (W and wide-angle M matrices from
   the Beutler hub) to make the BOSS component publication-grade.
