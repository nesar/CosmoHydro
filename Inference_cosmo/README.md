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
| A_mod | Preston, Amon & Efstathiou 2023 (0.858±0.052, DES Y3+Planck prior); Amon & Efstathiou 2022 (0.69, KiDS-1000, no σ) | published scalars |
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
* Linear theory (for A_mod template + RSD growth rates): CAMB tables over
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
* **A_mod (`kind: amod`)** — the constraint is ONE published number, so the
  emulated suppression is projected onto the template
  `1 + (A−1)(1 − P_L/P_go)` by least squares over 0.1 < k < 8, and the
  likelihood is Gaussian in that scalar. Never treat the template band as
  independent k-points — that would overcount a single datum.
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
| `Pk_amod_5subgrid_fidcosmo.yaml` | 5 subgrid | A_mod |
| `Pk_kids_amod_7p.yaml` | all 7 | KiDS + A_mod |
| `Pk_boss_z1z3_2cosmo.yaml` | ω_m, σ₈ (+b₁, P_sn ×2) | BOSS NGC z1+z3 |

Parameters not freed are fixed at the **project fiducial** (subgrid
3, 0.5, 0.8, 0.51, 0.13 scaled; cosmology ω_m=0.14176, σ₈=0.8102), never
design midpoints.

Results land in `results/` as `samples_*.npy` / `params_list_*.npy` /
`config_*.yaml` — identical format to `Inference/results/`.

## Roadmap

1. **HMF targets** — `data/Halo_mass_function_targets/` is currently empty;
   HMF multi-z emulators already exist (`models/HMF_multiz/`), so once
   observational cluster-abundance data are chosen, an `hmf` likelihood
   component slots into the same `targets:` list.
2. **Joint cosmology+astrophysics** — combine these components with the
   GSMF/CGD likelihoods from `Inference/` in one posterior.
3. **Sequential fixing** — cosmology from Pk (this directory) → freeze →
   subgrid from astrophysics (`Inference/`), via `fixed_params`/
   `gaussian_priors` handoff of the posterior medians.
4. **BOSS upgrade** — window convolution (W and wide-angle M matrices from
   the Beutler hub) to make the BOSS component publication-grade.
