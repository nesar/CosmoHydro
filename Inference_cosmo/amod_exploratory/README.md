# A_mod — EXPLORATORY ONLY (quarantined from the standard pipeline)

Everything A_mod-related lives here and is **not** part of the standard
`Inference_cosmo/` code path. Do not mix these results with the headline
constraints.

## Why quarantined

The published constraint (DES Y3 + Planck prior: A_mod = 0.858 ± 0.052,
Preston, Amon & Efstathiou 2023) comes from fitting cosmic shear with

    P = P_L + A_mod (P_NL^DMO − P_L)

at fixed Planck cosmology. Three reasons it is not a clean target for this
project:

1. **P_hydro/P_GO is not observable.** P_GO is a counterfactual (a universe
   without baryons acting on structure); only the total matter spectrum is
   physical. A_mod is quoted against a *theory* DMO spectrum.
2. **A_mod modulates the nonlinear boost, not the baryonic ratio.** It
   rescales (P_NL^DMO − P_L), i.e. nonlinear growth relative to linear
   theory. P_GO itself contains nonlinear growth, so "suppressed boost" is
   NOT equivalent to "P_hydro/P_GO suppression" — mapping one onto the
   other (as the likelihood here does via template projection) is an
   interpretive choice, not an identity.
3. **Cosmology-conditioned and degenerate.** The number is derived at
   Planck cosmology; the deficit it summarizes can equally be a lower
   σ8/S8, neutrinos, or new physics. Preston et al. themselves note the
   required suppression extends to k ~ 0.2 h/Mpc, larger scales than
   plausible feedback reaches. Using it as a feedback constraint attributes
   ALL of the deficit to baryons.

The only quasi-self-consistent use is the fixed-fiducial-cosmology subgrid
run (our fiducial is Planck-like), and even that inherits caveat 2.

## Contents

- `amod_likelihood.py` — `load_amod`, `AMOD_CONSTRAINTS`, `AmodLikelihood`
  (template projection of the emulated suppression onto the A_mod form).
- `run_mcmc_amod.py` — thin runner that registers the `amod` target kind
  into the standard driver and delegates to it:
      python run_mcmc_amod.py configs/Pk_amod_5subgrid_fidcosmo.yaml
- `compare_amod.py` — the suppression-plane comparison figure
  (design envelope vs A_mod template bands).
- `plot_summary_amod.py` — marg-vs-fixed summary figures for the archived
  runs below.
- `configs/` — all configs whose likelihood includes an `amod` component:
  `Pk_amod_5subgrid_fidcosmo`, `Pk_kids_amod_7p`, `Pk_kids_amod_hmf_7p`.
- `results/` — samples/params/config/corner/summary plots of those runs
  (completed 2026-07-22/23). Their *cosmology* posteriors are essentially
  KiDS(+HMF)-driven (the 2D sweeps show the A_mod likelihood is nearly flat
  in (ω_m, σ8)), but they are archived here because the subgrid posteriors
  are A_mod-shaped.
- `plots/` — compare_amod.png and the 4-column 2D sweep that included the
  A_mod panel.

Amod-free replacements for the joint runs are in the standard configs:
`configs/Pk_kids_7p.yaml` and `configs/Pk_kids_hmf_7p.yaml`.

The raw published numbers remain in
`data/Power_spec_targets/nonlinear_pk_targets/amod/` (data provenance).
