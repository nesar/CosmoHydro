# CosmoHydro

GP-based emulator and MCMC inference framework for summary statistics from cosmological hydrodynamic simulations, varying 5 subgrid + 2 cosmology parameters.

## Parameters

7 parameters total (5 subgrid + 2 cosmology):

| Parameter | Symbol | Scaled by |
|-----------|--------|-----------|
| AGN wind coupling | kappa_w | -- |
| AGN energy efficiency | e_w | -- |
| BH seed mass | M_seed | 1e6 |
| Kinetic feedback velocity | v_kin | 1e4 |
| Kinetic feedback efficiency | eps_kin | 1e1 |
| Matter density | omega_m | -- |
| Amplitude of fluctuations | sigma_8 | -- |

Design: 110 simulations (400 Mpc/h boxes, from `data/FinalDesign.txt`) -- a 40-point D-optimal core (0--39) plus three space-filling extensions (40--89, 90--99, 100--110). Test set held out: `[3, 11, 19, 27, 35]`.

Project fiducial cosmology (for `param_mode: subgrid` runs): `omega_m = 0.14176`, `sigma_8 = 0.8102`.

## Data

Simulation outputs in `data/scidac-400MPC_RUNS_5SG_2COSMO_PARAM-extracts_20260323/RUN{001-110}/extract/`.
Power spectrum data in `data/scidac-olcf-pk_3/`.

## Emulated Quantities

**Primary observables (multi-z):**
- **GSMF** -- Galaxy Stellar Mass Function (11 snapshots, z=0--2)
- **HMF** -- Halo Mass Function (11 snapshots, z=0--2)
- **fGas** -- Cluster Gas Fraction (7 snapshots, z<=1.0)
- **Pk** -- Power Spectrum Suppression ratio (z=0 only)
- **CSFR** -- Cosmic Star Formation Rate history (z=0 only)

**Cluster profile statistics (multi-z, z<=0.5):**
- **CGD** -- Cluster Gas Density Profile
- **CGED** -- Cluster Gas Electron Density Profile
- **CPP** -- Cluster Gas Pressure Profile
- **CTP** -- Cluster Gas Temperature Profile
- **CEP** -- Cluster Gas Entropy Profile
- **CEEP** -- Cluster Electron Entropy Profile
- **CMP** -- Cluster Gas Metallicity Profile
- **CYP** -- Cluster Compton-y (tSZ) Profile

## Emulator Package

Core modules in `codes/cosmo_hydro_emu/`:

| Module | Description |
|--------|-------------|
| `load_hacc.py` | Data loading for all observables (single-z and multi-z readers) |
| `pca.py` | PCA decomposition of summary statistics |
| `gp.py` | Gaussian Process training and prediction (SEPIA) |
| `emu.py` | Emulator wrapper: `emulate()`, `emu_redshift()`, `load_model_multiple()`, `load_model_autosync()` |
| `viz.py` | Plotting and visualization helpers |
| `snapshot_utils.py` | Multi-redshift snapshot handling (11 snapshots, z_initial=200) |
| `mcmc.py` | MCMC inference: likelihood, priors, emcee sampler, with multi-z support |

Unused/experimental modules are kept in `codes/cosmo_hydro_emu/_unused/`.

## Training Notebooks

| Notebook | Observables | Snapshots |
|----------|-------------|-----------|
| `codes/01_train_emulators_csfr.ipynb` | CSFR | z=0 only |
| `codes/02_train_emulators_multiz.ipynb` | GSMF, HMF, fGas, Pk-ratio | multi-z (Pk: z=0 only) |
| `codes/03_train_emulators_profiles_multiz.ipynb` | CGD, CGED, CPP, CTP, CEP, CEEP, CMP, CYP | multi-z (z<=0.5) |

Trained models are saved to `models/`:
- Single-z models: `models/<OBS>_multivariate_model_z_index0.pkl`
- Multi-z models: `models/<OBS>_multiz/multivariate_model_z_index{i}.pkl`

PCA basis size is auto-synced from each pickle at load time (`load_model_autosync`) so chains and emulators stay consistent across retrains.

Previous notebooks are preserved in `codes/_old/`.

## Inference

YAML-config-driven MCMC framework in `Inference/`. Trial configs are deep-merged on top of `Inference/configs/_defaults.yaml`, so shared data/MCMC/observation-path settings live in one place.

### Quick start

```bash
cd Inference/

# Dry run (loads data + models, tests likelihood, no sampling)
python run_mcmc.py configs/GSMF_7p.yaml --dry-run

# Run MCMC
python run_mcmc.py configs/GSMF_7p.yaml

# Plot a single chain or compare multiple
python plot_mcmc.py results/samples_GSMF_7p.npy
python plot_mcmc.py results/samples_GSMF_7p.npy results/samples_GSMF_CGD_fGas_7p.npy \
    --labels "GSMF" "GSMF+CGD+fGas" --output results/comparison.png
```

### Available trial configs

| Config | Observables | Parameters |
|--------|-------------|------------|
| `GSMF_5p_fid_cosmo.yaml` | GSMF | 5 subgrid, cosmology fixed at project fiducial |
| `GSMF_7p.yaml` | GSMF | all 7 (subgrid + cosmo) |
| `GSMF_subgrid.yaml` | GSMF | 5 subgrid only |
| `GSMF_CGD_7p.yaml` | GSMF + CGD | all 7 |
| `GSMF_CGD_subgrid.yaml` | GSMF + CGD | 5 subgrid only |
| `GSMF_fGas_7p.yaml` | GSMF + fGas | all 7 |
| `GSMF_CGD_fGas_7p.yaml` | GSMF + CGD + fGas | all 7 |
| `GSMF_CGD_fGas_7p_fidprior.yaml` | GSMF + CGD + fGas | all 7, fiducial-cosmology prior |
| `GSMF_CGD_fGas_2cosmo.yaml` | GSMF + CGD + fGas | 2 cosmology only |
| `GSMF_CGD_fGas_all.yaml` | GSMF + CGD + fGas | all 7, all-Gaussian priors |
| `CGD_CGED_cluster.yaml` | CGD + CGED | 5 subgrid only, cluster-only fit |
| `GSMF_CGD_bias.yaml` | GSMF + CGD | all 7 + 3 bias params |
| `GSMF_multiz_CGD.yaml` | GSMF(z=0,1) + CGD(z=0.4) | all 7, multi-z |
| `sigma8_vkin_custom.yaml` | GSMF + CGD | custom: v_kin + eps_kin + sigma_8 |

### Config options

- **`observables`**: list of names or `{name, redshift}` dicts
- **`param_mode`**: `subgrid` | `cosmo` | `subgrid+cosmo` | `custom`
- **`free_params`** (custom mode): subset of parameter keys to vary
- **`fixed_params`**: fix any parameter by short key
- **`bias_params`**: optional observational bias parameters (log_bstar, bCV, bHSE)
- **`flat_prior_indices`**: which free params get flat vs Gaussian priors

### MCMC settings

- Prior: Gaussian on all params except `eps_kin` (flat), centered at midpoint of design range with sigma = half-range
- Walker init: uniform across `[min, max]` per parameter
- Likelihood: chi-squared (`sigma2 = yerr**2`), summed over observables
- Sampler: emcee `EnsembleSampler` with multiprocessing Pool
- Defaults (`_defaults.yaml`): 400 walkers, 100 burn-in, 4000 production steps

### v1 (Flamingo) -- v2 (CosmoHydro) cross-check

`Inference/v1_v2_comparison/make_plots.py` generates a side-by-side
GSMF posterior overlay (`corner_overlay.png`), a posterior-predictive
comparison across GSMF / fGas / CGD (`summary_stats_compare.png`),
and a posterior-median table (`posterior_medians.txt`).

## Documentation

Four standalone LaTeX notes in `documentation/`:

| File | Contents |
|------|----------|
| `main.tex` | Simulation setup, parameter ranges, snapshot windows, design-matrix appendix tables |
| `summaries_emulation.tex` | P(k) suppression, HMF, GSMF, fGas, CSFR emulation figures |
| `cluster_profiles.tex` | Eight cluster-profile emulators (CGD, CGED, CEP, CEEP, CMP, CPP, CTP, CYP) |
| `calibration_results.tex` | MCMC posteriors and v1 -- v2 benchmarking |

The three companion notes are flat-layout: copy them next to `plots/`, `Inference/results/` (as `results/`), and `Inference/v1_v2_comparison/` (as `v1_v2_comparison/`) for an Overleaf-style build. `main.tex` additionally references `data/FinalDesign.txt` for the appendix tables.
