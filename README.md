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

Design: 39 simulations (400 Mpc/h boxes, from `data/FinalDesign.txt`).

## Data

Simulation outputs in `data/400MPC_RUNS_5SG_2COSMO_PARAM/HAvoCC/RUN{001-039}/extract/`.
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
| `emu.py` | Emulator wrapper: `emulate()`, `emu_redshift()`, `load_model_multiple()` |
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

Previous notebooks are preserved in `codes/_old/`.

## Inference

YAML-config-driven MCMC framework in `Inference/`. Run different trials by combining observables, parameter spaces, and redshifts without duplicating code.

### Quick start

```bash
cd Inference/

# Dry run (loads data + models, tests likelihood, no sampling)
python run_mcmc.py configs/GSMF_7p.yaml --dry-run

# Run MCMC
python run_mcmc.py configs/GSMF_7p.yaml

# Compare results from multiple trials
python plot_mcmc.py results/samples_GSMF_7p.npy results/samples_GSMF_CGD_7p.npy \
    --labels "GSMF" "GSMF+CGD" --output results/comparison.png
```

### Available trial configs

| Config | Observables | Parameters |
|--------|-------------|------------|
| `GSMF_7p.yaml` | GSMF | all 7 (subgrid + cosmo) |
| `GSMF_CGD_7p.yaml` | GSMF + CGD | all 7 |
| `GSMF_fGas_7p.yaml` | GSMF + fGas | all 7 |
| `GSMF_CGD_fGas_7p.yaml` | GSMF + CGD + fGas | all 7 |
| `GSMF_subgrid.yaml` | GSMF | 5 subgrid only |
| `GSMF_CGD_subgrid.yaml` | GSMF + CGD | 5 subgrid only |
| `CGD_CGED_cluster.yaml` | CGD + CGED | 5 subgrid only |
| `sigma8_vkin_custom.yaml` | GSMF + CGD | custom: v_kin + eps_kin + sigma_8 |
| `GSMF_CGD_bias.yaml` | GSMF + CGD | all 7 + 3 bias params |
| `GSMF_multiz_CGD.yaml` | GSMF(z=0,1) + CGD(z=0.4) | all 7, multi-z |

### Config options

- **`observables`**: list of names or `{name, redshift}` dicts
- **`param_mode`**: `subgrid` | `cosmo` | `subgrid+cosmo` | `custom`
- **`free_params`** (custom mode): subset of parameter keys to vary
- **`fixed_params`**: fix any parameter by short key
- **`bias_params`**: optional observational bias parameters (log_bstar, bCV, bHSE)
- **`flat_prior_indices`**: which free params get flat vs Gaussian priors

### MCMC settings (matching Flamingo reference)

- Prior: Gaussian on all params except eps_kin (flat), centered at midpoint of range
- Likelihood: chi-squared (`sigma2 = yerr**2`)
- Sampler: emcee EnsembleSampler with multiprocessing Pool
- Default: 100 walkers, 100 burn-in, 1000 production steps
