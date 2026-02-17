# CosmoHydro

GP-based emulator for summary statistics from cosmological hydrodynamic simulations, varying 5 subgrid + 2 cosmology parameters.

## Data

Simulation outputs should be placed in `data/400MPC_RUNS_5SG_2COSMO_PARAM/HAvoCC/`. The experimental design and parameter ranges are described in `data/README.txt`.

## Emulated Quantities

- **GSMF** -- Galaxy Stellar Mass Function
- **fGas** -- Cluster Gas Fraction (M_gas / M_500)
- **CGD** -- Cluster Gas Density Profile
- **CSFR** -- Cosmic Star Formation history (next)


## Emulator Package

Core modules in `codes/cosmo_hydro_emu/`:

- `load_hacc.py` -- Data loading utilities for HACC simulation outputs
- `pca.py` -- PCA decomposition of summary statistics
- `gp.py` -- Gaussian Process training and prediction
- `emu.py` -- Emulator wrapper (PCA + GP pipeline)
- `viz.py` -- Plotting and visualization helpers
- `snapshot_utils.py` -- Utilities for multi-redshift snapshot handling

Unused/experimental modules are kept in `codes/cosmo_hydro_emu/_unused/`.

## Notebooks

- `codes/gp_SGC_all.ipynb` -- Single-snapshot (z=0) emulation for all summary statistics, including sensitivity analysis and validation.
- `codes/gp_SGC_all_multiz.ipynb` -- Multi-snapshot emulation across redshifts, with redshift interpolation support.
