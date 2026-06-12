# diagnostics/

Scratch directory for a handful of sanity checks on the inference pipeline.
Each script only *reads* existing chains/artefacts — it never modifies the
pipeline — and writes its plots/summaries into this directory. Scripts skip any
chain that isn't on disk yet, so they're safe to run while MCMCs are going.

(Formerly `v1_v2_comparison/`; broadened beyond the v1↔v2 check.)

## Checks

| Script | Question | Chains | Outputs |
|---|---|---|---|
| `check_v1_v2.py` | Does v2 (CosmoHydro) reproduce v1 (Flamingo) on the shared 5 subgrid params? | v1 GSMF, v2 5p, v2 7p | `corner_overlay.png`, `summary_stats_compare.png`, `posterior_medians.txt` |
| `check_subgrid_marg_vs_fixed.py` | How much do the **subgrid** constraints change if cosmology is fixed vs. marginalized? | `GSMF_7p` (cosmo free) vs `GSMF_5p_fid_cosmo` (cosmo fixed) | `subgrid_marg_vs_fixed.png`, `..._medians.txt` |
| `check_cosmo_marg_vs_fixed.py` | How much do the **cosmology** constraints change if hydro is fixed vs. marginalized? | `GSMF_CGD_fGas_7p` (hydro free) vs `GSMF_CGD_fGas_2cosmo` (hydro fixed) | `cosmo_marg_vs_fixed.png`, `..._medians.txt` |

All MCMC chains use the shared project-default cosmology prior (moderate
fiducial-centered Gaussian; see `../configs/_defaults.yaml`).

> Note: `check_cosmo_marg_vs_fixed.py`'s marginalized partner — the 3-observable
> `GSMF_CGD_fGas_7p` chain — has not been run under the current prior yet, so
> that overlay is skipped until `configs/GSMF_CGD_fGas_7p.yaml` is run.

## Run

```bash
cd Inference
python diagnostics/check_subgrid_marg_vs_fixed.py
python diagnostics/check_cosmo_marg_vs_fixed.py
python diagnostics/check_v1_v2.py
```
