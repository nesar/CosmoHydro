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
| `check_subgrid_marg_vs_fixed.py` | How much do the **subgrid** constraints change if cosmology is fixed vs. marginalized? | `<suite>_7p` (cosmo free) vs `<suite>_5p_fid_cosmo` (cosmo fixed) | `subgrid_marg_vs_fixed{,_GSMF_CGD}.png` + `_medians.txt` |
| `check_cosmo_marg_vs_fixed.py` | How much do the **cosmology** constraints change if hydro is fixed vs. marginalized (moderate vs Planck prior)? | `<suite>_{7p,2cosmo}{,_pk}` | `cosmo_marg_vs_fixed{,_GSMF_CGD}.png` + `_medians.txt` |
| `likelihood_sweep_cosmo.py` | Where does the likelihood/posterior peak in (Ωₘ, σ₈), and under which prior? | GSMF, CGD emulators + obs (no chains) | `likelihood_sweep_omega_m.png`, `_sigma_8.png`, `likelihood_sweep_2d.png` (3 rows: likelihood / ×moderate / ×Planck-zoom), `prior_comparison.png` |

Both marg-vs-fixed checks now run for **both** observable suites (`GSMF` and
`GSMF+CGD`); the base filename is the GSMF version, the `_GSMF_CGD` suffix the
GSMF+CGD version. The cosmo check overlays the moderate-prior and Planck-prior
(`*_pk`) runs when present (each skipped until its chain exists).

> Retired: the `GSMF_CGD_fGas_*` runs (CGD+fGas railed cosmology to the box edge)
> and the hard-cut `*_trunc` priors (the ±1σ wall amputated the posterior) — both
> under `old_results/`. See `2p_cosmology_issue.md`.

## Run

```bash
cd Inference
python diagnostics/check_subgrid_marg_vs_fixed.py   # both suites
python diagnostics/check_cosmo_marg_vs_fixed.py     # both suites
python diagnostics/likelihood_sweep_cosmo.py        # sweeps + prior comparison (slow: emulator evals)
python diagnostics/check_v1_v2.py
```
