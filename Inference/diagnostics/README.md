# diagnostics/

Scratch directory for a handful of sanity checks on the inference pipeline.
Each script only *reads* existing chains/artefacts — it never modifies the
pipeline — and writes its plots/summaries into this directory. Scripts skip any
chain that isn't on disk yet, so they're safe to run while MCMCs are going.

(Formerly `v1_v2_comparison/`; broadened beyond the v1↔v2 check.)

## Checks

| Script | Question | Chains | Outputs |
|---|---|---|---|
| `check_v1_v2.py` | Does v2 (CosmoHydro) reproduce v1 (Flamingo) on the shared 5 subgrid params? | v1 GSMF, v2 5p, v2 7p_pk | `corner_overlay.png`, `summary_stats_compare.png`, `posterior_medians.txt` |
| `check_pk_marg_vs_fixed.py` | How much do the constraints change with marginalization vs. fixing — **Planck-prior (`*_pk`) chains only**, the clean 2-chain contrast. Cosmology *and* subgrid. | cosmo: `<suite>_7p_pk` (hydro marg) vs `<suite>_2cosmo_pk` (hydro fixed); subgrid: `<suite>_7p_pk` (cosmo marg) vs `<suite>_5p_fid_cosmo` (cosmo fixed) | **written to `results/`** (headline result): `pk_cosmo_marg_vs_fixed{,_GSMF_CGD}.png`, `pk_subgrid_marg_vs_fixed{,_GSMF_CGD}.png` + `_medians.txt` |
| `likelihood_sweep_cosmo.py` | Where does the likelihood/posterior peak in (Ωₘ, σ₈), and under which prior? | GSMF, CGD emulators + obs (no chains) | `likelihood_sweep_omega_m.png`, `_sigma_8.png`, `likelihood_sweep_2d.png` (3 rows: likelihood / ×moderate / ×Planck-zoom), `prior_comparison.png` |
| `select_fixed_hydro_points.py` | Is the fixed-hydro point actually the 7p peak? (**No** — the project/Frontier-E fiducial sits ~22σ away.) Picks 4 replacement points from the 7p posterior and writes their configs. | `GSMF_CGD_7p_pk` | `fixed_hydro_scan_points.txt`, `configs/GSMF_CGD_2cosmo_hyd{A,B,C,D}.yaml` |
| `check_fixed_hydro_scan.py` | 2p cosmology posterior vs fixed subgrid point, **Planck prior** (dotted). All chains share the Planck prior. | `GSMF_CGD_{7p_pk, 2cosmo_pk, 2cosmo_hydA..D}` | `gsmf_cgd_fixed_hydro_scan_pk{,_noFE}.png` + `_summary.txt` |
| `check_fixed_hydro_scan_wide.py` | Same, but **flat cosmology prior** (data-only). Marginalized reference is the flat-prior `GSMF_CGD_7p_wide` run. | `GSMF_CGD_{7p_wide, 2cosmo_wide, 2cosmo_hyd{A..D}_wide}` | `gsmf_cgd_fixed_hydro_scan_wide{,_noFE}.png` + `_summary.txt` |
| `plot_fixed_points_design.py` | **Where** do the fixed points sit in the full 7p space? Design (LHC) + 7p posterior + A/B/C/D/Frontier-E markers in every panel. | `GSMF_CGD_7p_pk` + the 2cosmo medians + `FinalDesign.txt` | `gsmf_cgd_fixed_points_design.png` |

> The fixed-hydro scan needs the `GSMF_CGD_2cosmo_hyd{A..D}` (Planck) and
> `..._wide` (flat) runs, plus `GSMF_CGD_7p_wide` (flat-prior 7p reference).
> `_noFE` variants drop the Frontier-E outlier and auto-zoom to A-D. All scan
> figures use the `gsmf_cgd_` prefix and draw the prior as a **dotted** curve
> (Planck Gaussian) or **dotted** horizontal line (flat).

Both `check_pk_marg_vs_fixed.py` checks run for **both** observable suites (`GSMF`
and `GSMF+CGD`); the base filename is the GSMF version, the `_GSMF_CGD` suffix the
GSMF+CGD version.

> **The `*_pk` (Planck-prior) runs are the canonical result** — they give complete,
> closed posteriors and are the most intuitive to read. The moderate-prior
> marg-vs-fixed checks (`check_{cosmo,subgrid}_marg_vs_fixed.py` and their
> `{cosmo,subgrid}_marg_vs_fixed*` plots) plus the moderate 7p/2cosmo chains were
> retired to `old_results/retired_moderate_prior/`.
>
> Also retired (earlier): the `GSMF_CGD_fGas_*` runs (CGD+fGas railed cosmology to
> the box edge) and the hard-cut `*_trunc` priors (the ±1σ wall amputated the
> posterior). See `2p_cosmology_issue.md`.

## Run

```bash
cd Inference
python diagnostics/check_pk_marg_vs_fixed.py        # both suites, Planck-prior chains only
python diagnostics/likelihood_sweep_cosmo.py        # sweeps + prior comparison (slow: emulator evals)
python diagnostics/check_v1_v2.py

# fixed-hydro scan: pick points, run the MCMCs (via screen wrappers), then plot
python diagnostics/select_fixed_hydro_points.py
./run_fixed_hydro_scan.sh          # 4 Planck-prior points A-D (MCMC_NWORKERS=4)
./run_7p_wide.sh                   # flat-prior 7p reference (for the wide plot)
# (plus the *_wide fixed points, run the same way)
python diagnostics/check_fixed_hydro_scan.py       # gsmf_cgd_..._pk{,_noFE}.png
python diagnostics/check_fixed_hydro_scan_wide.py  # gsmf_cgd_..._wide{,_noFE}.png
python diagnostics/plot_fixed_points_design.py     # gsmf_cgd_fixed_points_design.png
```

### The fixed-hydro scan

The "hydro fixed" 2p runs pin subgrid at the project/Frontier-E fiducial
`(3, 0.5, 0.8, 0.51, 0.13)`. That is **not** the peak of the marginalized 7p
posterior — it is ~22σ away (M_seed −9.6σ, v_kin −6.2σ). So the scan re-runs the
2p cosmology fit with hydro pinned at 4 points drawn from the 7p posterior itself
(Mahalanobis radius ≈0, 1, 2, 2σ) to test whether the cosmology posterior is an
artefact of the hydro choice.
