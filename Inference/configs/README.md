# configs/

YAML trial definitions for `run_mcmc.py`. Each file describes one MCMC trial:
which observables, which parameters are free/fixed, and any prior overrides.

## How configs work

```bash
cd Inference
python run_mcmc.py configs/<trial>.yaml            # full run
python run_mcmc.py configs/<trial>.yaml --dry-run  # print priors/likelihood, skip sampling
```

`run_mcmc.py` loads **`_defaults.yaml` first**, then deep-merges the trial config
on top (trial values win, nested dicts merge key-by-key). So a trial file only
specifies what differs from the defaults. Outputs are written to
`results/` as `samples_<trial_name>.npy`, `params_list_<trial_name>.npy`, and a
copy `config_<trial_name>.yaml`.

### `_defaults.yaml` (shared, not a trial)
Holds the data paths (`DirIn`, `design_file`, `num_sims=109`, `model_dir`),
observation dirs, MCMC settings (`nwalkers=400`, `nburn=100`, `nrun=4000`), and
the **shared cosmology prior** used by every trial unless overridden:

```yaml
gaussian_priors:
  omega_m: {mu: 0.14176, sigma: 0.005}   # moderate, fiducial-centered
  sigma_8: {mu: 0.8102,  sigma: 0.03}
```

Trials where `omega_m`/`sigma_8` are fixed/absent ignore these automatically.

### `param_mode`
- `subgrid` — 5 subgrid (hydro) params free; cosmology **fixed** at the project
  fiducial (`omega_m=0.14176`, `sigma_8=0.8102`).
- `subgrid+cosmo` — all 7 params free (5 subgrid + 2 cosmo). "7p".
- `custom` — explicitly list the free params under `free_params`.

`flat_prior_indices: [4]` gives `eps_kin` (index 4) a flat prior instead of the
default Gaussian, matching the legacy `mcmc_hacc.py` convention; the subgrid
params otherwise use the broad default Gaussian (midpoint, σ=half-range).

## Trial catalogue

### 7-parameter (subgrid + cosmo) fits
| Config | Observables | Notes |
|---|---|---|
| `GSMF_7p.yaml` | GSMF | 7p; `eps_kin` flat. Main blue chain / subgrid-marg partner. |
| `GSMF_CGD_7p.yaml` | GSMF + CGD | 7p; `eps_kin` flat. Marginalized partner for the cosmo marg-vs-fixed check. |
| `GSMF_fGas_7p.yaml` | GSMF + fGas | 7p; `eps_kin` flat. |

### Subgrid-only fits (cosmology fixed at fiducial)
| Config | Observables | Notes |
|---|---|---|
| `GSMF_subgrid.yaml` | GSMF | 5 subgrid free. |
| `GSMF_5p_fid_cosmo.yaml` | GSMF | 5 subgrid free; `eps_kin` flat. Reproduces the legacy `mcmc_hacc.py` "GSMF only" result. Green chain / fixed partner for the subgrid marg-vs-fixed check. |
| `GSMF_CGD_5p_fid_cosmo.yaml` | GSMF + CGD | 5 subgrid free; `eps_kin` flat. GSMF+CGD analog of `GSMF_5p_fid_cosmo` (was `GSMF_CGD_subgrid`). |
| `CGD_CGED_cluster.yaml` | CGD + CGED | Cluster-only statistics, subgrid free. |

### Cosmology-only / custom
| Config | Observables | Notes |
|---|---|---|
| `GSMF_2cosmo.yaml` | GSMF | `omega_m`, `sigma_8` free, hydro fixed at fiducial. GSMF-only control; pairs with `GSMF_7p` for a GSMF-only cosmo marg-vs-fixed check. |
| `GSMF_CGD_2cosmo.yaml` | GSMF + CGD | `omega_m`, `sigma_8` free, hydro fixed at fiducial. Fixed partner for the GSMF+CGD cosmo marg-vs-fixed check (pairs with `GSMF_CGD_7p`). |
| `sigma8_vkin_custom.yaml` | GSMF + CGD | Custom: `v_kin`, `eps_kin`, `sigma_8` free. |

#### Planck-prior variants (`*_pk`)
`GSMF_7p_pk`, `GSMF_2cosmo_pk`, `GSMF_CGD_7p_pk`, `GSMF_CGD_2cosmo_pk` — same as
the base configs but the cosmology prior is a **Planck-width Gaussian**
(σ_ωm=0.0011, σ_σ8=0.006, via `gaussian_priors`), bounded only by the design box
(no hard cut). Use these for a Planck-informed, complete cosmology posterior.

### Bias / multi-redshift
| Config | Observables | Notes |
|---|---|---|
| `GSMF_CGD_bias.yaml` | GSMF + CGD | 7p + observational bias params; flat priors on indices `[4, 7, 8, 9]`. |
| `GSMF_multiz_CGD.yaml` | GSMF (z=0, z=1) + CGD (z=0.4) | 7p, multi-redshift (per-observable `redshift` keys). |

> All cosmology priors come from `_defaults.yaml` (moderate fiducial-centered
> Gaussian). The former tight Planck-pin variants (`*_fidprior`, `*_planck`) and
> the broad-default `*_match7p` 2cosmo variant were retired; their old chains
> live under `old_results/results_pre_fid_cosmo/`.
>
> The `GSMF_CGD_fGas_*` trio (`7p`, `2cosmo`, `all`) was retired to
> `old_results/retired_GSMF_CGD_fGas/`: adding CGD **and** fGas drove cosmology
> to the upper design-box edge (Ωₘ→0.155, σ₈→0.9) in both hydro-free and
> hydro-fixed runs. The `GSMF_CGD_*` trio above is the fGas-free replacement.
>
> The hard-truncated `*_trunc` cosmology priors were retired to
> `old_results/retired_trunc/`: the ±1σ hard wall amputated the real posterior
> (see `diagnostics/2p_cosmology_issue.md`). Use `*_pk` (Gaussian, no hard cut).
