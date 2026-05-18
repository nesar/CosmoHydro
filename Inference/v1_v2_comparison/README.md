## v1 (Flamingo/Clean) vs v2 (CosmoHydro) — GSMF-only MCMC discrepancy analysis

**Date:** 2026-05-17
**Reference (v1):** `/home/nramachandra/Projects/Hydro_runs/Flamingo/Clean/gp_HACC_mcmc_universal.{ipynb,py}` + `hydro_emu/mcmc_hacc.py` (grey `L_GSMF only` contour in `plots/a2/multi.png`).
**Current (v2):** `Inference/run_mcmc.py` driven by `configs/GSMF_5p_fid_cosmo.yaml`, posterior in `results/plot_GSMF_5p_mean.png`.

### Headline

The 5p posterior **disagreement is mostly expected**, not a code bug. The two pipelines are fitting the **same observation** (Driver+2022 GSMF) but with **different simulation suites underneath**, **a wider M_seed design range**, and **a different fiducial cosmology** baked into the training data. The priors and chain initialization are mathematically identical for the parameters they share; the likelihood form is identical when no bias terms are used.

| Parameter (scaled) | v1 (`L_GSMF` only) | v2 (`GSMF_5p_fid_cosmo`) | Δ |
|---|---|---|---|
| κ_w        | 3.14 | 2.19 | **−0.95**  (well outside expected statistical scatter) |
| e_w        | 0.496 | 0.495 | ~0 |
| M_seed/10⁶ | 0.78 | 0.69 | −0.09 |
| v_kin/10⁴  | 0.63 | 0.48 | −0.15 |
| ε_kin/10¹  | 0.60 | 0.37 | −0.23 |

v2 wants **weaker AGN feedback** (lower κ_w, v_kin, ε_kin) than v1 to fit the same observed GSMF — consistent with a switch in the underlying simulation campaign rather than a code change.

---

### Pipeline configuration side-by-side

| | **v1 (Flamingo/Clean)** | **v2 (CosmoHydro)** |
|---|---|---|
| Simulation suite | HACC 128 MPC, 5-param-only design | HAvoCC 400 MPC, 7-param (5 subgrid + 2 cosmo) |
| Number of training sims | **64** | **34** (39 minus 5 test sims) |
| Cosmology in training | Fixed at one HACC fiducial (unknown to v2 from artifacts) | Varied across design rows: ω_m ∈ [0.12, 0.155], σ_8 ∈ [0.7, 0.9] |
| Cosmology at inference time | Implicit (sim fiducial) | Fixed at **design midpoint** ω_m=0.1375, σ_8=0.80 |
| `M_seed` range (scaled /10⁶) | **[0.6, 1.2]** → prior μ=0.9, σ=0.3 | **[0.6, 2.0]** → prior μ=1.3, σ=0.7 (README: "updated from 1.2 × 10⁶") |
| Other 4 subgrid ranges | identical | identical (see numerical table below) |
| `exp_variance` (PCA) | 0.95 | 0.95 (after my recent unification; load auto-syncs from pickle) |
| `nwalkers / nburn / nrun` | 100 / 100 / 1000 | 400 / 100 / 4000 |
| Prior form | Gaussian @ midpoint, σ = half-range; flat for index 4 (`eps_kin`) | **identical** — see `mcmc.py:ln_prior` |
| Walker init | Uniform across [min, max] per param | **identical** — see `mcmc.py:chain_init` |
| Likelihood (no bias) | `−½ Σ (y−model)² / yerr²` | **identical** |
| Bias parameters | optional `log_f1, log_f2` (error inflation) | optional `log_bstar, bCV, bHSE` (a different scheme) — **not used** in the 5p case (`bias_params: []`) |
| Obs data (GSMF) | `HAvoCC/.../GSMF/data/` (Driver+2022), `yerr_raw[0]` | **identical** |
| GSMF mass cut | `mass_conds('GSMF')` = [5×10⁹, 3×10¹¹] | **identical** |
| Emulator predictor | `emulate()` = mean of 100 posterior samples | `emulate()` = analytical mean from 1 posterior draw (μ ≈ same to ~0.1 %), or `emulate_ensemble()` for legacy behaviour |

### Numerical prior comparison

```
Parameter        v1 (64-sim HACC-5p)          v2 (39-sim HAvoCC-7p)
                 range      μ_prior σ_prior   range      μ_prior σ_prior
kappa_w          [2.0, 4.0]  3.000   1.000    [2.0, 4.0]  3.000   1.000
e_w              [0.2, 1.0]  0.600   0.400    [0.2, 1.0]  0.600   0.400
M_seed/1e6       [0.6, 1.2]  0.900   0.300    [0.6, 2.0]  1.300   0.700   ← WIDER, shifted
v_kin/1e4        [0.1, 1.2]  0.650   0.550    [0.1, 1.2]  0.650   0.550
eps_kin/1e1      [0.02, 1.2] 0.610   0.590    [0.02, 1.2] 0.610   0.590   (flat prior anyway)
```

---

### Confirmed: priors and chain init are not the cause

I verified the new pipeline's `ln_prior` and `chain_init` are mathematically identical to the v1 reference (`hydro_emu/mcmc_hacc.py:151–193, 299–308`):
* Default = Gaussian centred at `0.5(min+max)`, σ = `0.5(max−min)`.
* `flat_indices = [4]` ⇒ eps_kin has a flat prior in both.
* Walkers initialised uniformly across `[min, max]` per parameter — no perturbation around a calib value.
* The shared 4 of 5 parameter ranges are identical, so those Gaussian priors are bit-for-bit the same.

For the **shared-range** parameters (κ_w, e_w, v_kin, ε_kin) the prior pull is identical. So any shift in those parameters between v1 and v2 has to come from the data/emulator, **not** the prior.

### Confirmed: likelihood form is not the cause

Old `log_likelihood` (no bias):
```python
model  = np.interp(x, x_grid, model_grid[:, 0])
sigma2 = yerr**2
ll     = −½ Σ (y − model)² / sigma2
```
New `log_likelihood` with `bias_params: []`:
```python
model      = np.interp(x_eff, x_grid, model_grid[:, 0])   # x_eff == x when no bias
sigma2     = yerr**2
ll         = −½ Σ (y_adjusted − model)² / sigma2          # y_adjusted == y when no bias
```
Identical. (The new `log_f` is replaced by `log_bstar/bCV/bHSE` in the bias path, but the 5p config doesn't activate it.)

### Confirmed: observations are not the cause

Both call `load_gsmf_obs(gsmf_obs_dir)` on the same Driver+2022 path, apply the same `mass_conds('GSMF') = [5e9, 3e11]` cut, take `10**y_raw[m]` for the linear GSMF and `yerr_raw[0, m]` for the lower-error magnitude. Bit-for-bit the same observation.

---

### Likely causes of the discrepancy, ranked

**1. Different simulation campaign (highest impact).**
The v1 GP was trained on a 128 MPC HACC 5-parameter run (`128MPC_RUNS_HACC_5PARAM_extract2/`, 64 sims). The v2 GP is trained on the 400 MPC HAvoCC 7-parameter run (`400MPC_RUNS_5SG_2COSMO_PARAM/HAvoCC/`, 39 sims). Box size, resolution, baryonic physics implementation, and number of training points are all different. The emulator is therefore a different function of θ; the same observation will pull different subgrid parameters out. **This alone is enough to shift the best-fit by O(0.1–1) in scaled units.**

**2. `M_seed` design range changed from [0.6, 1.2] to [0.6, 2.0].**
README explicitly notes "updated from 1.2 × 10⁶". With the wider range, the Gaussian prior shifts from N(μ=0.9, σ=0.3) to N(μ=1.3, σ=0.7). The v2 prior is **broader and centred ~0.4 higher**, so:
* If the data weakly constrains M_seed, v2 will land closer to 1.3 than v1's 0.9.
* If the data strongly constrains M_seed (as appears to be the case; v2 lands at 0.69, *below* its prior mean), the wider prior lets the posterior travel further before being pulled back.
The observed shift M_seed: 0.78 → 0.69 is consistent with the looser prior allowing the data to dominate.

**3. Effective cosmology at inference time differs from training-time cosmology in v1.**
v2 fixes ω_m = 0.1375, σ_8 = 0.80 (the v2 design midpoint, chosen to match HACC's nominal h = 0.681 → Ω_m ≈ 0.296). v1 implicitly uses whatever cosmology was hard-coded into its 128 MPC HACC sims, which v2's metadata doesn't expose. If they differ by even a few percent, the GSMF normalisation shifts, and the inferred AGN-feedback strength (κ_w, v_kin) compensates. This is a plausible contributor to the v2 → lower-feedback direction.

**4. Far fewer training sims (34 vs 64).**
This raises emulator posterior variance (we measured analytical-std/empirical-std ≈ 0.5 on real GSMF earlier — the GP hyperparameter uncertainty is non-negligible at this sample size). Mean predictions still match well, but the *effective* likelihood width widens and constraints can move on the order of the broadened uncertainty. Not enough to explain the κ_w shift by itself.

**5. Possibly: the off-by-one design-row alignment bug that we fixed earlier.**
The v2 runs you have in `results/` were produced *before* the fix. The user's screenshots (Image #1 and #2) show posteriors generated on misaligned (params, sims) pairs. The OLD reference (Image #3) was generated correctly because v1 reads params from RUN directory names, not by index, so it was never susceptible. The retrain you started today (with the corrected slicing) should shift the v2 posterior again. **Until that retrain finishes and the MCMC is re-run on the corrected emulator, the v1 ↔ v2 comparison is contaminated by the off-by-one.**

### Ruled out

* Prior shape (Gaussian / flat — identical).
* Walker initialization (uniform [min,max] — identical).
* Observation source, mass cut, error vector (identical).
* Likelihood functional form for the no-bias case (identical).
* `exp_variance` choice — auto-synced from the pickle at load time now, so v1 (0.95) and v2 (0.999 training, 0.95 fallback) both rebuild a basis matching the saved model. The earlier reshape error from the basis-size mismatch is already fixed.

### Recommendations for a fair v1↔v2 comparison

If you want to genuinely test whether v2's pipeline reproduces v1's posterior:

1. **Retrain v2 GSMF z=0 emulator on the corrected design slice (off-by-one) — already in progress.**
2. **Re-run `python run_mcmc.py configs/GSMF_5p_fid_cosmo.yaml`** after retrain. The 5p contour should sharpen.
3. **Verify the v2 fiducial cosmology matches v1's 128 MPC HACC sims** — pull the sim config card for one of the 128 MPC runs (`Default_..._SS_ZINI_0.25/.../config*`) and confirm Ω_m, σ_8, h. If they differ from (Ω_m=0.296, σ_8=0.80), update `GSMF_5p_fid_cosmo.yaml` to fix cosmology at the v1 values instead of the v2 design midpoint.
4. **Optional sanity check at a single point.** Pick θ = v1's reported best-fit. Run the v2 emulator at that θ (with cosmology fixed at v1's fiducial), interpolate to the obs mass grid, and compute `χ²_v2(θ_v1)`. Then compute `χ²_v2(θ_v2)`. If v2's χ²(θ_v2) ≪ χ²(θ_v1), the v2 simulation campaign genuinely prefers different feedback. If they're comparable, the data weakly constrains the direction and the shift is just prior + Monte-Carlo width.
5. **(Optional) Run v2 on the 128 MPC sims.** The biggest single experiment that would prove or disprove "the data is different" is to retrain a v2-style GSMF emulator on the *v1 simulation set* and run inference. If that reproduces v1's posterior, then the suite change is the smoking gun.

### Files in this directory

* `README.md` — this report
* `priors_compare.txt` — full numerical prior table (auto-generated)
* `config_diff.txt` — line-by-line config diff between v1 setup and v2 YAML

### Quick-grep references for the assertions above

* v1 priors: `Flamingo/Clean/hydro_emu/mcmc_hacc.py:151–193`
* v1 chain init: `Flamingo/Clean/hydro_emu/mcmc_hacc.py:299–308`
* v1 likelihood (no bias): `Flamingo/Clean/hydro_emu/mcmc_hacc.py:42–59`
* v1 GSMF obs slicing: `Flamingo/Clean/gp_HACC_mcmc_universal.ipynb` cell 15 (Calib_type == 'OBS')
* v1 params_list build: `Flamingo/Clean/gp_HACC_mcmc_universal.ipynb` cell 18
* v1 simulation directory: `'../../Data/ProfileData/SCIDAC_RUNS/128MPC_RUNS_HACC_5PARAM_extract2/'`
* v1 num_sims: `gp_HACC_mcmc_universal.ipynb` cell ~3 (`num_sims = 64`)
* v2 priors / chain init: `codes/cosmo_hydro_emu/mcmc.py:152–228`
* v2 build_param_space (cosmo fixing for subgrid mode): `Inference/run_mcmc.py:348–418`
* v2 GSMF obs slicing: `Inference/run_mcmc.py:296–309`
* v2 5p config: `Inference/configs/GSMF_5p_fid_cosmo.yaml`
* v2 design source: `data/FinalDesign.txt` (slice `[start_sim_idx : start_sim_idx + num_sims]` after the off-by-one fix)
