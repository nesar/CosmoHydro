# The "2-parameter cosmology rails to the box edge" issue

A running log of the investigation into why the **2-cosmology-parameter** MCMC
runs (cosmology free, hydro **fixed**) push Ωₘ (and σ₈) to the edges of the
simulation design box, and what we concluded. Written for future-us.

---

## Symptom

The `*_2cosmo` runs (vary only `omega_m`, `sigma_8`; hold the 5 hydro/subgrid
params fixed) produce posteriors pinned against the design-box edges instead of
sitting near the project fiducial (Ωₘ=0.14176, σ₈=0.8102). Examples:

| Run | hydro | Ωₘ median | note |
|---|---|---|---|
| `GSMF_CGD_fGas_2cosmo` | fixed @ midpoints | 0.1548 | box edge 0.155 |
| `GSMF_CGD_fGas_7p`     | **free**          | 0.1524 | rails even hydro-free |
| `GSMF_2cosmo`          | fixed @ midpoints | 0.1548 | box edge |
| `GSMF_7p`             | **free**          | 0.1418 | = fiducial ✓ |

Design box: Ωₘ ∈ [0.12, 0.155], σ₈ ∈ [0.70, 0.90].

---

## What we tried, in order

1. **Stronger prior.** Replaced the old weak fallback (midpoint Gaussian,
   σ = half-range) with a moderate fiducial-centered prior as the project
   default (`configs/_defaults.yaml`): Ωₘ ~ N(0.14176, 0.005), σ₈ ~ N(0.8102, 0.03).
   → Did **not** stop the railing. (And we deliberately did *not* go to a
   Planck-tight pin, which would just impose the answer.)

2. **Free hydro vs fixed hydro (3-observable).** The hydro-*free* 3-obs run
   (`GSMF_CGD_fGas_7p`) **also** rails → so fixing hydro is not the only cause;
   the cluster-gas observables pull too. Retired the whole `GSMF_CGD_fGas_*`
   trio (chains in `old_results/retired_GSMF_CGD_fGas/`) and switched to the
   fGas-free `GSMF_CGD_*` trio.

3. **GSMF-only control.** `GSMF_7p` (hydro free) sits at fiducial, but
   `GSMF_2cosmo` (hydro fixed) rails → for GSMF alone, fixing hydro is what
   does it.

4. **Hydro fixed at midpoints vs fiducial.** The 2cosmo configs were fixing
   hydro at **design midpoints** (3.0, 0.6, 1.3, 0.65, 0.61), not the project
   **fiducial hydro** (3, 0.5, 0.8, 0.51, 0.13). Fixed that. But the likelihood
   sweep then showed the railing *direction flips* with the hydro choice (see
   below), so this corrected the values without removing the railing.

5. **Likelihood sweeps** (`likelihood_sweep_cosmo.py`) — the decisive step.

---

## What the likelihood sweeps showed (hydro fixed at fiducial)

1D profiles (other param at fiducial) looked benign — GSMF peaked near fiducial.
But that was a **conditional-slice artifact**. The 2D maps over (Ωₘ, σ₈) tell the
real story:

| Case | likelihood peak | with MCMC prior |
|---|---|---|
| GSMF      | (0.120, 0.900)  | (0.1215, 0.897) |
| CGD       | (0.155, 0.726)  | (0.1545, 0.735) |
| GSMF+CGD  | (0.155, 0.723)  | (0.1545, 0.729) |

- **Both observables rail, to *opposite* corners.** GSMF → low Ωₘ / high σ₈;
  CGD → high Ωₘ / low σ₈. Each has a strong Ωₘ–σ₈ degeneracy running to a box
  corner; the fiducial point is strongly disfavoured by *both* individually.
- **The MCMC prior is nearly powerless.** Including the exact MCMC prior shifts
  the posterior peak by only ~0.001–0.002 — the fixed-hydro likelihood is so
  steep toward the corners (ΔlnP > 30 across the box) that it dominates.
- The railing **direction depends on the fixed hydro point** (midpoint pulled
  GSMF high; fiducial pulls GSMF low) — proof that cosmology is just absorbing
  whatever the frozen hydro cannot fit.

Plots: `likelihood_sweep_omega_m.png`, `likelihood_sweep_sigma_8.png`,
`likelihood_sweep_2d.png` (2 rows: likelihood / posterior).

---

## Root cause

Fixing hydro **breaks the hydro↔cosmology degeneracy**. We fit *real*
observations; the data's best-fit hydro is generally **not** the fiducial (or
midpoint) hydro, so when hydro is frozen, cosmology is the only remaining
freedom and it slides along the degeneracy until it hits a box corner. CGD (and
fGas) make it worse — they are strongly constraining (CGD assumes a flat 5%
error, `yerr = 0.05*y`) and pull hardest toward the high-Ωₘ / low-σ₈ corner,
into emulator-extrapolation territory beyond the design box.

This is **not a bug** — it is the expected behaviour of a degeneracy-broken fit.

---

## Lessons learned / recommendations

- **Trust the hydro-free 7p runs for cosmology.** With hydro marginalized, the
  hydro DOF absorb the misfit and Ωₘ lands at fiducial (`GSMF_7p` → 0.142). The
  `*_2cosmo` cosmology is *not* a measurement — it is a demonstration of how
  badly fixing hydro biases cosmology.
- **A prior cannot rescue a fixed-hydro 2cosmo run.** Our moderate prior barely
  moves the peak; the only prior that would "fix" it is one tight enough to just
  impose the fiducial answer.
- **Don't read 1D likelihood slices as the full story** — check the 2D map;
  degeneracies hide in the conditional slices.
- **Fix nuisance params at the data's best-fit, not at a convention** (midpoint
  or even the simulation fiducial), if you must fix them at all. Better: don't
  fix them — marginalize.
- **CGD's 5% error assumption is worth revisiting** — it makes CGD dominate the
  joint likelihood and drags cosmology to the box edge. Likewise check whether
  the emulator is even trustworthy out near Ωₘ = 0.155.
- The `GSMF_CGD_fGas` combination was retired because CGD **and** fGas together
  railed hardest; `GSMF_CGD` is the current multi-observable working set.

---

## Update: hard-truncated (`*_trunc`) priors retired in favour of `*_planck`

We tried a **hard ±1σ truncation** of the Planck Gaussian (`*_trunc` configs:
σ=0.0011/0.006, with `param_ranges` forcing `ln_prior → -inf` outside the
window). The sampler respected it (every point stayed inside), but the result
was misleading: the prior-comparison sweep showed the true GSMF+CGD posterior
peaks at **σ₈ ≈ 0.786**, which lies *below* the cut window [0.804, 0.816]. So the
hard wall **amputated** the real posterior and piled 54% of σ₈ samples against
the lower wall. A hard ±1σ wall is an arbitrary fence — not a physical bound — so
it has no place in inference.

**Resolution:** retired all `*_trunc` configs/chains to
`old_results/retired_trunc/`. Use **`*_planck`** instead — same Planck-width Gaussian
(σ=0.0011/0.006) but **no hard cut** (only the design box bounds it), giving a
complete (closed) posterior at (Ωₘ≈0.1415, σ₈≈0.786). The deeper signal is a
**σ₈ tension**: the data (GSMF+CGD, hydro fixed) prefer σ₈≈0.77 vs Planck 0.81,
and the likelihood peaks *inside* the box (it is not trying to escape it — the
"railing" was the Ωₘ–σ₈ degeneracy ridge). See `diagnostics/prior_comparison.png`.

---

## How to reproduce

```bash
cd Inference
# per-observable likelihood + posterior sweeps over (omega_m, sigma_8):
python diagnostics/likelihood_sweep_cosmo.py
# overlay triangles (GSMF and GSMF+CGD) + the GSMF-vs-GSMF+CGD 7p comparison:
python plot_7p_vs_2cosmo_5subgrid.py
```

Relevant configs:
- moderate prior (from `_defaults.yaml`): `GSMF_7p`, `GSMF_2cosmo`,
  `GSMF_5p_fid_cosmo`, `GSMF_CGD_7p`, `GSMF_CGD_2cosmo`, `GSMF_CGD_5p_fid_cosmo`.
- Planck prior (Planck-width Gaussian, no hard cut): `GSMF_7p_planck`,
  `GSMF_2cosmo_planck`, `GSMF_CGD_7p_planck`, `GSMF_CGD_2cosmo_planck`.
- retired hard-cut (`*_trunc`) and `GSMF_CGD_fGas_*` live under `old_results/`.

The `*_2cosmo*` runs fix hydro at the **fiducial** values (3, 0.5, 0.8, 0.51, 0.13).
