# Plan: cosmology constraints from (quasi-)linear scales in the joint calibration

Status: **planning only** (2026-08-07). Nothing here is implemented yet.
Companion audit: `audit_joint_calibration.py` (run it for the numbers quoted).

## Motivation

The joint GSMF+CGD+Pk runs show that the KiDS P_m(k) term barely constrains
cosmology (Δχ² ≈ 3 across the chains) while GSMF/CGD — galaxy/cluster
observables entangled with subgrid physics — dominate. The cleanest cosmology
information in P_m lives at **(quasi-)linear scales**, where baryonic
suppression S(k) → 1 and P_hydro ≈ P_GO depends on (ω_m, σ₈) alone. Our
current setup is weakest exactly there.

## What the current box can and cannot do

Box: **L = 400 Mpc/h, 1600³ particles** (HAvoCC SciDAC design).

| quantity | value | consequence |
|---|---|---|
| fundamental mode k_f = 2π/L | **0.0157 h/Mpc** | nothing below this exists in the box |
| emulator k-grid | 0.0222 – 8.03 h/Mpc (511 bins) | lowest bins = 1.4 k_f |
| particle Nyquist πN/L | 12.6 h/Mpc | not the binding limit |
| P(k) mesh Nyquist (1024³ FFT) | 8.03 h/Mpc | grid max |
| half mesh-Nyquist (aliasing-safe) | **~4.0 h/Mpc** | current k_max = 7.0 is beyond it |

Mode counts in the KiDS log bins (Δln k ≈ 0.40):

| k [h/Mpc] | k/k_f | N modes | sample noise ~ √(2/N) |
|---|---|---|---|
| 0.0101 | 0.6 | ~1 | dropped (k_min=0.03) |
| 0.0150 | 1.0 | ~4 | dropped |
| 0.0224 | 1.4 | ~15 | dropped |
| **0.0334** | 2.1 | ~48 | **~20 %** |
| **0.0498** | 3.2 | ~160 | **~11 %** |
| 0.0742 | 4.7 | ~530 | ~6 % |

**Conclusions (apply now, independent of the extension):**
1. `k_min = 0.03` is *marginal*: the two lowest kept points carry 10–20 %
   box-realization noise on the model side. This — not a physical baryon or
   emulator failure — is the likely source of the visually larger low-k
   model–data offsets. (At the emu-var best fit the low-k points are NOT the
   statistically dominant misfit: mean |z| = 0.47 at k<0.1 vs 0.63 at k>0.1.)
   A safer floor is **k_min ≈ 0.05** (≥3 k_f, ≥150 modes) unless the GO
   emulator's low-k box variance is explicitly modeled.
2. `k_max = 7.0` exceeds the aliasing-safe half mesh-Nyquist; prefer
   **k_max ≈ 4.0** unless the extraction corrected mass-assignment aliasing.
3. The KiDS deprojection itself assumed Ω_m = 0.305 ± 0.012 in its lensing
   kernel. Solutions far from that (e.g. the railed ω_m = 0.154 → Ω_m = 0.337)
   are internally inconsistent with the data product being fit. Any future
   low-k pipeline should treat |Ω_m − 0.305| ≫ 0.012 regions with suspicion.

## Proposed extension (two stages)

### Stage 1 — replace the low-k GO model, keep current data
Below k ≈ 0.1 h/Mpc the box-trained GO emulator is noise/variance limited, but
the physics is (quasi-)linear and cosmology-only. Replace it there:

- **Model**: P_GO(k<k_switch) from **CosmicEmu (Mira-Titan IV)** — covers
  k ≈ 0.001–5 h/Mpc, sub-percent accurate, and is HACC-calibrated, so it is
  *the natural in-family choice* — or CAMB/CLASS linear + halofit as a
  cross-check. Set S(k)=1 below k ≈ 0.3 h/Mpc (baryon suppression is <0.5 %
  there in all design models — verify with the ratio emulator).
- **Stitching**: blend log P over a window (e.g. k ∈ [0.08, 0.15]) with a
  smooth taper; validate CosmicEmu vs our GO emulator in the overlap band
  [0.05, 1] h/Mpc across the 109 design cosmologies (expect agreement within
  the box sample variance; document residuals).
- **Payoff**: the existing KiDS points at 0.033, 0.050 become usable with a
  trustworthy model, and k down to ~0.01 (currently dropped) can be restored.

### Stage 2 — add genuinely linear-scale data (Planck)
The strongest σ₈/ω_m lever is CMB-scale linear physics:

- Option A (simple, robust): a **compressed Planck Gaussian prior** on
  (ω_m, σ₈) or (ω_b, ω_c, n_s, A_s) — we already have the machinery
  (`gaussian_priors`); this is what the `_planck` runs did. Cheap, but it is a
  *prior*, not a likelihood with scales.
- Option B (the idea here): a **linear-band P_m likelihood**, e.g. the
  Planck-2018 linear matter power reconstruction (or DESI/BOSS post-recon
  linear P(k)) at k ≈ 0.005–0.05 h/Mpc, modeled by Stage-1's CosmicEmu/CAMB
  branch. This adds real data at scales our box cannot reach, breaking the
  ω_m–σ₈ degeneracy from the data side rather than by prior.
- **Circularity guard**: never combine Option B data with an Option A prior
  derived from the same Planck chain; pick one per run and label configs
  accordingly (`_planckprior` vs `_lowkpk`).

### Implementation sketch (when we do it)
1. `linear_theory.py`: add `P_lin(k, z; omega_m, sigma_8)` (CAMB or tabulated
   CosmicEmu wrapper); new `lowk_go.py` with the stitched
   `P_go_ext(k, z, params7)`.
2. `PkEmulator.P_hydro`: use `P_go_ext` and S→1 below k_switch; flag via
   config key `lowk_model: cosmicemu | camb | none`.
3. `targets.py`: restore the dropped KiDS bins (k ≥ 0.01) behind a config
   switch; later add the Planck linear-band target as `kind: planck_lin`.
4. Validation notebook/script: overlap-band comparison + recovery test on a
   held-out design sim.

## Open questions
- Are the design sims fixed-phase? (If yes, low-k box variance is common-mode
  and partially cancels in parameter *dependence*; the absolute spectrum is
  still biased by the one realization.)
- CosmicEmu parameter coverage vs our design box corners (σ₈ ∈ [0.70, 0.90],
  ω_m ∈ [0.12, 0.155]) — inside Mira-Titan's design, but verify h/n_s/w
  handling at our fixed values.
- Whether to model the KiDS kernel's Ω_m=0.305 conditioning explicitly
  (importance-reweight or add a systematic) before pushing cosmology with it.
