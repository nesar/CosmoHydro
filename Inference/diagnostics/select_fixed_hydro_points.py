"""Pick the fixed-hydro points for the 2p-cosmology scan, and verify what the
CURRENT fixed point actually is.

Question being answered
-----------------------
The "hydro fixed" 2p-cosmology runs pin the 5 subgrid parameters at the project
fiducial (3, 0.5, 0.8, 0.51, 0.13) -- the Frontier-E-style subgrid point. Is that
the same as the peak of the marginalized 7p posterior? (Answer: no, it is many
sigma away in M_seed and v_kin.) If the fixed point is far from where the data
actually want the hydro, the 2p cosmology posterior is forced to compensate, and
the apparent cosmology "constraint" is really an artefact of the hydro choice.

So: scan the fixed point. Choose 4 REASONABLE fixed hydro points drawn from the
7p posterior itself (Mahalanobis distance ~0, ~1, ~2 sigma in different
directions), run a 2p cosmology MCMC at each, and see how much the cosmology
posterior moves.

Using actual posterior samples (rather than analytic offsets) guarantees every
point is inside the design box and is a genuinely plausible hydro configuration.

Writes: configs/<suite>_2cosmo_hyd{A,B,C,D}.yaml  + a summary table here.

    python diagnostics/select_fixed_hydro_points.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INFER = os.path.dirname(HERE)
RES = os.path.join(INFER, 'results')
CONFIGS = os.path.join(INFER, 'configs')

SUITE = 'GSMF_CGD'                 # the suite showing the strong fixed-hydro bias
MARG_TRIAL = f'{SUITE}_7p_pk'
OBSERVABLES = ['GSMF', 'CGD']

SG_NAMES = ['kappa_w', 'e_w', 'M_seed', 'v_kin', 'eps_kin']
SG_PRETTY = ['kappa_w', 'e_w', 'M_seed/1e6', 'v_kin/1e4', 'eps_kin/1e1']
FIDUCIAL = np.array([3.0, 0.5, 0.8, 0.51, 0.13])     # current fixed point
SEED = 12345                                          # reproducible selection

# Target Mahalanobis radii (in sigma) for the 4 chosen points.
TARGET_RADII = [0.0, 1.0, 2.0, 2.0]
LETTERS = ['A', 'B', 'C', 'D']

CONFIG_TEMPLATE = """# Trial: {obs_str}, 2 cosmology params free, hydro FIXED at scan point {letter}.
# Part of the fixed-hydro scan (diagnostics/select_fixed_hydro_points.py):
#   {desc}
# Planck-width Gaussian cosmology prior, no hard truncation (design box only).
# Shared data/mcmc/obs_dirs/output_dir come from _defaults.yaml.
trial_name: {trial}

observables:
{obs_block}
param_mode: custom
free_params:
  - omega_m
  - sigma_8

# Fixed subgrid point {letter} (scaled units), drawn from the {marg} posterior at
# Mahalanobis radius {radius:.2f} sigma.
fixed_params:
  kappa_w: {p0:.4f}
  e_w: {p1:.4f}
  M_seed: {p2:.4f}
  v_kin: {p3:.4f}
  eps_kin: {p4:.4f}

gaussian_priors:
  omega_m: {{mu: 0.14176, sigma: 0.0011}}
  sigma_8: {{mu: 0.8102, sigma: 0.006}}
"""


def main():
    s = np.load(os.path.join(RES, f'samples_{MARG_TRIAL}.npy'))
    sg = s[:, :5]
    med = np.median(sg, axis=0)
    std = sg.std(axis=0)
    cov = np.cov(sg, rowvar=False)
    inv = np.linalg.inv(cov)

    lines = []
    def out(t=''):
        print(t); lines.append(t)

    out('Fixed-hydro scan: point selection')
    out('=' * 72)
    out(f'marginalized reference: {MARG_TRIAL}  ({s.shape[0]:,} samples)')
    out('')
    out('1) Is the CURRENT fixed point (project fiducial) the 7p peak?')
    out('')
    out(f'   {"param":12s} {"fiducial":>9s} {"7p median":>10s} {"7p std":>8s} {"offset":>9s}')
    for i, n in enumerate(SG_PRETTY):
        out(f'   {n:12s} {FIDUCIAL[i]:9.4f} {med[i]:10.4f} {std[i]:8.4f} '
            f'{(FIDUCIAL[i]-med[i])/std[i]:8.2f}s')
    d_fid = float(np.sqrt((FIDUCIAL - med) @ inv @ (FIDUCIAL - med)))
    out('')
    out(f'   => Mahalanobis distance of the fiducial from the 7p posterior: '
        f'{d_fid:.1f} sigma')
    out('   => NO: the current fixed point is NOT the 7p peak. It is the project/'
        'Frontier-E')
    out('      fiducial subgrid point, far outside the 7p posterior (esp. M_seed, '
        'v_kin).')
    out('')

    # Mahalanobis radius of every sample; pick points near the target radii,
    # spread out in direction so the 4 points probe different parts of the shell.
    d = sg - med
    r = np.sqrt(np.einsum('ij,jk,ik->i', d, inv, d))
    rng = np.random.default_rng(SEED)

    out('2) Chosen fixed points (drawn from the 7p posterior itself)')
    out('')
    chosen, chosen_r = [], []
    for k, (letter, target) in enumerate(zip(LETTERS, TARGET_RADII)):
        if target == 0.0:
            pt, rad = med.copy(), 0.0
        else:
            band = np.where(np.abs(r - target) < 0.08)[0]
            # push successive points away from the ones already chosen so the
            # shell is sampled in different directions, not the same corner
            if chosen and len(band):
                prev = np.array(chosen[1:]) if len(chosen) > 1 else np.array(chosen)
                cand = sg[band]
                sep = np.min([[float(np.sqrt((c - p) @ inv @ (c - p))) for p in prev]
                              for c in cand], axis=1)
                idx = band[int(np.argmax(sep))]
            else:
                idx = int(rng.choice(band))
            pt, rad = sg[idx].copy(), float(r[idx])
        chosen.append(pt); chosen_r.append(rad)
        out(f'   {letter}: radius {rad:4.2f}s  ' +
            '  '.join(f'{n}={v:.4f}' for n, v in zip(SG_PRETTY, pt)))
    out('')
    out(f'   (for scale, the current fiducial fixed point sits at {d_fid:.1f}s)')
    out('')

    # ---- write configs -------------------------------------------------
    obs_block = ''.join(f'  - {o}\n' for o in OBSERVABLES)
    obs_str = ' + '.join(OBSERVABLES)
    out('3) Configs written')
    out('')
    for letter, pt, rad in zip(LETTERS, chosen, chosen_r):
        trial = f'{SUITE}_2cosmo_hyd{letter}'
        desc = (f'point {letter}, Mahalanobis radius {rad:.2f} sigma from the '
                f'{MARG_TRIAL} subgrid posterior')
        txt = CONFIG_TEMPLATE.format(
            obs_str=obs_str, letter=letter, desc=desc, trial=trial,
            obs_block=obs_block, marg=MARG_TRIAL, radius=rad,
            p0=pt[0], p1=pt[1], p2=pt[2], p3=pt[3], p4=pt[4])
        path = os.path.join(CONFIGS, f'{trial}.yaml')
        with open(path, 'w') as f:
            f.write(txt)
        out(f'   configs/{trial}.yaml')
    out('')
    out('Run them with:')
    for letter in LETTERS:
        out(f'   python run_mcmc.py configs/{SUITE}_2cosmo_hyd{letter}.yaml')

    with open(os.path.join(HERE, 'fixed_hydro_scan_points.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwrote {os.path.join(HERE, "fixed_hydro_scan_points.txt")}')


if __name__ == '__main__':
    main()
