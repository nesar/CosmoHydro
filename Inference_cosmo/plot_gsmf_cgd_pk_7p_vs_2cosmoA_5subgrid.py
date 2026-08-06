"""Pk analogue of Inference/diagnostics/plot_7p_lines_vs_hydA_5subgrid.py, for the
joint GSMF + CGD + KiDS power-spectra (2-redshift) runs.

Overlays, in the 7-parameter triangle:
  - GSMF_CGD_Pk_7p            : all 7 free, RED unfilled outline (marginalized)
  - GSMF_CGD_Pk_5p_fid_cosmo  : 5 subgrid, cosmology fixed at fiducial, GREEN fill
  - GSMF_CGD_Pk_2cosmo_hydA   : 2 cosmology, hydro fixed at A (7p peak), BLUE fill

Reuses the plotting helpers in Inference/plot_7p_vs_2cosmo_5subgrid.py, but loads
the chains from Inference_cosmo/results/ and draws the BROAD (default Gaussian)
cosmology prior these runs actually used — NOT the Planck-tight prior.

Output: gsmf_cgd_pk_7p_vs_2cosmoA_5subgrid.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))        # Inference_cosmo/
ROOT = os.path.dirname(HERE)                             # CosmoHydro/
sys.path.insert(0, os.path.join(ROOT, 'Inference'))
import plot_7p_vs_2cosmo_5subgrid as P                   # triangle helpers

RESULTS = os.path.join(HERE, 'results')

# Broad DEFAULT Gaussian cosmology prior: N(midpoint, half-range) over the design
# box — exactly what ln_prior uses when no gaussian_priors override is set.
BROAD_PRIOR = {'omega_m': (0.1375, 0.0175, 0.12, 0.155),
               'sigma_8': (0.80,  0.10,   0.70, 0.90)}


def load(trial):
    s = os.path.join(RESULTS, f'samples_{trial}.npy')
    p = os.path.join(RESULTS, f'params_list_{trial}.npy')
    if not (os.path.exists(s) and os.path.exists(p)):
        print(f'  skip (missing): {trial}')
        return None
    samples = np.load(s)
    pl = np.load(p, allow_pickle=True).tolist()
    names = [q[0] for q in pl]
    ranges = {q[0]: (float(q[2]), float(q[3])) for q in pl}
    print(f'  loaded {trial}: {samples.shape}')
    return samples, names, ranges


def main():
    # (trial, color, filled, linewidth, label)
    specs = [
        ('GSMF_CGD_Pk_7p',           '#d62728', False, 2.6,
         '7 params (GSMF+CGD+Pk, broad prior, cosmo + hydro free)'),
        ('GSMF_CGD_Pk_5p_fid_cosmo', '#2ca02c', True, 1.5,
         '5 subgrid only (GSMF+CGD+Pk, cosmology fixed at fiducial)'),
        ('GSMF_CGD_Pk_2cosmo_hydA',  '#1f77b4', True, 1.5,
         '2 cosmo only (GSMF+CGD+Pk, hydro fixed at A = 7p peak)'),
    ]
    avail = []
    for trial, color, filled, lw, label in specs:
        l = load(trial)
        if l is not None:
            avail.append((l, color, filled, lw, label))
    if not avail:
        print('  no chains — nothing to plot')
        return

    anchor = max(avail, key=lambda t: len(t[0][1]))       # anchor on the 7p run
    names, ranges = anchor[0][1], anchor[0][2]
    mcs    = [P.mcsamples(l, label) for (l, _, _, _, label) in avail]
    colors = [c for (_, c, _, _, _) in avail]
    filled = [f for (_, _, f, _, _) in avail]
    lws    = [w for (_, _, _, w, _) in avail]
    labels = [lab for (_, _, _, _, lab) in avail]

    g = P._plotter()
    g.triangle_plot(mcs, params=names, filled=filled, legend_labels=labels,
                    param_limits=P.axis_limits([l for (l, _, _, _, _) in avail], names),
                    contour_colors=colors,
                    line_args=[{'color': c, 'lw': w} for c, w in zip(colors, lws)])
    P.fiducial_crosshairs(g, names)
    P.overlay_priors(g, names, ranges, BROAD_PRIOR)
    plt.suptitle('GSMF+CGD+Pk (KiDS, 2 redshifts), broad prior — 7p (outline) vs '
                 '2-cosmology @ A vs 5-subgrid MCMC (dashed = prior)',
                 y=1.005, fontsize=13)

    out = os.path.join(HERE, 'gsmf_cgd_pk_7p_vs_2cosmoA_5subgrid.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
