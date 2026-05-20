"""Combined v1 (Flamingo) ↔ v2 (CosmoHydro) comparison plots.

Outputs (in this directory):
  corner_overlay.png         Triangle plot, v1 5p vs v2 5p vs v2 7p on 5 shared subgrid params.
  gsmf_compare.png           GSMF observation + model curves at each posterior's median.
                             v1 emulator @ v1 median, v2 emulator @ v2 5p median, v2 emulator @ v2 7p median.

No code in the inference pipeline is modified — this script only reads
existing artefacts.
"""
import os, sys, contextlib, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import plots as gd_plots
from getdist import MCSamples

V1 = '/home/nramachandra/Projects/Hydro_runs/Flamingo/Clean'
V2 = '/home/nramachandra/Projects/Hydro_runs/CosmoHydro'
OUT = os.path.join(V2, 'Inference/v1_v2_comparison')


@contextlib.contextmanager
def quiet():
    s = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w'); yield
    finally:
        sys.stdout.close(); sys.stdout = s


# ---------------------------------------------------------------- chains
v1_chain = np.load(f'{V1}/plots/universal/npy/samples_HACC_fixed_None_obs_GSMF.npy')
v2_5p    = np.load(f'{V2}/Inference/results/samples_GSMF_5p_fid_cosmo.npy')
v2_7p    = np.load(f'{V2}/Inference/results/samples_GSMF_7p.npy')

print(f'v1 GSMF chain: {v1_chain.shape}')
print(f'v2 5p chain:   {v2_5p.shape}')
print(f'v2 7p chain:   {v2_7p.shape}')

# Shared subgrid columns (first 5 of each)
labels_sg = [r'\kappa_\mathrm{w}', r'e_\mathrm{w}',
             r'M_\mathrm{seed}/10^{6}', r'v_\mathrm{kin}/10^{4}',
             r'\epsilon_\mathrm{kin}/10^{1}']
names_sg  = ['kappa_w', 'e_w', 'M_seed_e6', 'v_kin_e4', 'eps_kin_e1']

# v2 ranges (current design)
RANGES_V2 = {'kappa_w': (2.0, 4.0), 'e_w': (0.2, 1.0),
             'M_seed_e6': (0.6, 2.0), 'v_kin_e4': (0.1, 1.2),
             'eps_kin_e1': (0.02, 1.2)}
# v1 ranges (older HACC 5p design) — used to set common plot limits
RANGES_V1 = {'kappa_w': (2.0, 4.0), 'e_w': (0.2, 1.0),
             'M_seed_e6': (0.6, 1.2), 'v_kin_e4': (0.1, 1.2),
             'eps_kin_e1': (0.02, 1.2)}
# Plot limits — use v2 (wider) where ranges differ
PLOT_RANGES = {k: RANGES_V2[k] for k in names_sg}


def make_samples(chain, label):
    return MCSamples(samples=chain[:, :5], names=names_sg, labels=labels_sg,
                     label=label, ranges=PLOT_RANGES)


s_v1   = make_samples(v1_chain, r'v1 (Flamingo 128 MPC, $\mathcal{L}_{\mathrm{GSMF}}$ only)')
s_v25p = make_samples(v2_5p,    r'v2 5p, cosmo fixed ($\mathcal{L}_{\mathrm{GSMF}}$ only)')
s_v27p = make_samples(v2_7p,    r'v2 7p, cosmo free ($\mathcal{L}_{\mathrm{GSMF}}$ only)')

# ---------------------------------------------------------------- triangle
g = gd_plots.get_subplot_plotter(width_inch=11)
g.settings.alpha_filled_add = 0.55
g.settings.legend_fontsize = 13
g.settings.axes_fontsize   = 12
g.settings.lab_fontsize    = 14
g.triangle_plot([s_v1, s_v25p, s_v27p], names_sg, filled=True,
                contour_colors=['gray', 'tab:blue', 'tab:red'],
                line_args=[{'color': 'gray', 'ls': '-', 'lw': 1.5},
                           {'color': 'tab:blue', 'ls': '-', 'lw': 1.5},
                           {'color': 'tab:red', 'ls': '-', 'lw': 1.5}],
                legend_loc='upper right')
g.fig.suptitle('GSMF-only posteriors: v1 vs v2 (5p, fixed cosmo) vs v2 (7p, free cosmo)',
               y=1.02, fontsize=14)
out1 = os.path.join(OUT, 'corner_overlay.png')
g.export(out1)
print(f'wrote {out1}')

# ---------------------------------------------------------------- GSMF data + model curves
# Load observation (same on both sides)
sys.path.insert(0, f'{V2}/codes')
from cosmo_hydro_emu.load_hacc import (
    load_gsmf_obs, mass_conds, sepia_data_format, seed_mass_scale, vkin_scale, eps_scale,
    fill_nan_with_interpolation, read_gsmf,
)
from cosmo_hydro_emu.emu import emulate, load_model_autosync
import pandas as pd

GSMF_OBS_DIR = '/home/nramachandra/Projects/Hydro_runs/HAvoCC/havocc/analysis/modules/galaxy_modules/GalStellarMassFunction/data/'
x_obs_raw, y_obs_raw, yerr_obs_raw = load_gsmf_obs(GSMF_OBS_DIR)
m_lo, m_hi = mass_conds('GSMF')
obs_mask = (x_obs_raw > m_lo) & (x_obs_raw < m_hi)
x_obs   = x_obs_raw[obs_mask]
y_obs   = 10 ** y_obs_raw[obs_mask]
yerr_obs = yerr_obs_raw[0, obs_mask]
print(f'GSMF obs: {len(x_obs)} points in [{m_lo:.1e}, {m_hi:.1e}]')

# Build the v2 emulator (autosync from saved pickle)
df = pd.read_csv(f'{V2}/data/FinalDesign.txt')
p_v2 = df.values.astype(float)[1:40]
for k, sc in {2: seed_mass_scale, 3: vkin_scale, 4: eps_scale}.items():
    p_v2[:, k] /= sc
DirIn_v2 = f'{V2}/data/400MPC_RUNS_5SG_2COSMO_PARAM/HAvoCC/'
x_grid_v2, y_arr_v2 = read_gsmf(DirIn_v2, 39, start_sim_idx=1)
g_mask = (x_grid_v2 > m_lo) & (x_grid_v2 < m_hi)
y_vals_v2 = y_arr_v2[:, g_mask]
y_ind_v2  = x_grid_v2[g_mask]

# Prefer the multi-z z=0 model if present (newer training), else single-z
v2_model_path = f'{V2}/models/GSMF_multiz/multivariate_model_z_index10'
if not os.path.exists(v2_model_path + '.pkl'):
    v2_model_path = f'{V2}/models/GSMF_multivariate_model_z_index0'
print(f'v2 emulator: {v2_model_path}')
with quiet():
    sd_v2 = sepia_data_format(p_v2, y_vals_v2, y_ind_v2)
    em_v2 = load_model_autosync(v2_model_path, sd_v2)
print(f'   pu = {em_v2.num.pu}')

# Build the v1 emulator (from the 128 MPC HACC-5p sims)
import importlib.util
spec = importlib.util.spec_from_file_location('hydro_emu_v1_load',
                                              f'{V1}/hydro_emu/load_hacc.py')
v1mod = importlib.util.module_from_spec(spec)
sys.modules['hydro_emu_v1_load'] = v1mod
try:
    with quiet():
        spec.loader.exec_module(v1mod)
    v1_loadable = True
except Exception as e:
    print(f'v1 load_hacc not importable directly ({e}); using inline loader')
    v1_loadable = False

DirIn_v1 = '/home/nramachandra/Projects/Hydro_runs/Data/ProfileData/SCIDAC_RUNS/128MPC_RUNS_HACC_5PARAM_extract2/'

# Inline param + GSMF reader for v1 (mirrors gp_HACC_mcmc_universal.ipynb)
pattern = re.compile(r'KAPPA_(\d+\.?\d*)_EGW_(\d+\.?\d*)_SEED_([\d\.eE\+\-]+)_VKIN_([\d\.]+)_EPS_([\d\.eE\+\-]+)')
params_v1 = []
gsmf_v1   = []
stellar_mass_v1 = None
for d in sorted(os.listdir(DirIn_v1)):
    m = pattern.match(d)
    if not m: continue
    p = [float(g) for g in m.groups()]
    fileIn = os.path.join(DirIn_v1, d, 'analysis_pipeline/extract/GalStellarMassFunction_624.txt')
    if not os.path.exists(fileIn):
        # try shallower / different layout
        candidates = glob.glob(os.path.join(DirIn_v1, d, '**/GalStellarMassFunction_624.txt'),
                               recursive=True)
        if not candidates: continue
        fileIn = candidates[0]
    arr = np.loadtxt(fileIn)
    if stellar_mass_v1 is None: stellar_mass_v1 = arr[:, 0]
    params_v1.append(p)
    gsmf_v1.append(arr[:, 1])
params_v1 = np.array(params_v1)
gsmf_v1   = np.array(gsmf_v1)
# Apply v1 scalings
params_v1_scaled = params_v1.copy()
params_v1_scaled[:, 2] /= seed_mass_scale
params_v1_scaled[:, 3] /= vkin_scale
params_v1_scaled[:, 4] /= eps_scale
print(f'v1 training set: {gsmf_v1.shape} sims, mass bins = {stellar_mass_v1.shape}')

# Mass cut + 10**
gsmf_v1_extra = fill_nan_with_interpolation(gsmf_v1, 'linear')
gm = (stellar_mass_v1 > m_lo) & (stellar_mass_v1 < m_hi)
y_vals_v1 = 10 ** gsmf_v1_extra[:, gm]
y_ind_v1  = stellar_mass_v1[gm]

with quiet():
    sd_v1 = sepia_data_format(params_v1_scaled, y_vals_v1, y_ind_v1)
    em_v1 = load_model_autosync(f'{V1}/model/GSMF_multivariate_model_z_index0', sd_v1)
print(f'v1 emulator loaded, pu = {em_v1.num.pu}')


def predict_v2(theta_sg, omega_m=0.1375, sigma_8=0.8):
    """v2 emulator: pad with cosmo fixed at design midpoint."""
    theta = np.append(theta_sg, [omega_m, sigma_8])
    with quiet():
        mean, _ = emulate(em_v2, theta)
    return np.interp(x_obs, y_ind_v2, mean[:, 0])

def predict_v1(theta_sg):
    with quiet():
        mean, _ = emulate(em_v1, theta_sg)
    return np.interp(x_obs, y_ind_v1, mean[:, 0])


# Posterior medians
med_v1   = np.median(v1_chain,   axis=0)
med_v25p = np.median(v2_5p,      axis=0)
med_v27p = np.median(v2_7p,      axis=0)
print('v1 median:    ', med_v1.round(3))
print('v2 5p median: ', med_v25p.round(3))
print('v2 7p median: ', med_v27p.round(3))

# Bands: sample 200 random posterior points each, compute model spread
rng = np.random.default_rng(0)
def model_band(chain, predictor, sg_slice=slice(0, 5), extra_for_predict=None, n=200):
    idx = rng.choice(chain.shape[0], size=n, replace=False)
    preds = np.zeros((n, x_obs.size))
    for k, i in enumerate(idx):
        sg = chain[i, sg_slice]
        if extra_for_predict is None:
            preds[k] = predictor(sg)
        else:
            extra = chain[i, extra_for_predict]
            # 7p chain: cols 5,6 are omega_m, sigma_8
            preds[k] = predictor(sg, omega_m=extra[0], sigma_8=extra[1])
    return preds.mean(axis=0), np.percentile(preds, 16, axis=0), np.percentile(preds, 84, axis=0)

print('Computing model bands (this may take ~1 min)…')
m_v1,  lo_v1,  hi_v1  = model_band(v1_chain, predict_v1)
m_v25, lo_v25, hi_v25 = model_band(v2_5p,    predict_v2)
m_v27, lo_v27, hi_v27 = model_band(v2_7p,    predict_v2,
                                   extra_for_predict=[5, 6])
print('  done')

# ---------------------------------------------------------------- plot 2: GSMF
fig, (ax_main, ax_resid) = plt.subplots(2, 1, figsize=(8.5, 6.8), sharex=True,
                                         gridspec_kw={'height_ratios': [3, 1],
                                                      'hspace': 0.05})

ax_main.errorbar(x_obs, y_obs, yerr=yerr_obs, fmt='o', color='k',
                 ms=4, capsize=2, label='Driver+ 2022', zorder=10)
for (med, lo, hi, c, lab) in [
    (m_v1,  lo_v1,  hi_v1,  'gray',     'v1 (Flamingo 128 MPC), median'),
    (m_v25, lo_v25, hi_v25, 'tab:blue', 'v2 5p (cosmo fixed), median'),
    (m_v27, lo_v27, hi_v27, 'tab:red',  'v2 7p (cosmo free), median'),
]:
    ax_main.plot(x_obs, med, color=c, lw=2, label=lab)
    ax_main.fill_between(x_obs, lo, hi, color=c, alpha=0.18)

ax_main.set_xscale('log'); ax_main.set_yscale('log')
ax_main.set_xlim(m_lo, m_hi)
ax_main.set_ylabel(r'$\mathrm{d}n / \mathrm{d}\log_{10} M_\star \; [1/(h^{-1}\mathrm{Mpc})^3]$',
                   fontsize=13)
ax_main.set_title('GSMF: observation + posterior-median model from each pipeline',
                  fontsize=13)
ax_main.legend(fontsize=10, loc='lower left')
ax_main.grid(True, ls=':', alpha=0.3)

# Residuals: (model - data) / data
for (med, c, lab) in [(m_v1, 'gray', 'v1'),
                      (m_v25, 'tab:blue', 'v2 5p'),
                      (m_v27, 'tab:red', 'v2 7p')]:
    ax_resid.plot(x_obs, (med - y_obs) / y_obs, color=c, lw=2, label=lab)
ax_resid.axhline(0, ls=':', color='k', lw=1)
ax_resid.set_xscale('log')
ax_resid.set_xlim(m_lo, m_hi)
ax_resid.set_ylim(-0.5, 0.5)
ax_resid.set_xlabel(r'$M_\star \; [M_\odot]$', fontsize=13)
ax_resid.set_ylabel(r'$\Delta / \mathrm{obs}$', fontsize=12)
ax_resid.grid(True, ls=':', alpha=0.3)
ax_resid.legend(fontsize=10, ncol=3, loc='lower left')

plt.tight_layout()
out2 = os.path.join(OUT, 'gsmf_compare.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f'wrote {out2}')

# ---------------------------------------------------------------- numerical summary
summary = os.path.join(OUT, 'posterior_medians.txt')
with open(summary, 'w') as f:
    f.write('Posterior medians (linear / scaled units) — GSMF-only fits\n\n')
    f.write(f'{"":18s}  {"v1":>8s} {"v2 5p":>8s} {"v2 7p":>8s}\n')
    for i, nm in enumerate(['kappa_w','e_w','M_seed/1e6','v_kin/1e4','eps_kin/1e1']):
        f.write(f'  {nm:18s} {med_v1[i]:8.4f} {med_v25p[i]:8.4f} {med_v27p[i]:8.4f}\n')
    f.write(f'  {"omega_m":18s} {"--":>8s} {"--":>8s} {med_v27p[5]:8.4f}\n')
    f.write(f'  {"sigma_8":18s} {"--":>8s} {"--":>8s} {med_v27p[6]:8.4f}\n')
print(f'wrote {summary}')
