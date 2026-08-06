"""Combined v1 (Flamingo) ↔ v2 (CosmoHydro) comparison plots.

Outputs (in this directory):
  corner_overlay.png         Triangle plot, v1 5p vs v2 5p vs v2 7p on 5 shared subgrid params.
  summary_stats_compare.png  3-panel observation + posterior-median model + 16/84% band,
                             one panel each for GSMF / fGas / CGD, three chains overlaid.
  posterior_medians.txt      Tabular medians for all parameters.

No inference-pipeline code is modified — this script only reads existing artefacts.
"""
import os, sys, contextlib, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import plots as gd_plots
from getdist import MCSamples

V1  = '/home/nramachandra/Projects/Hydro_runs/Flamingo/Clean'
V2  = '/home/nramachandra/Projects/Hydro_runs/CosmoHydro'
OUT = os.path.join(V2, 'Inference/diagnostics')

sys.path.insert(0, f'{V2}/codes')
from cosmo_hydro_emu.load_hacc import (
    load_gsmf_obs, load_fgas_obs, load_cgd_obs,
    mass_conds, sepia_data_format, fill_nan_with_interpolation,
    seed_mass_scale, vkin_scale, eps_scale,
    read_gsmf, read_gasfr_all_snaps, read_cgd_all_snaps,
)
from cosmo_hydro_emu.emu import emulate, load_model_autosync
from cosmo_hydro_emu.snapshot_utils import SNAPSHOT_IDS
import pandas as pd


@contextlib.contextmanager
def quiet():
    s = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w'); yield
    finally:
        sys.stdout.close(); sys.stdout = s


# ============================================================ chains
v1_chain = np.load(f'{V1}/plots/universal/npy/samples_HACC_fixed_None_obs_GSMF.npy')
v2_5p    = np.load(f'{V2}/Inference/results/samples_GSMF_5p_fid_cosmo.npy')
v2_7p    = np.load(f'{V2}/Inference/results/samples_GSMF_7p_planck.npy')  # Planck-prior 7p (subgrid marginals are prior-independent)
print(f'v1 GSMF chain: {v1_chain.shape}')
print(f'v2 5p chain:   {v2_5p.shape}')
print(f'v2 7p chain:   {v2_7p.shape}')

# ============================================================ corner overlay
labels_sg = [r'\kappa_\mathrm{w}', r'e_\mathrm{w}',
             r'M_\mathrm{seed}/10^{6}', r'v_\mathrm{kin}/10^{4}',
             r'\epsilon_\mathrm{kin}/10^{1}']
names_sg  = ['kappa_w', 'e_w', 'M_seed_e6', 'v_kin_e4', 'eps_kin_e1']
PLOT_RANGES = {'kappa_w': (2.0, 4.0), 'e_w': (0.2, 1.0),
               'M_seed_e6': (0.6, 2.0), 'v_kin_e4': (0.1, 1.2),
               'eps_kin_e1': (0.02, 1.2)}

def _samples(chain, label):
    return MCSamples(samples=chain[:, :5], names=names_sg, labels=labels_sg,
                     label=label, ranges=PLOT_RANGES)

s_v1   = _samples(v1_chain, r'v1 (Flamingo 128 MPC, $\mathcal{L}_{\mathrm{GSMF}}$ only)')
s_v25p = _samples(v2_5p,    r'v2 5p, cosmo fixed ($\mathcal{L}_{\mathrm{GSMF}}$ only)')
s_v27p = _samples(v2_7p,    r'v2 7p, cosmo free ($\mathcal{L}_{\mathrm{GSMF}}$ only)')

g = gd_plots.get_subplot_plotter(width_inch=11)
g.settings.alpha_filled_add = 0.55
g.settings.legend_fontsize  = 13
g.settings.axes_fontsize    = 12
g.settings.lab_fontsize     = 14
g.triangle_plot([s_v1, s_v25p, s_v27p], names_sg, filled=True,
                contour_colors=['gray', 'tab:blue', 'tab:red'],
                line_args=[{'color': 'gray',     'ls': '-', 'lw': 1.5},
                           {'color': 'tab:blue', 'ls': '-', 'lw': 1.5},
                           {'color': 'tab:red',  'ls': '-', 'lw': 1.5}],
                legend_loc='upper right')
g.fig.suptitle('GSMF-only posteriors: v1 vs v2 (5p, fixed cosmo) vs v2 (7p, free cosmo)',
               y=1.02, fontsize=14)
out_corner = os.path.join(OUT, 'corner_overlay.png')
g.export(out_corner)
print(f'wrote {out_corner}')


# ============================================================ emulator helpers
def load_v1_raw(file_glob, col_y, drop_first=False):
    """Read v1 128MPC HACC-5p sim tree. Returns (params_scaled, raw_ys, x_grid).
    NaN handling and any log/linear conversion is left to the caller.
    """
    pattern_5p = re.compile(
        r'KAPPA_(\d+\.?\d*)_EGW_(\d+\.?\d*)_SEED_([\d\.eE\+\-]+)_VKIN_([\d\.]+)_EPS_([\d\.eE\+\-]+)')
    DirIn_v1 = '/home/nramachandra/Projects/Hydro_runs/Data/ProfileData/SCIDAC_RUNS/128MPC_RUNS_HACC_5PARAM_extract2/'
    params, ys = [], []
    x_grid = None
    for d in sorted(os.listdir(DirIn_v1)):
        m = pattern_5p.match(d)
        if not m: continue
        p = [float(x) for x in m.groups()]
        cand = glob.glob(os.path.join(DirIn_v1, d, '**', file_glob), recursive=True)
        if not cand: continue
        arr = np.loadtxt(cand[0])
        if x_grid is None:
            x_grid = arr[:, 0]
            if drop_first: x_grid = x_grid[1:]
        params.append(p)
        col = arr[:, col_y]
        ys.append(col[1:] if drop_first else col)
    params = np.array(params); ys = np.array(ys)
    params[:, 2] /= seed_mass_scale
    params[:, 3] /= vkin_scale
    params[:, 4] /= eps_scale
    return params, ys, x_grid


# ============================================================ load obs + data per stat

GSMF_OBS_DIR = '/home/nramachandra/Projects/Hydro_runs/HAvoCC/havocc/analysis/modules/galaxy_modules/GalStellarMassFunction/data/'
CGD_OBS_DIR  = '/home/nramachandra/Projects/Hydro_runs/HAvoCC/havocc/analysis/modules/halo_profile_modules/ClusterGasDensityProfile/data/'

# --- v2 design matrix (corrected slicing, 5p subset is cols 0..4) ---
df = pd.read_csv(f'{V2}/data/FinalDesign.txt')
p_v2 = df.values.astype(float)[1:40]
for k, sc in {2: seed_mass_scale, 3: vkin_scale, 4: eps_scale}.items():
    p_v2[:, k] /= sc
DirIn_v2 = f'{V2}/data/400MPC_RUNS_5SG_2COSMO_PARAM/HAvoCC/'

# --- v1 obs loaders/mass cuts mirror new pipeline ---
def obs_gsmf():
    x_raw, y_raw, yerr_raw = load_gsmf_obs(GSMF_OBS_DIR)
    m_lo, m_hi = mass_conds('GSMF')
    m = (x_raw > m_lo) & (x_raw < m_hi)
    return x_raw[m], 10 ** y_raw[m], yerr_raw[0, m], (m_lo, m_hi)

def obs_fgas():
    x_raw, y_raw, yerr_raw = load_fgas_obs()
    m_lo, m_hi = mass_conds('fGas')
    m = (x_raw > m_lo) & (x_raw < m_hi)
    return x_raw[m], y_raw[m], yerr_raw[m], (m_lo, m_hi)

def obs_cgd():
    data = load_cgd_obs(CGD_OBS_DIR)
    x_raw = data['mcdonald2017_avg'][0]
    y_raw = data['mcdonald2017_avg'][1][:, 0]
    r_lo, r_hi = mass_conds('CGD')
    m = (x_raw > r_lo) & (x_raw < r_hi)
    y = y_raw[m]
    return x_raw[m], y, 0.05 * y, (r_lo, r_hi)


# ============================================================ build per-stat emulators
def _build(short_name, v1_kw, v2_x_grid, v2_ys_per_sim, v2_mask, x_units_v1,
           y_post_v1, kind_nan, v1_model, v2_multiz_dir, v2_single_file):
    """Generic per-stat constructor.

    v1_kw          : kwargs for load_v1_raw
    v2_x_grid, v2_ys_per_sim, v2_mask : pre-NaN-interped v2 arrays + mask
    x_units_v1     : callable applied to v1 x-grid (e.g. lambda x: 10**x for log axis)
    y_post_v1      : callable applied to interpolated v1 ys (e.g. lambda y: 10**y for GSMF)
    kind_nan       : 'linear' or 'cubic' for fill_nan_with_interpolation
    """
    # v1 side
    p_v1, raw_ys, x_v1_raw = load_v1_raw(**v1_kw)
    ys_v1 = fill_nan_with_interpolation(raw_ys, kind_nan)
    ys_v1 = y_post_v1(ys_v1)
    x_v1  = x_units_v1(x_v1_raw)
    # apply same mass cut used at v2 training
    if short_name == 'GSMF':
        lo, hi = mass_conds('GSMF')
        m = (x_v1 > lo) & (x_v1 < hi)
    elif short_name == 'fGas':
        lo, hi = mass_conds('fGas')
        m = (x_v1 > lo) & (x_v1 < hi)
    else:  # CGD
        lo, hi = mass_conds('CGD')
        m = (x_v1 > lo) & (x_v1 < hi)
    ys_v1 = ys_v1[:, m]; x_v1 = x_v1[m]
    # drop any column that still has NaN/inf (interp may fail at extreme edges)
    good = np.isfinite(ys_v1).all(axis=0)
    if not good.all():
        print(f'   {short_name}: dropping {(~good).sum()} bad columns from v1')
    ys_v1 = ys_v1[:, good]; x_v1 = x_v1[good]
    print(f'   v1 {short_name} training set: {ys_v1.shape}, x_grid: {x_v1.shape}')
    with quiet():
        sd_v1 = sepia_data_format(p_v1, ys_v1, x_v1)
        em_v1 = load_model_autosync(f'{V1}/model/{v1_model}', sd_v1)
    print(f'   v1 {short_name} emulator: {v1_model}  pu={em_v1.num.pu}')

    # v2 side
    y_v2 = v2_ys_per_sim[:, v2_mask]
    good2 = np.isfinite(y_v2).all(axis=0)
    y_v2 = y_v2[:, good2]; x_v2 = v2_x_grid[v2_mask][good2]
    path = os.path.join(V2, 'models', v2_multiz_dir, 'multivariate_model_z_index10')
    if not os.path.exists(path + '.pkl'):
        path = os.path.join(V2, 'models', v2_single_file.rsplit('.', 1)[0])
    with quiet():
        sd_v2 = sepia_data_format(p_v2, y_v2, x_v2)
        em_v2 = load_model_autosync(path, sd_v2)
    print(f'   v2 {short_name} emulator: {path}  pu={em_v2.num.pu}')
    return em_v1, em_v2, x_v1, x_v2


# ---- GSMF
print('\nbuilding GSMF emulators…')
x_g, y_arr_g = read_gsmf(DirIn_v2, 39, start_sim_idx=1)
gm_lo, gm_hi = mass_conds('GSMF')
gmask = (x_g > gm_lo) & (x_g < gm_hi)
# nb #02 cell: gsmf_arr is log10 → 10** for v2 y_vals
y_vals_v2_g = 10 ** fill_nan_with_interpolation(y_arr_g, 'linear')
em_v1_g, em_v2_g, x_v1_g, x_v2_g = _build(
    'GSMF',
    v1_kw=dict(file_glob='GalStellarMassFunction_624.txt', col_y=1),
    v2_x_grid=x_g, v2_ys_per_sim=y_vals_v2_g, v2_mask=gmask,
    x_units_v1=lambda x: x,            # v1 stellar_mass already linear
    y_post_v1=lambda y: 10 ** y,       # v1 col1 is log10 GSMF
    kind_nan='linear',
    v1_model='GSMF_multivariate_model_z_index0',
    v2_multiz_dir='GSMF_multiz',
    v2_single_file='GSMF_multivariate_model_z_index0.pkl')

# ---- fGas (use snapshot 624 / index 10 = z=0)
print('\nbuilding fGas emulators…')
mh_log, fgas_arr_all = read_gasfr_all_snaps(DirIn_v2, 39, SNAPSHOT_IDS, start_sim_idx=1)
fm_lo, fm_hi = mass_conds('fGas')
fmask = (10 ** mh_log > fm_lo) & (10 ** mh_log < fm_hi)
for s in range(fgas_arr_all.shape[1]):
    fgas_arr_all[:, s, :] = fill_nan_with_interpolation(fgas_arr_all[:, s, :], 'cubic')
v2_x_fgas      = 10 ** mh_log
v2_ys_fgas_z0  = fgas_arr_all[:, -1, :]   # z=0 slice, all bins
em_v1_f, em_v2_f, x_v1_f, x_v2_f = _build(
    'fGas',
    v1_kw=dict(file_glob='Mgas_M500_Ratio_624.txt', col_y=5),
    v2_x_grid=v2_x_fgas, v2_ys_per_sim=v2_ys_fgas_z0, v2_mask=fmask,
    x_units_v1=lambda x: 10 ** x,      # v1 col0 is log10 halo mass
    y_post_v1=lambda y: y,             # v1 col5 is already linear M_gas/M_500c
    kind_nan='cubic',
    v1_model='fGas_multivariate_model_z_index0',
    v2_multiz_dir='fGas_multiz',
    v2_single_file='fGas_multivariate_model_z_index0.pkl')

# ---- CGD profile (z=0)
print('\nbuilding CGD emulators…')
rad, cgd_arr_all = read_cgd_all_snaps(DirIn_v2, 39, SNAPSHOT_IDS, start_sim_idx=1)
cm_lo, cm_hi = mass_conds('CGD')
cmask = (rad > cm_lo) & (rad < cm_hi)
for s in range(cgd_arr_all.shape[1]):
    cgd_arr_all[:, s, :] = fill_nan_with_interpolation(cgd_arr_all[:, s, :], 'linear')
v2_ys_cgd_z0 = cgd_arr_all[:, -1, :]
em_v1_c, em_v2_c, x_v1_c, x_v2_c = _build(
    'CGD',
    v1_kw=dict(file_glob='ClusterGasDensityProfile_624.txt', col_y=1, drop_first=True),
    v2_x_grid=rad, v2_ys_per_sim=v2_ys_cgd_z0, v2_mask=cmask,
    x_units_v1=lambda x: 10 ** x,      # v1 col0 is log10 radius
    y_post_v1=lambda y: y,             # v1 col1 already linear rho/rho_crit
    kind_nan='linear',
    v1_model='CGD_multivariate_model_z_index0',
    v2_multiz_dir='CGD_multiz',
    v2_single_file='CGD_multivariate_model_z_index0.pkl')


# ============================================================ predictors + bands
# NOTE on GSMF units:
# The training/likelihood code both apply an extra 10** to the *already-linear*
# col-1 values of GalStellarMassFunction_624.txt (and to load_gsmf_obs's _phi).
# This cancels in the likelihood — sim and obs are transformed identically —
# but the emulator output is "10**(linear GSMF)". For a *physically labelled*
# plot we undo it with log10() and use _phi directly for the obs.
GSMF_UNDO_DOUBLE_10 = True

def _emu_at(em, theta):
    with quiet():
        mean, _ = emulate(em, theta)
    return mean[:, 0]

def predict_v2_pad(theta_sg, em, omega_m=0.14176, sigma_8=0.8102, undo_log=False):
    y = _emu_at(em, np.append(theta_sg, [omega_m, sigma_8]))
    return np.log10(y) if undo_log else y

def predict_v1(theta_sg, em, undo_log=False):
    y = _emu_at(em, theta_sg)
    return np.log10(y) if undo_log else y


rng = np.random.default_rng(0)
def model_band(chain, predict_fn, x_grid, n=200, pad_cosmo=False):
    """Sample n posterior draws, return (median, 16%, 84%) on x_grid."""
    idx = rng.choice(chain.shape[0], size=n, replace=False)
    preds = np.zeros((n, x_grid.size))
    for k, i in enumerate(idx):
        sg = chain[i, :5]
        if pad_cosmo and chain.shape[1] >= 7:
            preds[k] = predict_fn(sg, omega_m=chain[i, 5], sigma_8=chain[i, 6])
        else:
            preds[k] = predict_fn(sg)
    return (np.median(preds, axis=0),
            np.percentile(preds, 16, axis=0),
            np.percentile(preds, 84, axis=0))


# ============================================================ figure: 3 panels
fig, axes = plt.subplots(3, 1, figsize=(8.5, 12))

def panel(ax, x_obs, y_obs, yerr_obs, model_bands, title, ylabel, xlabel,
          ylog=True, xlog=True):
    """model_bands = list of (label, color, x_grid, (median, lo, hi)).
    Each pipeline is plotted on its own native x_grid (full training range)."""
    ax.errorbar(x_obs, y_obs, yerr=yerr_obs, fmt='o', color='k',
                ms=4, capsize=2, label='Observation', zorder=10)
    for lab, c, xg, (med, lo, hi) in model_bands:
        ax.plot(xg, med, color=c, lw=2, label=lab)
        ax.fill_between(xg, lo, hi, color=c, alpha=0.18)
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, ls=':', alpha=0.3)

# ----- GSMF (undo double-10** to plot true physical units) -----
x_obs_g, y_obs_g_raw, yerr_obs_g_raw, _ = obs_gsmf()
# obs_gsmf returns 10**_phi (run_mcmc convention). _phi is already linear, so
# take log10 to get back to physical and use the symmetric upper error.
if GSMF_UNDO_DOUBLE_10:
    y_obs_g = np.log10(y_obs_g_raw)              # _phi (linear GSMF)
    yerr_obs_g = yerr_obs_g_raw                  # same vector — error is small at this scale
else:
    y_obs_g, yerr_obs_g = y_obs_g_raw, yerr_obs_g_raw
bands_g = [
    ('v1 (Flamingo 128 MPC)', 'gray', x_v1_g,
     model_band(v1_chain, lambda th: predict_v1(th, em_v1_g, undo_log=GSMF_UNDO_DOUBLE_10), x_v1_g)),
    ('v2 5p (cosmo fixed)',   'tab:blue', x_v2_g,
     model_band(v2_5p, lambda th: predict_v2_pad(th, em_v2_g, undo_log=GSMF_UNDO_DOUBLE_10), x_v2_g)),
    ('v2 7p (cosmo free)',    'tab:red', x_v2_g,
     model_band(v2_7p,
                lambda th, omega_m, sigma_8: predict_v2_pad(th, em_v2_g, omega_m, sigma_8,
                                                            undo_log=GSMF_UNDO_DOUBLE_10),
                x_v2_g, pad_cosmo=True)),
]
panel(axes[0], x_obs_g, y_obs_g, yerr_obs_g, bands_g,
      'GSMF: emulator over full training range vs. observation',
      r'$\Phi \; [1/(h^{-1}\mathrm{Mpc})^3]$',
      r'$M_\star \; [M_\odot]$', ylog=False, xlog=True)
# GSMF: y is linear log10 GSMF (i.e. ~-5 to -1), plot on linear y to show range
axes[0].set_ylabel(r'$\log_{10}\,\Phi \; [1/(h^{-1}\mathrm{Mpc})^3]$', fontsize=12)

# ----- fGas (already in physical units) -----
x_obs_f, y_obs_f, yerr_obs_f, _ = obs_fgas()
bands_f = [
    ('v1 (Flamingo 128 MPC)', 'gray', x_v1_f,
     model_band(v1_chain, lambda th: predict_v1(th, em_v1_f), x_v1_f)),
    ('v2 5p (cosmo fixed)',   'tab:blue', x_v2_f,
     model_band(v2_5p, lambda th: predict_v2_pad(th, em_v2_f), x_v2_f)),
    ('v2 7p (cosmo free)',    'tab:red', x_v2_f,
     model_band(v2_7p,
                lambda th, omega_m, sigma_8: predict_v2_pad(th, em_v2_f, omega_m, sigma_8),
                x_v2_f, pad_cosmo=True)),
]
panel(axes[1], x_obs_f, y_obs_f, yerr_obs_f, bands_f,
      r'$f_{\rm gas}$: emulator over full training range vs. observation',
      r'$M_{\rm gas}/M_{500c}$', r'$M_{500c}\; [h^{-1} M_\odot]$',
      ylog=False, xlog=True)

# ----- CGD (already in physical units) -----
x_obs_c, y_obs_c, yerr_obs_c, _ = obs_cgd()
bands_c = [
    ('v1 (Flamingo 128 MPC)', 'gray', x_v1_c,
     model_band(v1_chain, lambda th: predict_v1(th, em_v1_c), x_v1_c)),
    ('v2 5p (cosmo fixed)',   'tab:blue', x_v2_c,
     model_band(v2_5p, lambda th: predict_v2_pad(th, em_v2_c), x_v2_c)),
    ('v2 7p (cosmo free)',    'tab:red', x_v2_c,
     model_band(v2_7p,
                lambda th, omega_m, sigma_8: predict_v2_pad(th, em_v2_c, omega_m, sigma_8),
                x_v2_c, pad_cosmo=True)),
]
panel(axes[2], x_obs_c, y_obs_c, yerr_obs_c, bands_c,
      'CGD profile: emulator over full training range vs. observation',
      r'$\rho_{\rm gas}/\rho_{\rm crit}$', r'$r/R_{500c}$', ylog=True, xlog=True)

plt.tight_layout()
out_stats = os.path.join(OUT, 'summary_stats_compare.png')
plt.savefig(out_stats, dpi=150, bbox_inches='tight')
plt.close()
print(f'wrote {out_stats}')


# ============================================================ numerical summary
med_v1   = np.median(v1_chain,   axis=0)
med_v25p = np.median(v2_5p,      axis=0)
med_v27p = np.median(v2_7p,      axis=0)
summary = os.path.join(OUT, 'posterior_medians.txt')
with open(summary, 'w') as f:
    f.write('Posterior medians (linear / scaled units) — GSMF-only fits\n\n')
    f.write(f'{"":18s}  {"v1":>8s} {"v2 5p":>8s} {"v2 7p":>8s}\n')
    for i, nm in enumerate(['kappa_w','e_w','M_seed/1e6','v_kin/1e4','eps_kin/1e1']):
        f.write(f'  {nm:18s} {med_v1[i]:8.4f} {med_v25p[i]:8.4f} {med_v27p[i]:8.4f}\n')
    f.write(f'  {"omega_m":18s} {"--":>8s} {"--":>8s} {med_v27p[5]:8.4f}\n')
    f.write(f'  {"sigma_8":18s} {"--":>8s} {"--":>8s} {med_v27p[6]:8.4f}\n')
print(f'wrote {summary}')
