#!/usr/bin/env python
"""
Plot and compare MCMC inference results from different trials.

Usage:
    # Single trial — triangle plot + best-fit overlays
    python plot_mcmc.py results/samples_GSMF_subgrid.npy

    # Compare multiple trials
    python plot_mcmc.py results/samples_GSMF_subgrid.npy results/samples_GSMF_CGD_subgrid.npy

    # Custom labels and output path
    python plot_mcmc.py results/samples_GSMF_subgrid.npy results/samples_GSMF_CGD_subgrid.npy \
        --labels "GSMF only" "GSMF+CGD" --output plots/comparison.png
"""

import sys
import os
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from getdist import plots, MCSamples
from PIL import Image
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codes'))

from cosmo_hydro_emu.pca import *
from cosmo_hydro_emu.load_hacc import (
    PARAM_NAME, PARAM_NAME_SG, PARAM_NAME_COSMO,
    seed_mass_scale, vkin_scale, eps_scale,
    sepia_data_format, mass_conds, plot_strings, fill_nan_with_interpolation,
    read_gsmf, read_gasfr, read_cgd, read_cged, read_hmf,
    load_gsmf_obs, load_fgas_obs, load_cgd_obs,
)
from cosmo_hydro_emu.emu import emulate
from cosmo_hydro_emu.gp import gp_load

from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# mcmc_results (copied from plot_mcmc_final_dec16.py)
# ---------------------------------------------------------------------------

def mcmc_results(samples, peak=1):
    """Extract mode (peak) values from MCMC samples jointly using GMM.

    Args:
        samples: array-like, shape (n_samples, n_params)
        peak: int, which peak to return (1 = most dominant, 2 = second, etc.)

    Returns:
        tuple of parameter modes
    """
    samples = np.array(samples, dtype=float)

    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm.fit(samples)
    labels = gmm.predict(samples)

    cluster_modes = []
    cluster_weights = []
    for k in range(gmm.n_components):
        cluster_samples = samples[labels == k]
        cluster_weights.append(len(cluster_samples))
        modes = []
        for i in range(samples.shape[1]):
            kde = gaussian_kde(cluster_samples[:, i])
            x_eval = np.linspace(cluster_samples[:, i].min(),
                                 cluster_samples[:, i].max(), 1000)
            density = kde(x_eval)
            modes.append(x_eval[np.argmax(density)])
        cluster_modes.append(tuple(modes))

    sorted_clusters = np.argsort(cluster_weights)[::-1]
    if peak > len(sorted_clusters):
        raise ValueError(f"Requested peak {peak} but only {len(sorted_clusters)} peaks found.")

    chosen_modes = cluster_modes[sorted_clusters[peak - 1]]
    print('mcmc results (modes):', ' '.join(f'{mode:.4f}' for mode in chosen_modes))
    return chosen_modes


def mcmc_results_percentile(samples):
    """Calculate median and 16/84 percentile uncertainties."""
    samples = np.array(samples, dtype=float)
    results = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                       zip(*np.percentile(samples, [16, 50, 84], axis=0))))
    print('mcmc results:', ' '.join(str(result[0]) for result in results))
    return tuple(result[0] for result in results)


# ---------------------------------------------------------------------------
# Data loading helpers (from plot_mcmc_final_dec16.py)
# ---------------------------------------------------------------------------

def load_design(design_file, start_sim_idx=1, num_sims=None):
    """Load parameter design matrix from CSV, apply scaling."""
    import pandas as pd
    df = pd.read_csv(design_file)
    params = df.values.astype(float)
    start_row = start_sim_idx - 1
    end_row = start_row + num_sims if num_sims else params.shape[0]
    params = params[start_row:end_row]
    params[:, 2] = params[:, 2] / seed_mass_scale
    params[:, 3] = params[:, 3] / vkin_scale
    params[:, 4] = params[:, 4] / eps_scale
    return params


def load_model(model_filename, p_all, y_vals, y_ind, exp_variance):
    sepia_data = sepia_data_format(p_all, y_vals, y_ind)
    sepia_model_i = do_pca(sepia_data, exp_variance=exp_variance)
    sepia_model = gp_load(sepia_model_i, model_filename)
    return sepia_model


def load_obs_data_for_plot(obs_dirs):
    """Load observational data for all available observables."""
    obs_data = {}

    if obs_dirs.get('gsmf'):
        mlim1, mlim2 = mass_conds('GSMF')
        x_raw, y_raw, yerr_raw = load_gsmf_obs(obs_dirs['gsmf'])
        m = (x_raw > mlim1) & (x_raw < mlim2)
        obs_data['gsmf'] = {'x': x_raw[m], 'y': 10**y_raw[m], 'yerr': yerr_raw[:, m]}

    try:
        mlim1, mlim2 = mass_conds('fGas')
        x1_raw, y1_raw, yerr1_raw = load_fgas_obs()
        m2 = (x1_raw > mlim1) & (x1_raw < mlim2)
        obs_data['fgas'] = {'x': x1_raw[m2], 'y': y1_raw[m2], 'yerr': yerr1_raw[m2]}
    except Exception:
        pass

    if obs_dirs.get('cgd'):
        data_cgd = load_cgd_obs(obs_dirs['cgd'])
        rlim1, rlim2 = mass_conds('CGD')
        x2_raw = data_cgd['mcdonald2017_avg'][0]
        y2_raw = data_cgd['mcdonald2017_avg'][1][:, 0]
        r = (x2_raw > rlim1) & (x2_raw < rlim2)
        obs_data['cgd'] = {'x': x2_raw[r], 'y': y2_raw[r], 'yerr': 0.05 * y2_raw[r]}

    return obs_data


def load_all_data(cfg):
    """Load all data and models needed for plotting."""
    data_cfg = cfg['data']
    dir_in = os.path.join(os.path.dirname(__file__), data_cfg['DirIn'])
    design_file = os.path.join(os.path.dirname(__file__), data_cfg['design_file'])
    model_dir = os.path.join(os.path.dirname(__file__), data_cfg['model_dir'])
    num_sims = data_cfg['num_sims']
    start_sim_idx = data_cfg.get('start_sim_idx', 1)
    exp_variance = data_cfg['exp_variance']
    z_index = data_cfg['z_index']

    params32 = load_design(design_file, start_sim_idx=start_sim_idx, num_sims=num_sims)

    # Load simulation data
    stellar_mass, gsmf_arr = read_gsmf(dir_in, num_sims, params32)
    log_halo_mass, gas_fr_arr = read_gasfr(dir_in, num_sims, params32)
    radius, cgd_arr = read_cgd(dir_in, num_sims, params32)

    gsmf_arr_extra = fill_nan_with_interpolation(gsmf_arr, 'linear')
    gas_fr_arr_extra = fill_nan_with_interpolation(gas_fr_arr, 'cubic')

    # Prepare datasets
    datasets = {}

    mlim1, mlim2 = mass_conds('GSMF')
    mass_cond = np.where((stellar_mass > mlim1) & (stellar_mass < mlim2))
    datasets['gsmf'] = {
        'y_vals': 10**gsmf_arr_extra[:, mass_cond[0]],
        'y_ind': stellar_mass[mass_cond]
    }

    mlim1, mlim2 = mass_conds('fGas')
    mass_cond2 = np.where((10**log_halo_mass > mlim1) & (10**log_halo_mass < mlim2))
    datasets['fgas'] = {
        'y_vals': gas_fr_arr_extra[:, mass_cond2[0]],
        'y_ind': 10**log_halo_mass[mass_cond2]
    }

    rlim1, rlim2 = mass_conds('CGD')
    rad_cond = np.where((radius > rlim1) & (radius < rlim2))
    datasets['cgd'] = {
        'y_vals': cgd_arr[:, rad_cond[0]],
        'y_ind': radius[rad_cond]
    }

    # Load models
    models = {}
    for name in ['GSMF', 'fGas', 'CGD']:
        key = name.lower()
        model_file = os.path.join(model_dir,
                                  f'{name}_multivariate_model_z_index{z_index}')
        models[key] = load_model(model_file, params32,
                                 datasets[key]['y_vals'],
                                 datasets[key]['y_ind'],
                                 exp_variance)

    obs_data = load_obs_data_for_plot(cfg.get('obs_dirs', {}))

    return {
        'params32': params32,
        'stellar_mass': stellar_mass,
        'gsmf_arr': gsmf_arr,
        'log_halo_mass': log_halo_mass,
        'gas_fr_arr': gas_fr_arr,
        'radius': radius,
        'cgd_arr': cgd_arr,
        'models': models,
        'datasets': datasets,
        'obs_data': obs_data,
    }


# ---------------------------------------------------------------------------
# Plotting functions (from plot_mcmc_final_dec16.py)
# ---------------------------------------------------------------------------

def configure_axes(axes, obs_list, data_dict):
    """Configure axis properties based on which observables are plotted."""
    obs_key_map = {'GSMF': 'gsmf', 'fGas': 'fgas', 'CGD': 'cgd',
                   'CGED': 'cged', 'HMF': 'hmf', 'Pk': 'pk'}

    for i, obs in enumerate(obs_list):
        plt_title, x_label, y_label = plot_strings(obs)
        mlim1, mlim2 = mass_conds(obs)
        axes[i].set_xlim(mlim1, mlim2)
        axes[i].set_xscale('log')
        axes[i].set_xlabel(x_label, fontsize=14)
        axes[i].set_ylabel(y_label, fontsize=14)

        if obs in ('CGD', 'CGED', 'HMF'):
            axes[i].set_yscale('log')


def plot_mcmc_bestfit(p_mcmc_list, chains_labels, fig, axes,
                      fixed_params, data_dict, obs_list, param_names):
    """Plot emulator predictions at MCMC best-fit, overlaid on obs data."""
    models = data_dict['models']
    datasets = data_dict['datasets']
    obs_data = data_dict['obs_data']

    line_styles = ['-', '--', '-.', ':']
    if len(chains_labels) == 2:
        colors = ["#E03424", "#006FED", "#009966", "#000866"]
    elif len(chains_labels) == 1:
        colors = ["#006FED", "#009966", "#000866"]
    else:
        colors = ["gray", "#E03424", "#006FED", "#009966", "#000866"]

    obs_key_map = {'GSMF': 'gsmf', 'fGas': 'fgas', 'CGD': 'cgd'}

    for chain_idx, p_mcmc in enumerate(p_mcmc_list):
        theta_params = []
        p_mcmc_array = np.array(p_mcmc)
        theta_index = 0
        for pname in param_names:
            if pname in fixed_params:
                theta_params.append(fixed_params[pname])
            else:
                theta_params.append(p_mcmc_array[theta_index])
                theta_index += 1
        theta_params = np.array(theta_params)

        linestyle = line_styles[chain_idx % len(line_styles)]
        color = colors[chain_idx % len(colors)]
        label = f'MCMC best fit: {chains_labels[chain_idx]}'

        for i, obs in enumerate(obs_list):
            key = obs_key_map.get(obs)
            if key and key in models:
                model_grid, model_var_grid = emulate(models[key], theta_params)

                if obs == 'GSMF':
                    axes[i].plot(datasets[key]['y_ind'], np.log10(model_grid),
                                label=label, lw=3, linestyle=linestyle, color=color)
                    axes[i].fill_between(datasets[key]['y_ind'],
                                         np.log10(model_var_grid[:, 0, 0]),
                                         np.log10(model_var_grid[:, 0, 1]),
                                         alpha=0.2, color=color)
                else:
                    axes[i].plot(datasets[key]['y_ind'], model_grid,
                                label=label, lw=3, linestyle=linestyle, color=color)
                    axes[i].fill_between(datasets[key]['y_ind'],
                                         model_var_grid[:, 0, 0],
                                         model_var_grid[:, 0, 1],
                                         alpha=0.2, color=color)

    # Plot observational data
    for i, obs in enumerate(obs_list):
        key = obs_key_map.get(obs, obs.lower())
        if key in obs_data:
            if obs == 'GSMF':
                axes[i].errorbar(obs_data[key]['x'], np.log10(obs_data[key]['y']),
                                 yerr=obs_data[key]['yerr'], fmt=".k", capsize=2, zorder=3)
            else:
                axes[i].errorbar(obs_data[key]['x'], obs_data[key]['y'],
                                 yerr=obs_data[key]['yerr'], fmt=".k", capsize=2, zorder=3)

    configure_axes(axes, obs_list, data_dict)
    for ax in axes:
        ax.legend(fontsize='small')

    return fig, axes


def add_parameter_textboxes(ax_triangle, p_mcmc_list, chains_labels, params_range_list):
    """Add parameter value textboxes to the triangle plot."""
    if len(chains_labels) == 2:
        colors = ["#E03424", "#006FED", "#009966", "#000866"]
    elif len(chains_labels) == 1:
        colors = ["#006FED", "#009966", "#000866"]
    else:
        colors = ["gray", "#E03424", "#006FED", "#009966", "#000866"]

    param_display_names = list(params_range_list.keys())
    n_chains = len(p_mcmc_list)
    y_positions = np.linspace(0.9, 0.9 - 0.12 * n_chains, n_chains)
    x_positions = np.array([0.65] * n_chains)
    if n_chains != 2:
        x_positions = np.array([0.475, 0.7, 0.7])[:n_chains]
        y_positions = np.array([0.88, 0.88, 0.65])[:n_chains]

    for i, (p_mcmc, label) in enumerate(zip(p_mcmc_list, chains_labels)):
        text_color = colors[i % len(colors)]
        props = dict(boxstyle='round4', facecolor='white', alpha=0.8, edgecolor=text_color)
        chain_text = f"{label}:\n"
        for name, value in zip(param_display_names, p_mcmc):
            chain_text += f"{name}: {value:.4f}\n"
        ax_triangle.text(x_positions[i], y_positions[i], chain_text,
                         transform=ax_triangle.transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=props, color=text_color, weight='bold')


# ---------------------------------------------------------------------------
# combined_plot (from plot_mcmc_final_dec16.py)
# ---------------------------------------------------------------------------

def combined_plot(chains_samples, chains_labels, params_list, p_mcmc_list,
                  save_path, fixed_params, data_dict, obs_list, param_names):
    """Create combined triangle plot + best-fit overlay for multiple chains."""
    param_display_names = [param[0] for param in params_list]

    # Build param_limits from params_list ranges
    params_range_list = {}
    for param in params_list:
        params_range_list[param[0]] = (float(param[2]), float(param[3]))

    samples_all = [
        MCSamples(samples=s,
                  names=param_display_names,
                  label=l,
                  settings={'mult_bias_correction_order': 0.5,
                            'smooth_scale_2D': 4,
                            'smooth_scale_1D': 4})
        for s, l in zip(chains_samples, chains_labels)
    ]

    g = plots.get_subplot_plotter(subplot_size=2)
    g.settings.axes_fontsize = 14
    g.settings.axes_labelsize = 14
    g.settings.legend_fontsize = 14
    g.settings.fontsize = 14
    g.settings.alpha_filled_add = 0.9
    g.settings.solid_contour_palefactor = 0.5
    g.settings.num_plot_contours = 3

    g.triangle_plot(samples_all,
                    param_display_names,
                    filled=True,
                    legend_labels=chains_labels,
                    param_limits=params_range_list)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        triangle_plot_path = tmpfile.name
        g.export(triangle_plot_path)

    triangle_image = Image.open(triangle_plot_path)

    n_obs_panels = min(len(obs_list), 3)
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.1)

    ax_triangle = fig.add_subplot(gs[0, 0])
    ax_triangle.imshow(triangle_image)
    ax_triangle.axis('off')

    gs_subplots = gs[0, 1].subgridspec(n_obs_panels, 1, hspace=0.5)
    axs = [fig.add_subplot(gs_subplots[i]) for i in range(n_obs_panels)]

    plot_mcmc_bestfit(p_mcmc_list, chains_labels, fig, axs,
                      fixed_params, data_dict, obs_list[:n_obs_panels], param_names)

    add_parameter_textboxes(ax_triangle, p_mcmc_list, chains_labels, params_range_list)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved as: {save_path}")

    os.unlink(triangle_plot_path)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot MCMC inference results')
    parser.add_argument('samples', nargs='+',
                        help='Path(s) to samples .npy files')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each chain (default: trial name from config)')
    parser.add_argument('--output', default=None,
                        help='Output plot path (default: results/plot_<trial>.png)')
    parser.add_argument('--peak', type=int, default=1,
                        help='Which GMM peak to report (1=dominant)')
    args = parser.parse_args()

    # Load configs for each trial
    all_samples = []
    all_labels = []
    all_params_list = []
    cfg_ref = None

    for i, samples_path in enumerate(args.samples):
        samples = np.load(samples_path)
        all_samples.append(samples)

        trial_name = os.path.basename(samples_path).replace('samples_', '').replace('.npy', '')

        # Try loading the config copy
        config_path = samples_path.replace('samples_', 'config_').replace('.npy', '.yaml')
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            if cfg_ref is None:
                cfg_ref = cfg
        else:
            cfg = cfg_ref

        if args.labels and i < len(args.labels):
            all_labels.append(args.labels[i])
        else:
            all_labels.append(trial_name)

        # Load params_list
        params_file = samples_path.replace('samples_', 'params_list_')
        if os.path.exists(params_file):
            params_list = np.load(params_file, allow_pickle=True).tolist()
            all_params_list.append(params_list)

    if cfg_ref is None:
        print("ERROR: No config file found. Place config_<trial>.yaml alongside samples.")
        sys.exit(1)

    # Use the first trial's params_list as reference
    params_list = all_params_list[0] if all_params_list else []
    param_names = list(PARAM_NAME)

    # Determine fixed_params from config
    from run_mcmc import build_param_space, SHORT_KEY_TO_LABEL
    _data_cfg_ref = cfg_ref['data']
    _, fixed_params, param_names_used, _ = build_param_space(cfg_ref,
        load_design(os.path.join(os.path.dirname(__file__), _data_cfg_ref['design_file']),
                    start_sim_idx=_data_cfg_ref.get('start_sim_idx', 1),
                    num_sims=_data_cfg_ref['num_sims']))

    # Determine observables to plot (use union of all configs, limited to 3 with available models)
    obs_list = cfg_ref.get('observables', ['GSMF', 'fGas', 'CGD'])
    plot_obs = [o for o in ['GSMF', 'fGas', 'CGD'] if True][:3]

    print(f"Loading data for plotting...")
    data_dict = load_all_data(cfg_ref)

    # Compute best-fit values
    p_mcmc_list = []
    for samples in all_samples:
        # Only use the columns matching the first params_list
        ncols = min(samples.shape[1], len(params_list))
        p_mcmc = mcmc_results(samples[:, :ncols], peak=args.peak)
        p_mcmc_list.append(p_mcmc)

    # Determine output path
    if args.output:
        save_path = args.output
    else:
        names = '_vs_'.join(all_labels)
        save_path = os.path.join(os.path.dirname(__file__), 'results',
                                 f'plot_{names}.png')

    print(f"Creating combined plot...")
    combined_plot(
        chains_samples=[s[:, :len(params_list)] for s in all_samples],
        chains_labels=all_labels,
        params_list=params_list,
        p_mcmc_list=p_mcmc_list,
        save_path=save_path,
        fixed_params=fixed_params,
        data_dict=data_dict,
        obs_list=plot_obs,
        param_names=param_names_used,
    )

    print("Done!")


if __name__ == '__main__':
    main()
