__all__ = ['ln_prior', 'ln_like', 'ln_prob', 'chain_init', 'define_sampler',
           'do_mcmc', 'mcmc_results', 'log_likelihood']

import numpy as np
import emcee
import time
from cosmo_hydro_emu.emu import emulate, emu_redshift
from cosmo_hydro_emu.load_hacc import PARAM_NAME


def log_likelihood(theta,
                   x_grid,
                   sepia_model,
                   x, y, yerr,
                   fixed_params=None,
                   with_underestimation_bias=False,
                   case_label=None,
                   param_names=None,
                   redshift=None,
                   z_all=None):
    """
    Calculate log likelihood for a single observable.

    Parameters
    ----------
    theta : array-like
        Free parameter values (physical params, then optional bias params).
    x_grid : array
        x-values on which the emulator is defined.
    sepia_model : SepiaModel or list of SepiaModel
        Single trained GP emulator (z_index=0) or list of models across
        snapshots for redshift interpolation.
    x, y, yerr : arrays
        Observational data (x-values, y-values, y-errors).
    fixed_params : dict or None
        Dict mapping PARAM_NAME entries to fixed values.
    with_underestimation_bias : bool
        If True, last 3 entries of theta are [log_bstar, bCV, bHSE].
    case_label : str
        Observable label (e.g. 'GSMF', 'fGas', 'CGD').
    param_names : list or None
        Parameter name list to use. Defaults to PARAM_NAME.
    redshift : float or None
        Target redshift for this observable. If None or 0, uses the single
        z_index=0 model directly. Otherwise uses emu_redshift interpolation
        (requires sepia_model to be a list and z_all to be set).
    z_all : array or None
        Redshifts corresponding to each model in sepia_model list.
        Required when redshift is not None/0.
    """
    if fixed_params is None:
        fixed_params = {}
    if param_names is None:
        param_names = PARAM_NAME

    full_params = []
    theta_index = 0
    for param in param_names:
        if param in fixed_params:
            full_params.append(fixed_params[param])
        else:
            full_params.append(theta[theta_index])
            theta_index += 1

    log_bstar = 0.0
    bCV = 1.0
    bHSE = 1.0
    if with_underestimation_bias:
        log_bstar = theta[-3]
        bCV = theta[-2]
        bHSE = theta[-1]
        if bCV <= 0.0 or bHSE <= 0.0:
            return -np.inf

    full_params = np.array(full_params)

    # Choose emulation mode based on redshift
    if redshift is not None and redshift > 0 and z_all is not None:
        # Multi-z interpolation: sepia_model is a list of models
        params_with_z = np.append(full_params, [redshift])[np.newaxis, :]
        model_grid, model_var_grid = emu_redshift(params_with_z,
                                                  sepia_model, None, z_all)
    else:
        # Single-snapshot model (z_index=0)
        model_grid, model_var_grid = emulate(sepia_model, full_params)

    if with_underestimation_bias:
        if case_label == 'GSMF':
            x_eff = x * (10.0 ** log_bstar)
        elif case_label == 'fGas':
            x_eff = x / bHSE
        else:
            x_eff = x
    else:
        x_eff = x

    model = np.interp(x_eff, x_grid, model_grid[:, 0])

    if with_underestimation_bias and case_label == 'GSMF':
        y_adjusted = y * bCV
    else:
        y_adjusted = y

    sigma2 = yerr ** 2
    ll = -0.5 * np.sum((y_adjusted - model) ** 2 / sigma2)
    return ll


def ln_like(theta,
            x_grids,
            sepia_models,
            data,
            fixed_params=None,
            with_underestimation_bias=False,
            case_labels=None,
            param_names=None,
            redshifts=None,
            z_all_list=None):
    """Total log likelihood across all observables.

    Parameters
    ----------
    redshifts : list of float or None
        Per-observable target redshifts. Entry of None or 0 means z=0 model.
    z_all_list : list of (array or None)
        Per-observable snapshot redshifts for multi-z interpolation.
        Entry of None means z=0 single-model observable.
    """
    log_likelihoods = []
    labels = case_labels.split('_')

    for i in range(len(sepia_models)):
        z_i = redshifts[i] if redshifts is not None else None
        z_all_i = z_all_list[i] if z_all_list is not None else None
        ll = log_likelihood(theta,
                            x_grids[i],
                            sepia_models[i],
                            data[i]['x'],
                            data[i]['y'],
                            data[i]['yerr'],
                            fixed_params=fixed_params,
                            with_underestimation_bias=with_underestimation_bias,
                            case_label=labels[i],
                            param_names=param_names,
                            redshift=z_i,
                            z_all=z_all_i)
        log_likelihoods.append(ll)

    return sum(log_likelihoods)


def ln_prior(theta, params_list, flat_indices=None):
    """
    Mixed prior: Gaussian for most params, flat for specified indices.

    Parameters
    ----------
    theta : array-like
        Parameter values.
    params_list : list
        Each entry: [name, initial_value, lower_bound, upper_bound].
    flat_indices : list or None
        Indices that get flat priors. If None, all get Gaussian.
    """
    if flat_indices is None:
        flat_indices = []

    pdf_sum = 0
    for i, (p, param) in enumerate(zip(theta, params_list)):
        if not (param[2] < p < param[3]):
            return -np.inf
        if i in flat_indices:
            continue
        p_mu = 0.5 * (param[3] - param[2]) + param[2]
        p_sigma = 1 * (param[3] - p_mu)
        pdf_sum += (np.log(1.0 / (np.sqrt(2 * np.pi) * p_sigma))
                    - 0.5 * (p - p_mu) ** 2 / p_sigma ** 2)
    return pdf_sum


def ln_prob(theta,
            params_list,
            x_grids,
            sepia_models,
            data,
            fixed_params=None,
            with_underestimation_bias=False,
            case_labels=None,
            flat_indices=None,
            param_names=None,
            redshifts=None,
            z_all_list=None):
    """Log probability = log prior + log likelihood."""
    lp = ln_prior(theta, params_list, flat_indices=flat_indices)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta, x_grids, sepia_models, data,
                        fixed_params=fixed_params,
                        with_underestimation_bias=with_underestimation_bias,
                        case_labels=case_labels,
                        param_names=param_names,
                        redshifts=redshifts,
                        z_all_list=z_all_list)


def chain_init(params_list, ndim, nwalkers):
    """Initialize walker positions uniformly across parameter ranges."""
    pos0 = []
    for _ in range(nwalkers):
        walker_position = []
        for param in params_list:
            min_val, max_val = param[2], param[3]
            init_val = np.random.uniform(min_val, max_val)
            walker_position.append(init_val)
        pos0.append(walker_position)
    return np.array(pos0)


def define_sampler(ndim, nwalkers, params_list, x_grids, sepia_models, data,
                   fixed_params=None, with_underestimation_bias=False,
                   case_labels=None, flat_indices=None, param_names=None,
                   redshifts=None, z_all_list=None,
                   pool=None):
    """Define emcee EnsembleSampler with optional parallel pool."""
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, ln_prob,
        args=(params_list, x_grids, sepia_models, data, fixed_params,
              with_underestimation_bias, case_labels, flat_indices, param_names,
              redshifts, z_all_list),
        pool=pool)
    return sampler


def do_mcmc(sampler, pos, nrun, ndim, if_burn=False):
    """Run MCMC sampling (burn-in or production)."""
    time0 = time.time()
    pos, prob, state = sampler.run_mcmc(pos, nrun)
    print('time (minutes):', (time.time() - time0) / 60.)
    samples = sampler.chain.reshape((-1, ndim))
    if if_burn:
        print('Burn-in phase')
        sampler.reset()
    else:
        print('Sampling phase')
    return pos, prob, state, samples, sampler


def mcmc_results(samples):
    """Calculate median and 16/84 percentile uncertainties from MCMC samples."""
    results = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                       zip(*np.percentile(samples, [16, 50, 84], axis=0))))
    print('mcmc results:', ' '.join(str(result[0]) for result in results))
    return tuple(result[0] for result in results)
