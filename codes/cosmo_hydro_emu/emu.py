__all__ = ['emulate', 'emulate_ensemble', 'load_model_multiple', 'emu_redshift',
           'blockPrint', 'enablePrint', 'pu_from_saved_model', 'load_model_autosync']

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia.SepiaPredict import SepiaEmulatorPrediction
import numpy as np
import pickle
import sys
import os
from cosmo_hydro_emu.pca import do_pca
from cosmo_hydro_emu.gp import gp_load
from cosmo_hydro_emu.load_hacc import sepia_data_format


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def pu_from_saved_model(model_filename):
    """Inspect a saved SEPIA model pickle and return the number of PCA basis
    components (`pu`) used at training. Returns None if the file is missing or
    doesn't expose betaU samples (in which case the caller should fall back to
    an explicit exp_variance).
    """
    path = model_filename if model_filename.endswith('.pkl') else model_filename + '.pkl'
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as fh:
            blob = pickle.load(fh)
    except Exception:
        return None
    samples = blob.get('samples') if isinstance(blob, dict) else None
    betaU = samples.get('betaU') if isinstance(samples, dict) else None
    if betaU is None or getattr(betaU, 'ndim', 0) < 3:
        return None
    return int(betaU.shape[2])


def load_model_autosync(model_filename, sepia_data, exp_variance=0.95):
    """Load a trained SEPIA model with the basis size auto-synced to the saved
    pickle. Uses the pickle's `samples['betaU'].shape[2]` as the integer `n_pc`
    so the reconstructed K basis always matches what the MCMC samples expect.
    Falls back to `exp_variance` (float) if the basis count can't be detected.
    """
    pu = pu_from_saved_model(model_filename)
    n_pc = pu if pu is not None else exp_variance
    blockPrint()
    sepia_model = do_pca(sepia_data, exp_variance=n_pc)
    sepia_model = gp_load(sepia_model, model_filename)
    enablePrint()
    return sepia_model


def emulate_ensemble(sepia_model: SepiaModel = None,
                     input_params: np.array = None,
                     ) -> tuple:  # (mean, 5/95 quantile band) from posterior samples
    """Posterior-sample-based predictor (legacy).

    Returns (pred_mean, pred_err) where pred_err is the [0.05, 0.95] quantile
    band with shape (p, n_inputs, 2). Uses 100 posterior samples of the
    hyperparameters.
    """
    if len(input_params.shape) == 1:
        ip = np.expand_dims(input_params, axis=0)
    else:
        ip = input_params

    pred_samples = sepia_model.get_samples(numsamples=100)
    pred = SepiaEmulatorPrediction(t_pred=ip, samples=pred_samples, model=sepia_model)
    pred_samps = pred.get_y()

    pred_mean = np.mean(pred_samps, axis=0).T
    pred_err = np.quantile(pred_samps, [0.05, 0.95], axis=0).T

    return pred_mean, pred_err


def emulate(sepia_model: SepiaModel = None,
            input_params: np.array = None,
            sepia_data: SepiaData = None,
            ) -> tuple:  # (mean, std), both shape (p, n_inputs)
    """Analytical GP predictor: returns mean and std on the original y-scale.

    Uses the latent GP posterior (mu, Sigma) and projects through the K basis:
        y_mu  = K^T mu
        y_std = sqrt(diag(K^T Sigma K))
    then undoes the (orig_y_mean, orig_y_sd) standardization. If sepia_data is
    not supplied it is taken from sepia_model.data so this is a drop-in
    replacement signature-wise versus the legacy emulate_ensemble.
    """
    if input_params.ndim == 1:
        input_params = np.expand_dims(input_params, 0)

    if sepia_data is None:
        sepia_data = sepia_model.data

    K = sepia_data.sim_data.K
    y_sd = sepia_data.sim_data.orig_y_sd
    y_mean = sepia_data.sim_data.orig_y_mean

    pred_samples = sepia_model.get_samples(numsamples=1)

    K_T = K.T  # sepia stores K as (pu, p); y = K^T x in latent->output

    means, stds = [], []
    for param in input_params:
        pred = SepiaEmulatorPrediction(t_pred=param[None, :], samples=pred_samples,
                                       model=sepia_model, storeMuSigma=True)
        mu = pred.mu[0]
        Sigma = pred.sigma[0]

        y_mu = K_T @ mu
        y_cov = K_T @ Sigma @ K
        y_std = np.sqrt(np.clip(np.diag(y_cov), 0, None))

        y_mu = y_sd * y_mu + y_mean
        y_std = y_sd * y_std

        means.append(y_mu)
        stds.append(y_std)

    return np.stack(means, axis=1), np.stack(stds, axis=1)


def load_model_multiple(model_dir:str=None, # Pickle directory path
                        p_train_all:np.array=None, # Parameter array
                        y_vals_all:np.array=None, # Target y-values array
                        y_ind_all:np.array=None, # x-values
                        z_index_range:np.array=None, # Snapshot indices for training
                        exp_variance:float=0.95, # Fallback if pickle has no betaU info
                   ) -> None:

    model_list = []
    data_list = []

    for z_index in z_index_range:

        sepia_data = sepia_data_format(p_train_all, y_vals_all[:, z_index, :], y_ind_all)

        model_filename = model_dir + 'multivariate_model_z_index' + str(z_index)
        sepia_model_z = load_model_autosync(model_filename, sepia_data,
                                            exp_variance=exp_variance)
        model_list.append(sepia_model_z)
        data_list.append(sepia_data)

    print('Number of models loaded: ' + str(len(model_list)) + ' from: ' + model_dir)

    return model_list, data_list


def emu_redshift(input_params_and_redshift: np.array = None,
                 sepia_model_list: list = None,
                 sepia_data_list: list = None,
                 z_all: np.array = None):
    """Linearly interpolate the emulator across z snapshots.

    Returns (mean, std) on the original y-scale (matches new emulate()).
    sepia_data_list is optional; when None, sepia_data is taken from each
    model's .data attribute.
    """
    z = input_params_and_redshift[:, -1]
    input_params = input_params_and_redshift[:, :-1]

    snap_idx_nearest = (np.abs(z_all - z)).argmin()
    if z > z_all[snap_idx_nearest]:
        snap_ID_z1 = snap_idx_nearest - 1
    else:
        snap_ID_z1 = snap_idx_nearest
    snap_ID_z2 = snap_ID_z1 + 1

    z1 = z_all[snap_ID_z1]
    z2 = z_all[snap_ID_z2]

    sd1 = sepia_data_list[snap_ID_z1] if sepia_data_list is not None else None
    sd2 = sepia_data_list[snap_ID_z2] if sepia_data_list is not None else None

    Bk_z1, Bk_z1_err = emulate(sepia_model_list[snap_ID_z1], input_params,
                               sepia_data=sd1)
    Bk_z2, Bk_z2_err = emulate(sepia_model_list[snap_ID_z2], input_params,
                               sepia_data=sd2)

    Bk_interp = Bk_z2 + (Bk_z1 - Bk_z2) * (z - z2) / (z1 - z2)
    Bk_interp_err = Bk_z2_err + (Bk_z1_err - Bk_z2_err) * (z - z2) / (z1 - z2)

    return Bk_interp, Bk_interp_err
