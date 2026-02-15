__all__ = ['emulate', 'load_model_multiple', 'emu_redshift', 'blockPrint', 'enablePrint']

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData
from sepia.SepiaPredict import SepiaEmulatorPrediction
import numpy as np
import sys
import os
from cosmo_hydro_emu.pca import do_pca
from cosmo_hydro_emu.gp import gp_load
from cosmo_hydro_emu.load_hacc import sepia_data_format


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def emulate(sepia_model:SepiaModel=None, # Input data in SEPIA format
        input_params:np.array=None, #Input parameter array
       ) -> tuple: # 2 np.array of mean and (0.05,0.95) quantile in prediction

    if len(input_params.shape) == 1:
        ip = np.expand_dims(input_params, axis=0)

    else:
        ip = input_params

    pred_samples= sepia_model.get_samples(numsamples=100)

    pred = SepiaEmulatorPrediction(t_pred=ip, samples=pred_samples, model=sepia_model)

    pred_samps = pred.get_y()

    pred_mean = np.mean(pred_samps, axis=0).T
    pred_err = np.quantile(pred_samps, [0.05, 0.95], axis=0).T

    return pred_mean, pred_err


def load_model_multiple(model_dir:str=None, # Pickle directory path
                        p_train_all:np.array=None, # Parameter array
                        y_vals_all:np.array=None, # Target y-values array
                        y_ind_all:np.array=None, # x-values
                        z_index_range:np.array=None, # Snapshot indices for training
                   ) -> None:

    blockPrint()

    model_list = []
    data_list = []

    for z_index in z_index_range:

        sepia_data = sepia_data_format(p_train_all, y_vals_all[:, z_index, :], y_ind_all)

        sepia_model_pca_i = do_pca(sepia_data, exp_variance=0.999)

        model_filename = model_dir + 'multivariate_model_z_index' + str(z_index)
        sepia_model_z = gp_load(sepia_model_pca_i, model_filename)
        model_list.append(sepia_model_z)
        data_list.append(sepia_data)

    enablePrint()

    print('Number of models loaded: ' + str(len(model_list)) + ' from: ' + model_dir)

    return model_list, data_list


def emu_redshift(input_params_and_redshift:np.array=None, # Input parameters (along with redshift)
                 sepia_model_list:list=None,
                 sepia_data_list:list=None,
                 z_all:np.array=None): # All the trained models

    z = input_params_and_redshift[:, -1]
    input_params = input_params_and_redshift[:, :-1]

    # Linear interpolation between z1 < z < z2
    snap_idx_nearest = (np.abs(z_all - z)).argmin()
    if (z > z_all[snap_idx_nearest]):
        snap_ID_z1 = snap_idx_nearest - 1

    else:
        snap_ID_z1 = snap_idx_nearest
    snap_ID_z2 = snap_ID_z1 + 1

    z1 = z_all[snap_ID_z1]
    z2 = z_all[snap_ID_z2]

    sepia_model_z1 = sepia_model_list[snap_ID_z1]
    Bk_z1, Bk_z1_err = emulate(sepia_model_z1, input_params)

    sepia_model_z2 = sepia_model_list[snap_ID_z2]
    Bk_z2, Bk_z2_err = emulate(sepia_model_z2, input_params)

    Bk_interp = np.zeros_like(Bk_z1)
    Bk_interp = Bk_z2 + (Bk_z1 - Bk_z2)*(z - z2)/(z1 - z2)

    Bk_interp_err = np.zeros_like(Bk_z1_err)
    Bk_interp_err = Bk_z2_err + (Bk_z1_err - Bk_z2_err)*(z - z2)/(z1 - z2)

    return Bk_interp, Bk_interp_err
