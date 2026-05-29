__all__ = ['do_gp_train', 'gp_load', 'do_gp_train_multiple']

import numpy as np
from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData

from cosmo_hydro_emu.pca import do_pca
from cosmo_hydro_emu.load_hacc import sepia_data_format


def do_gp_train(sepia_model: SepiaModel = None,   # SEPIA model after PCA
                model_file: str = None,           # pickle file path
                ) -> SepiaModel:
    sepia_model.tune_step_sizes(50, 20, update_vals=True)
    sepia_model.do_mcmc(1000)
    sepia_model.save_model_info(model_file)
    return sepia_model


def gp_load(sepia_model: SepiaModel,              # SEPIA data container
            model_file: str,                      # pickle file path (no extension)
            ) -> SepiaModel:
    try:
        sepia_model.restore_model_info(model_file)
        return sepia_model
    except FileNotFoundError as e:
        print(e)


def do_gp_train_multiple(model_dir: str = None,           # pickle directory path
                         p_train_all: np.ndarray = None,  # parameter array
                         y_vals_all: np.ndarray = None,   # target y-values
                         y_ind_all: np.ndarray = None,    # x-values
                         z_index_range: np.ndarray = None,  # snapshot indices
                         exp_variance: float = 0.999,     # PCA explained-variance threshold
                         ) -> None:

    for z_index in z_index_range:
        sepia_data = sepia_data_format(p_train_all, y_vals_all[:, z_index, :], y_ind_all)
        model_filename = model_dir + 'multivariate_model_z_index' + str(z_index)

        sepia_model = do_pca(sepia_data, exp_variance=exp_variance)
        sepia_model = do_gp_train(sepia_model, model_filename)
        print('Training complete for snapshot ' + str(z_index))
        print('Model saved at ' + model_dir)
        print(30 * '=*')

    return None
