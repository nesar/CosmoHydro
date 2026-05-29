__all__ = ['do_pca']

from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData


def do_pca(sepia_data: SepiaData = None,    # input data in SEPIA format
           exp_variance: float = 0.99,      # explained-variance threshold for PCA basis
           do_discrepancy: bool = False,    # add a discrepancy basis
           ) -> SepiaModel:
    sepia_data.transform_xt()
    sepia_data.standardize_y()
    sepia_data.create_K_basis(n_pc=exp_variance)
    if do_discrepancy:
        sepia_data.create_D_basis()
    return SepiaModel(sepia_data)
