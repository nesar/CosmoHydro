__all__ = ['PARAM_NAME',
           'load_params', 'sepia_data_format', 'read_gsmf', 'read_cged', 'read_bhmsm', 'load_cged_obs',
           'load_cgd_obs', 'read_gal_ssfr', 'read_gasfr', 'read_csfr', 'read_cgd', 'read_pk', 'plot_strings', 'mass_conds',
           'load_gsmf_obs', 'load_fgas_obs', 'load_fgas_other_sims', 'load_bhmsm_other_sims',
           'fill_nan_with_interpolation', 'eps_scale', 'seed_mass_scale', 'vkin_scale',
           'load_delta_cgd', 'delta_cgd', 'DirIn_128_256',
           'read_gsmf_all_snaps', 'read_gasfr_all_snaps', 'read_cgd_all_snaps', 'read_cged_all_snaps']


import numpy as np
import glob
import os
import pkg_resources
from sepia.SepiaData import SepiaData
from scipy.interpolate import interp1d



DATA_DIR = "data/"
# LIBRARY_ZK_FILE = pkg_resources.resource_stream("CubicGalileonEmu", DATA_DIR + "z_k.txt").name
# LIBRARY_BK_FILE = pkg_resources.resource_stream("CubicGalileonEmu", DATA_DIR + "Boost.npy").name
# LIBRARY_PARAM_FILE = pkg_resources.resource_stream("CubicGalileonEmu", DATA_DIR + "cosmo_newdesign.txt").name


seed_mass_scale = 1e6
vkin_scale = 1e4
eps_scale = 1e1

# PARAM_NAME = ['KAPPA', 'EGW', 'SEED' + '/' + str( int(seed_mass_scale)), 'VKIN' + '/' + str( int(vkin_scale)), 'EPS' + '/' + str( int(eps_scale))] ## HACC-5p


def load_delta_cgd(DirIn, EPS_val = 2.4):
    Vkin_val = 6300 if EPS_val == 2.4 else 3500
    fileIn128 = DirIn + 'Default_128MPC_NEW_KIN_JET_E_'+str(EPS_val)+'_V_'+str(Vkin_val)+'_S_8e5_SS_ZINI_0.25/analysis_pipeline/extract/ClusterGasDensityProfile_624.txt'
    fileIn256 = DirIn + 'Default_256MPC_NEW_KIN_JET_E_'+str(EPS_val)+'_V_'+str(Vkin_val)+'_S_8e5_SS_ZINI_0.25/analysis_pipeline/extract/ClusterGasDensityProfile_624.txt'
    cgd_128 = np.loadtxt(fileIn128, skiprows=0)
    cgd_256 = np.loadtxt(fileIn256, skiprows=0)
    cgd_128_d = cgd_128[1:, 1] 
    cdg_128_r = 10**cgd_128[1:, 0]
    cgd_256_d = cgd_256[1:, 1] 
    cdg_256_r = 10**cgd_256[1:, 0]

    delta_cgd_add = cgd_256_d - cgd_128_d

    return delta_cgd_add

DirIn_128_256 = '/home/nramachandra/Projects/Hydro_runs/Data/ProfileData/SCIDAC_RUNS/128_256/'
delta_cgd = load_delta_cgd(DirIn = DirIn_128_256, EPS_val = 2.4)


# PARAM_NAME = [
#     r'$\kappa$', 
#     r'$E_{GW}$', 
#     r'$M_{seed}/10^{' + str(int(np.log10(seed_mass_scale))) + '}$', 
#     r'$V_{kin}/10^{' + str(int(np.log10(vkin_scale))) + '}$', 
#     r'$E_{PS}/10^{' + str(int(np.log10(eps_scale))) + '}$'
# ]


# PARAM_NAME = [
#     r'$\kappa_w$', 
#     r'$e_w$', 
#     r'$m_\text{BH,seed}/10^{' + str(int(np.log10(seed_mass_scale))) + '}$', 
#     r'$v_{kin}/10^{' + str(int(np.log10(vkin_scale))) + '}$', 
#     r'$\epsilon/10^{' + str(int(np.log10(eps_scale))) + '}$'
# ]


PARAM_NAME =  [r'$\kappa_\text{w}$', 
               r'$e_\text{w}$', 
               r'$M_\text{seed}/10^{' + str(int(np.log10(seed_mass_scale))) + '}$', 
               r'$v_\text{kin}/10^{' + str(int(np.log10(vkin_scale))) + '}$', 
               r'$\epsilon_\text{kin}/10^{' + str(int(np.log10(eps_scale))) + '}$'
               ]


def load_params(p_fileIn:str, # Input file for parameters
               ) -> np.array: # Parameters
    p_all = np.loadtxt(p_fileIn)
    return p_all



def sepia_data_format(design:np.array=None, # Params array of shape (num_simulation, num_params)
                     y_vals:np.array=None, # Shape (num_simulation, num_y_values)
                     y_ind:np.array=None # Shape (num_y_values,)
                     ) -> SepiaData: #Sepia data format
    sepia_data = SepiaData(t_sim=design, y_sim=y_vals, y_ind_sim=y_ind)
    return sepia_data


######################################################


def read_csfr(DirIn, num_sims, params=None, start_sim_idx=1):
    """Note: CSFR files not available in 400MPC_RUNS_5SG_2COSMO_PARAM dataset."""

    csfr_arr = None

    for file_indx in range(num_sims):
        sim_idx = file_indx + start_sim_idx
        fileIn = os.path.join(DirIn, f'RUN{sim_idx:03d}', 'extract', 'CSFR.txt')
        csfr_all = np.loadtxt(fileIn)

        if csfr_arr is None:
            csfr_arr = np.zeros(shape=(num_sims, csfr_all.shape[0]))

        csfr_arr[file_indx, :] = csfr_all[:, 1]

    scale_factor = csfr_all[:, 0]

    return scale_factor, csfr_arr



def read_gsmf(DirIn, num_sims, params=None, start_sim_idx=1):

    gsmf_arr = None

    for file_indx in range(num_sims):
        sim_idx = file_indx + start_sim_idx
        fileIn = os.path.join(DirIn, f'RUN{sim_idx:03d}', 'extract', 'GalStellarMassFunction_624.txt')
        gsmf_all = np.loadtxt(fileIn)

        if gsmf_arr is None:
            gsmf_arr = np.zeros(shape=(num_sims, gsmf_all.shape[0]))

        gsmf_arr[file_indx, :] = gsmf_all[:, 1]

    stellar_mass = gsmf_all[:, 0]

    return stellar_mass, gsmf_arr


def read_pk(DirIn, num_sims, params=None, start_sim_idx=1):
    """Note: Power spectrum files not available in 400MPC_RUNS_5SG_2COSMO_PARAM dataset."""

    pk_arr = None

    for file_indx in range(num_sims):
        sim_idx = file_indx + start_sim_idx
        fileIn = os.path.join(DirIn, f'RUN{sim_idx:03d}', 'extract', 'm000p.pk.624')
        f_all = np.loadtxt(fileIn)

        if pk_arr is None:
            pk_arr = np.zeros(shape=(num_sims, f_all.shape[0]))

        pk_arr[file_indx, :] = f_all[:, 1]

    k = f_all[:, 0]

    fileIn_go = os.path.join(DirIn, 'GRAV_ONLY', 'power_spectrum', 'm000p.pk.624')
    pk_go = np.loadtxt(fileIn_go)[:, 1]

    pk_ratio = pk_arr / pk_go

    return k, pk_arr, pk_ratio


def read_bhmsm(DirIn, num_sims, params=None, start_sim_idx=1):
    """Note: BHMass_StellarMass files not available in 400MPC_RUNS_5SG_2COSMO_PARAM dataset."""

    bhmsm_arr = None

    for file_indx in range(num_sims):
        sim_idx = file_indx + start_sim_idx
        fileIn = os.path.join(DirIn, f'RUN{sim_idx:03d}', 'extract', 'BHMass_StellarMass_624.txt')
        f_all = np.loadtxt(fileIn)

        if bhmsm_arr is None:
            bhmsm_arr = np.zeros(shape=(num_sims, f_all.shape[0]))

        bhmsm_arr[file_indx, :] = f_all[:, 3]

    log_bhmsm_mass = f_all[:, 0]

    return log_bhmsm_mass, bhmsm_arr


def read_gal_ssfr(DirIn, num_sims, params=None, start_sim_idx=1):
    """Note: GalaxySSFR files not available in 400MPC_RUNS_5SG_2COSMO_PARAM dataset."""

    gal_ssfr_arr = None

    for file_indx in range(num_sims):
        sim_idx = file_indx + start_sim_idx
        fileIn = os.path.join(DirIn, f'RUN{sim_idx:03d}', 'extract', 'GalaxySSFR_624.txt')
        f_all = np.loadtxt(fileIn)

        if gal_ssfr_arr is None:
            gal_ssfr_arr = np.zeros(shape=(num_sims, f_all.shape[0]))

        gal_ssfr_arr[file_indx, :] = f_all[:, 1]

    log_ssfr_mass = f_all[:, 0]

    return log_ssfr_mass, gal_ssfr_arr



def read_gasfr(DirIn, num_sims, params=None, start_sim_idx=1):

    gas_fr_mean = None

    for file_indx in range(num_sims):
        sim_idx = file_indx + start_sim_idx
        fileIn = os.path.join(DirIn, f'RUN{sim_idx:03d}', 'extract', 'Mgas_M500_Ratio_624.txt')
        f_all = np.loadtxt(fileIn)

        if gas_fr_mean is None:
            gas_fr_mean = np.zeros(shape=(num_sims, f_all.shape[0]))

        gas_fr_mean[file_indx, :] = f_all[:, 5]  ## 3 for mean, 5 for median

    log_halo_mass = f_all[:, 0]

    return log_halo_mass, gas_fr_mean


def read_cged(DirIn, num_sims, params=None, start_sim_idx=1):

    cged_arr = None

    for file_indx in range(num_sims):
        sim_idx = file_indx + start_sim_idx
        fileIn = os.path.join(DirIn, f'RUN{sim_idx:03d}', 'extract', 'ClusterGasElectronDensityProfile_624.txt')
        cged_all = np.loadtxt(fileIn)

        if cged_arr is None:
            cged_arr = np.zeros(shape=(num_sims, cged_all.shape[0]))

        cged_arr[file_indx, :] = cged_all[:, 1]

    log_radius = cged_all[:, 0]

    return 10**log_radius[1:], cged_arr[:, 1:]


def read_cgd(DirIn, num_sims, params=None, start_sim_idx=1):

    cgd_arr = None

    for file_indx in range(num_sims):
        sim_idx = file_indx + start_sim_idx
        fileIn = os.path.join(DirIn, f'RUN{sim_idx:03d}', 'extract', 'ClusterGasDensityProfile_624.txt')
        cgd_all = np.loadtxt(fileIn)

        if cgd_arr is None:
            cgd_arr = np.zeros(shape=(num_sims, cgd_all.shape[0]))

        cgd_arr[file_indx, :] = cgd_all[:, 1]

    log_radius = cgd_all[:, 0]

    return 10**log_radius[1:], cgd_arr[:, 1:]


######################################################
## Multi-snapshot read functions
######################################################


def read_gsmf_all_snaps(DirIn, num_sims, snapshot_ids, start_sim_idx=1):
    """Read GSMF for all simulations and all snapshots.
    Returns: stellar_mass (1D), gsmf_arr (num_sims, num_snaps, num_bins)
    """
    num_snaps = len(snapshot_ids)
    gsmf_arr = None

    for file_indx in range(num_sims):
        sim_idx = file_indx + start_sim_idx
        for snap_idx, snap_id in enumerate(snapshot_ids):
            fileIn = os.path.join(DirIn, f'RUN{sim_idx:03d}', 'extract',
                                  f'GalStellarMassFunction_{snap_id}.txt')
            data = np.loadtxt(fileIn)

            if gsmf_arr is None:
                n_bins = data.shape[0]
                gsmf_arr = np.zeros(shape=(num_sims, num_snaps, n_bins))

            gsmf_arr[file_indx, snap_idx, :] = data[:, 1]

    stellar_mass = data[:, 0]
    return stellar_mass, gsmf_arr


def read_gasfr_all_snaps(DirIn, num_sims, snapshot_ids, start_sim_idx=1):
    """Read gas fraction for all simulations and all snapshots.
    Returns: log_halo_mass (1D), gas_fr_arr (num_sims, num_snaps, num_bins)
    """
    num_snaps = len(snapshot_ids)
    gas_fr_arr = None

    for file_indx in range(num_sims):
        sim_idx = file_indx + start_sim_idx
        for snap_idx, snap_id in enumerate(snapshot_ids):
            fileIn = os.path.join(DirIn, f'RUN{sim_idx:03d}', 'extract',
                                  f'Mgas_M500_Ratio_{snap_id}.txt')
            data = np.loadtxt(fileIn)

            if gas_fr_arr is None:
                n_bins = data.shape[0]
                gas_fr_arr = np.zeros(shape=(num_sims, num_snaps, n_bins))

            gas_fr_arr[file_indx, snap_idx, :] = data[:, 5]  ## 3 for mean, 5 for median

    log_halo_mass = data[:, 0]
    return log_halo_mass, gas_fr_arr


def read_cgd_all_snaps(DirIn, num_sims, snapshot_ids, start_sim_idx=1):
    """Read cluster gas density profiles for all simulations and all snapshots.
    Returns: radius (1D, first bin removed), cgd_arr (num_sims, num_snaps, num_bins-1)
    """
    num_snaps = len(snapshot_ids)
    cgd_arr = None

    for file_indx in range(num_sims):
        sim_idx = file_indx + start_sim_idx
        for snap_idx, snap_id in enumerate(snapshot_ids):
            fileIn = os.path.join(DirIn, f'RUN{sim_idx:03d}', 'extract',
                                  f'ClusterGasDensityProfile_{snap_id}.txt')
            data = np.loadtxt(fileIn)

            if cgd_arr is None:
                n_bins = data.shape[0]
                cgd_arr = np.zeros(shape=(num_sims, num_snaps, n_bins))

            cgd_arr[file_indx, snap_idx, :] = data[:, 1]

    log_radius = data[:, 0]
    return 10**log_radius[1:], cgd_arr[:, :, 1:]


def read_cged_all_snaps(DirIn, num_sims, snapshot_ids, start_sim_idx=1):
    """Read cluster gas electron density profiles for all simulations and all snapshots.
    Returns: radius (1D, first bin removed), cged_arr (num_sims, num_snaps, num_bins-1)
    """
    num_snaps = len(snapshot_ids)
    cged_arr = None

    for file_indx in range(num_sims):
        sim_idx = file_indx + start_sim_idx
        for snap_idx, snap_id in enumerate(snapshot_ids):
            fileIn = os.path.join(DirIn, f'RUN{sim_idx:03d}', 'extract',
                                  f'ClusterGasElectronDensityProfile_{snap_id}.txt')
            data = np.loadtxt(fileIn)

            if cged_arr is None:
                n_bins = data.shape[0]
                cged_arr = np.zeros(shape=(num_sims, num_snaps, n_bins))

            cged_arr[file_indx, snap_idx, :] = data[:, 1]

    log_radius = data[:, 0]
    return 10**log_radius[1:], cged_arr[:, :, 1:]


######################################################


def mass_conds(summary_stat):
        
    if (summary_stat == 'Common'):
        
        ## Flamingo limits
        mlim1 = 10**9
        mlim2 = 2*10**12 
        
    elif (summary_stat == 'GSMF'):

        mlim1 = 5*10**9
        mlim2 = 3*10**11 
        
    if (summary_stat == 'BHMSM'): 
          
        mlim1 = 10**10
        mlim2 = 2*10**12 
        
    if (summary_stat == 'gSSFR'): 
 
        mlim1 = 10**9
        mlim2 = 10**13
        
    if (summary_stat == 'fGas'): 

        ## Flamingo limits
        mlim1 = 10**(13.5)
        mlim2 = 10**(14.3)

    # mass_cond = np.where( (target_xvals > mlim1)  &  (target_xvals < mlim2) ) 
    
    if (summary_stat == 'CGED'): 

        ## Actually radius for CGED
        mlim1 =  0.025
        mlim2 = 1.2
   
    if (summary_stat == 'CGD'): 

        ## Actually radius for CGED
        mlim1 =  0.015
        mlim2 = 2.75


    if (summary_stat == 'Pk'): 

        ## Based on 

        # k_min = 2*np.pi/side_length
        # delta_x = side_length/Npart
        # k_max = np.pi/delta_x #Nyquist

        mlim1 =  0.04908738521234052
        mlim2 = 12.566370614359172
    
    if (summary_stat == 'CSFR'): 

        mlim1 =  0.0
        mlim2 = 1.0
    
    return mlim1, mlim2


'''
def plot_strings(summary_stat):
     
    if(summary_stat == 'GSMF'):

        plt_title = 'Galaxy stellar mass function'
        x_label = r"$\log_{10} \left[ M_\text{stars} / \text{M}_\odot  \right]$"
        y_label = r"$\text{d}n \, / \, \text{d}\log_{10} M_\text{stars}  \left[1 / (h^{-1}\text{Mpc})^3  \right]$"

        
    elif(summary_stat == 'BHMSM'): 
        
        plt_title = 'Black hole mass-stellar mass'
        x_label = r"$M_*$ [$M_\odot$]"
        y_label = r"$M_\text{BH}$ [$M_\odot$]"   
        
    elif(summary_stat == 'gSSFR'): 

        plt_title = 'Galaxy specific star formation rate'
        x_label = r'$M_\star\,/\,M_\odot$'
        y_label = r'$sSFR$'
                
    elif(summary_stat == 'fGas'): 

        plt_title = 'Cluster gas fraction'
        y_label = r"$M_{\text{500c}} / h^{-1}{\text{M}}_\odot \quad [<R_{\text{500c}}]$"
        x_label = r"$M_{\text{500c}} / h^{-1}{\text{M}}_\odot$"
        
    elif(summary_stat == 'CGED'): 

        plt_title = 'Cluster gas electron density'
        y_label = r'$n_\text{e}$'
        x_label = r'$r$'

    elif(summary_stat == 'CGD'): 

        plt_title = 'Cluster gas density'
        y_label = r"$\rho_\text{gas} \,/\, \rho_\text{crit}$"
        x_label = r"$r/R_{500c}$"

    elif(summary_stat == 'Pk'): 

        plt_title = 'Total power spectra ratio'
        y_label = r'$P_\text{sub}(k)\,/\,P_\text{grav}(k)$'
        x_label = r'$k [h\/Mpc]$'

    elif(summary_stat == 'CSFR'): 

        plt_title = 'Total power spectra ratio'
        y_label = r"{\text{cosmic SFR}} \[${\text{M}}_\odot / {\text{yr}} / (h^{-1}{\text{Mpc}})^3$\]"
        x_label = r"$a$"
        
    else: 
        print('Not implemented')
        
    
    return plt_title, x_label, y_label

'''

def plot_strings(summary_stat):
    
    if summary_stat == 'GSMF':
        plt_title = 'Galaxy stellar mass function'
        x_label = r"$\log_{10} \left[ M_{\mathrm{stars}} / \mathrm{M}_{\odot}  \right]$"
        y_label = r"$\mathrm{d}n \, / \, \mathrm{d}\log_{10} M_{\mathrm{stars}}  \left[1 / (h^{-1}\mathrm{Mpc})^3  \right]$"
        
    elif summary_stat == 'BHMSM':
        plt_title = 'Black hole mass-stellar mass'
        x_label = r"$M_{\ast}$ [$\mathrm{M}_{\odot}$]"
        y_label = r"$M_{\mathrm{BH}}$ [$\mathrm{M}_{\odot}$]"   
        
    elif summary_stat == 'gSSFR':
        plt_title = 'Galaxy specific star formation rate'
        x_label = r'$M_{\ast}\,/\,\mathrm{M}_{\odot}$'
        y_label = r'$\mathrm{sSFR}$'
                
    elif summary_stat == 'fGas':
        plt_title = 'Cluster gas fraction'
        x_label = r"$M_{\mathrm{500c}} / h^{-1}\mathrm{M}_{\odot}$"
        y_label = r"$M_{\mathrm{gas}} / M_{\mathrm{500c}} \quad [<R_{\mathrm{500c}}]$"
        
    elif summary_stat == 'CGED':
        plt_title = 'Cluster gas electron density'
        x_label = r'$r / R_{\mathrm{500c}}$'
        y_label = r'$n_{\mathrm{e}}$ [$\mathrm{cm}^{-3}$]'

    elif summary_stat == 'CGD':
        plt_title = 'Cluster gas density'
        x_label = r"$r/R_{\mathrm{500c}}$"
        y_label = r"$\rho_{\mathrm{gas}} \,/\, \rho_{\mathrm{crit}}$"

    elif summary_stat == 'Pk':
        plt_title = 'Total power spectra ratio'
        x_label = r'$k \, [h\,\mathrm{Mpc}^{-1}]$'
        y_label = r'$P_{\mathrm{sub}}(k)\,/\,P_{\mathrm{grav}}(k)$'

    elif summary_stat == 'CSFR':
        plt_title = 'Cosmic star formation rate'
        x_label = r"$a$"
        y_label = r"$\mathrm{CSFR} \, [\mathrm{M}_{\odot} \, \mathrm{yr}^{-1} \, (h^{-1}\mathrm{Mpc})^{-3}]$"
        
    else:
        print('Not implemented')
        
    return plt_title, x_label, y_label

######################################################

def load_gsmf_obs(gsmf_obs_dir):
    
    # _data = np.loadtxt( gsmf_obs_dir + "Moustakas2013_z0.1.txt")
    # ax.plot(10**_data[:, 0], 10 ** _data[:, 1] / hubble**3, 'rx-', label=r"Moustakas et al. 2013", alpha=0.3)
    # # ax.fill_between(10**_data[:, 0], 10 ** (_data[:, 1] - _data[:, 2]) / hubble**3, 10 ** (_data[:, 1] + _data[:, 2]) / hubble**3)

    hubble=0.681
    _data = np.loadtxt( gsmf_obs_dir + "Driver2022.txt")
    # _h = 0.73
    # _data[:, :3] -= 2 * np.log10(_h)  # not sure if we need a correction here?
    _phi = 10 ** _data[:, 1] / hubble**3
    _yerr_range = (10 ** np.array([_data[:, 1] - _data[:, 2], _data[:, 1] + _data[:, 2]])/ hubble**3)
    _yerr = np.abs(_yerr_range - _phi)
    
    return 10**_data[:, 0], _phi, _yerr
 

def load_fgas_obs():
    # Table 5 from Kugel+ 2023
    # M500c fgas_500c fgas_500c_err
    # [log10 Msun] [-] [-]
    
    
    
    hubble=0.681
    
    M500c = np.array([13.89, 14.06, 14.23, 14.40, 14.57, 14.74, 14.91])
    fgas = np.array([0.083, 0.094, 0.105, 0.115, 0.130, 0.130, 0.139])
    err_fgas = np.array([0.002, 0.003, 0.005, 0.008, 0.002, 0.002, 0.003])
    
    return (10**M500c)*(hubble), fgas, err_fgas



def load_fgas_other_sims(directory, pattern='*.txt', exclude=['SalmanCollection', 'Chiu2018Sample'] ):

    hubble=0.681
    data = {}  # Initialize an empty dictionary to store transformed x and original y values for each source
    for fileIn in glob.glob(directory + pattern):
        sourceIn = fileIn.split('/')[-1].split('.txt')[0]
        print(sourceIn)
        
        if not any(excl in sourceIn for excl in exclude):
            a = np.loadtxt(fileIn)
            x_transformed = a[:, 0]
            y = a[:, 1]
            data[sourceIn] = (x_transformed, y)  # Store transformed x and original y as a tuple in the dictionary
    return data


def load_cged_obs(directory, pattern='*.txt', exclude=[], rho_cgs_scale=True):
    
    
    if rho_cgs_scale:
        RHO_C = 2.77536627e11  # Critical density in (M_sun/h) / (Mpc/h)^3
        MPC_IN_CM = 3.24077927e-25
        G_IN_MSUN = 1.9885e33
        MP = 1.67262158e-24  # Mass of proton in grams (using 1.00794 amu )
        _rhoc_cgs = RHO_C * 0.7**2  # in proper (Msun/h) / (Mpc/h)^3 = h^2 Msun/Mpc^3
        _rhoc_cgs *= MPC_IN_CM**3 * G_IN_MSUN
    
    data = {}
    
    
    for fileIn in glob.glob(directory + pattern):
        sourceIn = fileIn.split('/')[-1].split('.txt')[0]
        if not any(excl in sourceIn for excl in exclude):
            a = np.loadtxt(fileIn)
            x_transformed = a[:, 0]
            if rho_cgs_scale: y = a[:, 1:]*_rhoc_cgs / MP / 1.397 * 1.199
            else: y = a[:, 1:]
            
            data[sourceIn] = (x_transformed, y)
            
    return data

def load_cgd_obs(directory, pattern='*.txt', exclude=[]):
    
    data = {}
    
    for fileIn in glob.glob(directory + pattern):
        sourceIn = fileIn.split('/')[-1].split('.txt')[0]
        if not any(excl in sourceIn for excl in exclude):
            a = np.loadtxt(fileIn)
            x_transformed = a[:, 0]
            y = a[:, 1:]
            
            data[sourceIn] = (x_transformed, y)

    return data

    
def load_bhmsm_other_sims(directory, pattern='*.txt', exclude='Moustakas2013_z0.2-z1.0'):
    data = {}  # Initialize an empty dictionary to store x and y values for each source
    for fileIn in glob.glob(directory + pattern):
        sourceIn = fileIn.split('/')[-1].split('.txt')[0]
        print(sourceIn)
        
        if not any(excl in sourceIn for excl in exclude):
            a = np.loadtxt(fileIn)
            x = 10 ** a[:, 0]
            y = 10 ** a[:, 1]
            data[sourceIn] = (x, y)  # Store x and y as a tuple in the dictionary
    return data

######################################################

def fill_nan_with_interpolation(data, kind):
    extrapolated_data = np.copy(data)  # Make a copy of the input array to avoid modifying it directly

    for i in range(data.shape[0]):  # Iterate over rows
        # Identify the indices of NaN and non-NaN values
        # nan_indices = np.where(np.isnan(data[i]))[0]
        # non_nan_indices = np.where(~np.isnan(data[i]))[0]
        
        nan_indices = np.where(np.isnan(data[i]) | (data[i] < 1e-6))[0]
        non_nan_indices = np.where(~np.isnan(data[i]) & (data[i] > 1e-6))[0]

        # Continue only if there are both NaN and non-NaN values in the row
        if len(nan_indices) > 0 and len(non_nan_indices) > 0:
            # Use non-NaN values to create an interpolation function
            # 'kind' can be adjusted depending on the desired interpolation type (e.g., linear, quadratic)
            # 'fill_value'='extrapolate' allows for extrapolation
            interpolation_function = interp1d(non_nan_indices, data[i][non_nan_indices], kind=kind, fill_value='extrapolate')

            # Compute interpolated/extrapolated values for NaN indices
            interpolated_values = interpolation_function(nan_indices)

            # Replace NaN values with interpolated/extrapolated values
            extrapolated_data[i][nan_indices] = interpolated_values

    return extrapolated_data