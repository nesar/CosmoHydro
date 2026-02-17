__all__ = ['DATA_DIR', 'LIBRARY_ZK_FILE', 'LIBRARY_BK_FILE', 'LIBRARY_PARAM_FILE', 'PARAM_NAME',
           'load_params', 'sepia_data_format', 'read_gsmf', 'read_bhmsm', 'read_gal_ssfr', 'read_gasfr', 'plot_strings', 'mass_conds', 'load_gsmf_obs', 'load_fgas_obs', 'load_fgas_other_sims', 'load_bhmsm_other_sims', 'fill_nan_with_interpolation' ]


import numpy as np
import glob
import pkg_resources
from sepia.SepiaData import SepiaData
from scipy.interpolate import interp1d



DATA_DIR = "data/"
LIBRARY_ZK_FILE = pkg_resources.resource_stream("CubicGalileonEmu", DATA_DIR + "z_k.txt").name
LIBRARY_BK_FILE = pkg_resources.resource_stream("CubicGalileonEmu", DATA_DIR + "Boost.npy").name
LIBRARY_PARAM_FILE = pkg_resources.resource_stream("CubicGalileonEmu", DATA_DIR + "cosmo_newdesign.txt").name
PARAM_NAME = [r'$f_{SN}$', r'$log(\nu_{SN})$',  r'$log(T_{AGN})$', r'$\beta_{BH}$']




def load_params(p_fileIn:str=LIBRARY_PARAM_FILE, # Input file for parameters
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

def read_gsmf(DirIn, num_sims, params):
    
    gsmf_arr = np.zeros(shape=(num_sims, 39))


    for file_indx in range(num_sims): 
        # print(file_indx)
        kappa_i = params[:, 0][file_indx]

        str_cond = str(kappa_i.astype(int)) if kappa_i.is_integer() else str(round(kappa_i, 4))
        fileSearch = DirIn + '**' + 'FSN_' + str_cond + '**/**/GalStellarMassFunction_624.txt'
        fileIn = glob.glob(fileSearch, recursive=True)
        gsmf_all = np.loadtxt(fileIn[0], skiprows=0)


        gsmf = gsmf_all[:, 1] 
        gsmf_arr[file_indx, :] = gsmf

    stellar_mass = gsmf_all[:, 0] 
    # gsmf_um = gsmf_all[:, 3] 
    # gsmf_um = gsmf_arr[target_indx] ## just choosing one for now 
    
    return stellar_mass, gsmf_arr

def read_cged(DirIn, num_sims, params):
    
    cged_arr = np.zeros(shape=(num_sims, 20))


    for file_indx in range(num_sims): 
        # print(file_indx)
        kappa_i = params[:, 0][file_indx]

        str_cond = str(kappa_i.astype(int)) if kappa_i.is_integer() else str(round(kappa_i, 4))
        fileSearch = DirIn + '**' + 'FSN_' + str_cond + '**/**/ClusterGasElectronDensityProfile_624.txt'
        fileIn = glob.glob(fileSearch, recursive=True)
        cged_all = np.loadtxt(fileIn[0], skiprows=0)


        cged = cged_all[:, 1] 
        cged_arr[file_indx, :] = gsmf

    log_radius = gsmf_all[:, 0] 
    # gsmf_um = gsmf_all[:, 3] 
    # gsmf_um = gsmf_arr[target_indx] ## just choosing one for now 
    
    return log_radius, cged_arr



def read_bhmsm(DirIn, num_sims, params):

    bhmsm_arr = np.zeros(shape=(num_sims, 20))

    for file_indx in range(num_sims): 
        kappa_i = params[:, 0][file_indx]

        str_cond = str(kappa_i.astype(int)) if kappa_i.is_integer() else str(round(kappa_i, 4))
        fileSearch = DirIn + '**' + 'FSN_' + str_cond + '**/**/BHMass_StellarMass_624.txt'
        fileIn = glob.glob(fileSearch, recursive=True)
        f_all = np.loadtxt(fileIn[0], skiprows=0)


        f = f_all[:, 3] 
        bhmsm_arr[file_indx, :] = f

    log_bhmsm_mass = f_all[:, 0] 
    # bhmsm_um = f_all[:, 4] 
    # bhmsm_um = bhmsm_arr[target_indx] ## just choosing one for now 
    
    return log_bhmsm_mass, bhmsm_arr


def read_gal_ssfr(DirIn, num_sims, params):

    gal_ssfr_arr = np.zeros(shape=(num_sims, 20))

    for file_indx in range(num_sims): 
        kappa_i = params[:, 0][file_indx]

        str_cond = str(kappa_i.astype(int)) if kappa_i.is_integer() else str(round(kappa_i, 4))
        fileSearch = DirIn + '**' + 'FSN_' + str_cond + '**/**/GalaxySSFR_624.txt'
        fileIn = glob.glob(fileSearch, recursive=True)
        f_all = np.loadtxt(fileIn[0], skiprows=0)


        f = f_all[:, 1] 
        gal_ssfr_arr[file_indx, :] = f

    log_ssfr_mass = f_all[:, 0] 
    # gal_ssfr_um = f_all[:, 4] 
    # gal_ssfr_um = gal_ssfr_arr[target_indx]  ## just choosing one for now 
    
    return log_ssfr_mass, gal_ssfr_arr



def read_gasfr(DirIn, num_sims, params):

    gas_fr_mean = np.zeros(shape=(num_sims, 40))

    for file_indx in range(num_sims): 
        kappa_i = params[:, 0][file_indx]

        str_cond = str(kappa_i.astype(int)) if kappa_i.is_integer() else str(round(kappa_i, 4))
        # fileSearch = DirIn + '**' + 'FSN_' + str_cond + '**/**/GasFractionR500_624.txt'
        fileSearch = DirIn + '**' + 'FSN_' + str_cond + '**/**/Mgas_M500_Ratio_624.txt'  ## Design2
        fileIn = glob.glob(fileSearch, recursive=True)
        f_all = np.loadtxt(fileIn[0], skiprows=0)


        f = f_all[:, 5] ## 3 for mean, 5 for median 
        gas_fr_mean[file_indx, :] = f

    log_halo_mass = f_all[:, 0] 
    # gas_fr_um = gas_fr_mean[target_indx] ## just choosing one for now 
    
    return log_halo_mass, gas_fr_mean


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
    
    return mlim1, mlim2


def plot_strings(summary_stat):
     
    if(summary_stat == 'GSMF'):

        plt_title = 'Galaxy stellar mass function'
        x_label = r'$M_\star\,/\,M_\odot$'
        y_label = r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$'
        
    elif(summary_stat == 'BHMSM'): 
        
        plt_title = 'Black hole mass-stellar mass'
        x_label = r'$M_\star\,/\,M_\odot$'
        y_label = r'$M_{BH}\,/\,M_\odot$'     
        
    elif(summary_stat == 'gSSFR'): 

        plt_title = 'Galaxy specific star formation rate'
        x_label = r'$M_\star\,/\,M_\odot$'
        y_label = r'$sSFR$'
                
    elif(summary_stat == 'fGas'): 

        plt_title = 'Cluster gas fraction'
        y_label = r'$M_{gas}\,/\,M_{500} [< R_{500c}]$'
        x_label = r'$M_{500c}\,/\,h^{-1}M_\odot$'
        
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
    
    hubble=0.681
    
    M500c = np.array([13.89, 14.06, 14.23, 14.40, 14.57, 14.74, 14.91])
    fgas = np.array([0.083, 0.094, 0.105, 0.115, 0.130, 0.130, 0.139])
    err_fgas = np.array([0.002, 0.003, 0.005, 0.008, 0.002, 0.002, 0.003])
    
    return (10**M500c)*(hubble), fgas, err_fgas


def load_fgas_other_sims(directory, pattern='*.txt', exclude='none'):
    data = {}  # Initialize an empty dictionary to store transformed x and original y values for each source
    for fileIn in glob.glob(directory + pattern):
        sourceIn = fileIn.split('/')[-1].split('.txt')[0]
        
        if sourceIn != exclude:
            a = np.loadtxt(fileIn)
            x_transformed = a[:, 0]
            y = a[:, 1]
            data[sourceIn] = (x_transformed, y)  # Store transformed x and original y as a tuple in the dictionary
    return data

    
    
def load_bhmsm_other_sims(directory, pattern='*.txt', exclude='Moustakas2013_z0.2-z1.0'):
    data = {}  # Initialize an empty dictionary to store x and y values for each source
    for fileIn in glob.glob(directory + pattern):
        sourceIn = fileIn.split('/')[-1].split('.txt')[0]
        
        if sourceIn != exclude:
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