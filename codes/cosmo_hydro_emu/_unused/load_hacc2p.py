__all__ = ['PARAM_NAME',
           'load_params', 'sepia_data_format', 'read_gsmf', 'read_cged', 'read_bhmsm', 'load_cged_obs', 
           'load_cgd_obs', 'read_gal_ssfr', 'read_gasfr', 'read_cgd', 'read_cgd_cc', 'read_pk', 'plot_strings', 'mass_conds', 
           'load_gsmf_obs', 'load_fgas_obs', 'load_fgas_other_sims', 'load_bhmsm_other_sims', 'load_cgd_cc_obs', 
           'fill_nan_with_interpolation', 'eps_scale', 'seed_mass_scale', 'vkin_scale']


import numpy as np
import glob
import pkg_resources
from sepia.SepiaData import SepiaData
from scipy.interpolate import interp1d
import os



DATA_DIR = "data/"
# LIBRARY_ZK_FILE = pkg_resources.resource_stream("CubicGalileonEmu", DATA_DIR + "z_k.txt").name
# LIBRARY_BK_FILE = pkg_resources.resource_stream("CubicGalileonEmu", DATA_DIR + "Boost.npy").name
# LIBRARY_PARAM_FILE = pkg_resources.resource_stream("CubicGalileonEmu", DATA_DIR + "cosmo_newdesign.txt").name


seed_mass_scale = 1e6
vkin_scale = 1e4
eps_scale = 1e1

# PARAM_NAME = ['KAPPA', 'EGW', 'SEED' + '/' + str( int(seed_mass_scale)), 'VKIN' + '/' + str( int(vkin_scale)), 'EPS' + '/' + str( int(eps_scale))] ## HACC-5p


# def load_delta_cgd(DirIn, EPS_val = 2.4):
#     Vkin_val = 6300 if EPS_val == 2.4 else 3500
#     fileIn128 = DirIn + 'Default_128MPC_NEW_KIN_JET_E_'+str(EPS_val)+'_V_'+str(Vkin_val)+'_S_8e5_SS_ZINI_0.25/analysis_pipeline/extract/ClusterGasDensityProfile_624.txt'
#     fileIn256 = DirIn + 'Default_256MPC_NEW_KIN_JET_E_'+str(EPS_val)+'_V_'+str(Vkin_val)+'_S_8e5_SS_ZINI_0.25/analysis_pipeline/extract/ClusterGasDensityProfile_624.txt'
#     cgd_128 = np.loadtxt(fileIn128, skiprows=0)
#     cgd_256 = np.loadtxt(fileIn256, skiprows=0)
#     cgd_128_d = cgd_128[1:, 1] 
#     cdg_128_r = 10**cgd_128[1:, 0]
#     cgd_256_d = cgd_256[1:, 1] 
#     cdg_256_r = 10**cgd_256[1:, 0]

#     delta_cgd_add = cgd_256_d - cgd_128_d

#     return delta_cgd_add

# DirIn_128_256 = '/home/nramachandra/Projects/Hydro_runs/Data/ProfileData/SCIDAC_RUNS/128_256/'
# delta_cgd = load_delta_cgd(DirIn = DirIn_128_256, EPS_val = 2.4)


# PARAM_NAME = [
#     r'$\kappa$', 
#     r'$E_{GW}$', 
#     r'$M_{seed}/10^{' + str(int(np.log10(seed_mass_scale))) + '}$', 
#     r'$V_{kin}/10^{' + str(int(np.log10(vkin_scale))) + '}$', 
#     r'$E_{PS}/10^{' + str(int(np.log10(eps_scale))) + '}$'
# ]

# PARAM_NAME = [
#     r'$V_{kin}/10^{' + str(int(np.log10(vkin_scale))) + '}$', 
#     r'$E_{PS}/10^{' + str(int(np.log10(eps_scale))) + '}$'
# ]


# PARAM_NAME = [
#     r'$v_{kin}/10^{' + str(int(np.log10(vkin_scale))) + '}$', 
#     r'$\epsilon/10^{' + str(int(np.log10(eps_scale))) + '}$'
# ]


PARAM_NAME =  [
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

def read_gsmf(DirIn, num_sims, params):
    
    gsmf_arr = np.zeros(shape=(num_sims, 39))


    for file_indx in range(num_sims): 
        # print(file_indx)
        kappa_i = params[:, 0][file_indx]

        str_cond = str(kappa_i.astype(int)) if kappa_i.is_integer() else str(round(kappa_i, 4))
        fileSearch = DirIn + '**' + 'VKIN_' + str_cond + '**/**/GalStellarMassFunction_624.txt'
        fileIn = glob.glob(fileSearch, recursive=True)
        gsmf_all = np.loadtxt(fileIn[0], skiprows=0)


        gsmf = gsmf_all[:, 1] 
        gsmf_arr[file_indx, :] = gsmf

    stellar_mass = gsmf_all[:, 0] 
    # gsmf_um = gsmf_all[:, 3] 
    # gsmf_um = gsmf_arr[target_indx] ## just choosing one for now 
    
    return stellar_mass, gsmf_arr



def read_pk(DirIn, num_sims, params):

    pk_arr = np.zeros(shape=(num_sims, 886))

    for file_indx in range(num_sims): 
        vkin_i = params[:, 0][file_indx]*10000
        print(vkin_i)

        # Minimal change: round to integer for cleaner string formatting
        vkin_rounded = int(round(vkin_i))
        str_cond = str(vkin_rounded)
        
        fileSearch = DirIn + '**' + 'VKIN_' + str_cond + '**/**/m000p.pk.624'
        fileIn = glob.glob(fileSearch, recursive=True)
        print(fileIn)
        f_all = np.loadtxt(fileIn[0], skiprows=0)

        f = f_all[:, 1] 
        pk_arr[file_indx, :] = f

    k = f_all[:, 0] 

    fileIn_go = DirIn + 'GRAV256_ONLY/power_spectrum/' + 'm000p.pk.624'
    pk_go = np.loadtxt(fileIn_go, skiprows=0)[:, 1]

    pk_ratio = pk_arr/pk_go
    
    return k, pk_arr, pk_ratio



def read_bhmsm(DirIn, num_sims, params):

    bhmsm_arr = np.zeros(shape=(num_sims, 20))

    for file_indx in range(num_sims): 
        kappa_i = params[:, 0][file_indx]

        str_cond = str(kappa_i.astype(int)) if kappa_i.is_integer() else str(round(kappa_i, 4))
        fileSearch = DirIn + '**' + 'VKIN_' + str_cond + '**/**/BHMass_StellarMass_624.txt'
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
        fileSearch = DirIn + '**' + 'VKIN_' + str_cond + '**/**/GalaxySSFR_624.txt'
        fileIn = glob.glob(fileSearch, recursive=True)
        f_all = np.loadtxt(fileIn[0], skiprows=0)


        f = f_all[:, 1] 
        gal_ssfr_arr[file_indx, :] = f

    log_ssfr_mass = f_all[:, 0] 
    # gal_ssfr_um = f_all[:, 4] 
    # gal_ssfr_um = gal_ssfr_arr[target_indx]  ## just choosing one for now 
    
    return log_ssfr_mass, gal_ssfr_arr



# def read_gasfr(DirIn, num_sims, params):

#     gas_fr_mean = np.zeros(shape=(num_sims, 40))

#     for file_indx in range(num_sims): 
#         kappa_i = params[:, 0][file_indx]

#         str_cond = str(kappa_i.astype(int)) if kappa_i.is_integer() else str(round(kappa_i, 4))
#         # fileSearch = DirIn + '**' + 'FSN_' + str_cond + '**/**/GasFractionR500_624.txt'
#         fileSearch = DirIn + '**' + 'VKIN_' + str_cond + '**/**/Mgas_M500_Ratio_624.txt'  ## Design2
#         fileIn = glob.glob(fileSearch, recursive=True)
#         f_all = np.loadtxt(fileIn[0], skiprows=0)


#         f = f_all[:, 5] ## 3 for mean, 5 for median 
#         gas_fr_mean[file_indx, :] = f

#     log_halo_mass = f_all[:, 0] 
#     # gas_fr_um = gas_fr_mean[target_indx] ## just choosing one for now 
    
#     return log_halo_mass, gas_fr_mean


def read_gasfr(DirIn, num_sims, params):  # Adjust eps_scale as needed
    fgas_arr = np.zeros(shape=(num_sims, 40))

    for file_indx in range(num_sims): 
        vkin_i = params[:, 0][file_indx] * vkin_scale  # Apply scaling factor to VKIN
        eps_i = params[:, 1][file_indx] * eps_scale    # Apply scaling factor to EPS

        # Format VKIN as integer or float with full precision
        vkin_str = f"{int(vkin_i)}" if vkin_i.is_integer() else f"{vkin_i:.4f}".rstrip('0').rstrip('.')
        
        # Ensure EPS is always represented with exactly 3 decimal places (even if zeros)
        eps_str = f"{eps_i:.3f}"

        # Adjust the file search path based on VKIN and EPS values
        fileSearch = os.path.join(DirIn, f'VKIN_{vkin_str}_EPS_{eps_str}', '**', 'Mgas_M500_Ratio_624.txt')
        
        # Search for the file recursively
        fileIn = glob.glob(fileSearch, recursive=True)
        print(fileIn[0])
        
        if fileIn:
            fgas_all = np.loadtxt(fileIn[0], skiprows=0)
            fgas_cc = fgas_all[:, 5] ## 3 for mean, 5 for median 
            fgas_arr[file_indx, :] = fgas_cc
        else:
            print(f"File not found for VKIN_{vkin_str}_EPS_{eps_str}")

    log_m500 = fgas_all[:, 0] 
    
    return log_m500[1:], fgas_arr[:, 1:]


def read_cged(DirIn, num_sims, params):
    
    cged_arr = np.zeros(shape=(num_sims, 20))


    for file_indx in range(num_sims): 
        # print(file_indx)
        kappa_i = params[:, 0][file_indx]

        str_cond = str(kappa_i.astype(int)) if kappa_i.is_integer() else str(round(kappa_i, 4))
        fileSearch = DirIn + '**' + 'VKIN_' + str_cond + '**/**/ClusterGasElectronDensityProfile_624.txt'
        
        fileIn = glob.glob(fileSearch, recursive=True)
        # print(fileIn)
        cged_all = np.loadtxt(fileIn[0], skiprows=0)

        cged = cged_all[:, 1] 
        cged_arr[file_indx, :] = cged

    log_radius = cged_all[:, 0] 
    
    return 10**log_radius[1:], cged_arr[:, 1:]


# def read_cgd(DirIn, num_sims, params):  # Adjust eps_scale as needed
#     cgd_arr = np.zeros(shape=(num_sims, 20))

#     for file_indx in range(num_sims): 
#         vkin_i = params[:, 0][file_indx] * vkin_scale  # Apply scaling factor to VKIN
#         eps_i = params[:, 1][file_indx] * eps_scale    # Apply scaling factor to EPS

#         # Format VKIN as integer or float with full precision
#         vkin_str = f"{int(vkin_i)}" if vkin_i.is_integer() else f"{vkin_i:.4f}".rstrip('0').rstrip('.')
        
#         # Ensure EPS is always represented with exactly 3 decimal places (even if zeros)
#         eps_str = f"{eps_i:.3f}"

#         # Adjust the file search path based on VKIN and EPS values
#         fileSearch = os.path.join(DirIn, f'VKIN_{vkin_str}_EPS_{eps_str}', '**', 'ClusterGasDensityProfile_624.txt')
        
#         # Search for the file recursively
#         fileIn = glob.glob(fileSearch, recursive=True)
#         print(fileIn[1])
        
#         if fileIn:
#             cgd_all = np.loadtxt(fileIn[1], skiprows=0)
#             cgd = cgd_all[:, 1]
#             cgd_arr[file_indx, :] = cgd
#         else:
#             print(f"File not found for VKIN_{vkin_str}_EPS_{eps_str}")

#     log_radius = cgd_all[:, 0] 
    
#     return 10**log_radius[1:], cgd_arr[:, 1:]


import os, glob
import numpy as np

def read_cgd(DirIn, num_sims, params):
    cgd_arr = np.zeros((num_sims, 20))
    log_radius_ref = None

    for file_indx in range(num_sims):
        vkin_i = float(params[file_indx, 0]) * vkin_scale
        eps_i  = float(params[file_indx, 1]) * eps_scale

        vkin_i = round(vkin_i)
        eps_i  = round(eps_i, 3)

        vkin_str = f"{int(vkin_i)}"
        eps_str  = f"{eps_i:.3f}"

        simdir = os.path.join(DirIn, f"VKIN_{vkin_str}_EPS_{eps_str}")
        if not os.path.isdir(simdir):
            eps_str2 = eps_str.rstrip("0").rstrip(".")  # handles EPS_3 vs EPS_3.000
            simdir = os.path.join(DirIn, f"VKIN_{vkin_str}_EPS_{eps_str2}")

        fileSearch = os.path.join(simdir, "**", "ClusterGasDensityProfile_624.txt")
        files = sorted(glob.glob(fileSearch, recursive=True))

        if not files:
            print(f"File not found for {simdir}")
            continue

        file_path = files[0]
        cgd_all = np.loadtxt(file_path, skiprows=0)

        if log_radius_ref is None:
            log_radius_ref = cgd_all[:, 0]

        cgd = cgd_all[:, 1]
        n = min(cgd_arr.shape[1], cgd.shape[0])
        cgd_arr[file_indx, :n] = cgd[:n]

    if log_radius_ref is None:
        raise FileNotFoundError("No ClusterGasDensityProfile_624.txt files found for any sim.")

    return 10**log_radius_ref[1:], cgd_arr[:, 1:]



def read_cgd_cc(DirIn, num_sims, params):  # Adjust eps_scale as needed
    cgd_arr = np.zeros(shape=(num_sims, 20))

    for file_indx in range(num_sims): 
        vkin_i = params[:, 0][file_indx] * vkin_scale  # Apply scaling factor to VKIN
        eps_i = params[:, 1][file_indx] * eps_scale    # Apply scaling factor to EPS

        # Format VKIN as integer or float with full precision
        vkin_str = f"{int(vkin_i)}" if vkin_i.is_integer() else f"{vkin_i:.4f}".rstrip('0').rstrip('.')
        
        # Ensure EPS is always represented with exactly 3 decimal places (even if zeros)
        eps_str = f"{eps_i:.3f}"

        # Adjust the file search path based on VKIN and EPS values
        fileSearch = os.path.join(DirIn, f'VKIN_{vkin_str}_EPS_{eps_str}', '**', 'ClusterGasDensityProfile_624.txt')
        
        # Search for the file recursively
        fileIn = glob.glob(fileSearch, recursive=True)
        print(fileIn[0])
        
        if fileIn:
            cgd_all = np.loadtxt(fileIn[0], skiprows=0)
            cgd_cc = cgd_all[:, 4]
            cgd_arr[file_indx, :] = cgd_cc
        else:
            print(f"File not found for VKIN_{vkin_str}_EPS_{eps_str}")

    log_radius = cgd_all[:, 0] 
    
    return 10**log_radius[1:], cgd_arr[:, 1:]



######################################################


def mass_conds(summary_stat):
        
    if (summary_stat == 'Common'):
        
        ## Flamingo limits
        mlim1 = 10**9
        mlim2 = 2*10**12 
        
    elif (summary_stat == 'GSMF'):

        mlim1 = 5*10**9
        mlim2 = 3*10**11 
        
    elif (summary_stat == 'BHMSM'): 
          
        mlim1 = 10**10
        mlim2 = 2*10**12 
        
    elif (summary_stat == 'gSSFR'): 
 
        mlim1 = 10**9
        mlim2 = 10**13
        
    elif (summary_stat == 'fGas'): 

        ## Flamingo limits
        mlim1 = 10**(13.5)
        mlim2 = 10**(14.3)

    # mass_cond = np.where( (target_xvals > mlim1)  &  (target_xvals < mlim2) ) 
    
    elif (summary_stat == 'CGED'): 

        ## Actually radius for CGED
        mlim1 =  0.025
        mlim2 = 1.2
   
    elif (summary_stat == 'CGD'): 

        ## Actually radius for CGD
        mlim1 =  0.015
        mlim2 = 2.75
    
    elif (summary_stat == 'CGD_CC'): 

        ## Actually radius for CGD_CC
        mlim1 =  0.016
        mlim2 = 2.75

    if (summary_stat == 'Pk'): 

        ## Based on 

        # k_min = 2*np.pi/side_length
        # delta_x = side_length/Npart
        # k_max = np.pi/delta_x #Nyquist

        mlim1 =  0.02454369260617026
        mlim2 = 12.566370614359172

    else: 
        print('Not implemented')
    
    return mlim1, mlim2

'''
def plot_strings(summary_stat):
     
    if(summary_stat == 'GSMF'):

        plt_title = 'Galaxy stellar mass function'
        x_label = r'$M_\star\,/\,M_\odot$'
        y_label = r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$'
        
    elif(summary_stat == 'BHMSM'): 
        
        plt_title = 'Black hole mass-stellar mass'
        x_label = r'$M_\star\,/\,M_\odot$'
        y_label = r'$M_\text{BH}\,/\,M_\odot$'     
        
    elif(summary_stat == 'gSSFR'): 

        plt_title = 'Galaxy specific star formation rate'
        x_label = r'$M_\star\,/\,M_\odot$'
        y_label = r'$sSFR$'
                
    elif(summary_stat == 'fGas'): 

        plt_title = 'Cluster gas fraction'
        y_label = r'$M_\text{gas}\,/\,M_{500} [< R_{500c}]$'
        x_label = r'$M_\text{500c}\,/\,h^{-1}M_\odot$'
        
    elif(summary_stat == 'CGED'): 

        plt_title = 'Cluster gas electron density'
        y_label = r'$n_\text{e}$'
        x_label = r'$r$'

    elif(summary_stat == 'CGD'): 

        plt_title = 'Cluster gas density'
        y_label = r'$\rho_\text{gas}\,/\,\rho_\text{crit}$'
        x_label = r'$r$'

    elif(summary_stat == 'CGD_CC'): 

        plt_title = 'Cluster gas density - cool cores'
        y_label = r'$\rho_\text{gas, cc}\,/\,\rho_\text{crit}$'
        x_label = r'$r$'

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

    elif(summary_stat == 'CGD_CC'): 

        plt_title = 'Cluster gas density - cool cores'
        y_label = r"$\rho_{\mathrm{gas, cc}} \,/\, \rho_{\mathrm{crit}}$"
        x_label = r"$r/R_{\mathrm{500c}}$"

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

def load_cgd_cc_obs(directory, pattern='*.txt', exclude=[]):
    
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