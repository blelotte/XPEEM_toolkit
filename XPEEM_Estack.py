# -*- coding: utf-8 -*-
"""
Module to handle XPEEM E-stack

Created on Thu Jul 10 14:26:42 2025

@author: lelotte_b
"""

# Basics
import time
import numpy as np
import pandas as pd
import ast

# Display in python
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_context('paper', font_scale=2.2) # to change the font settings in the graphs
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina') # High resolution plots
from typing import List, Tuple, Union


# Docstrings, test and warnings
np.seterr(all='warn') # to be able to handle warnings
from typing import Optional

# Folder/system management
import psutil
import os


def is_program_running(program_name):
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == program_name:
            return True
    return False

# Statistics
from savitzkygolay import filter3D
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter

# My modulesES
import XPEEM_utils as utils



""" 
Opening files, defining parameters
_________________________________________________________________________"""
# TODO Swathi add the load I0 and energy from stack folder.
def load_Estack(edgeFd_path,stackFd_path):
    """
    Load the I0, energy and E-stack

    Parameters
    ----------
    stackFd_path (str): the path to the undistrdd folder.

    Returns
    -------
    n_x,n_y,n_z (tuple): dimensions of the E-stack
    E_z (np.ndarray): energy.
    I0_z (np.ndarray): incoming flux.
    I_xyz (np.ndarray): E-stack
    ImNames (list): the E-stack slice labels
    """
    I_xyz,ImNames=utils.open_sequence(stackFd_path)
    
    (n_x,n_y,n_z)=np.shape(I_xyz)
    
    E_z,I0_z=utils.load_E_I0(edgeFd_path)
    
    n_z,E_z,I0_z,I_xyz=removeDup_Estack(E_z,I0_z,I_xyz)
    
    return (n_x,n_y,n_z),E_z,I0_z,I_xyz,ImNames


def removeDup_Estack(E_z: np.ndarray,I0_z: np.ndarray,I_xyz: np.ndarray,energyResolution=0.1) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Average duplicate images in a 3D stack that share the same energy values, fully vectorized.

    Parameters
    ----------
    E_z : np.ndarray, shape (n_z,)
        Energy values for each slice.
    I0_z : np.ndarray, shape (n_z,)
        Reference I0 values for each slice.
    I_xyz : np.ndarray, shape (n_x, n_y, n_z)
        Stack of images (x, y, z).
    energy_resolution : float
        Energy resolution of the beamline (default 0.1 eV)
    
    Returns
    -------
    n_k : int
        Number of unique energies.
    unique_E_k : np.ndarray, shape (k,)
        The sorted unique energy values.
    I0_k : np.ndarray, shape (k,)
        The averaged I0 per unique energy.
    I_xyz_avg : np.ndarray, shape (n, m, k)
        The averaged image stack for each unique energy.
    """
    # 1) Find unique energies (within energy resolution) and an index map
    E_z = np.round(E_z / energyResolution) * energyResolution
    unique_E_k, inverse = np.unique((E_z,), return_inverse=True)
    n_k = unique_E_k.size

    # 2) Counts and sum of I0 per group
    counts_k = np.bincount(inverse)
    sum_I0_k  = np.bincount(inverse, weights=I0_z)
    avg_I0_k  = sum_I0_k / counts_k

    # 3) Sum up I_xyz across z-axis for each group
    #    Build a boolean mask: mask[i, j] = True if slice i belongs to group j
    mask = inverse[:, None] == np.arange(n_k)[None, :]
    #    Tensordot over z-axis (axis=2 of I_xyz) against mask’s axis=0
    #    yields shape (n, m, k): sums of all I_xyz[..., i] where inverse[i] == j
    sum_Ixyk = np.tensordot(I_xyz, mask, axes=(2, 0))

    # 4) Divide by counts to get the average
    avg_I_xyz = sum_Ixyk / counts_k[np.newaxis, np.newaxis, :]  # shape (n, m, k)

    return n_k, unique_E_k, avg_I0_k, avg_I_xyz

def average_goldSpectra(gold: List[str], E: np.ndarray, details: bool) -> np.ndarray:
    """
    This function averages the gold spectra by interpolating the TEY values for each gold spectrum 
    and then averaging these interpolated values. The averaged spectrum can be plotted if details is set to True.

    Parameters:
    gold (List[str]): The list of file paths for the gold spectra.
    E (np.ndarray): The energy values for interpolation.
    details (bool): Whether to plot the averaged spectrum.

    Returns:
    np.ndarray: The averaged TEY values.
    """

    TEY_interp=np.zeros(len(E))
    n=np.size(gold)
    for i in range(0,n):
        goldi=np.loadtxt(gold[i], delimiter=';', skiprows=9, usecols= (0, 1, 2, 3))
        E_gold=np.array(goldi[:,0])
        TEY_gold=np.array(goldi[:,2])/np.array(goldi[:,1])
        TEY_inter_i=np.interp(E,E_gold,TEY_gold)
        
        TEY_interp+=TEY_inter_i
    TEY_interp=TEY_interp/n
    
    if details :
        fig, ax = plt.subplots()
        ax.plot(E,TEY_interp, linewidth=3)
        plt.title("gold_averaging: spectrum")
        
    return TEY_interp

def normalise(x: List[float], y: List[float], x_std: float) -> List[float]:
    """
    Normalise a list of values by a standard value.

    Parameters:
    x (List[float]): The list of values to find the standard value in.
    y (List[float]): The list of values to normalise.
    x_std (float): The standard value.

    Returns:
    List[float]: The list of normalised values.
    """
    x_std_ind=min(range(len(x)), key=lambda i: abs(x[i]-x_std)) 
    return y/y[x_std_ind]

def load_params_excels(sheetName,edgeFd):
    df = pd.read_excel(utils.path_join(os.getcwd(),'_Input/1_args_ES.xlsx',dt='f'),sheet_name=sheetName)

    # Remove rows that contain '--' in any column
    df = df[~df.apply(lambda row: row.astype(str).str.contains('--').any(), axis=1)]

    # Filter the data based on the edge
    matcharg = []
    values = [edgeFd]
    columns = ['edgeFd']
    matcharg = []
    for arg,col in zip(values,columns):
        colmatch=[]
        for _, row in df.iterrows():
            valcol=row[col]
            if pd.isnull(valcol):
                condition = True
            elif isinstance(valcol,str) : 
                condition = valcol == arg or '*' in valcol and valcol.replace('*', '') in arg
            else:
                raise ValueError('The arguments.xlsx should only contain None or string in the column edge_name')
            colmatch.append(condition)  
        matcharg.append(colmatch)
    param = df[np.array(np.all(matcharg,axis=0))]
    
    # If there is no matching data, return None
    if param.empty:
        raise NotImplementedError(f'Error: you tried processing an experiment that is not configurated: \n Edge: {edgeFd}.')   

    # Otherwise, return the first matching row as a dictionary
    return param.iloc[0].to_dict()

def correct_Eshift(E_z: np.ndarray,II0_kz: np.ndarray,dE_k: np.ndarray) -> np.ndarray:
    """
    Correct each spectrum in II0_kz for the energy dispersion plane by interpolating
    from a shifted energy grid back onto the original E_z axis.

    Parameters
    ----------
    E_z : np.ndarray, shape (n_z,)
        The original energy axis.
    II0_kz : np.ndarray, shape (n_k, n_z)
        Array of spectra (rows) indexed by k over the same energy bins.
    dE_k : np.ndarray, shape (n_k,)
        Energy drift for each spectrum.

    Returns
    -------
    aligned_II0 : np.ndarray, shape (n_k, n_z)
        The drift-corrected spectra, each one interpolated onto E_z.
    """
    n_k, n_z = II0_kz.shape
    indices = np.arange(n_k, dtype=int)

    aligned_II0 = np.zeros((n_k, n_z), dtype=II0_kz.dtype)

    # Build a (n_k, n_z) grid of shifted energies
    shifted_E = E_z[None, :] - dE_k[:, None]

    # Interpolate each row back onto E_z
    for out_row, spec_row, shift_row in zip(indices, II0_kz, shifted_E):
        aligned_II0[out_row] = np.interp(
            E_z,
            shift_row,
            spec_row,
            left=spec_row[0],
            right=spec_row[-1]
        )

    return aligned_II0

def fit_preEdge_spectrum(E_z: np.ndarray,Algnd_II0_kz: np.ndarray, inputParams, edge_name: str,
    calcMask_k: Optional[np.ndarray] = None) -> Tuple[LinearRegression, float]:
    """
    Fit a linear background to the pre-edge region of a spectrum and compute the normalized background slope.

    Parameters
    ----------
    edge_name : str
        Identifier used to load parameters (must match an entry in the parameter Excel).
    E_z : np.ndarray, shape (p,)
        Array of energy values for each channel/slice.
    Algnd_II0_kz : np.ndarray, shape (n_pixels, p)
        Aligned II0 spectra for each pixel (rows) over energy bins (columns).
    calcMask_k : Optional[np.ndarray], shape (n_pixels,)
        Boolean mask indicating which pixels to include in the mean spectrum.
        If None, all pixels are used.

    Returns
    -------
    model : LinearRegression
        Fitted linear regression model on (energy - E_0) vs. mean II0 in pre-edge.
    bkg_slope : float
        The normalized background slope in units of (counts per eV), computed as
        model.coef_[0] divided by the mean pre-edge signal per pixel.
    """
    # 1. Load parameters
    preEdge_range = eval(inputParams['preEdge_range'])

    # 2. Compute mean II0 spectrum over masked pixels
    if isinstance(calcMask_k, np.ndarray):
        meanII0_z = np.nanmean(
            np.where(calcMask_k[:, np.newaxis], Algnd_II0_kz, np.nan),
            axis=0
            )
    else:
        meanII0_z = np.nanmean(Algnd_II0_kz, axis=0)

    # 3. Handle sentinel value for lower bound
    if preEdge_range[0] == -1:
        preEdge_range[0] = E_z[0]

    # 4. Find index bounds
    preEdge_low  = utils.find_closest_index(E_z, preEdge_range[0]) + 1
    preEdge_high = utils.find_closest_index(E_z, preEdge_range[1])

    preEdge_E_z     = E_z[preEdge_low:preEdge_high]
    E_0             = (preEdge_range[0] + preEdge_range[1]) / 2
    preEdge_II0_z   = meanII0_z[preEdge_low:preEdge_high]

    # 5. Fit linear background
    model = LinearRegression()
    X = (preEdge_E_z - E_0).reshape(-1, 1)
    y = preEdge_II0_z
    model.fit(X, y)

    # 6. Compute normalized slope
    B_k = np.nanmean(Algnd_II0_kz[:, preEdge_low:preEdge_high], axis=1)
    bkgSlope = model.coef_[0] / np.mean(B_k)

    print("Background slope:")
    print(f"{bkgSlope:.3f} counts/eV")

    return E_0, bkgSlope, B_k

def bkg_substraction_spectrum(E_z: np.ndarray,II0_kz:np.ndarray,params:pd.DataFrame,edge:str,
                              slope=None,intercept_k=None,E_0=None)->Tuple[np.ndarray,np.ndarray,List[Union[np.ndarray,List[str]]]]:
    """
    remove background and calculate integrals over specified energy ranges. 
    The normalization and integration ranges are determined based on the "edge" argument.

    Parameters:
    E_z (np.ndarray): The energy values.
    
    II0_z (np.ndarray): The PEEM spectrum.
    
    params (pd.DataFrame): The arguments for the fit.
    
    edge (str): The edge to use for normalization and integration.

    slope (): the slope of the pre-edge linear background
    
    intercept_k (): the intensity of the pre-edge.
    
    E_0 (float): the pre-edge energy.

    Returns:
    Tuple[np.ndarray, np.ndarray, List[Union[np.ndarray, List[str]]]]: 
        The energy values, the normalized PEEM spectrum, and a list containing 
        the integrals and their names.
    """
    (n_k,n_z)=np.shape(II0_kz)
    
    function = params.get('function')
    isFlipped = params.get('Flip spectra')
    print(isFlipped)
    
    if function=='double-fermi':
        ratioL3onL2 = float(params.get('L3/L2 ratio'))
        
        postEdgeL3_E_range = ast.literal_eval(params.get('postEdgeL3_E_range'))
        assert isinstance(postEdgeL3_E_range, list), "Expected list for postEdge_E_range"
        if postEdgeL3_E_range[0] == -1 :
            postEdgeL3_E_range[0] = E_z[0]
        postEdgeL2_E_range = ast.literal_eval(params.get('postEdgeL2_E_range'))
        if postEdgeL2_E_range[1] == -1 :
            postEdgeL2_E_range[1] = E_z[-1]
        assert isinstance(postEdgeL2_E_range, list), "Expected list for postEdge_E_range"
        L3_E0 = float(params.get('L3_E0'))
        L2_E0 = float(params.get('L2_E0'))
        L3_kT = float(params.get('L3_kT'))
        L2_kT = float(params.get('L2_kT'))
    elif function=='fermi' or function=='fermi_C_Ka':
        postEdgeK_E_range = ast.literal_eval(params.get('postEdge_E_range'))
        if postEdgeK_E_range[1] == -1 :
            postEdgeK_E_range[1] = E_z[-1]
        K_E0 = float(params.get('E0'))
        K_kT = float(params.get('kT'))
        
        assert isinstance(postEdgeK_E_range, list), "Expected list for postEdge_E_range"
    elif function == 'pass' or function == 'constant':
        pass
    else:
        raise ValueError(f"The edge f{edge} wasn't defined.")
        
    # Background subtraction
    if function == 'pass' : # No background subtraction
        rho_kz=np.zeros_like(II0_kz)
        P_kz=II0_kz
    elif function == 'constant' : # Constant background
        fit_borders=[E_z[0],E_z[3]]
        evalE_range=np.logical_and(E_z<fit_borders[1],E_z>fit_borders[0])
        rho_z = np.nanmean(II0_kz[:,evalE_range],axis=1)
        P_kz=II0_kz/intercept_k[:,np.newaxis]-rho_z[:,np.newaxis]
        rho_kz = np.ones((n_k,n_z))*rho_z[:,np.newaxis]
    elif function == 'fermi' : # K-edge
        B_z = slope * (E_z-E_0) + 1
        preEdge_P_kz=II0_kz/intercept_k[:,np.newaxis]-B_z[np.newaxis,:]

        if isFlipped :
            preEdge_P_kz*=-1

        rho_kInf=(np.nanmean(preEdge_P_kz[:,(E_z>postEdgeK_E_range[0])&(E_z<postEdgeK_E_range[1])],axis=1))

        # TODO Add argument - clear outliers with negative post-edge are set to a positive value.
        negMask_k=rho_kInf<0
        slope_correction_k = (rho_kInf[negMask_k] / (np.mean(postEdgeK_E_range)-E_0))
        preEdge_P_kz[negMask_k,:]-=slope_correction_k[:,np.newaxis] * (E_z-E_0)[np.newaxis,:]
        rho_kInf[negMask_k]=0

        rho_kz=rho_kInf[:,np.newaxis]/(1+np.exp((K_E0-E_z)/K_kT))[np.newaxis,:]
        P_kz=preEdge_P_kz-rho_kz
        
        rho_k = rho_kInf
    elif function == 'double-fermi' : # L-edge
        B_z = slope * (E_z-E_0) + 1
        preEdge_P_kz=II0_kz/intercept_k[:,np.newaxis]-B_z[np.newaxis,:]

        if isFlipped :
            preEdge_P_kz*=-1

        rho_kL3=(np.nanmean(preEdge_P_kz[:,(E_z>postEdgeL3_E_range[0])&(E_z<postEdgeL3_E_range[1])],axis=1))/ratioL3onL2

        # TODO Add argument - clear outliers with negative post-edge are set to a positive value.        
        negMask_k=rho_kL3<0
        slope_correction_k = rho_kL3[negMask_k] / (np.mean(postEdgeL3_E_range)-E_0)
        preEdge_P_kz[negMask_k,:]-=slope_correction_k[:,np.newaxis] * (E_z-E_0)[np.newaxis,:]
        rho_kL3[negMask_k]=0#removal_fraction*np.nanmean(preEdge_P_kz[negMask_k,(E_z>postEdgeL2_E_range[0])&(E_z<postEdgeL2_E_range[1])],axis=1)

        rho_kL2=np.nanmean(preEdge_P_kz[:,(E_z>postEdgeL2_E_range[0])&(E_z<postEdgeL2_E_range[1])],axis=1)
        rho_kInf=(rho_kL3*1.5 + rho_kL2)/2
        theoric_rho_kL3=rho_kInf*(2/3)
        theoric_rho_kL2=rho_kInf*(1/3)
        
        #TODO maybe removal fraction could affect here.
        rho_kz=theoric_rho_kL3[:,np.newaxis]/(1+np.exp((L3_E0-E_z[np.newaxis,:])/L3_kT))+theoric_rho_kL2[:,np.newaxis]/(1+np.exp((L2_E0-E_z[np.newaxis,:])/L2_kT))
        P_kz=preEdge_P_kz-rho_kz
        
        rho_k = rho_kInf
    elif function == 'fermi_C_Ka' : # Single edge function based on post edge for C K-edge
        B_z = slope * (E_z-E_0) + 1
        preEdge_P_kz=II0_kz/intercept_k[:,np.newaxis]/B_z[np.newaxis,:]
        
        if isFlipped :
            preEdge_P_kz*=-1
        
        rho_kinf=np.nanmean(preEdge_P_kz[:,(E_z>postEdgeK_E_range[0])&(E_z<postEdgeK_E_range[1])],axis=1)
        rho_kz=rho_kinf[:,np.newaxis]+(1-rho_kinf)[:,np.newaxis]/(1+np.exp((K_E0-E_z)/K_kT))[np.newaxis,:]
        P_kz=preEdge_P_kz-rho_kz+0.3
        
        rho_k = rho_kz[:,0]
    
    return P_kz,rho_kz,rho_k

def integrate_spectrum(E_z: np.ndarray,P_kz:np.ndarray,params:pd.DataFrame,edge:str,E_0=0)->Tuple[np.ndarray,List[str]]:
    """
    remove background and calculate integrals over specified energy ranges. 
    The normalization and integration ranges are determined based on the "edge" argument.

    Parameters:
        
    E_z (np.ndarray): The energy values.
    
    P_kz (np.ndarray): The PEEM spectrum.
    
    params (pd.DataFrame): The arguments for the fit.
    
    edge (str): The edge to use for normalization and integration.
    
    E_0 (np.ndarray): The pre-edge energy.    
    
    Returns:
    Tuple[np.ndarray, np.ndarray, List[Union[np.ndarray, List[str]]]]: 
        The energy values, the normalized PEEM spectrum, and a list containing 
        the integrals and their names.
    """
    revert = params.get('Revert spectra',False)
    integ_E_ranges_arg = ast.literal_eval(params.get('int_E_ranges'))
    assert isinstance(integ_E_ranges_arg, list), "Expected list for int_E_range_list"
    integ_E_ranges = [np.logical_or.reduce([(E_z >= a) & (E_z <= b) for a, b in (r if isinstance(r[0], list) else [r])])
         for r in integ_E_ranges_arg]
    integ_names = ast.literal_eval(params.get('int_names'))
    integ_types = ast.literal_eval(params.get('int_types'))
    assert isinstance(integ_names, list), "Expected list for int_names"
    assert isinstance(integ_types, list), "Expected list for int_types"

    # Integration
    n_integ=len(integ_E_ranges)
    n_centroids = integ_types.count("peak")
    i_centroid=0
    n_r=n_integ+n_centroids
    (n_k,n_z)=P_kz.shape
    calculationValues_kr=np.zeros((n_k,n_r))
    calculations_names=[]
    calculations_bounds=[]
    # n_centroids=0
    filteredP_kz=gaussian_filter(utils.median_filter3(P_kz,2),1,axes=1)
    for j in range(0,n_integ):
        label_j=integ_names[j]
        filteredPj_kz=filteredP_kz[:,integ_E_ranges[j]]
        Ej_z=E_z[integ_E_ranges[j]]-E_0
        
        # Calculate integral
        if len(Ej_z) < 2 or filteredPj_kz.shape[1] < 2:
            integ_jk=np.zeros(n_k)
            centroid_jk=Ej_z[0]*np.ones(n_k)
        else:
            integ_jk=np.trapz(filteredPj_kz,x=Ej_z)
        
        # Store the integral in arrays
        calculationValues_kr[:,j+i_centroid]=integ_jk
        calculations_names.append(label_j)
        calculations_bounds.append([None,None])
        
        # If this integral is a single peak
        if integ_types[j]=="peak":   
            # Calculate the centroid of the peak (eV)
            i_centroid+=1
            firstMoment_j_k=np.trapz(Ej_z[np.newaxis,:]*filteredPj_kz,x=Ej_z,axis=1)
            centroid_jk=firstMoment_j_k/integ_jk+E_0
            
            # Store the centroid in arrays
            calculationValues_kr[:,j+i_centroid]=centroid_jk
            integ_name_j=label_j
            if "Integral" not in integ_name_j:
                raise ValueError(f"The string {label_j} to label the integral does not contain 'Integral'")
            name_centroid=integ_name_j.replace('Integral','Centroid')
            calculations_names.append(name_centroid)
            calculations_bounds.append([Ej_z[0]+E_0,Ej_z[-1]+E_0])
            
    # Revert pixels with negative integrals
    maskPositive_k=calculationValues_kr[:,0] < 0
    if revert :
        P_kz[maskPositive_k,:] *= -1
        calculationValues_kr[maskPositive_k,:] *= -1
    else :
        calculationValues_kr[maskPositive_k,:] = 0

    return calculationValues_kr,calculations_names,calculations_bounds

def rescale_Estack(P_kz: np.ndarray, scale_k: np.ndarray, E_z: np.ndarray) -> np.ndarray:
    """
    Rescale each spectrum of the P_xyz energy stack based on their integral.

    Parameters:
    P_kz (np.ndarray): The 2D numpy array representing a stack of images after processing.
    scale_k (np.ndarray): A 1D numpy array with the integral/background of each spectrum.
    E_z (np.ndarray): A 1D numpy array with the energy for each image.

    Returns:
    np.ndarray: The rescaled stack of images.
    """
    
    print(np.sum((scale_k==0).astype(int)))

    D_kz = P_kz / scale_k[:,np.newaxis]

    _,meanD_z=utils.scatter_mean_stack(E_z,D_kz,"Scaled",None,norm=-1,plot=False)[:2]
    
    # Rescale to fit in a the E stack intensity in a reasonable range.
    # (arbitrary chosen to fit in 8-bit range of intensity 0-255).
    max_meanD=255/abs(max(meanD_z))
    variationScale = 10 # magic parameter fixing the final intensity [a.u.].
    D_kz *= max_meanD / variationScale
    D_kz[D_kz >255] = 0
    D_kz[D_kz <-50] = 0

    # Save in 32 bit precision.
    return D_kz.astype(np.float32)


def test_PEEM_rescale_spectra():
    directory=os.getcwd()
    # Load the stack and integral from files
    stackfolder=utils.path_join(directory,'Test/PEEM/Ni-edge-spectrum/1_Ni_undistrdd_median_filter')
    stack = utils.open_sequence(stackfolder,images_format='')[0]
    integralfolder=utils.path_join(directory,'Test/PEEM/Ni-edge-spectrum/Ni_La/Integral',dt='f')
    integral = np.expand_dims(utils.open_image(integralfolder),2)

    # Call the function
    rescaled_stack = rescale_Estack(stack, integral)

    # Save the result of the test.
    n=np.shape(stack)[2]
    directory=utils.path_join(directory,'Test/PEEM/Ni-edge-spectrum')
    folder='1_Ni_undistrdd_median_filter_adj'
    filenames=[]
    for i in range(0,n):
        xtra_zeros=3-len(str(i))
        filenames.append('_Ni_adjusted'+"0"*xtra_zeros+str(i))
    utils.save_stack_as_image_sequence(n,rescaled_stack,directory,folder,filenames,save_stack=True)

def Kalman_filter_Estack(stk_kz):
    (n_k,n_z)=np.shape(stk_kz)
    
    # Kalman filter parameters
    A = 1  # State transition matrix
    H = 5  # Observation matrix
    Q = 1e-1  # Process noise covariance
    R = 1  # Measurement noise covariance
    state_k = np.zeros(n_k)  # Initial state estimate
    P = np.ones(n_k) * 1e3  # Initial covariance estimate
    
    # Apply Kalman filter
    Kalman_stk_kz = np.zeros((n_k,n_z))
    for z in np.arange(0,n_z):
        # Prediction step
        state_z = A * state_k
        P = A * P * A + Q
    
        # Update step
        K = P * H / (H * P * H + R)  # Kalman gain
        state_z = state_z + K * (stk_kz[:,z] - H * state_k)
        P = (1 - K * H) * P
    
        Kalman_stk_kz[:,z] = state_z
    
    return Kalman_stk_kz 
    
def sg3d_filter_Estack(stk_xyz):
    
    """ 3D Savitzky-Golay -> high frequency noise.""" 
    sg3d_stk_xyz=filter3D(np.array(stk_xyz),5,3)
    sg3d_stk_xyz=filter3D(np.array(sg3d_stk_xyz),5,3)
    sg3d_stk_xyz=filter3D(np.array(sg3d_stk_xyz),5,3)
    sg3d_stk_xyz=filter3D(np.array(sg3d_stk_xyz),5,3)
    sg3d_stk_xyz=filter3D(np.array(sg3d_stk_xyz),5,3)
    
    return sg3d_stk_xyz 
    




if __name__ == '__main__':    
    # test_PEEM_rescale_spectra()
    print(os.getcwd)