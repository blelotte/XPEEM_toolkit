# -*- coding: utf-8 -*-
"""
Module XPEEM

Contains the main functions to process XPEEM module.

Useful commands:
    > To check what Python is being executed:
    print(sys.executable)
    > To activate the virtual environent in the terminal.
    C:/Users/lelotte_b/Modules/myenv/Scripts/activate
    > To install a missing module
    pip install ...

Created on Thu Jul 10 14:25:50 2025

@author: lelotte_b

"""

# Basics
import re
import numpy as np
import pandas as pd
import time


# Display in python
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_context('paper', font_scale=2.2) # to change the font settings in the graphs
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina') # High resolution plots

# Handle excel inputs and outputs

# Docstrings, test and warnings
from typing import List, Optional, Tuple
np.seterr(all='warn') # to be able to handle warnings

# Folder/system management
import psutil
import sys
sys.path.append('D:/Documents/a PSI/Data/Data analysis/spyder/Modules')
sys.path.append('D:/Documents/a PSI/Data/Data analysis/spyder/Modules/savitzkygolay-master')
import os
import cv2
from PIL import Image

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import scipy.stats as stats
from scipy.stats import rankdata
import hdbscan
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

def is_program_running(program_name):
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == program_name:
            return True
    return False

# Handle image inputs and outputs
import imageio as io # Unified I/O for images, volumes, GIFs, videos 
from tifffile import TiffFile, imwrite

# Statistics
from scipy.optimize import minimize

# Image processing
from scipy.ndimage import generic_filter
from savitzkygolay import filter3D

# My modules
import XPEEM_image2E as E2
import XPEEM_Estack as ES
import XPEEM_utils as utils
import OriginPlot as oplt
import mantis_xray.mantis as mt

# TODO Simplify version, transformation.
def align_images(SampleDirectory, reference_image_name, sample,params="1;10;10;0.5;Coarse;0.1;2000;0", rectify=False):
    assert isinstance(reference_image_name,list) and len(reference_image_name)==2
    basedir=os.getcwd()
    # >> Check if a pre-calibrated transform was given
    # 7 = classic, 8 with transform
    trsf_path = utils.path_join(SampleDirectory,'Transformation','Transformation.txt',dt='f')
    if os.path.exists(trsf_path):
        ve=8
    else:
        ve=7
        trsf_path  = "NONE"
        print("Using non-linear registration. Sometimes easier to use Non-linear registration followed by alignement with SIFT.")
    
    if reference_image_name[0] :
        if int(params[0])==0 or rectify :
            target_image1edge = utils.path_join(basedir, f'_Input/refImages/{sample}/Raw_undistrddEdge',reference_image_name[0]+'.tif',dt='f')
            assert os.path.exists(target_image1edge)
        elif int(params[0])==1  or int(params[0])==2:
            target_image1edge = utils.path_join(basedir, f'_Input/refImages/{sample}/Raw_undistrdd',reference_image_name[0]+'.tif',dt='f')
            assert os.path.exists(target_image1edge)
        params1 = params + ',' + target_image1edge
        if ve==8:
            params1 = params1 + ',' + trsf_path
        
        # >> Align the spectrum
        spectrumfolder='1'
        path1=utils.path_join(SampleDirectory,'1',dt='d')
        if os.path.exists(path1):
            print(spectrumfolder)
            assert os.path.exists(path1)
            if rectify :
                drtrsfpath=utils.path_join(SampleDirectory,'1_drtrsf',dt='d')
                drtrsfpath=E2.smoothtrsf(drtrsfpath)
                E2.IJ_applytrsf(path1, drtrsfpath, trsf_path,target_image1edge)
            elif not rectify :
                E2.IJ_bUnwarpJ(target_image1edge, path1, version = ve,args = params1)
                
    
    
    # >> Align the 2 energies
    if reference_image_name[1] :
        # Get the names of all folder in 2_energies that contain '_aligned'
        twoenergiesfolder=utils.path_join(SampleDirectory,'2_energies',dt='d')
        TwoEnergies = [f for f in os.listdir(twoenergiesfolder) if (not('_aligned' in f) and not('--' in f) and not('DIV' in f) and not ('_undistrdd' in f) and not ('_indrtrsf' in f) and not ('_drtrsf' in f) and not ('_filtered' in f) and not '.tif' in f )] 
        
        if int(params[0])==0 or rectify:
            target_image2edge = utils.path_join(basedir,f'_Input/refImages/{sample}/2E_undistrddEdge',reference_image_name[1]+'.tif',dt='f')
        if int(params[0])==1 or int(params[0])==2:
            target_image2edge = utils.path_join(basedir,f'_Input/refImages/{sample}/2E_undistrdd',reference_image_name[1]+'.tif',dt='f')


        params2 = params + ','+ target_image2edge
        if ve==8:
            params2 = params2 + ',' + trsf_path
        # assert os.path.exists(target_image2)
        assert os.path.exists(target_image2edge)
        for folder in TwoEnergies:
            # basetarget=target_image2
            path2E = utils.path_join(twoenergiesfolder,folder,dt='d')
            print(folder)
            assert os.path.exists(path2E)
            if rectify :
                drtrsfpath=utils.path_join(twoenergiesfolder,folder.rstrip('/')+'_drtrsf',dt='d')
                drtrsfpath=E2.smoothtrsf(drtrsfpath)
                E2.IJ_applytrsf(path2E, drtrsfpath, trsf_path,target_image2edge)
            elif not rectify :
                E2.IJ_bUnwarpJ(target_image2edge, path2E, version = ve,args = params2)

# TODO Move to utils, merge with broadMask, testMask
def load_masks(sample_name: str, sheet_name: Optional[str] = 'maskParams', loadAll_bool=False, kind=None) -> Tuple[str, List[str], List[pd.DataFrame]]:
    """
    This function opens a list of masks saved in the experiment/_Input folder.
    It extracts the list of the masks as images, their names and their corresponding
    ROIs. 
    It checks whether the column "sample" in excel matches the argument sample.

    Parameters:
        
    sample (str): The sample, i.e. "Uncy".
    
    sheet_name (str, optional): The name of the sheet in the 'arguments.xlsx' file. 
    Defaults to 'maskParams'.
    
    loadAll_bool (bool, optional): Set the argument "load" in the sheet.

    Returns:
    
    Tuple[str, List[str], List[str]]: A tuple containing the path of the sample 
    folder, the short name for each mask, the labels for each mask, and the list 
    of masks.
    """
    
    assert len(sample_name) < 5, 'The sample name must be shorter than 4 characters.'  
    assert isinstance(sheet_name,str), 'The sheet name should be a string.'
    assert isinstance(loadAll_bool,bool), ''
      
    inputFd_path=utils.path_join(os.getcwd(),'_Input',dt='d')
    
    # read excel file
    argFile_path=utils.path_join(inputFd_path,'arguments.xlsx',dt='f')
    sheetMasks_df = pd.read_excel(argFile_path,sheet_name=sheet_name)
    
    # create empty list for image names
    masks_list = []
    ROIs_list = []
    legends_list = []
    
    # loop through each row (except header) and load image
    for index, row in sheetMasks_df.iterrows():
        if (row['Sample'] == sample_name) and ((row['Load'] or loadAll_bool) and (row['Kind']==kind or kind is None)):
            masks_path = utils.path_join(inputFd_path,'masks',sample_name,row['Material'],row['Mask'],dt='f')
            masks_list.append(utils.open_image(masks_path))
            maskROIs_name = row['ROI_name']
            maskLegends_name = row['Legend']
            ROIs_list.append(maskROIs_name)
            legends_list.append(maskLegends_name)
        
    return ROIs_list , legends_list, masks_list

# TODO set-up test
def process_2E(directory: str, file, IDs: Tuple[int, int, int], dim: Tuple[int, int, int, int], save_stack: bool = False,bounds=[0,10],suffix='undistrdd',std_bkg=True, shift=False,scale=[None,None]) -> None:
    """
    This function opens two sequences of PEEM micrographs, i.e. Energy 1, Energy 2.
    
    It open the stacks and applies a median filter to each micrographs.
    It then calculates the average of the stack and saves.
    It also calculates the ratio of the first two images.

    Parameters:
    directory (str): The directory where the image files are located.
    folder (List[str]): The folders where the image files are located.
    file (List[str]): The names of the image files.
    format_file (str): The format of the image files.
    filename_z (str): The name of the output file.
    dim (Tuple[int, int, int, int]): The dimensions of the images. Format: (nr_images, nr_files, x_dimension, y_dimension).
    save_stack (bool, optional): Whether to save the stack of images. Defaults to False.
    
    
    Returns:
    None
    """
    # Change format of bounds
    bounds = np.array(bounds)
    # >> Dimensions
    (p,q,n,m)=dim
    
    
    # >> Load E and I0.
    # Assuming 'directory' is your current directory
    sample_folder = os.path.dirname(os.path.dirname(directory))
    # Find the 'file_E'
    E, I0 = utils.load_E_I0(sample_folder,processed=True)
    
    # >> Initialize names
    folder = [file + ID +'_'+suffix+'/' for ID in IDs]
    
    # >> Load the images.
    stack=np.zeros((n,m,p,q+1))
    final=np.zeros((n,m,q+1))
    #open file 1
    for i in range(0,q) :
        if suffix == 'undistrdd':
            im_i,E_i,name_in_sequence=utils.open_sequence(utils.path_join(directory,folder[i],dt='d'),returnEnergy=True)
            E_i=np.round(np.nanmean(E_i,axis=0),decimals=1)
            k = ES.find_closest_index(E,E_i)
            E[i]=E_i
        elif suffix == 'filtered':
            im_i,name_in_sequence=utils.open_sequence(utils.path_join(directory,folder[i],dt='d'),returnEnergy=False)
            im_i=im_i[:,:,0:p]

        im_i,msk=utils.image_well_defined(im_i)
        im_i=np.transpose(im_i,(1,2,0))
        if suffix == 'undistrdd':
            stack[:,:,:,i]=msk[:,:,np.newaxis]*filter3D(im_i/I0[k],5,3)
        elif suffix == 'filtered':
            stack[:,:,:,i]=im_i
        final[:,:,i]=np.nanmean(stack[:,:,:,i],axis=2)
        
    totmsk=np.ones((n,m))
    for k in range(0,p):
        # Add an arbitratry small number to prevent division by 0
        imageB=stack[:,:,k,1]
        if suffix == 'filtered' and shift == True:
            imageB += 0.1

        # Build a mask of non-zero locations
        images,msk=utils.image_well_defined([stack[:,:,k,0],imageB],axis=0)
        [imageA,imageB]=images
        totmsk=totmsk * msk.astype(int)
        boolean = totmsk == 1
        
        # Calculate ratio
        ratio = np.where(boolean,imageA/imageB,0)
        stack[:,:,k,2]=ratio
        
    ratios=stack[:,:,:,2]
    
    # Calculate image background
    if suffix == 'undistrdd':
        if std_bkg :
            bkg=1
        else:
            bkg=max(np.percentile(ratios[boolean], [1])[0],bounds[0])
        bounds[0]=bkg
    elif suffix == 'filtered':
        bkg=0
    print("Background:"+ str(np.percentile(ratios[boolean], [1])))
        
    # Remove image background
    stack[:,:,:,2]=np.where((ratios>bounds[0]) & (ratios < bounds[1]) & (boolean)[:,:,np.newaxis],ratios-bkg,0)
    final[:,:,2]=np.nanmean(stack[:,:,:,2],axis=2)

    fig, ax = plt.subplots(2, 1, figsize=(5, 10))
    im=ax[0].imshow(final[:,:,0])
    fig.colorbar(im, ax=ax[0], orientation='vertical')

    p1, p99 = np.percentile(final[:,:,0], [1, 99])
    hist, bins = np.histogram(final[:,:,0].flatten(), bins=1000, range=(p1+0.001, p99))
    width = (bins[1] - bins[0])
    ax[1].bar(bins[:-1], hist, align='center', width=width, color='r')
    if scale[0] and scale[1]: 
        ax[1].set_xlim(0,scale[0])        
        ax[1].set_ylim(0,scale[1])        
    if suffix == 'undistrdd':
        ax[0].set_title(str(E[0])+' [eV]')
    
    fig, ax = plt.subplots(2, 1, figsize=(5, 10))
    im=ax[0].imshow(final[:,:,1])
    fig.colorbar(im, ax=ax[0], orientation='vertical')
    p1, p99 = np.percentile(final[:,:,1], [1, 99])
    hist, bins = np.histogram(final[:,:,1].flatten(), bins=1000, range=(p1+0.001, p99))
    width = (bins[1] - bins[0])
    ax[1].bar(bins[:-1], hist, align='center', width=width, color='r')
    if scale[0] and scale[1]: 
        ax[1].set_xlim(0,scale[0])        
        ax[1].set_ylim(0,scale[1])        
    if suffix == 'undistrdd':
        ax[0].set_title(str(E[1])+' [eV]')

    fig, ax = plt.subplots(2, 1, figsize=(5, 10))
    im=ax[0].imshow(final[:,:,2])
    fig.colorbar(im, ax=ax[0], orientation='vertical')
    hist, bins = np.histogram(final[:,:,2].flatten(), bins=1000, range=(bounds[0]-bkg+0.001, bounds[1]-bkg))
    width = (bins[1] - bins[0])
    ax[1].bar(bins[:-1], hist, align='center', width=width, color='r')

    if suffix == 'undistrdd':
        bkg=np.round(bkg,decimals=2)
        ax[0].set_title(str(E[0])+' / '+str(E[1])+f' - {bkg} [a.u.]')
    
    # Save the comparison images
    if suffix == 'undistrdd':
        OutputName=[IDs[0]+"_"+str(E[0]).replace('.','_'),IDs[1]+"_"+str(E[1]).replace('.','_'),str(IDs[2])+'_'+str(IDs[0])+'_on_'+str(IDs[1])+'_filtered']
        utils.save_stack_as_image_sequence(q+1,final,directory,OutputName,save_stack)
    elif suffix == 'filtered':
        utils.save_stack_as_image_sequence(1,final[:,:,2:3],directory,[IDs[2]],save_stack)

        
    # Save the comparison stack
    if suffix == 'undistrdd':
        outputfolder=[file + ID +'_filtered/' for ID in IDs]
        utils.save_stack_as_image_sequence(p,stack[:,:,:,0],utils.path_join(directory,outputfolder[0]),name_in_sequence,save_stack)
        utils.save_stack_as_image_sequence(p,stack[:,:,:,1],utils.path_join(directory,outputfolder[1]),name_in_sequence,save_stack)
        utils.save_stack_as_image_sequence(p,stack[:,:,:,2],utils.path_join(directory,outputfolder[2]),name_in_sequence,save_stack)
    elif suffix == 'filtered':
        outputfolder=IDs[2]+'_filtered/'
        utils.save_stack_as_image_sequence(p,stack[:,:,:,2],utils.path_join(directory,outputfolder),name_in_sequence,save_stack)

def find_Eshift(edgeFd_path: str, stackFd_name: str, calcMask_xy: np.ndarray, name: str, peakE_range=[None, None], fitE_range=[3, 3], peakE=851.3) -> None:
    """
    Calculates the energy drift in PEEM image stacks by fitting a 2D plane to the 
    image stack and determining the drift relative to a peak energy.
    
    Parameters:
    ----------
    edgeFd_path : str
        The directory where the IxyE files are located.
    stackFd_name : str
        The folder name containing the image stack files.
    calcMask_xy : np.ndarray (n x m)
        A binary mask array indicating the valid regions (pixels) to include in the drift fitting.
    name : str
        The base name for saving output images and drift data.
    peakE_range : list of two elements [float or None, float or None], optional
        Energy range in which to calculate the maximum (expressed in eV). Defaults to [None, None], meaning full range.
    fitE_range : list of two integers, optional
        A further restriction to the range (expressed in eV), which will be used to calculate the final fit. Defaults to [3, 3].
    peakE : float, optional
        Expected peak energy position (in eV) used to determine drift reference. Default is 851.3 eV (Ni2+, Ni L-edge).

    Returns:
    -------
    None
        The function saves output files (images, drift data) but does not return a value.
    """
    stackFd_path=utils.path_join(edgeFd_path,stackFd_name)
    if not os.path.isdir(stackFd_path): raise ValueError('The directory does not exist:'+stackFd_path)    

    #Load raw images Estack in ndarray.
    (n_x,n_y,n_z),E_z,I0_z,I_xyz,_=ES.load_Estack(stackFd_path)
    
    # Bound the Estack to the supplied energy range for the peak.
    boundI_xyz = I_xyz
    if peakE_range[0] is not None : boundI_xyz = boundI_xyz[:,:,ES.find_closest_index(E_z, peakE_range[0]):]
    if peakE_range[1] is not None : boundI_xyz = boundI_xyz[:,:,:ES.find_closest_index(E_z, peakE_range[1])]
    
    # Find the slice (z) of the maximum and get the corresponding energy of the maximum
    boolCalcMask_xy = calcMask_xy==255
    maxI_z_xy = np.where(boolCalcMask_xy,np.argmax(boundI_xyz, axis=2),0)
    maxI_E_xy=E_z[maxI_z_xy]
    maxI_E_xy[(maxI_E_xy>(peakE+fitE_range[1])) | (maxI_E_xy<(peakE-fitE_range[0]))]=np.nan
    
    def nanMeanFilter(values):
        """ Carry the mean ignoring nan values, allowing to handle masks """
        # Reshape the flat array to a 2D window
        window = values.reshape((window_size, window_size))
        return np.nanmean(window)
    
    # Apply a mean filter (which decreases the quantization)
    window_size=3
    maxI_E_xy = generic_filter(maxI_E_xy,nanMeanFilter,size=(window_size, window_size),mode='wrap')


    def nanResiduals(params, xi, yi, zi, maski, mean = True):
        """ Calculate the residuals to the fitted plane ignoring nan values, allowing to handle masks """
        residuals_xy = np.abs(maski * (zi - (params[0] * xi + params[1] * yi + params[2]))).astype(float)
        if mean :
            return np.nanmean(np.nanmean(residuals_xy))
        else: 
            return residuals_xy

    def fitEDispersionPlane(I_xy, boolMask_xy):
        """ Fit a plane by least squarre based on data points in I_xy"""
        x, y = np.meshgrid(np.arange(I_xy.shape[1]), np.arange(I_xy.shape[0]))
        params_init = [1, 1, 1]
        fit = minimize(nanResiduals, params_init, args=(x, y, I_xy, boolMask_xy))
        return x,y,fit.x
    
    # Fit the energy dispersion plane energy of the maximum data in the image
    x,y,planeParams = fitEDispersionPlane(maxI_E_xy, boolCalcMask_xy)
    
    # Calculate the fitted plane and print the parameters
    fitted_E_xy = planeParams[0] * x + planeParams[1] * y + planeParams[2]
    print('Slope in x='+str(np.round(planeParams[0],decimals=10))+'eV/pixel')
    print('Slope in y='+str(np.round(planeParams[1],decimals=10))+'eV/pixel')
    print('Intercept='+str(np.round(planeParams[2],decimals=3))+'eV')
    
    # Substract the center value to the plane fit to get a relative energy shift.
    center_E = planeParams[0] * n_x/2 + planeParams[1] * n_y/2 + planeParams[2]
    dE_xy = fitted_E_xy - center_E
        
    
    # Calculate/print the mean energy, sum of square, sum of square for the residuals and R-squared value.
    mean_E=np.nanmean(maxI_E_xy)
    print('Mean energy:'+str(np.round(mean_E,decimals=2))+'eV')
    
    stat_dE_xy = maxI_E_xy - mean_E
    sumSquareTot = np.sum(stat_dE_xy[~np.isnan(stat_dE_xy)]**2)
    print('Sum of square total:'+str(np.round(sumSquareTot,decimals=2)))
    
    residuals = nanResiduals(planeParams, x, y, maxI_E_xy, boolCalcMask_xy, mean = False)
    sumSquareResidual = np.sum(residuals[~np.isnan(residuals)]**2)
    print('Sum of square for the residuals:'+str(np.round(sumSquareResidual,decimals=3)))
    
    Rsqurd=1-sumSquareResidual/sumSquareTot
    print('R-squared:'+str(np.round(Rsqurd,decimals=3)))
    
    # Plot the fitted plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, dE_xy, color='b', alpha=0.5)
    data_dE_xy = np.where(boolCalcMask_xy,maxI_E_xy-planeParams[2],np.nan)
    ax.scatter(x,y,data_dE_xy)
    plt.show()

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.join(edgeFd_path, 'E-shift'), exist_ok=True)
        
    # Save the the energies of the maximum (raw data) to an image
    maxI_E_xy=np.where(boolCalcMask_xy,maxI_E_xy,np.nan)
    im1 = Image.fromarray(maxI_E_xy.astype(np.float32))
    im1.save(utils.path_join(edgeFd_path, 'E-shift', name+"-raw-maxI_E_xy.tif",dt='f'))
    
    # To store with good precision in integer 32bit, multiply by 1E6 
    fitStatExport_dE_xy = 1e6*(dE_xy)
    im2 = Image.fromarray(fitStatExport_dE_xy.astype(np.int32))  
    im2.save(utils.path_join(edgeFd_path, 'E-shift', name+"-fit_dE_xy-times_1E6.tif",dt='f'))



def process_Estack(edgeFd_path:str, stackFd_name:str, save_stack:bool=False, 
                          test: Optional[bool]=False,verbiose: Optional[bool]=False,calib_dEonE: Optional[float]=0, 
                          dEAB: Optional[list]=[0.3,5,1],E_range=None,mantis_name='Mantis',preEdge_E_range=None) -> None:
    """
    This function process a stack of PEEM images from the edge folder, and applies a series of processing steps to 
    each image, and then saves the processed images to a new directory. 
    It also calculates and saves total integrated intensity, background height and peak centroid.

    Parameters:
    edgeFd_path (str): The directory where the experiment is located.
    stackFd_name (str): The folder where the stack is located.
    filename (str): The name of the image file.
    save_stack (bool, optional): Whether to save the stack of images. Defaults to False.
    test (bool, optional): Whether or not to speed up the processing by only computing a fraction of the points.
    verbiose (bool, optional): Whether or not to plot in the interactive window.
    calib_dEonE (float, optional): dE/E to correct the energy (determined from a calibration with Origin).
    dEAB (list, optional): For C K-edge, energy shift (dE) and linear coefficient (A*I0+B) for adjusting the gold I0.
    E_range (list, optional): A narrower energy range (in eV) to be used for Mantis. Example use: analyze the pre-edge features of the O K-edge.
    mantis_name (string, optional): A name added to the Mantis output for further reference. Example: NiUncy, test, ...
    preEdge_E_range (list, optional): An energy range (in eV) to be used to compute the pre-edge linear fit.

    Returns:
    None
    
    Note:
        The structure of the folder is assumed to be
        "../scriptFd/sampleFd/edgeFd/stackFd" (containing the image sequence to be processed)
        
        In ../scriptFd/sampleFd/edgeFd/, one can also provide 
            1. An energy dispersion plane in a folder "E-shift", with a name containing "-fit_dE_xy-times_1E6.tif".
            2. Masks in a folder "Masks", named 
                a. "testMask.tif" - to speed-up the processing by only analysing one region.
                b. "broadMask.tif" - to restrict the processing to a region.
                c. "calcMask.tif" - to restrict the pre-edge fit to a region.
            3. A different I0 used for normalization in a folder "Gold", as a txt file.
            4. A calibration weight image for the I0 gold as an image in a folder "Gold", with a name as "Intensity_gold.tif".
            5. A calibration weight image for the density E-stack as an image in a folder "weights", with a name as "weights.tif". (for example if higher quality 2E images estimate is available).
    """
    edgeFd_name=os.path.basename(os.path.normpath(edgeFd_path))
    stackFd_path=utils.path_join(edgeFd_path,stackFd_name)
    if not os.path.isdir(stackFd_path): raise ValueError('The directory does not exist:'+stackFd_path)    
        
    """
    >> Opening files and pre-processing.
    ________________________________________________________________________"""    
    # Load/define raw E-stack image sequence, energies and I0, edge name, dimensions. 
    (n_x,n_y,n_z),E_z,I0_z,I_xyz,ImNames=ES.load_Estack(edgeFd_path,stackFd_path)

    edge=utils.find_edge(utils.path_join(os.path.basename(os.path.dirname(edgeFd_path)).replace('/',''),stackFd_name,dt='d'),exp='SIM')

    # Pre-process the E-stack: average dupplicates and remove infinite/nan values.
    I_xyz, minMask_xy = utils.image_well_defined2(I_xyz)
    minMask_xy=minMask_xy==1
    I_xyz=np.where(minMask_xy[:,:,np.newaxis],I_xyz,0)
        
    # Set potential speckle/dead pixels to the local median value
    # (assuming the E-stack is relatively smooth).
    threshold_factor=3
    despeckled_I_xyz = utils.median_filter3(I_xyz, nr_pixel=4)
    speckles_xyz = I_xyz > threshold_factor * despeckled_I_xyz
    I_xyz[speckles_xyz]=despeckled_I_xyz[speckles_xyz]
    
    # Pre-process the energy: apply calibration
    E_z=E_z*(1+calib_dEonE)
        
    # In-line plot of the raw data
    ax,meanI_z,stdI_z,_=utils.scatter_mean_stack(E_z,I_xyz,"Raw stack",None,norm=1)
    
    """
    >> Open masks.
    ________________________________________________________________________"""    
    # Check if the broad-mask directory exists
    masksFd_path=utils.path_join(edgeFd_path, 'Masks')
    if os.path.exists(masksFd_path):
        # Get a list of all files in the directory
        mask_files = os.listdir(masksFd_path)
    
        # Filter the list to only include .tif files
        tif_files = [f for f in mask_files if f.endswith('.tif') and 'broadMask' in f]
    
        # If there is exactly one .tif file
        if len(tif_files) == 1 and not test:
            # Load the .tif file
            mask_image = Image.open(utils.path_join(masksFd_path, tif_files[0],dt='f'))
            
            # Check that the array only has 0 and 255 as values.
            uniqueValues_list = np.unique(mask_image)
            assert np.array_equal(uniqueValues_list, [0, 255]) or np.array_equal(uniqueValues_list, [255]), "Broad mask contains values other than 0 and 255."
    
            # Convert the image data to a numpy array
            mask_xy = np.array(mask_image)/255
            mask_xy = (mask_xy*minMask_xy)==1
        elif not test:
            raise ValueError('There is not exactly one broadmask.tif file in the directory.')
        
        # Filter the list to only include .tif files
        tif_files = [f for f in mask_files if f.endswith('.tif') and 'testMask' in f]
    
        # If there is exactly one .tif file
        if len(tif_files) == 1 and test==True:
            # Load the .tif file
            testMask_image = Image.open(os.path.join(masksFd_path, tif_files[0]))
            
            # Check that the array only has 0 and 255 as values.
            uniqueValues_list = np.unique(testMask_image)
            assert np.array_equal(uniqueValues_list, [0, 255]) or np.array_equal(uniqueValues_list, [255]), "Test mask contains values other than 0 and 255."
    
            # Convert the image data to a numpy array
            mask_xy = np.array(testMask_image)/255
            mask_xy = (mask_xy*minMask_xy)==1
        elif len(tif_files) > 1:
            raise ValueError('There is not exactly one testMask.tif file in the directory.')
        
        # Filter the list to only include .tif files
        tif_files = [f for f in mask_files if f.endswith('.tif') and 'preEdgeMask' in f]
    
        # If there is exactly one .tif file
        if len(tif_files) == 1:
            # Load the .tif file
            preEdgeMask_image = Image.open(os.path.join(masksFd_path, tif_files[0]))
            
            # Check that the array only has 0 and 255 as values.
            uniqueValues_list = np.unique(preEdgeMask_image)
            assert np.array_equal(uniqueValues_list, [0, 255]) or np.array_equal(uniqueValues_list, [255]), "Pre-edge mask contains values other than 0 and 255."
    
            # Convert the image data to a numpy array
            preEdgeMask_xy = np.array(preEdgeMask_image)/255
            preEdgeMask_xy = (preEdgeMask_xy*minMask_xy)==1
            preEdgeMask_k=preEdgeMask_xy[mask_xy]
        elif len(tif_files) > 1:
            raise ValueError('There is not exactly one preEdgeMask.tif file in the directory.')
        else:
            preEdgeMask_k=None
            
    else:
        mask_xy=minMask_xy
        preEdgeMask_xy=None
        print('The folder "Mask" in the edge folder could not be found.')
        
    I_kz=utils.flatten_Estack(I_xyz,mask_xy)
    n_k=I_kz.shape[0]
    
    """
    >> Divide by C K edge measured in XAS or in PEEM for the gold or by the I0 beam current.
    ________________________________________________________________________"""
    goldFd_path=utils.path_join(edgeFd_path, 'Gold',dt='d')
    if os.path.exists(goldFd_path) :
        AuRef_files = [file for file in os.listdir(goldFd_path) if file.endswith('.txt')]
        AuRef_path=[utils.path_join(edgeFd_path, 'Gold', AuRef_file, dt='f') for AuRef_file in AuRef_files]

        dEAu=dEAB[0]
        I0_z=ES.average_goldSpectra(AuRef_path, E_z+dEAu, False)

        A = dEAB[1]
        B = dEAB[2]
        min_I0_z=np.nanmin(I0_z)
        norm_I0_z=A*(I0_z-min_I0_z)/(max(I0_z)-min_I0_z)+B
        
        weightI0_xy= np.array(Image.open(utils.path_join(goldFd_path,'Intensity_gold.tif',dt='f')))
        weightI0_k=weightI0_xy[mask_xy]
        weightI0_k = weightI0_k / np.nanmean(weightI0_k,axis=0)
        
        
        I0_kz=norm_I0_z[np.newaxis,:]*weightI0_k[:,np.newaxis]
        
        II0_kz=I_kz/I0_kz
        II0_kz=np.where(np.isinf(I_kz), 0, I_kz)
        
        I0_z=np.nanmean(I0_kz,axis=0)
    else:
        II0_kz=I_kz/I0_z[np.newaxis,:]
    
    ax,meanII0_z,stdII0_z,_=utils.scatter_mean_stack(E_z,II0_kz,"/I0 + masked",ax,norm=1)
    plt.show()

    """
    >> Load energy dispersion plane (if it was calculated with find_drift).
    ________________________________________________________________________"""
    EshiftFd_path = utils.path_join(edgeFd_path, 'E-shift')
    if os.path.exists(EshiftFd_path) :
        # List files in the directory
        Eshift_files = os.listdir(EshiftFd_path)
        fit_dE_k=None
        for Eshift_file in Eshift_files:
            # Check if the filename includes the typical suffix
            if '-fit_dE_xy-times_1E6.tif' in Eshift_file :
                fit_dE_xy = utils.open_image(utils.path_join(EshiftFd_path,Eshift_file,dt='f'),format_file='').astype(np.float32)/1e6 # eV
                fit_dE_k=fit_dE_xy[mask_xy]
                print(fit_dE_k.shape)
            
    else:
        raise ValueError("Could not find the E-shift folder.")

    """
    >> Energy correction and background subtraction, followed by calculation of the total integrated intensity and background height.
    ________________________________________________________________________"""
    if mantis_name=='I_I0' :     
        (P_kz,mean_rho_z,rho_k,B_k,(calc_arrays,calc_labels,calc_ranges))=(II0_kz,np.ones((n_z)),np.ones((n_k)),np.ones((n_k)),(np.ones((n_z,1)),['Integral'],[None,None]))
    else :
        
        # Get dimensions
        t0 = time.time()        
        
        # Energy shift correction
        if fit_dE_k is None :
            Algnd_II0_kz=II0_kz
        else:
            Algnd_II0_kz=ES.correct_Eshift(E_z,II0_kz,fit_dE_k)

        # Linear fit of pre-edge.
        bkgParams=ES.load_params_excels('bkgParams',edgeFd_name)
        (E_0,bkgSlope,B_k)=ES.fit_preEdge_spectrum(E_z,Algnd_II0_kz,bkgParams,edge,calcMask_k=preEdgeMask_k)
            
        # Substract the spectra pre-edge and post-edge backgrounds
        (P_kz,rho_kz,rho_k)=ES.bkg_substraction_spectrum(E_z,Algnd_II0_kz,bkgParams,edge,
                                                slope=bkgSlope,intercept_k=B_k,E_0=E_0)
    
        # Integrate the E-stack and relevant peaks, calculate peak centroids for L-edges.
        integParams=ES.load_params_excels('integParams',edgeFd_name)
        (calc_arrays,calc_labels,calc_ranges)=ES.integrate_spectrum(E_z, P_kz, integParams, edge,E_0=E_0)
            
        # Plot processed E-stack
        ax=utils.scatter_mean_stack(E_z,II0_kz,"/I0 + masked",None,norm=1)[0]
        ax=utils.scatter_mean_stack(E_z,Algnd_II0_kz,"E. aligned",ax,norm=1)[0]
        plt.show()
        
        # Plot background
        ax2,mean_rho_z=utils.scatter_mean_stack(E_z,bkgSlope*(E_z-E_0)[np.newaxis,:]+B_k[:,np.newaxis]+rho_kz,"Background",None,norm=1)[:2]
        plt.show()
                    
        t1 = time.time()
        print("Runtime Background subtraction: "+repr((t1-t0))+" s")

    # Reshape calculation images and background (k) -> (x,y)
    rho_xy=utils.reshape_Image(rho_k,mask_xy)
    B_xy=utils.reshape_Image(B_k,mask_xy)
    calc_xyl=utils.reshape_Estack(calc_arrays,mask_xy)

    n_l=np.shape(calc_xyl)[2]
    n_l2=n_l*3
    calcRBF_xyl2=np.zeros((n_x,n_y,n_l2))
    calcRBF_labels=[]
    for i in range(n_l):
        calcRBF_xyl2[:,:,3*i]=calc_xyl[:,:,i]
        calcRBF_xyl2[:,:,3*i+1],calcRBF_xyl2[:,:,3*i+2],calcRBF_label,calcMask_label=utils.interpolate_missing_values(calc_xyl[:,:,i],bounds=calc_ranges[i],norm=False,mask=mask_xy,label=calc_labels[i])
        calcRBF_labels.append(calc_labels[i])
        calcRBF_labels.append(calcRBF_label)
        calcRBF_labels.append(calcMask_label)

    """
    >> Eliminate obvious outliers in the background height and total integrated intensity
    based on two criteria: not too high values overall, and values of both should be similar.
    - Hypothesis: the two quantities are largely correlated, and keeps the smallest value of both.
    ________________________________________________________________________"""
    # Divide the total integrated intensity (tii) by its median value.
    tiiP_xy=calcRBF_xyl2[:,:,1]
    median_tiiP=np.median(tiiP_xy[mask_xy])
    norm_tiiP_xy=np.where(mask_xy,tiiP_xy/median_tiiP,0)
    
    # Divide the background height (rho) by its median value.
    median_rho_xy=np.median(rho_xy[mask_xy])
    norm_rho_xy=np.where(mask_xy,rho_xy/median_rho_xy,0)
    
    # Criteria one: values higher than three median of the normalized tiiP_xy or rho_xy are unlikely.
    plt.figure(figsize=(10, 6))
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    criteriaOneMask_xy=(norm_tiiP_xy < 3) & (norm_rho_xy < 3)
    plt.hexbin(norm_tiiP_xy[criteriaOneMask_xy], norm_rho_xy[criteriaOneMask_xy], gridsize=(500,500), cmap='inferno', bins='log')
    
    plt.xlabel('Total integrated intensity')
    plt.ylabel('Background height')
    plt.title('Regression and Data Points')
    plt.legend()
    plt.show()
    
    # Criteria two: for small background heights (rho_xy) values much larger than the integral, 
    # replace the rho_xy by the tiiP_xy.
    isPositive_tiiP_xy=tiiP_xy > 0
    criteriaTwoMask_xy = (norm_rho_xy > 3 * norm_tiiP_xy) & (norm_rho_xy < 0.3) & isPositive_tiiP_xy
    count_bad = np.sum(criteriaTwoMask_xy)
    thresholdCountBad=50 # pixels
    
    # If numerous point to remove, plot for vizual check.
    if count_bad > thresholdCountBad :
        
        plt.figure(figsize=(8, 6))
        plt.imshow(norm_rho_xy, cmap='viridis')
        plt.colorbar(label='Interpolated Values')
        plt.title('Interpolated Norm Background Image')
        plt.show()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(norm_tiiP_xy, cmap='viridis')
        plt.colorbar(label='Interpolated Values')
        plt.title('Interpolated Norm Integral Image')
        plt.show()
        
    norm_rho_xy[criteriaTwoMask_xy]=np.max(np.array([norm_rho_xy[criteriaTwoMask_xy],norm_tiiP_xy[criteriaTwoMask_xy]]),axis=0)    
        
    # Criteria three: the integral should not have values which are too high either.
    criteriaThreeMask_xy=(norm_tiiP_xy < 0.3 * norm_rho_xy) & isPositive_tiiP_xy
    norm_tiiP_xy[criteriaThreeMask_xy]=norm_rho_xy[criteriaThreeMask_xy]
    norm_tiiP_xy[norm_tiiP_xy<0.2]=0.2
    norm_tiiP_k=norm_tiiP_xy[mask_xy]

    norm_rho_xy,rhoMask_xy,_,_=utils.interpolate_missing_values(norm_rho_xy,mask=mask_xy,norm=False,label="Background_processed")
    
    """
    >> Load a different weight than the elemental contrast (norm_background) 
    from another file (if argument provided).
    ________________________________________________________________________"""
    
    weightsFd_path=utils.path_join(edgeFd_path, 'weights')
    if os.path.exists(weightsFd_path):
        files = os.listdir(weightsFd_path)
    
        # Filter the list to only include .tif files
        tif_files = [f for f in files if f.endswith('.tif') and 'weights' in f]
    
        # If there is exactly one .tif file
        if len(tif_files) == 1:
            # Load the .tif file
            weights_xy = np.array(Image.open(os.path.join(weightsFd_path, tif_files[0])))    
            weight_k=weights_xy[mask_xy]
        else:
            raise ValueError('Weights not loaded, there is more than one .tif file containing weights in its name in the "weights" directory.')
    else:
        weight_k=norm_tiiP_k
        
    """
    >> Divide by integral (if weights were provided, they are used instead of the total integrated intensity).
    ________________________________________________________________________"""

    # TODO Case function=='Constant' not handled, I_I0 not used in most cases.
    if not mantis_name=='I_I0':
        D_kz = ES.rescale_Estack(P_kz, weight_k,E_z)
    else:
        D_kz = P_kz

    # Estimate error on the stack
    diff_kz = D_kz[:, 1:] - D_kz[:, :-1]
    StdError_k = np.sqrt(np.sum((diff_kz)**2,axis=1))
    StdError_xy = utils.reshape_Image(StdError_k, mask_xy)

    # Estimate SNR of the stack
    maxD_k=np.nanmax(D_kz,axis=1)
    SNR_k = maxD_k/StdError_k
    SNR_xy = utils.reshape_Image(SNR_k, mask_xy)
    
    # Inline plot
    ax,meanP_z,stdP_z=utils.scatter_mean_stack(E_z,P_kz,"Background subtracted",None)[0:3]        
    ax,meanD_z,stdD_z=utils.scatter_mean_stack(E_z,D_kz,"Divided by integral",ax)[0:3]
    plt.show()
    
    
    """
    >> Filtering
    ________________________________________________________________________"""
    # TODO add if.
    Kalman_D_kz=ES.Kalman_filter_Estack(D_kz)
    Kalman_D_xyz=utils.reshape_Estack(Kalman_D_kz,mask_xy)
    
    filteredD_xyz=ES.sg3d_filter_Estack(Kalman_D_xyz)
        
    ax=utils.scatter_mean_stack(E_z,D_kz,"Divided by post-edge",None)[0]
    ax,meanFilteredD_z,stdFilteredD_z,label=utils.scatter_mean_stack(E_z,filteredD_xyz,"Filtered",ax,mask_xy=mask_xy)
    plt.show()
            
    """
    Save image sequences
    ________________________________________________________________________"""
    # TODO define rangeE
    if save_stack :
        E_xprt=np.round(E_z,decimals=3)
        if E_range :
            maskE = (E_xprt >= E_range[0]) & (E_xprt <= E_range[1])
            Eselected=E_xprt[maskE]
        else:
            maskE=np.ones(len(E_xprt)).astype(bool)
            Eselected=E_xprt

        # TODO move output folder        
        if not mantis_name == 'Mantis_density':
            processing_name='_'+mantis_name
        else:
            processing_name=''
        output_path=utils.path_join(edgeFd_path,f'calculatedImages{processing_name}')
        if not os.path.exists(output_path):
            os.makedirs(output_path,exist_ok=True)

        # TODO uniformize with way to save bkg, integ,...
        # Save error.
        path_results=utils.path_join(output_path,'SNR.tif',dt='f')
        io.imsave(path_results,SNR_xy.astype(np.float32))   
        path_results=utils.path_join(output_path,'Standard_Error.tif',dt='f')
        io.imsave(path_results,StdError_xy.astype(np.float32))
        
        # Save the D_xyz for Mantis.
        mantisfolder=utils.path_join(edgeFd_path,'Mantis')
        os.makedirs(mantisfolder, exist_ok=True)        
        mantisstackdensity=np.transpose(filteredD_xyz.astype(np.float32)[:,:,maskE],(2,0,1))
        imwrite(utils.path_join(mantisfolder,f'{mantis_name}_density.tif',dt='f'), mantisstackdensity, photometric='minisblack')
        with open(utils.path_join(mantisfolder,f'{mantis_name}_density.txt',dt='f'), 'w') as f:
            for Ei in Eselected:
                f.write(f'{Ei}\n')
                
        # Save the D_xyz as image sequence
        folder_save=edgeFd_path+f'{stackFd_name}_processed{processing_name}/'
        path_save=utils.path_join(edgeFd_path,folder_save,dt='d')
        utils.save_stack_as_image_sequence(n_z,filteredD_xyz,path_save,ImNames,save_stack=save_stack)

        # Save the mean spectra and calculated images 
        # (XPEEM background, post-edge background, integrals and centroid)
        if not mantis_name=='I_I0':
            # Save stack energy, mean and I0
            data = np.column_stack((E_xprt, I0_z, meanI_z, stdI_z[0], stdI_z[1], meanII0_z, stdII0_z[0], stdII0_z[1], mean_rho_z, meanP_z,stdP_z[0],stdP_z[1],meanD_z,stdD_z[0],stdD_z[1], meanFilteredD_z,stdFilteredD_z[0], stdFilteredD_z[1]))
            np.savetxt(utils.path_join(edgeFd_path,f'mean_spectra{processing_name}.csv',dt='f'), data, delimiter=";", header="Energy;Keithley_1;Raw;Raw residuals below;Raw residuals above;Divided I0;Divided I0 residuals below;Divided I0 residuals above;Background;Processed;Processed residuals below;Processed residuals above;Density;Density residuals below;Density residuals below;Filtered;Filtered residuals below;Filtered residuals above", comments='')

            # TODO move up.
            ## Normalize the calculation images: total integrated intensity, background, peak centroid and branching ratio.
            n_l3=n_l2+4
            calcALL_labels=calcRBF_labels+["Background_"+edge,"Background_processed_"+edge,"Background_mask_"+edge,"Background_XPEEM_"+edge]
            if 'Ni_La' in edge or 'Co_La' in edge or 'Mn_La' in edge :
                n_l3+=1
            calcALL_xyl3=np.zeros((n_x,n_y,n_l3))
            for i in range(0,n_l2):
                calcRBF_xy=calcRBF_xyl2[:,:,i]
                # Normalize the processed integral image to its median (a.u.).
                if i%3==1 and ("Integral" in calcRBF_labels[i]):
                    median_calc_im_i=np.nanmedian(calcRBF_xy[mask_xy])
                    calcALL_xyl3[:,:,i]=calcRBF_xy/median_calc_im_i
                else:
                    calcALL_xyl3[:,:,i]=calcRBF_xy
            calcALL_xyl3[:,:,n_l2]=rho_xy
            calcALL_xyl3[:,:,n_l2+1]=norm_rho_xy
            calcALL_xyl3[:,:,n_l2+2]=rhoMask_xy
            calcALL_xyl3[:,:,n_l2+3]=B_xy
            
            # Calculation of branching ratio.
            if 'Ni_La' in edge or 'Co_La' in edge or 'Mn_La' in edge :
                calcALL_labels+=["Branching_ratio_"+edge]
                assert ("processed" in calcRBF_labels[1]) and ("Integral" in calcRBF_labels[1])
                assert ("processed" in calcRBF_labels[4]) and ("_L3" in calcRBF_labels[4])
                branching_ratio=calcRBF_xyl2[:,:,4]/calcRBF_xyl2[:,:,1]
                calcALL_xyl3[:,:,n_l2+4]=branching_ratio
            
            # Save calcultation images
            utils.save_stack_as_image_sequence(n_l3,calcALL_xyl3,output_path,calcALL_labels,save_stack=save_stack)

def export_Estack(edgeFd_path,folder, segm: Optional[Tuple[List[np.ndarray], List[str], List[str]]] = None, Originplot: Optional[str] = None,shift=0,Epre_edge=None,Eref_peak=None,normArg=-1,samplelabel='PEEM',mantisfolder='') :
    """
    >> Load stack (if argument provided)
    ________________________________________________________________________"""
    rawStackFd_name=folder
    stackFd_name=folder+f'_processed{mantisfolder}/'
    
    # Check if the broad-mask directory exists
    masksFd_path=utils.path_join(edgeFd_path, 'Masks')
    if os.path.exists(masksFd_path):
        # Get a list of all files in the directory
        mask_files = os.listdir(masksFd_path)
    
        # Filter the list to only include .tif files
        tif_files = [f for f in mask_files if f.endswith('.tif') and 'broadMask' in f]
    
        # If there is exactly one .tif file
        if len(tif_files) == 1:
            # Load the .tif file
            mask_image = Image.open(utils.path_join(masksFd_path, tif_files[0],dt='f'))
            
            # Check that the array only has 0 and 255 as values.
            uniqueValues_list = np.unique(mask_image)
            assert np.array_equal(uniqueValues_list, [0, 255]) or np.array_equal(uniqueValues_list, [255]), "Broad mask contains values other than 0 and 255."
    
            # Convert the image data to a numpy array
            mask = np.array(mask_image)==255
        else:
            raise ValueError('There is not exactly one broadmask.tif file in the directory.')
            
    # > Raw stack
    print(utils.path_join(edgeFd_path,stackFd_name))
    (n_x,n_y,n_z),rawE_z,_,I_xyz,_=ES.load_Estack(edgeFd_path,utils.path_join(edgeFd_path,rawStackFd_name))
    # raw_stack,ImNames=utils.open_sequence(utils.path_join(edgeFd_path,folder))
    # Eraw=utils.load_E_I0(edgeFd_path)[0]
    ax,raw_av,raw_std,raw_label=utils.scatter_mean_stack(rawE_z,I_xyz,'Raw_stack',None,mask_xy=mask,norm=1)
    plt.show()

    # > Processed stack
    stack,ImNames=utils.open_sequence(utils.path_join(edgeFd_path,stackFd_name))
    (n,m,p)=stack.shape
    E = utils.load_E_I0(edgeFd_path,processed=True)[0]

    ax,segm_av,segm_std,label=utils.scatter_mean_stack(E,stack,'Processed stack',None,mask_xy=mask,norm=-1)
    plt.show()

    """
    >> Segmentation (if argument provided)
    ________________________________________________________________________"""
    if isinstance(segm,list) :
        mask_list=segm[0]
        n_segm=len(mask_list)

        # Initialise
        mask_list=np.array(mask_list)==255
        segm_av=np.zeros((p,n_segm*3))

        
        # Label for Origin (supports reach text)        
        segm_label=segm[1]
        # Short name for the mask 
        segm_alias=segm[2]
        segm_alias_list=[]
        
        # Create stack for each input segmentation  
        for i in range(0,n_segm):
            mask=mask_list[i,:,:]

            if i == 0: 
                ax=None
                ax2=False
            ax,segm_av[:,3*i],segm_std,segm_alias_i=utils.scatter_mean_stack(E,stack,segm_alias[i],ax,ax2=ax2,isav=False,norm=normArg,mask_xy=mask,preEdge_E_range=Epre_edge,refPeak_E=Eref_peak)
            segm_av[:,3*i+1]=segm_std[0]
            segm_av[:,3*i+2]=segm_std[1]
            segm_alias_list.append(segm_alias_i)
            segm_alias_list.append(segm_alias_i+' residual below')
            segm_alias_list.append(segm_alias_i+' residual above')
        data = np.column_stack((E,segm_av))
        np.savetxt(utils.path_join(edgeFd_path,f'Results{mantisfolder}','Segmentation_Spectra.csv',dt='f'), data, delimiter=";", header=f'Energy;{";".join(segm_alias_list)}', comments='')
    else:
        segm_label=[label]
        segm_alias="All"


    """
    >> Export to Origin
    ________________________________________________________________________"""
    if Originplot :
        edge=utils.find_edge(utils.path_join(os.path.basename(os.path.dirname(edgeFd_path)).replace('/',''),stackFd_name,dt='d'),exp='SIM')
        
        loc=os.getcwd()
        filename='OriginPlots.opju'
        
        # Background substracted folder
        oplt.AddSheetOrigin(loc,filename,rawE_z,raw_av,['']+segm_label,foldername='PEEM_raw',bookname=Originplot+'_1--_raw',ShNam=edge+'_raw')
        oplt.AddSheetOrigin(loc,filename,E,segm_av[:,::2],['']+segm_label*2,foldername='PEEM_processed',bookname=Originplot+'_1',ShNam=edge)
        for i in range(n_segm) :
            oplt.AddSheetOrigin(loc,filename,E,[segm_av[:,2*i],segm_av[:,2*i+1],segm_av[:,2*i+2]],['',samplelabel,samplelabel,samplelabel],foldername=f'PEEMSummary/{edge}',bookname=edge+' Sum--mary_1',ShNam=segm_alias[i],shiftCol=shift)

# TODO Make simple example 
# TODO Split in simpler tasks - peakRatioMap, 2imageCorrelation, calculate_nnmfMap
def calculate_chemicalMap(path_ComparisonsFile, sample):
    """
    Iterates through an Excel sheet with the list of XPEEM 2E_image to compare.
    Loads the analysis arguments defined in the Excel document, and calls the function PEEM_2i_find_Imgs.

    Args:
        excel_path (str): The path to the Excel file that contains the arguments for the comparisons.
        sample (str): The XPEEM dataset (sample) label.
    Returns:
        None
    """
    
    # Open the Excel file
    df = pd.read_excel(path_ComparisonsFile, sheet_name='args_comparisons')
    df = df[sample == df['name_Sample']]
    
    # Initialise basedir
    path_Project=os.path.dirname(path_ComparisonsFile)
    
    # Iterate through the rows of the DataFrame
    for _, row in df[df['Process']].iterrows():

        # Call find_folders_and_compare with the sample, alias, and peakIDcouple
        (paths_2E_AB, row_args, path_Project)=E2.find_2E(path_Project, row)
        
        E2.create_outputFd(paths_2E_AB[0], paths_2E_AB[1], row_args, path_Project, labels=[ID.replace('_', '') for ID in paths_2E_AB])
        

def prepare_MLMap(samplefolder,mask=None,ROI=None,filename='Mantis_density.tif'):
    base_name, extension = os.path.splitext(filename)
    path=utils.path_join(samplefolder,'Mantis')
    imfile=utils.path_join(path,filename,dt='f')
    with TiffFile(imfile) as tif:
        images = tif.asarray()
    
    summ_ims=np.expand_dims(np.nanmean(images,axis=0),0)
    _, minmask = utils.image_well_defined(summ_ims,axis=0)
    if isinstance(mask,np.ndarray) :
        unique_values = np.unique(mask)
        assert np.array_equal(unique_values, [0, 255]) or np.array_equal(unique_values, [255]), "Broad mask contains values other than 0 and 255"
        mask = minmask * mask / 255
    else: 
        mask = Image.open(utils.path_join(samplefolder, 'broad_material_mask/broadmask.tif',dt='f'))
        unique_values = np.unique(mask)
        assert np.array_equal(unique_values, [0, 255]) or np.array_equal(unique_values, [255]), "Broad mask contains values other than 0 and 255"

        mask=minmask * mask / 255
    
    if ROI : 
        ROIname='_'+ROI
    else:
        ROIname=''
        
    # Save mask
    pathmask=utils.path_join(path,f'{base_name}{ROIname}_mask.tif',dt='f')
    io.imsave(pathmask, mask.astype(np.uint8))    

    nonzerodelements=images[:,mask == 1]
    meandensity=np.nanmean(nonzerodelements,axis=1)
    images[:,mask == 0]=meandensity[:,np.newaxis]
    

    
    imfile=utils.path_join(path,f'{base_name}{ROIname}.tif',dt='f')
    imwrite(imfile, images, photometric='minisblack')
    
    # Read the text file
    txtfile1=utils.path_join(path,f'{base_name}.txt',dt='f')
    with open(txtfile1, 'r') as f:
        lines = f.readlines()
        
    # Duplicate the text file
    imfile2=utils.path_join(path,f'{base_name}{ROIname}.txt',dt='f')
    with open(imfile2, 'w') as f:
        f.writelines(lines)
    
    # Add ',1' to each line
    lines = [line.strip() + ',1\n' for line in lines]
    
    # Write the result to a CSV file
    with open(utils.path_join(path,'I0.csv',dt='f'), 'w') as f:
        f.writelines(lines)
        
    return f'{base_name}{ROIname}.tif'
    
def calculate_MLMap(new_work_dir, new_n_clusters, nnma_components, filename='Mantis_density.tif', sample='XPEEM',s=0,Eselect=''):

    # Get the ROI name    
    base_name, extension = os.path.splitext(filename)
    match = re.search(rf'({base_name})(.*)({re.escape(extension)})', filename)
    ROI=match.group(1)
    
    # Create output directory
    mantisfolder=utils.path_join(new_work_dir,'Mantis',dt='f')
    outputdir=f'MantisBatchResults{ROI}'    
    os.makedirs(utils.path_join(mantisfolder,outputdir), exist_ok=True)
    
    # Read the settings file
    with open('Mantis_batch_settings.txt', 'r') as f:
        lines = f.readlines()

    # Replace the parameters
    for i, line in enumerate(lines):
        if line.startswith('WORK_DIR:'):
            lines[i] = f'WORK_DIR: {mantisfolder}\n'
        elif line.startswith('OUTPUT_DIR_NAME:'):
            lines[i] = f'OUTPUT_DIR_NAME: {outputdir}\n'
        elif line.startswith('FILENAME'): 
            lines[i] = f'FILENAME: {filename}\n'
        elif line.startswith('N_CLUSTERS:'):
            lines[i] = f'N_CLUSTERS: {new_n_clusters}\n'
        elif line.startswith('NNMA_COMPONENTS:'):
            lines[i] = f'NNMA_COMPONENTS: {nnma_components}\n'
        elif line.startswith('SAMPLE:'):
            lines[i] = f'SAMPLE: {sample}\n'
        elif line.startswith('ROI:'):
            # Set a ROI name for the export
            if ROI == '':
                roilabel = 'All'
            else:
                roilabel=ROI.replace('_','')
            lines[i] = f'ROI: {roilabel}\n'
        elif line.startswith('EDGE:'):
            edge=utils.find_edge(os.path.basename(os.path.dirname(new_work_dir)),exp='SIM')
            lines[i] = f'EDGE: {edge}\n'
        elif line.startswith('SHIFT:'):
            lines[i] = f'SHIFT: {s}\n'

    # Write the result back to the settings file
    with open('Mantis_batch_settings.txt', 'w') as f:
        f.writelines(lines)
        
    return mt.main()

# TODO Sort images per energy.
# TODO Re-organise and simplify.
def calculate_ppcaMap(originplot=False, plot_pca=True, source='peakRatio_ES', mat='NCM', sample='XPEEM', n_PCA=None,n_NNMA=3,n_clusters=3, method_whitening='rank',cluster_method='kmeans'):
    # >> LOAD
    # Load mask
    assert source in ['peakRatio_ES','peakRatio_2E','nnmf']
    inputFd_path=utils.path_join(os.getcwd(),f'_Input/4_ppca_{source}Maps/{sample}/{mat}',dt='d')
    broadMask_path=utils.path_join(inputFd_path,'Mask','mask.tif',dt='f')
    if os.path.exists(broadMask_path) :
        flag_mask=False
        mask_global=np.array(Image.open(broadMask_path))==255
        assert mask_global.any() and not np.isnan(mask_global).any(), "The global mask should contain 0 or 255 values."
    else:
        flag_mask=True
        mask_global=None
        
    # Initiate output directory
    outputFd_path=utils.path_join(os.getcwd(),f'_Output/4_ppca_{source}Maps/{sample}/{mat}',dt='d')
    if not os.path.exists(outputFd_path):
        os.makedirs(outputFd_path)

        
    # Load all images in the directory
    weights = []
    energies = []
    weights_files = [f for f in os.listdir(inputFd_path) if f.endswith('.tif')]
    n_weights = len(weights_files)*2
    for k, filename in enumerate(weights_files):        
        if filename.endswith('.tif'):
            
            # Load energies
            match = re.search('_E_(.+)\.tif', filename)
            if match:
                E_k = match.group(1).replace('_','.')
                energies.append(float(E_k))
                energies.append(float(E_k)+0.1)
            else:
                E_k = k
            
            # Load weights
            w_k = Image.open(utils.path_join(inputFd_path, filename,dt='f'))
            _ = {'F': np.float32}[w_k.mode]
            w_k = np.float32(np.copy(w_k))
            if k == 0 :
                n_x=w_k.shape[0]
                n_y=w_k.shape[1]
            else:
                assert n_x==w_k.shape[0] and n_y==w_k.shape[1], "Error: all weights should have the same size."
            if isinstance(mask_global,np.ndarray) and not flag_mask:
                w_k+=0.01
                w_k=np.nan_to_num(w_k,nan=0.01,posinf=255,neginf=0.01) 
            [w_k],mask_k=utils.image_well_defined(w_k)
            w_ref_k=np.ones_like(w_k) * mask_k
            
            
            # Update the mask_global
            if isinstance(mask_global,np.ndarray) and not flag_mask:
                pass
                # w_k[np.isnan(w_k)]=0.001
                # w_ref_k[np.isnan(w_k)]=1
            elif isinstance(mask_global,np.ndarray) and flag_mask:
                mask_global=mask_global & mask_k 
            else:
                mask_global=mask_k
            
            weights.append(w_k)
            weights.append(w_ref_k)

    fig,ax=plt.subplots()
    ax.imshow(mask_global)
    ax.set_title('Global mask for the provided weights')
    plt.show()

    # Identify the edges for colouring
    gaps = np.diff(energies) > 40
    edge_nr=0
    edges_categories=np.zeros(n_weights)
    for k in range(n_weights-1):
        if gaps[k] : edge_nr+=1
        edges_categories[k+1] = int(edge_nr)

    # >> PREPROCESS
    # Flatten and select only the values out of the mask.
    weights_flattened=[]
    for w_i in weights:
        w_i_flt=w_i[mask_global==1]
        if method_whitening=='mean':
            median=np.nanmean(w_i_flt[w_i_flt>0])
            w_i_flt/=median
        nan_mask = np.isnan(w_i_flt)
        print('Result:')
        print(np.sum(nan_mask.astype(np.int8)))
        weights_flattened.append(w_i_flt)

    # Get dimensions
    weights_flattened=np.array(weights_flattened)
    n_pixels = weights_flattened.shape[1]        

    # > NORMALIZE THE HISTOGRAMMS    
    # Normalize by each edge by its pre-edge or post-edge.
    edge_preedge_groups = np.array_split(weights_flattened, n_weights // 2)
    edge_preedge_groups = [edge_preedge / edge_preedge[-1,:] for edge_preedge in edge_preedge_groups]
    weights_flattened_normalized = np.concatenate(edge_preedge_groups)
    weights_flattened_normalized = weights_flattened_normalized.reshape(n_weights, n_pixels)
    
    fig,ax=plt.subplots()
    plt.scatter(np.arange(n_weights),np.nanmean(weights_flattened,axis=1),label='Before')
    plt.scatter(np.arange(n_weights),np.nanmean(weights_flattened_normalized+0.1,axis=1),label='After (+0.1)')
    ax.set_title('Weight before/after normalization')
    ax.legend()
    plt.show()
    
    # Whithen the data according to the "method_whitening" parameter
    weigths_mean=np.nanmean(weights_flattened_normalized,axis=1)
    weight_whitened=weights_flattened_normalized/weigths_mean[:,np.newaxis]
    if method_whitening=='rank':
        temp_rk_wt_dt = np.array([rankdata(weight_whitened[i,:], method='average') for i in range(n_weights)])
        weight_whitened = ((temp_rk_wt_dt)/n_pixels-0.5)*-1
    elif method_whitening=='log':
        weight_whitened = (np.log10(weight_whitened+1,where=(weight_whitened>=0)))*-1
    elif method_whitening=='scale':
        weight_whitened-=1
        weight_whitened=weight_whitened/np.nanstd(weight_whitened,axis=1)[:,np.newaxis]
        weight_whitened = np.nan_to_num(weight_whitened,posinf=0,neginf=0)*-1
    elif method_whitening=='mean':
        weight_whitened-=1
        weight_whitened = np.nan_to_num(weight_whitened,posinf=0,neginf=0)*-1
        
    # Create a mask of valid values for the PCA
    energies_PCA = np.array([energies[i] for i in range(n_weights)])
    edges_categories_PCA = np.array([edges_categories[i] for i in range(n_weights)])

    # >> PRINCIPAL COMPONENT ANALYSIS (PCA)
    print('PCA: '+str(int(np.sum(np.isnan(weight_whitened)))))
    ncomponent=n_weights
    pca = PCA(n_components=ncomponent)
    pca.fit(weight_whitened)
    weight_PCA = pca.components_

    # Print the explained variance ratio
    fig,ax = plt.subplots()
    variance_PCA=pca.explained_variance_ratio_
    ax.scatter(np.arange(len(variance_PCA)),variance_PCA)
    ax.set_title('Explained variance ratio')
    ax.set_xlabel('PCA principal component')
    plt.show()
    
    variance_PCA_rounded=np.round(variance_PCA*100,decimals=1)
    
    variance_PCA_cumulative = np.cumsum(variance_PCA)
    n_component_significant = max(np.where(variance_PCA_cumulative > 0.95)[0][0]-1,n_PCA)
    print('95% of variance included in: '+str(n_component_significant)+'PCA components')

    for i in range(0,ncomponent):
        # Reshape the first component back into the shape of the original image
        component_i_image = np.zeros((n_x,n_y))
        component_i_image[mask_global==1] = weight_PCA[i,:]
        
        output2=Image.fromarray((component_i_image).astype(np.float32))
        os.makedirs(os.path.join(outputFd_path, 'PCA_score'),exist_ok=True)
        output2.save(os.path.join(outputFd_path, 'PCA_score', f'PCA_score_{i}.tif'))
                        
    # Initialize arrays
    saved_clusters_PCA = np.zeros(n_pixels)
    saved_extrema_PCA = np.zeros(n_pixels)
    components3=list(range(0,min(3,n_component_significant)))
    print(components3)
    
    # Obtain the highest PCA component for each pixel
    scaler = StandardScaler()
    x_var = (scaler.fit_transform(weight_PCA[0,:].reshape(-1, 1)).flatten())/100
    y_var = (scaler.fit_transform(weight_PCA[1,:].reshape(-1, 1)).flatten())/100
    z_var = (scaler.fit_transform(weight_PCA[2,:].reshape(-1, 1)).flatten())/100
    comp_var=np.vstack((x_var,y_var,z_var))
    r = np.max((abs(x_var),abs(y_var),abs(z_var)),axis=0)
    trsh=50
    pt_r50=np.percentile(r, 100-trsh)
    pt_r50_domain=r>pt_r50
    x_var*=variance_PCA_rounded[0]
    y_var*=variance_PCA_rounded[1]
    z_var*=variance_PCA_rounded[2]
    
    # Save hard clusters PCA
    for i in components3:
        temp = np.zeros((n_x,n_y))
        X=weight_PCA[i, :]
        
        # Save component
        Xstd=np.nanstd(X)
        X_scaled = X/(3*Xstd)
        X_scaled = 127.5 * X_scaled + 127.5
        temp[mask_global==1] = X_scaled.reshape(1, -1).ravel()
        im_positive = Image.fromarray(temp.astype(np.int16))
        im_positive.save(os.path.join(outputFd_path, f'PCA_Rescaled_+{i+1}.tif'))
        
        os.makedirs(os.path.join(outputFd_path, 'PCA_mask'),exist_ok=True)
        
        other_weight_PCA=components3.copy()
        other_weight_PCA.pop(i)
        comp_i=[]
        for other_PCA_comp in other_weight_PCA:
            comp_i.append(np.abs(comp_var[i,:])>np.abs(comp_var[other_PCA_comp,:]))
        domain=np.logical_and.reduce(comp_i)
        
        # Save mask for highest PCA value
        cluster_high_i=(domain & (comp_var[i,:] > 0))
        saved_clusters_PCA = np.where(cluster_high_i, cluster_high_i.astype(np.int8) * (2*i+1),saved_clusters_PCA)
        PCAi_high50=pt_r50_domain & cluster_high_i
        mask_high50=PCAi_high50
        mask_high50_8bit=mask_high50.astype(np.int8)
        saved_extrema_PCA = np.where(mask_high50==1, mask_high50_8bit * (2*i+1),saved_extrema_PCA)
        temp = np.zeros((n_x,n_y))
        temp[mask_global==1] = mask_high50_8bit*255
        im_high = Image.fromarray(temp)
        im_high.save(os.path.join(outputFd_path, 'PCA_mask', f'PCA_msk_{i+1}_high.tif'))

        # Save mask for lowest PCA value
        cluster_low_i=(domain & (comp_var[i,:] < 0)).astype(np.int8)
        saved_clusters_PCA = np.where(cluster_low_i==1, cluster_low_i * (2*i+2),saved_clusters_PCA)
        PCAi_low50=pt_r50_domain & cluster_low_i
        mask_low50=PCAi_low50
        mask_low50_8bit=mask_low50.astype(np.int8)
        saved_extrema_PCA = np.where(mask_low50, mask_low50_8bit * (2*i+2),saved_extrema_PCA)        
        temp = np.zeros((n_x,n_y))
        temp[mask_global==1] = mask_low50_8bit*255
        im_low = Image.fromarray(temp)
        im_low.save(os.path.join(outputFd_path, 'PCA_mask', f'PCA_msk_{i+1}_low.tif'))
        
        if originplot : 
            loc=os.getcwd()
            filename = 'OriginPlots.opju'
            
            oplt.AddImageOrigin(temp, f'Mantis/Projection_{source}', 'G'+sample+source,f'GPCA{i}'+sample)
        
    overlay_cluster_PCA = mask_global.astype(np.int8)
    overlay_cluster_PCA[mask_global==1]+=saved_clusters_PCA.astype(np.int8)
    overlay_image = Image.fromarray(overlay_cluster_PCA)
    overlay_image.save(os.path.join(outputFd_path, 'PCA_mask', 'PCA_msk_overlay_clusters.tif'))

    overlay_extrema_PCA = mask_global.astype(np.int8)
    overlay_extrema_PCA[mask_global==1]+=saved_extrema_PCA.astype(np.int8)
    overlay_image = Image.fromarray(overlay_extrema_PCA)
    overlay_image.save(os.path.join(outputFd_path, 'PCA_mask', 'PCA_msk_overlay_extrema.tif'))
        
    PCA_weights=pca.transform(weight_whitened*-1)
    np.savetxt(os.path.join(outputFd_path,'PCA_weights.csv'), PCA_weights, delimiter=',')
    
    # >> NNMA
    # > For all spectra
    n_component_NNMA = n_NNMA
    maps=np.where(weights_flattened_normalized>0,weights_flattened_normalized,0)
    if method_whitening=='rank':
        raw_hists = np.array([rankdata(maps[i,:], method='average') for i in range(n_weights)])
    elif method_whitening=='scale' :
        raw_hists = maps
    elif method_whitening=='mean':
        raw_hists = maps/np.nanmean(maps,axis=1)[:,np.newaxis]
    elif method_whitening=='log':
        raw_hists = np.log10(maps+1)
    np.savetxt(os.path.join(outputFd_path,'raw_hists.csv'), raw_hists.T, delimiter=',')
    
    # BEST:
    model = NMF(n_components=n_component_NNMA, init='nndsvd', random_state=0,solver='cd', beta_loss=2,max_iter=10000,shuffle=True)
    model.fit_transform(maps)
        
    W_matrix_all = model.components_ 
    H_matrix_all = model.transform(maps)
    np.savetxt(os.path.join(outputFd_path,'NNMA_all_H_matrix.csv'), H_matrix_all, delimiter=',')
    
    # Calculate statistics from model
    maps_approx=H_matrix_all@W_matrix_all    
    NMF_meanerror=np.sqrt(np.mean((maps-maps_approx)**2,axis=1))
    maps_mean=np.mean(maps,axis=1)
    maps_approx_mean=np.mean(maps_approx,axis=1)
    
    # Calculate contribution to the mean of each spectra
    H_contrib=np.linalg.lstsq(H_matrix_all, maps_mean)[0]
    H_matrix_weighted=H_matrix_all*H_contrib[np.newaxis,:]
    # W_contrib=np.mean(W_matrix_all,axis=1)
    W_contrib=1/H_contrib
    W_matrix_weighted=W_contrib[:,np.newaxis]*W_matrix_all
    
    # Plot the decomposition of the mean
    plt.figure(figsize=(10, 6))
    x = np.arange(n_weights/2)
    width = 0.8 / n_component_NNMA
    plt.bar(x, maps_approx_mean[::2], width=3.3*width, label='Sum ratio')
    plt.errorbar(x, maps_approx_mean[::2], yerr=NMF_meanerror[::2]*weigths_mean[::2], fmt='none', color='black', capsize=2)
    for i in range(n_component_NNMA):
        if i == 0:
            plt.bar(x, H_matrix_weighted[::2, i], label=f'Component {i+1}')
        else:
            plt.bar(x, H_matrix_weighted[::2, i], bottom=np.sum(H_matrix_weighted[::2, :i], axis=1), label=f'Component {i+1}')

    plt.title('Normalised NNMA decomposition of the cluster')
    plt.xlabel('Selected energies')
    plt.ylabel('Decomposition value')
    plt.legend()
    plt.savefig(os.path.join(outputFd_path, 'W_matrix_spectra_plot.png'))
    plt.show()

    for i in range(0, n_component_NNMA):
    
        # Reshape the first component back into the shape of the original image
        component_i_image = np.zeros((n_x,n_y))
        component_i_image[mask_global == 1] = W_matrix_weighted[i,:]
    
        # Save the first component
        output = Image.fromarray(component_i_image)
        output.save(os.path.join(outputFd_path, f'NNMA_all_component_{i}.tif'))
    
        if originplot :
            oplt.AddImageOrigin(component_i_image, f'Mantis/Projection_{source}', 'G'+sample+source,f'GNMF{i}'+sample)

    
    # >> K-MEANS CLUSTERING

    if cluster_method=='kmeans':
        PCA_proj=np.vstack([x_var,y_var,z_var]).T
        kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=2).fit(PCA_proj)
        labels=kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        
        np.savetxt(os.path.join(outputFd_path,'kmeans_cluster_centers.csv'), cluster_centers, delimiter=',')

        vor = Voronoi(cluster_centers[:,:2])

        fig, ax = plt.subplots()
        voronoi_plot_2d(vor, ax=ax, show_vertices=True, line_colors='k', line_width=2, line_alpha=0.6, point_size=2)
        
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=100, label='Centers')
        ax.legend()
        ax.set_title('Voronoi Diagram with KMeans Clustering')
        plt.show()
    elif cluster_method=='kmeans_polar':
        def cartesian_to_polar(x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)
            phi = np.arccos(z / r)
            return np.vstack([r, theta, phi]).T

        PCA_proj_polar=cartesian_to_polar(x_var,y_var,z_var)

        kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=2).fit(PCA_proj_polar)
        labels=kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        
        np.savetxt(os.path.join(outputFd_path,'kmeans_cluster_centers.csv'), cluster_centers, delimiter=',')
    elif cluster_method=='hdbscan':
        PCA_proj=np.vstack([x_var,y_var,z_var]).T
        clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=100)
        labels = clusterer.fit_predict(PCA_proj) + 1
    elif cluster_method == 'gmm':  # New Gaussian Mixture Model method
        PCA_proj=np.vstack([x_var,y_var,z_var]).T
        gmm = GaussianMixture(n_components=n_clusters, random_state=2).fit(PCA_proj)
        labels = gmm.predict(PCA_proj)
    else:
        raise ValueError(f"Unknown clustering method: {cluster_method}")


    label_matrix=np.array([labels==i for i in range(n_clusters)],dtype=int)
    
    # > Sort clusters according to closeness to Eigenvectors in the PCA
    normlabels=(label_matrix-np.mean(label_matrix,axis=0)[np.newaxis,:])
    normlabels/=np.std(normlabels,axis=0)[np.newaxis,:]
    cluster_centers_pca0 = pca.transform(normlabels)[:,:n_component_significant]
    cluster_centers_pca = np.array([center/np.linalg.norm(center) for center in cluster_centers_pca0])
    eigenvalues=pca.explained_variance_[:n_component_significant]
    common_weight = cluster_centers_pca @ eigenvalues
    sorted_cluster_indices = np.argsort(abs(common_weight))
    print(sorted_cluster_indices)    

    # > Calculate mean spectra for each cluster    
    saved_cluster_kMeans=np.zeros(n_pixels)
    for i in range(n_clusters) :
        saved_cluster_kMeans += label_matrix[i,:]*(i+1)

    mean_all_spectra = np.mean(weights_flattened_normalized,axis=1)
    std_all_spectra = np.mean(weights_flattened_normalized,axis=1)
    curve_std_dev_all_spectra = std_all_spectra / np.sqrt(n_pixels)
    a = 0.01
    ci_all_spectra = stats.t.ppf(1 - a / 2, df=n_weights - 1) * curve_std_dev_all_spectra
    data = np.vstack([weights_flattened_normalized, saved_cluster_kMeans])
    print(weight_whitened.shape)
    print(saved_cluster_kMeans.shape)
    weight_whitened_with_label = np.vstack([weight_whitened[:2,:], saved_cluster_kMeans])
    df_weight_whitened_with_label = pd.DataFrame(weight_whitened_with_label.T)
    df_weight_whitened_with_label.to_csv(os.path.join(outputFd_path,'raw_weight_whitened_and_label.csv'), sep=';', index=False)

    data_clusters = [data[:-1, saved_cluster_kMeans==(label)+1] for label in range(n_clusters)]
    mean_values = np.array([np.mean(data_i,axis=1) for data_i in data_clusters]) 
    std_values = np.array([np.std(data_i,axis=1) for data_i in data_clusters]) 


        
    # > Sort clusters according to their weight in the PCA
    
    label_matrix=label_matrix[sorted_cluster_indices,:]
    cluster_centers_pca0=cluster_centers_pca0[sorted_cluster_indices,:]
    cluster_centers_pca=cluster_centers_pca[sorted_cluster_indices,:]
    mean_values=mean_values[sorted_cluster_indices,:]
    std_values=std_values[sorted_cluster_indices,:]
    data_clusters_arranged=[data_clusters[sorted_i] for sorted_i in sorted_cluster_indices]
    
    cluster_letter=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'][:n_clusters]

    # Initialize a list to store the correlation matrices
    correlation_matrices = []
    df = pd.DataFrame(weight_whitened.reshape(n_weights, n_pixels)[::2,:].T)
    correlation_matrix = df.corr(method='pearson')
    correlation_matrices.append(correlation_matrix)
    
    # Calculate the Pearson correlation matrix for each cluster
    for i in range(len(data_clusters_arranged)):
        data_cluster_i = data_clusters_arranged[i][::2].T  # Transpose to get variables as columns
        df = pd.DataFrame(data_cluster_i)
        correlation_matrix = df.corr(method='pearson')
        correlation_matrices.append(correlation_matrix)
            
    # Function to extract the lower triangle of a correlation matrix with indices
    def get_lower_triangle_with_indices(corr_matrix):
        mask = np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool)
        lower_triangle = corr_matrix.where(mask)
        lower_triangle = lower_triangle.stack()
        lower_triangle.index = [f'{i+1},{j+1}' for i, j in lower_triangle.index]
        return lower_triangle
    
    # Extract the lower triangle of each correlation matrix and concatenate them
    lower_triangles = [
        get_lower_triangle_with_indices(correlation_matrix).rename(f'Cluster {i + 1}')
        for i, correlation_matrix in enumerate(correlation_matrices)
    ]
    
    # Concatenate all lower triangles into a single DataFrame
    all_correlations = pd.concat(lower_triangles, axis=1)
    
    # Print the concatenated correlation table
    print(all_correlations)    
    # Save the concatenated correlation table to a CSV file
    all_correlations.to_csv(utils.path_join(outputFd_path,'correlation_matrices.csv',dt='f'), sep=';')

    
    # Plot the correlation matrices for the first three clusters using matplotlib
    for i in range(1+n_clusters):
        plt.figure(figsize=(8, 8))
        mask = np.triu(np.ones_like(correlation_matrices[i], dtype=bool))
        masked_matrix = np.ma.masked_where(mask, correlation_matrices[i])
        
        os.makedirs(utils.path_join(outputFd_path, 'Correlations'),exist_ok=True)
        np.savetxt(utils.path_join(outputFd_path, 'Correlations',f'correlation_matrix_{i}.csv',dt='f'), masked_matrix, delimiter=',')

        plt.imshow(masked_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        
        if i==0:
            LabelCorrelation='Entire set'
        elif i>0:
            LabelCorrelation='Cluster {i}'
        plt.title('Correlation Matrix for '+LabelCorrelation)
        plt.xticks(ticks=np.arange(n_weights//2), labels=[f'Var {j+1}' for j in range(n_weights//2)], rotation=45, fontsize=10)
        plt.yticks(ticks=np.arange(n_weights//2), labels=[f'Var {j+1}' for j in range(n_weights//2)], fontsize=10)
        
        # Annotate the heatmap with the correlation values
        for (j, k), val in np.ndenumerate(correlation_matrices[i].values):
            if j >= k:  # Only annotate the lower triangle
                plt.text(k, j, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)
    
        plt.tight_layout()
        plt.show()
    
    plt.figure()
    x=np.arange(n_weights//2)*3
    for i in range(len(data_clusters_arranged)):
        data_cluster_i=data_clusters_arranged[i][::2].T
        plt.boxplot(data_cluster_i, positions=i*0.9+x, widths=0.7, medianprops={'color':'black'}, showfliers=False)
    
    data_clusters=[sublist[im,:] for sublist in data_clusters_arranged for im in np.arange(n_weights//2)*2]
    print(data_clusters[0].shape)
    
    data_clusters = [data_clusters_arranged[i][::2].flatten() for i in range(len(data_clusters_arranged))]

    if originplot :
        loc=os.getcwd()
        filename = 'OriginPlots.opju'
        
        comments=str(x)
        data_clusters_arranged[i].shape[0]
        x_i=np.arange(n_pixels)
        oplt.AddSheetOrigin(loc, filename, x_i, data_clusters, comments=comments, foldername=f'Mantis/Projection_{source}', bookname=sample+source, ShNam=f'boxplot_Cluster {i}')

    # Set the x-axis labels
    plt.xticks(x[::3])
    plt.show()
    
    # Save cluster centers
    PCplane=np.eye(n_component_significant)
    PCplane[3:,3:]=0
    clusterproj=cluster_centers_pca0 @ PCplane
    cluster_centers_pca_cropped = np.array([center/np.linalg.norm(center) for center in clusterproj[:n_clusters,:3]])
    cluster_labels=np.expand_dims(np.arange(n_clusters)+1,axis=0)
    np.savetxt(utils.path_join(outputFd_path,'PCA_cluster_centers.csv',dt='f'), np.hstack((cluster_labels[:,0:n_clusters].T,np.zeros((n_clusters,3)),cluster_centers_pca_cropped)), delimiter=',')
    
    # >> BIPLOT FOR PCA
    scaler = StandardScaler()
    x = PCA_weights[::2, 0:1]/100
    y = PCA_weights[::2, 1:2]/100
    
    unique_categories = np.unique(edges_categories_PCA)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    
    # Define a list of markers
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    
    plt.figure(figsize=(10, 10))
    
    for category, color in zip(unique_categories, colors):
        ix = np.array(np.where(edges_categories[::2] == category))[0]
        energies_at_ix = [energies_PCA[::2][i] for i in ix]
        for energy in np.unique(energies_at_ix):
            ix_energy = np.where(energies_at_ix == energy)
            plt.scatter(x[ix][ix_energy], y[ix][ix_energy], color=color, label=energy, marker=markers[int(energy) % len(markers)], s=5*plt.rcParams['lines.markersize'] ** 2)        

    # Add the cluster centers to the plot
    for i, center in enumerate(cluster_centers_pca):
        plt.arrow(0, 0, center[0], center[1], color='black', alpha=0.5,width=0.05)
        plt.text(center[0], center[1], f'Cluster {cluster_letter[i]}', color='black', alpha=0.7)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.axis('equal')
    plt.title('PCA Weights')
    plt.xlabel(f'P.C. 1 ({variance_PCA_rounded[0]:.1f}%)')
    plt.ylabel(f'P.C. 2 ({variance_PCA_rounded[1]:.1f}%)')
    plt.legend(title='Energies [eV]')
    plt.legend(title='Energies [eV]', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.show()
    
            
    # z = scaler.fit_transform(PCA_weights[::2, 2:3])  # Third PCA weight
    z = PCA_weights[::2, 2:3]/100
    
    unique_categories = np.unique(edges_categories_PCA)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    
    if n_clusters > 2:
        # Define a list of markers
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        
        plt.figure(figsize=(10, 10))
        
        for category, color in zip(unique_categories, colors):
            ix = np.array(np.where(edges_categories[::2] == category))[0]
            energies_at_ix = [energies_PCA[::2][i] for i in ix]
            for energy in np.unique(energies_at_ix):
                ix_energy = np.where(energies_at_ix == energy)
                scatter = plt.scatter(x[ix][ix_energy], y[ix][ix_energy], c=z[ix][0], label=energy, marker=markers[int(energy) % len(markers)], s=5*plt.rcParams['lines.markersize'] ** 2, cmap='viridis', vmin=np.min(z), vmax=np.max(z))
        
        plt.axis('equal')
        plt.colorbar(scatter, label=f'P.C. 3 ({variance_PCA_rounded[2]:.1f}%)')
        plt.title('PCA Weights')
        plt.xlabel(f'P.C. 1 ({variance_PCA_rounded[0]:.1f}%)')
        plt.ylabel(f'P.C. 2 ({variance_PCA_rounded[1]:.1f}%)')
        plt.legend(title='Energies [eV]', bbox_to_anchor=(1.3, 1), loc='upper left')
        plt.show()

    if originplot:
        loc=os.getcwd()
        filename = 'OriginPlots.opju'
        
        comments=['']*3
        print(comments)
        oplt.AddSheetOrigin(loc, filename, np.arange(n_weights), variance_PCA, comments=comments, foldername=f'Mantis/Projection_{source}', bookname=sample+source, ShNam='PCvar')

    if originplot and n_clusters > 2:
        loc=os.getcwd()
        filename = 'OriginPlots.opju'
        
        comments=['']*2+variance_PCA_rounded[:3].astype(str).tolist()
        print(comments)
        oplt.AddSheetOrigin(loc, filename, np.array([edges_categories[::2]+1,energies_PCA[::2],x.ravel()]), np.array([y.ravel(),z.ravel()]), comments=comments, foldername=f'Mantis/Projection_{source}', bookname=sample+source, ShNam='PC3')


    # Project the data (with variance unit) onto the first two principal axes
    plt.figure(figsize=(10, 10))    

    grid_n=100
    hb = plt.hexbin(x_var, y_var, gridsize=[int(variance_PCA_rounded[0]/100*grid_n),int(variance_PCA_rounded[1]/100*grid_n)], cmap='viridis', mincnt=1)
    plt.colorbar(hb, label='Density')
    plt.gca().set_facecolor('black')
    
    plt.axis('equal')    
    plt.xlim(-2, 2)
    
    # Add the cluster centers to the plot
    for i, center in enumerate(cluster_centers_pca):
        plt.arrow(0, 0, center[0], center[1], color='white', alpha=1, width=0.02)
        plt.text(center[0]+0.1, center[1]+0.1, f'{cluster_letter[i]}', color='white', alpha=1)


    plt.title('PCA data')
    plt.xlabel(f'P.C. 1 ({variance_PCA_rounded[0]:.1f}%)')
    plt.ylabel(f'P.C. 2 ({variance_PCA_rounded[1]:.1f}%)')
    plt.show()

    # Project the standardized data onto the principal axes 2 and 3
    plt.figure(figsize=(10, 10))    

    grid_n=100
    hb = plt.hexbin(y_var, z_var, gridsize=[int(variance_PCA_rounded[1]/100*grid_n),int(variance_PCA_rounded[2]/100*grid_n)], cmap='viridis', mincnt=1)
    plt.colorbar(hb, label='Density')
    plt.gca().set_facecolor('black')

    
    # Add the cluster centers to the plot
    for i, center in enumerate(cluster_centers_pca):
        plt.arrow(0, 0, center[1], center[2], color='white', alpha=1, width=0.02)
        plt.text(center[1] + 0.1, center[2] + 0.1, f'{cluster_letter[i]}', color='white', alpha=1)

    plt.axis('equal')
    plt.xlim(-2, 2)

    plt.title('PCA data')
    plt.xlabel(f'P.C. 2 ({variance_PCA_rounded[1]:.1f}%)')
    plt.ylabel(f'P.C. 3 ({variance_PCA_rounded[2]:.1f}%)')
    plt.show()

    np.savetxt(utils.path_join(outputFd_path,'PCA_data.csv',dt='f'), np.vstack([x_var,y_var,z_var,saved_cluster_kMeans,saved_clusters_PCA,saved_extrema_PCA]).T, delimiter=',')    


    # KMEANS EXPORT CLUSTERS
    os.makedirs(utils.path_join(outputFd_path, 'kmeans',dt='d'),exist_ok=True)

    # > save the mask for each cluster
    size_cluster=np.zeros(n_clusters)
    k_img=[]
    for i in range(n_clusters) :
        saved_cluster_kMeans += label_matrix[i,:]*(i+1)
        size_cluster[i]=np.sum(label_matrix[i,:])

        # Create a cluster image where each pixel is the cluster number of the corresponding pixel in the images
        cluster_image = np.zeros((n_x,n_y))
        cluster_image[mask_global==1] = label_matrix[i,:]
        k_img.append(cluster_image)
        output2=Image.fromarray((255*cluster_image).astype(np.uint8))
        output2.save(utils.path_join(outputFd_path, 'kmeans', f'KMeans_{i}.tif',dt='f'))

    k_img=k_img[::-1]
    
    overlay_cluster_kmeans=np.zeros_like(mask_global).astype(np.int8)
    for i in range(n_clusters):
        overlay_cluster_kmeans+=(k_img[i]*(i+1)).astype(np.int8)
    overlay_image = Image.fromarray(overlay_cluster_kmeans)
    overlay_image.save(utils.path_join(outputFd_path, 'kmeans', 'kmeans_msk_overlay_clusters.tif',dt='f'))
    
    # Function to convert RGB to BGR
    def rgb_to_bgr(color):
        return np.array(color[::-1], dtype=np.float32)
    
    # Generate a list of colors for the clusters
    def generate_colors(n_clusters):
        cmap = plt.get_cmap('tab10')  # You can choose any colormap
        return [rgb_to_bgr(cmap(i)[:3]) for i in range(n_clusters)]
    
    # Generate colors for each cluster
    colors = generate_colors(n_clusters)
    
    # Apply the colors to the corresponding channels
    colored_channels = []
    for i in range(n_clusters):
        channel = k_img[i].astype(np.float32)
        channel = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR) * colors[i]
        colored_channels.append(channel)
    
    # Stack the channels to create the BGR image
    BGRimage = np.zeros_like(colored_channels[0])
    for channel in colored_channels:
        BGRimage = cv2.add(BGRimage, channel)
    
    # Inline plot
    RGBimage = cv2.cvtColor(BGRimage, cv2.COLOR_BGR2RGB)
    plt.imshow(RGBimage)
    plt.title('KMeans Clustering Overlay')
    
    legend_elements = [Patch(facecolor=to_rgb(colors[i][::-1]), edgecolor='k', label=f'{cluster_letter[i]}') for i in range(n_clusters)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.show()
    

    # Convert the image to 8-bit and save it
    BGRimage = (BGRimage * 255).astype(np.uint8)
    cv2.imwrite(utils.path_join(outputFd_path, 'kmeans', 'KMeans_overlay.tif',dt='f'), BGRimage)    
    
    
    # >> EXPORT CLUSTER DATA
    curve_std_dev = std_values / np.sqrt(size_cluster[:,np.newaxis])
    a = 0.01
    curve_ci = stats.t.ppf(1 - a / 2, df=n_weights - 1) * curve_std_dev
    
    plt.figure(figsize=(10, 6))
    x = np.arange(n_weights/2)
    width = 1 / n_clusters
    x_values = x + width
    y_values = mean_all_spectra[::2]
    y_err = ci_all_spectra[::2]
    plt.bar(x_values, y_values, color='gray', alpha=0.5, label=f'Average {mat}')
    plt.errorbar(x_values, y_values, yerr=y_err, fmt='none', ecolor='black', capsize=1)
    for i in range(n_clusters):
        plt.bar(x + i * width, mean_values[i, ::2], width=width, label=f'Cluster {cluster_letter[i]}')
        plt.errorbar(x + i * width, mean_values[i, ::2], yerr=curve_ci[i, ::2], fmt='none', color='black', capsize=2)
    plt.title("Mean ratio for each cluster")
    plt.xlabel('Selected energies')
    plt.ylabel('Ratio [-]')
    plt.legend()
    plt.show()
        
    if originplot :
        loc=os.getcwd()
        filename = 'OriginPlots.opju'
        
        comments=['Energy','Edge']+["Cluster "+cluster_letter[k] for k in range(n_clusters)]+['Mean']+["Cluster "+cluster_letter[k] for k in range(n_clusters)]
        oplt.AddSheetOrigin(loc, filename, np.array([np.array(energies)[::2],edges_categories[::2]]), np.vstack((mean_values[:,::2],mean_all_spectra[::2],curve_ci[:,::2])), comments=comments, foldername=f'Mantis/Projection_{source}', bookname=sample+source, ShNam='HistV')
        
    # >> NNMA regression
    os.makedirs(utils.path_join(outputFd_path, 'NNMAregression'),exist_ok=True)
    W_matrix_raw = (H_matrix_all).T
    
    # > Normalize each edge subgroup of the histogramm in the NNMF spectras W 
    subgroups = np.array_split(W_matrix_raw.T, W_matrix_raw.shape[1] // 2)
    subgroups = [subgroup / subgroup[-1,:] for subgroup in subgroups]
    W_matrix_norm = np.concatenate(subgroups)
    W_matrix_norm = W_matrix_norm.T

    W_contrib=np.linalg.lstsq(W_matrix_norm.T,mean_all_spectra)[0]
    W_weigthed=W_matrix_norm*W_contrib[:,np.newaxis]
    np.savetxt(utils.path_join(outputFd_path, 'NNMAregression', 'NNMA_clusters_W_matrix.csv',dt='f'), W_matrix_norm, delimiter=',')    
    H_matrix = np.linalg.lstsq(W_weigthed.T, mean_values.T, rcond=None)[0]
    np.savetxt(utils.path_join(outputFd_path, 'NNMAregression', 'NNMA_clusters_H_matrix.csv',dt='f'), H_matrix, delimiter=',')

    # > Project each cluster mean value onto the W_matrix base     
    common_weight = np.array([np.linalg.norm(np.dot(mean_values.T[i,:]**2,W_matrix_norm**2)) for i in range(n_clusters)])
    common_weight = np.array([mean_values[i,:]**2@W_matrix_norm.T**2 for i in range(n_clusters)])
    common_weight = np.array([x/np.sum(x) for x in common_weight.T])
    common_weight = np.array([x/np.sum(x) for x in common_weight.T])
    fig,ax0=plt.subplots()
    ax0.imshow(common_weight, cmap='viridis')
    ax0.set_title('Weight of each NNMA component in each cluster')
    ax0.set_xlabel('NNMA')
    ax0.set_ylabel('Cluster')

    plt.figure(figsize=(10, 6))
    x = np.arange(n_weights/2)
    width = 0.8 / n_component_NNMA
    x_values = x + width
    y_values = mean_all_spectra[::2]
    y_err = ci_all_spectra[::2]
    plt.bar(x_values, y_values, color='gray', alpha=0.5, label=f'Average {mat}')
    plt.errorbar(x_values, y_values, yerr=y_err, fmt='none', ecolor='black', capsize=1)    
    for i in range(n_component_NNMA):
        plt.bar(x + i * width, W_weigthed[i, ::2], width=width, label=f'Component {i+1}')
    plt.title('Normalised NNMA decomposition of the cluster')
    plt.xlabel('Selected energies')
    plt.ylabel('Ratio [-]')
    plt.legend()
    plt.savefig(utils.path_join(outputFd_path, 'NNMAregression', 'NNMA_W_matrix_spectra_plot.png',dt='f'))
    plt.show()

    if originplot :
        loc=os.getcwd()
        filename = 'OriginPlots.opju'
        
        comments=['Energy','Edge']+[f"Component {k+1}" for k in range(n_component_NNMA)]+['Mean peak']

    # Plotting the abundance (H matrix) as a histogram
    plt.figure(figsize=(10, 6))
    bar_width=1/n_component_NNMA
    H_matrix_norm=np.array([x/np.sum(x) for x in H_matrix])
    for i in range(n_component_NNMA):
        plt.bar(np.arange(n_component_NNMA) + (1-bar_width) + i * bar_width,H_matrix_norm[:,i], width=bar_width, label=f'Component {i+1}')
    plt.xticks(np.arange(n_clusters)+1,cluster_letter)
    plt.xlim([0.5, 3.5])
    plt.title("NNMA weight in each cluster's mean spectra")
    plt.xlabel('Cluster')
    plt.ylabel('Weight [%]')
    plt.legend()
    plt.savefig(utils.path_join(outputFd_path, 'NNMAregression', 'H_matrix_abundance_histogram.png',dt='f'))
    plt.show()

    if originplot :
        loc=os.getcwd()
        filename = 'OriginPlots.opju'
        
        comments=['Cluster']+["Component "+str(k+1) for k in range(n_component_NNMA)]
        oplt.AddSheetOrigin(loc, filename, np.arange(n_clusters+1), np.vstack((H_matrix_norm*100,np.zeros(n_component_NNMA))), comments=comments, foldername=f'Mantis/Projection_{source}', bookname=sample+source, ShNam='NNweights')
        
        