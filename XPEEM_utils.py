# -*- coding: utf-8 -*-
"""
Module with specific utility functions for XPEEM.
- Handling path
- Handling Estack I/O
- Typical data processing/image APIs
- Handling masks

Created on Thu Jul 10 09:44:22 2025

@author: lelotte_b
"""
import numpy as np

# Display in python
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns 
sns.set_context('paper', font_scale=2.2) # To change the font settings in the graphs
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina') # High resolution plots

# Handle inputs and outputs with tif and excel.
import pandas as pd
from tifffile import TiffFile

# Docstrings, test and warnings
from typing import List, Optional, Tuple, Union
np.seterr(all='warn') # to be able to handle warnings
import warnings

# Folder/system management
import psutil
import os



def is_program_running(program_name):
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == program_name:
            return True
    return False

# Handle image inputs and outputs
from PIL import Image # Simple, Python-native API for image I/O and basic transforms 

# Statistics
import scipy

# Image processing
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import generic_filter

from savitzkygolay import filter1D
from scipy.interpolate import RBFInterpolator





""" 
Handling paths
_________________________________________________________________________"""
# TODO use os in the implementation.
def path_join(*args: str, dt: str = 'd', sort: str = 'str_operations') -> str:
    """
    Join one or more path components intelligently.

    Parameters:
        
    *args (str): One or more path components.
    
    dt (str, optional): The type of the path. 'd' for directory, 'f' for file. 
    Defaults to 'd'.
    
    sort (str, optional): The type of rectification to perform. 'str_operations' 
    to replace backslashes with slashes, 'abs_path' to convert a relative path 
    to an absolute path. Defaults to 'str_operations'.

    Returns:
    str: The joined path.

    Raises:
    ValueError: If any component is an absolute path (starts with a slash).
    """
    for i,arg in enumerate(args):
        if arg[0]=='/' and i>0:
            raise ValueError('in path_join \n Does not handle absolute path. \n Remove the backslash.')
    path=os.path.join(*args)
    path=path_rectify(path,dt,sort=sort)
    return path

# TODO use os in the implementation.
def path_rectify(path: str, dt: str, sort: str = 'str_operations') -> str:
    """
    Rectify a file or directory path.

    Parameters:
        
    path (str): The path to rectify.
    
    dt (str): The type of the path. 'f' for file, 'd' for directory.
    
    sort (str, optional): The type of rectification to perform. 'str_operations' 
    to replace backslashes with slashes, 'abs_path' to convert a relative path 
    to an absolute path. Defaults to 'str_operations'.

    Returns:
    str: The rectified path.
    """
    if dt=='f'and sort=='str_operations': 
        path = path.replace("\\", "/")
    elif dt=='f' and sort=='abs_path':
        path = os.path.abspath(path)
    else:
        path = path.replace("\\", "/")
        path=path+"/"
    return path

def is_included(x: float, range_x: List[float]) -> bool:
    """
    Check if a number is within a given range.

    This function checks if a given number is within a given range of values. The range is specified as a list 
    of two numbers, where the first number is the lower bound and the second number is the upper bound.

    Parameters:
    x (float): The number to check.
    range_x (List[float]): The range to check within.

    Returns:
    bool: True if the number is within the range, False otherwise.
    """
    assert isinstance(x,float)
    assert isinstance(range_x,list) and len(range_x)==2
    
    if x<range_x[0] or x>range_x[1]:
        return False
    else:
        return True

def find_skipLine(filename: str, word: str = 'ox/red', maxLine: int = 300) -> int:
    """
    Find the line number in a file where a specific word appears.

    Parameters:
    filename (str): The path to the file.
    word (str): The word to find. Default is 'ox/red'.
    maxLine (int): The maximum line number to search. Default is 300.

    Returns:
    int: The line number where the word is found.
    """
    print(os.getcwd())
    assert os.path.exists(filename), 'File '+filename+' not found'
    
    if os.path.splitext(filename)[1] == '.txt': #for the ASCII files from Origin.
        return 1
    elif os.path.splitext(filename)[1] == '.mpt' or os.path.splitext(filename)[1] == '.nor': # Other handled files (.mpt Bio-logic, .nor Athena)
        fpointer = open(filename, "r");
        skipln=0
        for position, line in enumerate(fpointer):
            if word in line:
               skipln=position+1
               print(str(line))
               break
            if position>maxLine:
                skipln=-1
                raise ValueError(f'Could not find the number of line to skip until {maxLine} in {filename}')
        print('Number of line skipped: '+str(skipln))
        return skipln


""" 
Image and signal processing
_________________________________________________________________________"""
def save_stack_as_image_sequence(p: int, stack: np.ndarray, path: str, filenames: List[str], save_stack: bool = False) -> None:
    """
    Convert a sequence of images to a single image.

    This function takes a stack of images as input (ndarray) and saves it as in a image sequence in a the folder "directory". 
    The resulting image is saved to a specified directory. 
    If the rgb parameter is set to True, the images are converted to 8-bit RGB images before saving.

    Parameters:
    p (int): The number of images in the stack.
    stack (np.ndarray): The stack of images.
    path (str): The path of the directory where the image files are located.
    filenames (List[str]): The names of the image files.
    save_stack (bool, optional): Whether to save the stack of images. Defaults to False.

    Returns:
    None
    """
        
    dim=np.shape(stack)
    print(dim)
    if save_stack:

        for i in range(0,p):
            if not os.path.exists(path):
                os.makedirs(path)
                print("Folder %s created!" % path)
            
            im = Image.fromarray(stack[:,:,i],)
            if not filenames[i].endswith(".tif"):
                im.save(path+filenames[i]+".tif")
            else:
                im.save(path+filenames[i])

        print('The image series was sucessfully saved in '+path+filenames[i])

def edge_list(exp: str) -> Tuple[List[str], List[str]]:
    """
    Get the list of edges and their "long name" for a given experiment type.

    Parameters:
    exp (str): The experiment type. Either 'XANES' or 'Phoenix'.

    Returns:
    Tuple[List[str], List[str]]: A tuple containing the list of edges and the list of names.

    Raises:
    ValueError: If the experiment type is not recognised.
    """
    if exp=='SIM': 
        edges=["Ni_","Co_","F_","Mn_","O_","Nb_","C_","P_","S_","P_S_","S_Cl_","P_S_Cl_",]
        names=["Ni_La","Co_La","F_Ka","Mn_La","O_Ka","Nb_Ma","CK_Ka","P_La","S_La","P_S_La","S_Cl_La","P_S_Cl_La"]
        return (edges, names)
    elif exp=='Phoenix':
        edges=["_Co","_Mn","_S","_P","_Nb","_Cl","_Y"]
        names=["CoKa","MnKa","SKa","PKa","NbLa","ClKa","YLa"]
        return (edges, names)
    else: raise ValueError("Error: this experiment was not planed when coding. Please adapt the script.")
    
def find_edge(filename: str, exp: str = 'SIM') -> str:
    """
    Find out the edge based on a Phoenix or SIM XANES filename.

    Parameters:
    filename (str): The filename to check.
    exp (str, optional): The experiment type. Used to determine the list of edges to check for. Defaults to 'XANES'.

    Returns:
    str: The name corresponding to the found edge.

    Raises:
    ValueError: If no edge is found in the filename.
    """
    (edges,names)=edge_list(exp)
    for i, edge in enumerate(edges):
        if edge in filename:
            return names[i]
    raise ValueError(f"Error: {filename}'s edge was not planed when coding. Please adapt the script.")

def load_E_I0(edgeFd_path, processed=False):
    """ Load the energy values from the XPEEM raw or processed 'edge' folder """
    print(edgeFd_path)
    if not processed :
        def determine_delimiter(path):
            with open(path, 'r') as f: first_line = f.readline()
            if ';' in first_line: return ';'
            else: return ','
            
        # Find the .csv file containing 'AbsortionSpectrum'.
        files = os.listdir(edgeFd_path)
        for f in files:
            if 'AbsortionSpectrum' in f and f.endswith('.csv'):
                csvFile_name = f
                break
        csvFile_path= path_join(edgeFd_path,csvFile_name,dt='f')
        
        # Load the energy and I0 column into a ndarray
        df = pd.read_csv(csvFile_path,delimiter=determine_delimiter(csvFile_path))
        Eref = df['Energy'].values
        df.rename(columns={" Keithley_1_raw": "Keithley_1_raw"}, inplace=True)
        I0raw = df['Keithley_1_raw'].values
        
        # > Normalise I0 by its scale, smooth noisy I0 (only trend is important).
        meanI0=np.nanmean(I0raw)
        I0ref = savitzky_golay(I0raw/meanI0,15,3)
        
        # > I0 is defined positive, add minimum if        
        if (I0ref<0).any() :
            I0ref += np.nanmin(I0ref)
    elif processed :
        # Find the .csv file containing 'mean_spectra'.
        files = os.listdir(edgeFd_path)
        csvFile_name = None
        for f in files:
            if 'mean_spectra' in f and f.endswith('.csv'):
                csvFile_name = f
                break      
        assert csvFile_name is not None
        csvFile_path= path_join(edgeFd_path,csvFile_name,dt='f')
        
        # Load the energy and I0 column into a ndarray
        df = pd.read_csv(csvFile_path,delimiter=';', index_col=False)
        Eref = df['Energy'].values
        I0ref = df['Keithley_1'].values
        
    else: 
        raise ValueError('Could not find the requested E and I0 files')
    return Eref, I0ref

def open_sequence(sequence_path: str, images_format:str= ".tif", returnEnergy:bool=False) -> Tuple[np.ndarray,Optional[np.ndarray],list]:
    """
    Open a sequence of image files and convert them to a stack (numpy array).
    
    It follows the convention x,y,z = lateral dimension, vertical dimention.

    This function opens a sequence of image files with a specified format and converts them to a numpy array. 
    The image counting sequence is determined based on the parameter counts.

    Parameters:
    sequence_path (str): The path of the image sequence.
    images_format (str): The format of the image files, default is .tif
    returnEnergy (bool): if set to True, reads the metadata and returns the energies.

    Returns:
    np.ndarray: The image data as a numpy array.
    Optional[np.ndarray]: The array with the photon energies during image acquisition.
    list: The original file name of the image (in order to save it with the same name).
    """
    stack=[]
    EnergyArray=[]
    imnames=os.listdir(sequence_path)

    for imname in imnames:
        # Load the image
        im = Image.open(path_join(sequence_path, imname,dt='f'))        

        # Checking that the image is 8bit or 32bit (raise keyerror if not)
        _ = {'F': np.float32, 'L': np.uint8}[im.mode]

        # Store in the return array
        imarray = np.nan_to_num(np.array(im))
        stack.append(imarray)

        if returnEnergy :
            # Read the energy
            # Open the TIFF file
            with TiffFile(path_join(sequence_path.replace('_undistrdd','--'), imname,dt='f')) as tif:
                # Get the metadata
                metadata = tif.imagej_metadata
            
            # Print the metadata
            info_lines=metadata['Info'].split('\n')

            # Find the line with 'photon energy'
            for line in info_lines:
                if 'photon energy' in line:
                    EnergyArray.append(np.round(float(line.split(': ')[1]),decimals=1))
                    break

    if returnEnergy :
        return np.transpose(np.array(stack),(1,2,0)), np.array(EnergyArray), imnames
    else:
        return np.transpose(np.array(stack),(1,2,0)), imnames

def open_image(image_path: str, crop_min=None, crop_max=None, format_file:str='.tif') -> np.ndarray:
    """
    Open an image file and convert it to a numpy array.

    Parameters:
    image_path (str): The directory and filename of the image file.
    crop_min : (list, optional)
        Minimum crop index.
    crop_max : (list, optional)
        Maximum crop index.
    format_file (str, optional): The format of the image file. Defaults to '.tif'.

    Returns:
    np.ndarray: The image data as a numpy array.
    """
    if not image_path.endswith(format_file):
        image_path += format_file
    im = Image.open(image_path)
    # Checking that the image is 8bit or 32bit (raise keyerror if not)
    _ = {'F': np.float32, 'L': np.uint8, 'I': np.int32}[im.mode]
    imarray = np.nan_to_num(np.array(im))
    
    # Crop if requested.
    if crop_min :
        assert isinstance(crop_min, list)
        assert len(crop_min)==2
        imarray = np.array(imarray)[crop_min[0]:, crop_min[1]:]
    if crop_max :
        assert isinstance(crop_max, list)
        assert len(crop_min)==2
        imarray = np.array(imarray)[:crop_max[0]:, :crop_max[1]]
    
    return imarray

def find_closest_index(array, target):
    differences = np.abs(array - target)
    closest_index = np.argmin(differences)
    return closest_index

def flatten_Estack(stack_xyz,mask_xy):
    """
    Flattens a 3D stack to a 2D array using a boolean or integer mask.
    
    Parameters:
    - stack_xyz: 3D array of shape (nx, ny, nz), the full image stack.
    - mask_xy: 2D boolean or integer array of shape (nx, ny), mask selecting pixels to keep.
    
    Returns:
    - stack_kz: 2D array of shape (n_masked, nz), the flattened stack at masked pixels.
    """
    stack_kz=stack_xyz[mask_xy,:]
    return stack_kz

def reshape_Estack(stack_kz,mask_xy):
    """
    Reconstructs a 3D stack from a flattened stack using a mask.
    
    Parameters:
    - stack_kz: 2D array of shape (n_masked, nz), the flattened stack.
    - mask_xy: 2D boolean or integer array of shape (nx, ny), mask indicating where values go.
    
    Returns:
    - stack_xyz: 3D array of shape (nx, ny, nz), reconstructed stack with values in masked positions.
    """
    (n_x,n_y)=mask_xy.shape
    n_z=stack_kz.shape[1]
    stack_xyz=np.zeros((n_x,n_y,n_z))
    stack_xyz[mask_xy,:]=stack_kz
    return stack_xyz

def reshape_Image(image_k,mask_xy):
    """
    Reconstructs a 2D image from a flattened array using a mask.

    Parameters:
    - image_k: 1D array of masked pixel values.
    - mask_xy: 2D boolean or integer array of shape (nx, ny), mask indicating where values go.

    Returns:
    - image_xy: 2D array of shape (nx, ny), reconstructed image.
    """
    (n_x,n_y)=mask_xy.shape
    image_xy=np.zeros((n_x,n_y))
    image_xy[mask_xy]=image_k
    return image_xy

def scatter_mean_stack(E_z: np.ndarray, Estack: np.ndarray, stk_label: str, ax: Optional[plt.Axes] = None, ax2: Optional[bool] = None, norm: bool = 0, isav: bool = False,shift=0,mask_xy=None,preEdge_E_range=None,refPeak_E=None,plot = True) -> Tuple[plt.Axes, np.ndarray, str]:
    """
    Plot the mean of a stack of images against a given x-axis.

    This function calculates the mean of non-zero values for each segmented image in a stack, 
    normalises the mean if specified, and plots it against a given x-axis. It can also create 
    a legend plot.

    Parameters:
    E_z (np.ndarray): The energy values.
    
    Estack (np.ndarray): The raw or masked E-stack (_xyz or kz)
    
    stk_label (str): The label for the stack in the plot.
    
    ax (plt.Axes, optional): The Axes object to draw the plot on. If None, a new Axes object is created. Defaults to None.
    
    ax2 (bool, optional): Whether to create a legend plot. If True, a new Axes object is created for the legend plot. Defaults to None.
    
    norm (bool, optional): Whether to normalise the mean. If True, the mean is divided by its maximum value. Defaults to True.
    
    isav (bool, optional): Whether the stack is already averaged. If True, the function skips the averaging step. Defaults to False.
    

    shift (float, optional): a shift of the plot up/down for display.
    
    mask_xy (np.ndarray, optional): the ROI on which to average the stack.
    
    preEdge_E_range (list, optional): the energy range of the pre-edge for pre-edge re-normalization.
    
    refPeak_E (float, optional): the energy of a peak of reference to use for normalization.
    
    plot (bool, optional): Whether the graph need to be plotted.

    Returns:
    Tuple[plt.Axes, np.ndarray, str]: A tuple containing the Axes object, the averaged stack, and the stack label.
    """ 
    if len(Estack.shape)==3 : # Estack_xyz
        if not isinstance(mask_xy,np.ndarray):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mask_xy=np.any(Estack>0,axis=2)
        stk_kz=flatten_Estack(Estack,mask_xy)
    elif len(Estack.shape)==2 :
        stk_kz=Estack
    elif len(Estack.shape)==1 :
        stk_kz=Estack
        isav=True
    
    (_,n_z)=stk_kz.shape

    n_infinite = np.sum(np.isinf(stk_kz))
    print(f"{stk_label:} Number of infinite values: {n_infinite}")
    
    n_nan = np.sum(np.isnan(stk_kz))
    print(f"{stk_label:} Number of nan values: {n_nan}")

    # >> Create the figures
    if not ax and plot:
        fig, ax = plt.subplots()
        ax.set_ylabel("Intensity [-]")
        ax.set_xlabel("Energy [eV]")
    elif ax2 and plot:
        figlegend, ax2 = plt.subplots()
        ax2.spines[['top','right','left','bottom']].set_visible(False)
        ax2.tick_params(left=False,bottom=False)
        ax2.tick_params(labelbottom=False)
        ax2.tick_params(labelleft=False)
    
    # >> Average the stack and calculate the residuals above and below the mean.
    if not isav:        
        # Calculate the mean
        meanStk_z=np.nanmean(stk_kz,0)
        
        # Calculate the residuals
        aboveStk_kz=np.where(stk_kz >= (meanStk_z.T)[np.newaxis,:],stk_kz-(meanStk_z.T)[np.newaxis,:],np.nan)
        belowStk_kz=np.where((stk_kz < (meanStk_z.T)[np.newaxis,:]) & (stk_kz > 0),abs(stk_kz-(meanStk_z.T)[np.newaxis,:]),np.nan)
        n_above=np.sum(~np.isnan(aboveStk_kz),axis=0)
        n_above[n_above==0]=1
        resAboveStk_z=np.nansum((aboveStk_kz),axis=0)/n_above
        n_below=np.sum(~np.isnan(belowStk_kz),axis=0)
        n_below[n_below==0]=1
        resBelowStk_z=np.nansum(belowStk_kz,axis=0)/n_below
    else:
        # Calculate the mean
        meanStk_z=stk_kz
        
        # Calculate the residuals
        resAboveStk_z=np.zeros_like(meanStk_z)
        resBelowStk_z=np.zeros_like(meanStk_z)

    # >> Normalise the stack
    if np.all(np.isnan(meanStk_z)) :
        print(stk_label+': the stack is empty.')
        meanStk_z=np.zeros(n_z)
        resAboveStk_z=np.zeros(n_z)
        resBelowStk_z=np.zeros(n_z)
    elif isinstance(preEdge_E_range,list) and refPeak_E:
        mean_pre_edge=np.nanmean(meanStk_z[(E_z>preEdge_E_range[0])&(E_z<preEdge_E_range[1])])
        meanStk_z-=mean_pre_edge
        if isinstance(refPeak_E,str):
            I_refpeak=np.nanmax(abs(meanStk_z))
        else:
            I_refpeak=meanStk_z[find_closest_index(E_z, refPeak_E)]
        meanStk_z/=I_refpeak
        resAboveStk_z/=I_refpeak
        resBelowStk_z/=I_refpeak
    elif refPeak_E is not None :
        if isinstance(refPeak_E,str):
            refI_MeanStk=np.nanmax(abs(meanStk_z))
        else:
            refPeak_z=find_closest_index(E_z,refPeak_E)
            refI_MeanStk=meanStk_z[refPeak_z]
        meanStk_z/=refI_MeanStk
        resAboveStk_z/=refI_MeanStk
        resBelowStk_z/=refI_MeanStk
    else:
        if norm==0: # Divide by max (better after background subtraction)
            max_meanStk=np.nanmax(meanStk_z)
            meanStk_z/=max_meanStk
            resAboveStk_z/=max_meanStk
            resBelowStk_z/=max_meanStk
        elif norm==1: # Divide by min (better before background subtraction)
            min_meanStk=np.nanmin(meanStk_z)
            meanStk_z/=min_meanStk
            resAboveStk_z/=min_meanStk
            resBelowStk_z/=min_meanStk
        elif norm==2: # Divide by integral 
            int_meanStk=np.trapz(meanStk_z,E_z)
            meanStk_z/=int_meanStk
            resAboveStk_z/=int_meanStk
            resBelowStk_z/=int_meanStk
        elif norm==-1:
            pass

    
    # Inline plot
    if plot:
        ax.plot(E_z,meanStk_z+shift,label=stk_label,marker='.',markersize=5)
        ax.fill_between(E_z, meanStk_z-resBelowStk_z+shift, meanStk_z+resAboveStk_z+shift, alpha=0.2)
    
    bothResiduals_Stk_z=[resBelowStk_z,resAboveStk_z]
    # Return axis and average
    # Case 1: legend on separate plot
    if ax2:
        if plot:
            ax2.plot(E_z,meanStk_z,label=stk_label)
            ax2.set_xlim([0, 1])
            ax2.legend(loc='center')        
        return ax,ax2,meanStk_z,bothResiduals_Stk_z,stk_label
    # Case 2: legend on same plot
    else:
        if plot:
            ax.legend()
        return ax,meanStk_z,bothResiduals_Stk_z,stk_label

def plot_images(x, imarray, imarray_dy, DIV, max_y, min_y):
    """
    Plot the input image, the processed image, and the DIV image.

    Parameters
    ----------
    x : ndarray
        The x positions.
    imarray : ndarray
        The input image.
    imarray_dy : ndarray
        The processed image.
    DIV : ndarray
        The DIV image.
    max_y : float
        The max y value for the images.
    min_y : float
        The min y value for the images.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    ax.plot(x, imarray)
    ax.plot(x, imarray_dy + 2)

    fig, ax = plt.subplots()
    ax.plot(x, DIV)

    fig, ax = plt.subplots()
    ax.imshow(imarray, vmax=max_y, vmin=min_y)

    fig, ax = plt.subplots()
    ax.imshow(imarray_dy, vmax=max_y, vmin=min_y)

    fig, ax = plt.subplots()
    ax.imshow(DIV, vmax=1.05, vmin=0.95)



def image_well_defined(images: np.ndarray,axis=2) -> List:
    """
    This function takes an arbitrary number of images, removes extreme values (replaces NaNs, positive infinity, and negative infinity with 0),
    and applies a mask to all images. The mask is created by multiplying the binary masks of all images where the pixel value is greater than 0.

    Args:
    *images (np.ndarray or list): the images to be processed.

    Returns:
    list: A list containing the processed versions of all input images 
    mask of non-zero values in the array.
    """
    if isinstance(images, np.ndarray):
        if len(images.shape) == 2:
            images=np.expand_dims(images,axis=0)
        elif axis==2:
            images=np.transpose(images,(2,0,1))
    elif isinstance(images, list):   
        for im in images:
            if not isinstance(im, np.ndarray) :
                print(im)
                raise ValueError("Some of the images are not ndarray.")
                
        if len(set(img.shape for img in images)) > 1 :
            raise ValueError("All images must have the same dimensions.")
    else:
        raise ValueError("The input type of 'images' should be np.ndarray or list.")


    # Initialize mask with 1s
    mask = 1
    out=[]

    for i, img in enumerate(images):
        # Check if image is of numerical data type
        if not np.issubdtype(img.dtype, np.number):
            raise ValueError(f"Image {i} is not of a numerical data type")

        # Remove extreme values
        imgi = np.nan_to_num(img, nan=0, posinf=0, neginf=0)
        out.append(imgi)

        # Update the mask
        mask *= np.array(imgi != 0, dtype=np.uint8)

    # Apply the mask to all images
    for k in range(len(out)):
        out[k] *= mask

    return out, mask

def image_well_defined2(images: Union[np.ndarray, List[np.ndarray]]) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    This function takes an array or list of images, cleans extreme values (NaNs, ±Inf → 0),
    computes a common mask where all pixels are non-zero, and applies that mask to each image.

    Parameters
    ----------
    images : np.ndarray or list of np.ndarray
        Input images. If ndarray, must be either shape (H, W) for a single image,
        or (N, H, W) or (H, W, N) for multiple images.
    axis : int, default=2
        If `images` is an ndarray of shape (H, W, N), set axis=2 to interpret the third
        dimension as the image index.

    Returns
    -------
    cleaned_list : list of np.ndarray
        The cleaned and masked images, each of shape (H, W).
    mask : np.ndarray
        Boolean mask of shape (H, W) where all input images had non-zero values.
    """
    # Normalize to shape (N, H, W)
    if isinstance(images, np.ndarray):
        arr = images
        if arr.ndim == 2:
            arr = arr[...,np.newaxis]
        elif arr.ndim == 3:
            arr = arr
    elif isinstance(images, list):
        for i, im in enumerate(images):
            if not isinstance(im, np.ndarray):
                raise ValueError(f"Element {i} is not a numpy array.")
        shapes = {im.shape for im in images}
        if len(shapes) > 1:
            raise ValueError("All images must have the same shape.")
        arr = np.stack(images, axis=2)
    else:
        raise ValueError("`images` must be a numpy array or list of numpy arrays.")

    # 1) Clean extreme values
    cleaned = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) Compute mask where all images are non-zero
    mask = np.all(cleaned != 0, axis=2)

    # 3) Apply mask
    masked = cleaned * mask[:, :,np.newaxis]

    return masked, mask

def test_image_well_defined():
    # Test for condition 4
    image_with_extreme_values = np.array([np.finfo(np.float32).max * 2, 0, 0])
    try:
        image_well_defined(image_with_extreme_values)
    except ValueError as e:
        assert str(e) == "Image 0 contains extreme values"

    # Test for condition 5
    image_with_non_numerical_data = np.array(['a', 'b', 'c'])
    try:
        image_well_defined(image_with_non_numerical_data)
    except ValueError as e:
        assert str(e) == "Image 0 is not of a numerical data type"

def image_positivehist(im: np.ndarray, normalize: str = 'not', ext_vals:List[float]=None, mask: np.ndarray = None) -> np.ndarray:
    """
    Take a grayscale 32bit image (in a ndarray) as input and make sure that all 
    the values are positive by adding the minimum value + a small number.
    The minimum pixel value is set to scale / 255 so that the 8 bit intensity of that value
    would be 1.

    Parameters
    ----------
    im : np.ndarray
        The raw image
    normalize : str or list of two element
        If 8bit, normalize the images to the range 0 to 255.
        If absolute, normalize the images to the range 0 to 1.
        If not, the images have the initial maximum value + the minimum non-zero value.

    Returns
    -------
    imr : np.ndarray
        The image with only positive values
    """

    # Assert that mask and image are the same size
    imshape=im.shape
    n=len(imshape)
    if isinstance(mask,np.ndarray) : 
        assert mask.shape == imshape
        

    # Check if the input is a single image or a stack of images
    expandmask = False
    if n == 2:
        # Single image, add an extra dimension to make it 3D
        im = np.expand_dims(im, axis=0)
        expandmask = True
    
    # Initialize an empty array to hold the processed images
    imr = np.zeros_like(im)

    # Loop over each image in the stack (or the single image)
    for i, imi in enumerate(im):
        assert len(imi.shape)==2

        imapprox = median_filter3(imi, 2, 'mirror')

        # Load/compute the mask
        if isinstance(mask, np.ndarray) : 
            if expandmask :
                mask = np.expand_dims(mask, axis=0)
            
            maski = mask[i]
            assert len(maski.shape)==2
            imsk = imapprox[maski==1]
            if (imsk==0).all() :
                print(imsk==0)
                raise ValueError('Some values are still 0')
        else:
            [imapprox], maski = image_well_defined(imapprox)
            imsk = imapprox[maski==1]
            
        boolmsk = maski==1
        
        # Load/compute the minimum and maximum pixel values in the image
        if ext_vals :
            max_val, min_val = ext_vals[1],ext_vals[0]
        elif len(imsk > 2):
            
            max_val, min_val = np.max(imsk), np.min(imsk)
        else: 
            max_val, min_val = np.max(imsk), np.min(imsk)

        # Adjust the image values based on the maximum and minium value        
        imri = np.where(boolmsk, imi - min_val,0)
        
        if normalize == '8bit' :
            imri =np.where(boolmsk,(imri)/(max_val - min_val)*254 + 1,0)
        elif normalize == 'absolute':
            imri =np.where(boolmsk, (imri)/(max_val - min_val) + 1 / 255,0)
        elif not normalize :
            imri = np.where(boolmsk, (max_val - min_val) / 255,0)
        
        imr[i] = imri

    # If input was a single image, return a 2D array instead of a 3D array
    return imr 

def test_image_positivehist():
    # Create a test image with negative and positive pixel values
    test_image = np.array([[-10, 20], [30, -40]], dtype=np.float32)

    # Create a test image stack with negative and positive pixel values
    test_stack = np.array([[[-10, 20], [30, -40]], [[-50, 60], [70, -80]]], dtype=np.float32)

    # Run the function on the test image and image stack
    result_image =image_positivehist(test_image)
    result_stack =image_positivehist(test_stack)

    # Check that all pixel values in the result are positive
    assert np.all(result_image >= 0)
    assert np.all(result_stack >= 0)

    # Create a test image with negative and positive pixel values
    test_image = np.array([[-10, 20], [30, -40]], dtype=np.float32)

    # Create a test image stack with negative and positive pixel values
    test_stack = np.array([[[-10, 20], [30, -40]], [[-50, 60], [70, -80]]], dtype=np.float32)

    # Run the function on the test image and image stack with normalize set to True
    result_image =image_positivehist(test_image, normalize='8bit')
    # result_stack =image_positivehist(test_stack, normalize='8bit')

    # Check that all pixel values in the result are between 0 and 255
    assert np.all((result_image >= 0) & (result_image <= 256))
    # assert np.all((result_stack >= 0) & (result_stack <= 256))
    
    # Check that there is at least one value in the range [254, 256] in the result
    assert np.any((result_image > 254) & (result_image <= 255))
    # assert np.any((result_stack >= 254) & (result_stack <= 256))

    # Test with an image file
    path= r'D:\Documents\a PSI\Data\Data analysis\spyder\2107_Progress_work\0410_SIM beamline 1\PEEM_py\Uncycled\Ni_uncycled\2_energies'
    
    # Load two images from specified paths
    # Caution: nan value can mess it up quite a bit.
    test_image = np.array(Image.open(path_join(path,'i211025_009_851_3_undistrdd/i211025_009#001.tif',dt='f')))
        
    result_image = image_positivehist(test_image)
    assert np.all(result_image >= 0)

    result_image = image_positivehist(test_image, normalize='8bit')
    assert np.any((result_image >= 0) & (result_image <= 256))
    assert np.any((result_image >= 254) & (result_image <= 256))

    print('test_image_positivehist ran without mistake')

def interpolate_missing_values(image, bounds=[None,None], mask=None, label="", norm=False):
    if not isinstance(mask,np.ndarray):
        mask = image != 0
        
    image=np.nan_to_num(image, nan=np.nan, posinf=np.nan, neginf=np.nan)
    median_image=np.nanmedian(image[mask])
    if norm :
        image=image/median_image
        median_image=1
    if not bounds[0] and not bounds[1]:
        bounds=[0,10*median_image]
    elif not bounds[1] and not norm:
        bounds=[bounds[0],5*median_image]
    elif not bounds[0] and not norm:
        bounds=[0,bounds[1]]
        
    print(label)
    print(image.shape)
    print(mask.shape)
    print(bounds)
    print(norm)
    
    image=image.astype(float)

    mask_included = ((image <= bounds[1]) & (image >= bounds[0]) & mask) | ~mask
    mask_excluded = ~mask_included

    rows, cols = image.shape
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    x_grid, y_grid = np.meshgrid(x, y)
    
    known_points = np.array([x_grid[mask_included], y_grid[mask_included]]).T
    known_values = image[mask_included]
    
    rbfi = RBFInterpolator(known_points, known_values, smoothing=1, neighbors=20, kernel='thin_plate_spline')
    
    interp_points = np.array([x_grid[mask], y_grid[mask]]).T
    interpolated_values = rbfi(interp_points)
    
    np.zeros_like(image)
    image[mask] = interpolated_values
    
    return image, mask_excluded.astype(int) * 255, label+"_processed", label+"_mask"

def median_filter2(y, window_size):
    """
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    
    """
    try:
        window_size = np.abs(int(np.round(window_size,decimals=0)))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")

    m = len(y)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:window_size+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-window_size-1:-1][::-1] - y[-1])
    ypad = np.concatenate((firstvals, y, lastvals))
    ymed=median_filter3(ypad, nr_pixel=window_size)[window_size:window_size+m]
    return ymed

def median_filter3(I_xy: np.ndarray, nr_pixel: int = 2, boundary_condition: str = "nearest", mask=None) -> np.ndarray:
    """
    Apply a median filter to an image.

    This function applies a median filter to an image. The size of the filter is specified by the 
    'nr_pixel' parameter. The 'boundary_condition' parameter specifies how the filter should handle 
    the edges of the image.

    Parameters:
        
    I_xy (scipy.ndimage): The image to apply the filter to.
    
    nr_pixel (int, optional): The size of the filter. Defaults to 2.
    
    boundary_condition (str, optional): The boundary condition for the filter. Either 'reflect', 'constant', 
    'nearest', 'mirror' or 'wrap'. Defaults to 'nearest'.

    Returns:
    scipy.ndimage: The filtered image.
    """
    if not isinstance(mask, np.ndarray) :
        filtered_I_xy = scipy.ndimage.median_filter(I_xy, size=nr_pixel, mode=boundary_condition)
    
        return filtered_I_xy
    else:
        def local_median_calculation(values):
            """Calculate the median of values excluding zeros."""
            valid_values = values[values != 0]
            if len(valid_values) == 0:
                return 0
            return np.median(valid_values)

        def masked_median_filter(image:np.ndarray, mask, size: int=3):
            """Apply a median filter to the image considering only pixels included in the mask."""            
            # Apply the median filter only to the masked regions
            filtered_image = generic_filter(image, local_median_calculation, size=nr_pixel)
            
            return filtered_image

        filtered_I_xy = masked_median_filter(I_xy, mask, size=nr_pixel)
        
        return filtered_I_xy

def savitzky_golay(y, window_size, order, deriv=0, rate=1,coeffs=None):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    
    try:
        window_size = np.abs(int(np.round(window_size,decimals=0)))
        order = np.abs(int(np.round(order,decimals=0)))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    # order_range = range(order+1)
    half_window = (window_size+1) // 2
    m = len(y)

    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    ypad = np.concatenate((firstvals, y, lastvals))
    if isinstance(coeffs,np.ndarray):
        ysg=np.convolve(ypad,coeffs,mode='full')[window_size:window_size+m]
    else:
        ysg=filter1D(ypad, window_size, order)[window_size:window_size+m]
    assert y.shape==ysg.shape
    return ysg



def images_outliers2nan(image_set1, image_set2):
    init1, init2 = image_set1, image_set2
    assert image_set1.shape == image_set2.shape
    n=len(image_set1.shape)
    # Check if the input is a single image or a stack of images
    if n == 2:
        # Single image, add an extra dimension to make it 3D
        image_set1 = np.expand_dims(image_set1, axis=0)
        image_set2 = np.expand_dims(image_set2, axis=0)
        
    # set infinite value to nan
    [image_set1], mskset1 = image_well_defined(image_set1)
    [image_set2], mskset2 = image_well_defined(image_set2)
    
    results1 = []
    results2 = []
    nrparam = 3
    for imA, imB, mskA, mskB in zip(image_set1,image_set2, mskset1, mskset2):
        # Check the image and the mask
        # If one of the mask is mostly 0, return the initial images
        assert np.all(imA.shape) <= 7 , ' SSIM only work with images containing more than 7 pixels.'
        if np.sum(mskA) < 2 or np.sum(mskB) < 2 :
            return np.array(init1), np.array(init2)

        
        val = np.zeros((nrparam,np.shape(imA)[0],np.shape(imA)[0]))
        nr=np.zeros(nrparam) # nr of sigme which will yield exclusion

        # Compute MSE between two images
        val[0] = np.where(mskA*mskB==1, np.abs((imA - imB) ** 2), 0)
        nr[0]=10
        
        if np.all(val[0]==0) :
            raise ValueError('The input images have no standard deviation')
        
        # Compute division product
        val[1] = np.where(mskB==1, np.abs(imA / imB), 0)
        nr[1]=3
        
        # Compute SSIM
        normAk,normBk=image_positivehist(imA, normalize='8bit', mask = mskA),image_positivehist(imB,normalize='8bit', mask = mskB)
        _, val[2] = ssim(normAk[0], normBk[0], data_range=255,multichannel=False,full=True)
        nr[2]=3


        # Compute mean and standard deviation of SSIM, MSE, and division product
        stat=np.zeros((nrparam,2))
        for i in range(0,nrparam):
            stat[i,:]=np.array([np.nanmean(val[i]), np.nanstd(val[i])])
            imA[val[i,:,:]-stat[i,0]>abs(nr[i]*stat[i,1])]=np.nan
            imB[val[i,:,:]-stat[i,0]>abs(nr[i]*stat[i,1])]=np.nan

        results1.append(imA)
        results2.append(imB)
    return np.array(results1), np.array(results2)



def test_images_outliers2nan():
    # Load the images
    path= r'D:\Documents\a PSI\Data\Data analysis\spyder\2107_Progress_work\0410_SIM beamline 1\PEEM_py\Uncycled\Ni_uncycled\2_energies'
    testim2 = np.array(Image.open(path_join(path,'i211025_009_851_3_undistrdd/i211025_009#001.tif',dt='f')))
    testim1 = np.array(Image.open(path_join(path,'i211025_012_853_1_undistrdd/i211025_012#001.tif',dt='f')))


    # Plot the images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(testim1, cmap='gray')
    axs[0].set_title('Test image 1')
    axs[1].imshow(testim2, cmap='gray')
    axs[1].set_title('Image 2')
    plt.show()

    im1,im2=images_outliers2nan(testim1,testim2)

    # Check if 97% of the pixels are not NaN
    assert np.sum(~np.isnan(im1)) / im1.size >= 0.85
    assert np.sum(~np.isnan(im2)) / im2.size >= 0.85
    print('Test passed.')

    # Create masks for NaN values
    mask1 = np.isnan(np.nanmean(im1,axis=0))
    mask2 = np.isnan(np.nanmean(im2,axis=0))

    # Plot the images + mask
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.nanmean(im1,axis=0), cmap='gray', norm=colors.Normalize(vmin=0, vmax=1))
    axs[0].imshow(mask1, cmap='jet', alpha=0.5)
    axs[0].set_title('Image 1')
    axs[1].imshow(np.nanmean(im2,axis=0), cmap='gray', norm=colors.Normalize(vmin=0, vmax=1))
    axs[1].imshow(mask2, cmap='jet', alpha=0.5)
    axs[1].set_title('Image 2')
    plt.show()
    
    # Plot the images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.nanmean(im1,axis=0), cmap='gray')
    axs[0].set_title('Image 1')
    axs[1].imshow(np.nanmean(im2,axis=0), cmap='gray')
    axs[1].set_title('Image 2')
    plt.show()
    
    # Case 1: Empty image sets
    im1, im2 = images_outliers2nan(np.array([]), np.array([]))
    assert im1.size == 0 and im2.size == 0, "Failed on case 1: Empty image sets"

    # Case 2: Single-pixel images (9 pixels here)
    im1, im2 = images_outliers2nan(np.random.rand(8, 8), np.random.rand(8, 8))
    assert np.any(im1) and np.any(im2), "Failed on case 2: Single-pixel images"

    # Case 3: Images with all pixels the same
    try: 
        im1, im2 = images_outliers2nan(np.full((10, 10), 5), np.full((10, 10), 5))
    except ValueError:
        pass

    # Case 4: Images with extreme pixel values
    testim1,testim2=np.random.rand(10, 10) * 1e6, np.random.rand(10, 10) * 1e6
    im1, im2 = images_outliers2nan(testim1,testim2)
    assert np.sum(~np.isnan(im1)) / im1.size >= 0.85 and np.sum(~np.isnan(im2)) / im2.size >= 0.85, "Failed on case 4: Images with extreme pixel values"

    # Case 5: Images with NaN or Inf values
    im1, im2 = images_outliers2nan(np.full((10, 10), np.nan), np.full((10, 10), np.inf))
    assert np.all(np.isnan(im1)) and np.all(np.isnan(im1)), "Failed on case 5: Images with NaN or Inf values"
    assert np.all(np.isinf(im2)) and np.all(np.isinf(im2)), "Failed on case 5: Images with NaN or Inf values"

    print('All tests passed.')

if __name__ == '__main__':
    # test_image_well_defined()
    print(os.getcwd())
