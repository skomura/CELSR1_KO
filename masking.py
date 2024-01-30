import numpy as np

import matplotlib.pyplot as plt

import os 

from scipy import ndimage

from skimage import io
from skimage import util 
from skimage import exposure
from skimage import morphology
from skimage.filters import threshold_otsu

import tifffile
from tqdm import tqdm




def ncad_mask(ncad, sigma = 6, square = 12):
    '''
    Returns neural cadherin mask based on Otsu thresholding.
    
    Steps:
    1. Image Normalization via CLAHE 
    2. Slight Gaussian blur to remove small divots (choose sigma)
    3. Otsu thresholding
    4. Binary closing
    5. Sort by largest area and pick

    Parameters
    ----------
    ncad : numpy array
        neural cadherin channel

    Returns
    -------       
    final: numpy array
        masked neural cadherin attached with ncad channel

    '''
    
    hist_eq = exposure.equalize_adapthist(ncad)
    blurred = ndimage.gaussian_filter(hist_eq, sigma = sigma)
    thresh = threshold_otsu(blurred)

    inverted_mask = blurred
    mask = np.zeros(blurred.shape)
    for i in range(0, blurred.shape[0]):
        inverted_mask[i,:,:] = ndimage.binary_closing(blurred[i] < thresh, morphology.square(square))
        inverted_mask[i,:,:] = (ndimage.label(inverted_mask[i,:,:])[0] == 1)
        mask[i,:,:] = ndimage.binary_closing(util.invert(inverted_mask[i]), morphology.square(square))
    mask = (ndimage.label(mask)[0] == 1)

    border = np.zeros(blurred.shape)
    for i in range(0, blurred.shape[0]):
        border[i,:,:] = ndimage.binary_dilation(mask[i], iterations=4).astype('int') - ndimage.binary_erosion(mask[i]).astype('int')
    
    joined = np.array([ncad, border])

    shaping = np.zeros((blurred.shape[0], 2, blurred.shape[1], blurred.shape[2]))
    for j in range(blurred.shape[0]):
        shaping[j,0,:,:] = joined[0][j]
        shaping[j,1,:,:] = joined[1][j]
    final = shaping.astype('uint16')
    mask = mask.astype('uint16')
    
    return mask, final

def mask_folder(folder_path, ecad_ch, ncad_ch, bcat_ch, sox10_ch, start_num, end_num, sigma=6, square=12):
    '''
    Iterates ncad_mask() through entire folder specified.

    Expects tifs to have shape corresponding to (z, y, x, channels), where channels = 4.
    
    
    Parameters
    ----------
    folder_path : raw string literal
        path to current data folder
        e.g. r'C:\organoids'
    start_num : int
        image number to start on in loop
    end_num : int
        image number to end on in loop 
    ecad_ch : int
        channel number
    ncad_ch : int
        channel number
    bcat_ch : int
        channel number
    sox10_ch : int 
        channel number

    
    Returns
    -------
    masks are output in newly created folders.
    
    '''

    for n in tqdm(range(start_num, end_num + 1)):
        tif_name = f'TileScan_1_Position_{n}' # Expecting tif stacks to take this form. Change if needed.
        path = os.path.join(folder_path, tif_name)

        data = io.imread(path + '.tif').T

        bcat = data[bcat_ch - 1].T
        sox10 = data[sox10_ch - 1].T
        ecad = data[ecad_ch - 1].T
        ncad = data[ncad_ch - 1].T

        ncad_masked, composite = ncad_mask(ncad, sigma=sigma,square=square)
        os.mkdir(path)

        tifffile.imwrite(path + f'\\TileScan 1_Position {n}_ncad_masked.tif', data=ncad_masked, imagej=True)
        tifffile.imwrite(path + f'\\TileScan 1_Position {n}_composite.tif', data=composite, metadata={'Composite mode': 'composite'}, imagej=True)
        f = open(path + '\\parameters.txt', 'a')
        f.write(f'sigma = {sigma}, square = {square}')
        f.close

def merge_components(file_path, mask_num, num_to_merge):
    initial = tifffile.imread(file_path + '\\' + f'Mask{mask_num}_1' + '.tif')
    for i in range(2, num_to_merge + 1):
        final = initial + tifffile.imread(file_path + '\\' + f'Mask{mask_num}_{num_to_merge}' + '.tif')
    
    return final  

# mask_num = 7
# output = merge_components(file_path, mask_num, 3)
# tifffile.imwrite(file_path + '\\' + f'Mask{mask_num}.tif', data=output, imagej=True) 
        
mask_folder(r'C:\Users\seank\Folders\lab\organoids\72H_Ch1-Bcat_Ch2-Sox10_Ch3-Ecad_Ch4-Ncad_Exp6_11012023\TileScan1_Position1', 3, 4, 1, 2, 1, 1, sigma=6, square=12)

