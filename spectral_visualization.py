"""
Hyperspectral Image Analysis and Visualization

This script performs detailed analysis of hyperspectral images by extracting specific
patches from a corresponding mask image, applying flatfield corrections, and analyzing
spectral data within those patches. The purpose is to visualize the reflectance statistics
across specific wavelengths for selected samples.

Key Processes:
1. Load hyperspectral image data, associated mask, and RGB visualization images.
2. Apply flatfield correction to correct for the uneven illumination and sensor response.
3. Extract random patches where the mask meets a specific target color criteria.
4. Compute statistical metrics for the reflectance within these patches.
5. Visualize the extracted patches on RGB images.
6. Plot the mean reflectance and the 25th to 75th percentile reflectance range across all wavelengths for the selected samples.

The output includes visualizations of the targeted patches superimposed on the RGB images and plots showing the spectral reflectance statistics across the samples.

Functions:
- extract_random_patches: Extracts patches from a mask where all pixels match a specified color.
- main: Coordinates the loading, processing, visualization, and statistical analysis of hyperspectral data.

Usage:
To run the script, modify the 'folder_path' to point to the directory containing the '.hsimage' files
and their corresponding mask and RGB images. Adjust 'sample_list' to include the specific samples to analyze.

Dependencies:
- numpy: For numerical operations.
- matplotlib.pyplot: For plotting graphs and images.
- cv2 (OpenCV): For image manipulation and reading.
- elements.common.data.datatypes.scandata: For handling hyperspectral scan data.
- elements.preprocess: For applying preprocessing techniques like flatfield correction.

Example:
Ensure the script is in a directory with access to the necessary 'elements' modules and the hyperspectral data.
Running this script will process specified hyperspectral images and output visualizations and a plot of spectral data.

Authors:
- [Milad Isakhani Zakaria]
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from elements.common.data.datatypes.scandata import ScanData
import os
from elements.preprocess import compute_ffc_statistics, apply_flatfield_correction, extract_random_patches


def main(directory_path,sample_list,patch_size):
    all_wavelengths = np.linspace(935.61, 1720.23, 224)
    all_ffc_stats = []

    for sample in sample_list:
        hs_image_path = os.path.join(directory_path, sample+'.hsimage')
        mask_path = os.path.join(directory_path, sample+'_mask.png')
        rgb_path = os.path.join(directory_path, sample+'.png')

        scan_data = ScanData()
        scan_data.load(hs_image_path)
        ffc = apply_flatfield_correction(scan_data.get_raw(), scan_data.get_whiteref(), scan_data.get_darkref())
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)

        target_color = np.array([255, 255, 255])
        coords = extract_random_patches(mask, patch_size, 1, target_color)

        # display extracted tile
        tiled_image = rgb.copy()
        for x1, y1, x2, y2 in coords:
            cv2.rectangle(tiled_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.figure(figsize=(10, 10))
        plt.imshow(tiled_image)
        plt.title(f'Extracted tile on {sample}')
        plt.axis('off')
        plt.show()

        # prepare spectral data for plot
        x1, y1, x2, y2 = coords[0]
        ffc_patch = ffc[y1:y2, x1:x2]
        mean, _, p25, p75 = compute_ffc_statistics(ffc_patch)
        all_ffc_stats.append((mean, p25, p75))

    # plot spectral data for all samples
    plt.figure(figsize=(10, 6))
    for i, (mean, p25, p75) in enumerate(all_ffc_stats):
        plt.plot(all_wavelengths, mean, linewidth=0.5, label=f'{sample_list[i]} mean')
        plt.fill_between(all_wavelengths, p25, p75, alpha=0.5, label=f'{sample_list[i]} 25-75 Percentile')

    plt.title('Mean and 25-75 percentile of spectrum in region for all samples')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('ffc reflectance (R-D / B-D)')
    plt.xlim((950, 1700))
    plt.ylim((0, 1.2))
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # directory path containing .hsimage files
    directory_path = '/home/student/myprojects/HIT1/dataset/data'

    # list of the samples file names from the .hsimage folder
    sample_list = ['sample001', 'sample015', 'sample066']

    # size of the extracted random patch to aggregate the spectral
    patch_size = (50, 50)

    # run the main pipeline
    main(directory_path = directory_path,
         sample_list = sample_list,
         patch_size = patch_size)