"""
Hyperspectral Image Dataset Creation Pipeline

This script automates the creation of hyperspectral image (HSI) datasets from provided CSV files that list the HSI files and
their metadata. It supports the generation of datasets in various forms: tiled datasets for convolutional neural networks (CNNs)
and raw datasets for general use.

The pipeline processes the listed HSI files in three different configurations:
1. Smart Tiling: Extracts uniform tiles from HSI files for training UNet architectures, using non-overlapping tiles of specified dimensions.
2. Smart Patching: Extracts small, densely overlapped patches from HSI files for training 1D-CNN models, intended to capture fine-grained spectral information.
3. Raw Extraction: Copies raw HSI data into a dataset format without any tiling or patching, suitable for test scenarios or applications requiring full image data.

Functions:
- create_hsi_dataset_from_csv: Creates a dataset from HSI files listed in a CSV, with configurable extraction modes and preprocessing options.

Key Parameters for `create_hsi_dataset_from_csv`:
- csv_file: Path to the CSV file listing HSI images and metadata.
- hsimage_folder: Directory containing the HSI files referenced in the csv_file.
- dataset_path: Output path for the created dataset file.
- extraction_mode: Method of data extraction ('smart_tiling', 'smart_patching', or 'raw').
- tile_size: Dimensions of the tiles or patches to be extracted (applicable in tiling and patching modes).
- stride: Overlap between consecutive tiles or patches (applicable in tiling and patching modes).
- use_aggregation: Whether to aggregate spectral data (applicable in patching mode).
- preprocessing: List of preprocessing operations to apply to each image before extraction.

Usage:
The script is executed as the main module and processes all listed configurations sequentially. Adjustments to the dataset creation parameters should be made in accordance with the specific requirements of the planned usage of the datasets.

Dependencies:
- elements.load_data: Module that includes functions for loading and creating HSI datasets.
- elements.preprocess: Module containing preprocessing classes and functions specific to hyperspectral data.

Example:
Running this script will create three different datasets as specified in the CSV files and parameters set in the calls to `create_hsi_dataset_from_csv`.

Author:
- [Milad Isakhani Zakaria]

"""

import os
from elements.utils import LoggerSingleton

# working dir
def _get_working_dir():
    return  '/home/student/myprojects/HIT1/'

# configure logging
LoggerSingleton.setup_logger(_get_working_dir())
logger = LoggerSingleton.get_logger()

from elements.load_data import load_hsi_dataset, create_hsi_dataset_from_csv
from elements.preprocess import HyperHuePreprocessor, SpectralNorm

# executiona
if __name__ == '__main__':

    create_hsi_dataset_from_csv(
        csv_file= '/home/student/myprojects/HIT1/dataset/train.csv',                # csv file containing file names and fabric compositions
        hsimage_folder= '/home/student/myprojects/HIT1/dataset/data',               # folder of hsimage files
        dataset_path= '/home/student/myprojects/HIT1/dataset/train_tiled_new.npz',  # datapath to save the dataset
        extraction_mode= 'smart_tiling',                                # extraction mode: smart_tiling, smart_patching, raw
        tile_size= (64, 64),                                            # tile size for smart_tiling and smart_patching modes
        stride= 64,                                                     # stride for smart_tiling and smart_patching modes
        use_aggregation= False,                                         # True for smart_patching otherwise False
        preprocessing= [],                                              # preprocessing list either empty or HyperHuePreprocessor
    )
    create_hsi_dataset_from_csv(
        csv_file='/home/student/myprojects/HIT1/dataset/test.csv',  # Assuming you have a test.csv
        hsimage_folder='/home/student/myprojects/HIT1/dataset/data',  # <-- UPDATED
        dataset_path='/home/student/myprojects/HIT1/dataset/test.npz',  # <-- MISSING FILE CREATED HERE
        extraction_mode='smart_tiling',
        tile_size=(64, 64),
        stride=64,
        use_aggregation=False,
        preprocessing=[],
    )

