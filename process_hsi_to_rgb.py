"""
Hyperspectral Image Processing Pipeline

This script defines a processing pipeline for hyperspectral images, converting them into RGB format for easier visualization and making annotations.
The pipeline handles the loading of hyperspectral image data, applies flatfield correction
to adjust for variations in lighting and sensor response, converts the corrected hyperspectral data to RGB,
and saves the output as PNG images.

The pipeline is designed to process all hyperspectral image files within a specified directory,
making it suitable for batch processing tasks. Each file with the '.hsimage' extension is processed individually,
and the resulting RGB images are stored in the same location with a '.png' extension.

Functions:
- process_hsimage_file(hsimage_file: Path): Processes a single hyperspectral image file.
- process_directory(directory_path: str): Processes all hyperspectral image files in the specified directory.

Usage:
This script is intended to be run as the main module, with the directory path of hyperspectral images provided at runtime.
It utilizes the `Path` object from `pathlib` for directory and file operations, and the `Image` class from `PIL` to handle image saving operations.

Example:
Running the script from the command line with a specified directory path will process all '.hsimage' files within that directory and
 save the corresponding RGB images.

Author:
- [Milad Isakhani Zakaria]

Dependencies:
- PIL: Used for creating and saving RGB images.
- elements.common.data.datatypes.scandata: Custom module for loading and handling scan data.
- elements.preprocess: Custom module providing preprocessing functions like flatfield correction and RGB conversion.
"""

from pathlib import Path
from PIL import Image
from elements.common.data.datatypes.scandata import ScanData
from elements.preprocess import convert_hsi_to_rgb, apply_flatfield_correction


def process_hsimage_file(hsimage_file: Path):
    """Load, process, and save an HSI image as an RGB PNG."""
    print(f"Processing {hsimage_file.name}")

    # load data
    scan_data = ScanData()
    scan_data.load(str(hsimage_file))
    raw_data = scan_data.get_raw()
    dark_frame = scan_data.get_darkref()
    white_frame = scan_data.get_whiteref()

    # apply flatfield correction
    corrected_data = apply_flatfield_correction(raw_data, white_frame, dark_frame)

    # convert to rgb
    rgb_image = convert_hsi_to_rgb(corrected_data)

    # save as png
    output_path = hsimage_file.with_suffix(".png")
    Image.fromarray(rgb_image).save(output_path)
    print(f"Saved {output_path.name}")


def process_directory(directory_path: str):
    """Process all .hsimage files in the specified directory."""
    directory = Path(directory_path)
    for hsimage_file in directory.glob("*.hsimage"):
        process_hsimage_file(hsimage_file)


# execution
if __name__ == '__main__':

    # directory path containing .hsimage files
    directory_path = '/home/student/myprojects/HIT1/dataset/data'

    # run pipeline
    process_directory(directory_path)
