"""
Segmentation Mask Generation from LabelMe JSON Annotations

This script processes JSON files annotated with LabelMe, a tool used for marking up images with polygons,
to generate segmentation masks. The script loads each JSON, reads its associated image, applies annotations
as masks, and saves these masks as separate images.

The pipeline supports different labels within the JSON files and converts these labels to specific colors
in the output masks:
- 'background' labeled polygons are filled with red (255, 0, 0).
- 'fabric' labeled polygons are filled with white (255, 255, 255).
- Unannotated regions are left as black (0, 0, 0), representing the 'void' class.

The masks are intended for use in machine learning models where segmentation of different classes is required.
The output mask will be saved with a '_mask' suffix.

Usage:
The script is designed to be run as the main module, processing all JSON files in a specified directory
and its subdirectories. The directory path should contain JSON files and their corresponding image files
in PNG format, named identically to the JSON files.

Example:
Suppose you have a directory with JSON files and their corresponding PNG images. Running this script will
generate a mask for each image based on the annotations in its corresponding JSON file and save it alongside
the original image with a '_mask' suffix.

Dependencies:
- json: For loading and parsing JSON files.
- numpy: For array operations.
- cv2 (OpenCV): For image reading, mask creation, and image writing.
- pathlib: For handling filesystem paths.
- elements.visualize: For visualizing images and their corresponding masks.

Functions:
- generate_mask_from_labelme_json(json_path: Path): Generates and saves a mask for a single JSON file.
- process_json_files(directory_path: str): Processes all JSON files within a directory to generate masks.

Authors:
- [Milad Isakhani Zakaria]
"""

import json
import numpy as np
import cv2
from pathlib import Path
from elements.visualize import visualize_image_and_mask


def generate_mask_from_labelme_json(json_path: Path):
    """
    Generate a segmentation mask from a LabelMe JSON file and save it as an image.

    Annotation Guidelines:
      - During LabelMe annotation, use polygons with the label 'background' to mark background areas.
        These regions will be filled with the color (255, 0, 0) in the mask.
      - Use polygons with the label 'fabric' to mark fabric areas. These regions will be filled
        with the color (255, 255, 255) in the mask.
      - Pixels outside of annotated polygons will be considered as the 'void' class, remaining
        black (0, 0, 0) by default in the mask.

    File Structure Requirements:
      - Each JSON file should have a corresponding PNG image in the same directory, with the same
        base filename. For example, if the JSON file is named 'image1.json', the associated PNG
        file should be 'image1.png' and located in the same directory. This PNG file is loaded
        as the original image for visualization and mask creation.

    Args:
        json_path (Path): Path to the LabelMe JSON file to process.

    Saves:
        A PNG mask image with the same name as the JSON file, appended with '_mask'.
    """
    print(f"Processing {json_path.name}")

    # load json file
    with json_path.open("r") as file:
        data = json.load(file)

    # path to image and future mask
    image_path = json_path.with_suffix('.png')
    mask_output_path = json_path.with_name(f"{json_path.stem}_mask.png")

    # load image and create zero mask
    image = cv2.imread(str(image_path))
    mask = np.zeros_like(image, dtype=np.uint8)

    # now generating mask based on the polygons
    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = np.array(shape.get("points"), dtype=np.int32)

        # color (0,0,0) is reserved for the "void" class
        if label == "background":
            cv2.fillPoly(mask, [points], (255, 0, 0))
        elif label == "fabric":
            cv2.fillPoly(mask, [points], (255, 255, 255))

    # visualize
    visualize_image_and_mask(image, mask, json_path.stem)
    cv2.imwrite(str(mask_output_path), mask)
    print(f"Saved mask as {mask_output_path.name}")


def process_json_files(directory_path: str):
    """Process all LabelMe JSON files in a directory to generate segmentation masks."""
    directory = Path(directory_path)
    for json_file in directory.rglob("*.json"):
        generate_mask_from_labelme_json(json_file)


if __name__ == "__main__":

    # directory path containing .hsimage files
    directory_path = '/home/student/myprojects/HIT1/dataset/data'

    # run pipeline
    process_json_files(directory_path)
