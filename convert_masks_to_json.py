import cv2
import numpy as np
import json
import os


def create_json_from_mask(mask_path, data_folder):
    """
    Creates a LabelMe-style JSON file from a 3-color segmentation mask.

    The mask is expected to have:
    - [255, 255, 255] (White) for 'fabric'
    - [255, 0, 0] (Red) for 'background'
    - [0, 0, 0] (Black) for 'void'
    """

    # Read the mask image
    mask_image = cv2.imread(mask_path)
    if mask_image is None:
        print(f"Warning: Could not read {mask_path}. Skipping.")
        return

    height, width = mask_image.shape[:2]
    base_name = os.path.basename(mask_path)
    rgb_image_name = base_name.replace('_mask.png', '.png')
    json_output_path = mask_path.replace('_mask.png', '.json')

    # Define the colors and labels
    # Note: Colors are BGR for OpenCV
    label_colors = {
        "fabric": [255, 255, 255],
        "background": [0, 0, 255],  # Red is (0,0,255) in BGR
        "void": [0, 0, 0]
    }

    shapes = []

    # Find contours for each label
    for label, color_bgr in label_colors.items():
        # Create a binary mask for the current color
        lower_bound = np.array(color_bgr, dtype=np.uint8)
        upper_bound = np.array(color_bgr, dtype=np.uint8)
        binary_mask = cv2.inRange(mask_image, lower_bound, upper_bound)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to LabelMe polygon format
        for contour in contours:
            # Simplify the contour slightly to reduce number of points
            contour = cv2.approxPolyDP(contour, epsilon=1.0, closed=True)

            # Reshape to a list of [x, y] points
            points = contour.reshape(-1, 2).tolist()

            # LabelMe requires at least 2 points for a polygon
            if len(points) > 2:
                shape = {
                    "label": label,
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                shapes.append(shape)

    # Create the final LabelMe-style dictionary
    labelme_data = {
        "version": "5.4.1",  # You can set this to any version string
        "flags": {},
        "shapes": shapes,
        "imagePath": rgb_image_name,  # Path to the corresponding RGB image
        "imageData": None,  # We don't need to embed the image data
        "imageHeight": height,
        "imageWidth": width
    }

    # Save the JSON file
    with open(json_output_path, 'w') as f:
        json.dump(labelme_data, f, indent=2)

    print(f"Successfully created {json_output_path}")


# --- Main Execution ---
if __name__ == "__main__":
    # !! IMPORTANT !!
    # Set this to the folder where your .hsimage, .png, and _mask.png files are
    DATA_DIRECTORY = '/home/student/myprojects/HIT1/dataset/masks'

    print(f"Scanning for masks in: {DATA_DIRECTORY}")

    # Find all mask files in the directory
    mask_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('_mask.png')]

    if not mask_files:
        print("Error: No '_mask.png' files found in that directory.")

    for mask_file in mask_files:
        full_mask_path = os.path.join(DATA_DIRECTORY, mask_file)
        create_json_from_mask(full_mask_path, DATA_DIRECTORY)

    print(f"\nDone. Processed {len(mask_files)} mask files.")