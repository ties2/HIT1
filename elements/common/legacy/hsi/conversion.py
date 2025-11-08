import glob
import json
import os

import numpy as np
import spectral.io.envi as envi
from skimage.io import imread
from tqdm import tqdm


def cubes_segmap_to_npz(folder, out_folder, samples_sub_folder="samples", segmap_sub_folder="annotations",
                        segmap_meta_file="annot_color.json"):
    """
    Convert ENVI HSI cubes and PixelAnnotationTool SegMaps to npz file format.

    Should be layed out as: "/media/public_data/Projects/intern/Plastics/Dataset/Flakes_Basic"
    """

    os.makedirs(out_folder, exist_ok=True)

    samples = os.path.join(folder, samples_sub_folder, "*")
    capture = "capture"
    segmap_folder = os.path.join(folder, segmap_sub_folder)
    segmap_meta = os.path.join(folder, segmap_folder, segmap_meta_file)
    with open(segmap_meta, "r") as f:
        annot_list = json.load(f)

    for folder in tqdm(glob.glob(samples)):
        # Find filenames
        files = [file for file in glob.glob(os.path.join(folder, capture, "*.hdr"))]
        dark_fn = [file for file in files if os.path.basename(file)[:4] == "DARK"][0]
        white_fn = [file for file in files if os.path.basename(file)[:4] == "WHIT"][0]
        cube_fn = [file for file in files if os.path.basename(file)[:4] != "DARK" and os.path.basename(file)[:4] != "WHIT"][0]
        segmap_fn = os.path.join(segmap_folder, os.path.splitext(os.path.basename(cube_fn))[0] + '.png')

        # Read all cubes
        cube = envi.open(cube_fn)
        cube_meta = cube.metadata
        white = envi.open(white_fn)
        white_meta = white.metadata
        dark = envi.open(dark_fn)
        dark_meta = dark.metadata
        class_names = np.array([item["name"] for item in annot_list])

        # Convert to normal np arrays
        dark = dark[:, :, :]
        white = white[:, :, :]
        cube = cube[:, :, :]

        # Convert segmap from color to ids
        segmap = imread(segmap_fn)
        segmap_ids = np.empty(segmap.shape[:2], dtype=np.int32)
        for item in annot_list:
            color = [int(x) for x in item["color"]]
            select = np.sum(segmap[:, :, :3] == color, axis=2) == 3
            segmap_ids[select] = item["id"]

        out_npz_fn = os.path.join(out_folder, os.path.splitext(os.path.basename(cube_fn))[0] + ".npz")
        empty = np.array([])

        # Perform background correction
        dark_mn = np.mean(dark, axis=0, keepdims=True)
        dark_ext = np.resize(dark_mn, cube.shape)

        white_mn = np.mean(white, axis=0, keepdims=True)
        white_ext = np.resize(white_mn, cube.shape)

        scan = (cube - dark_ext) / (white_ext - dark_ext)

        # Save compressed npz
        np.savez_compressed(out_npz_fn,
                            scan=scan,
                            scan_shape=scan.shape,
                            raw=cube,
                            raw_shape=cube.shape,
                            white=white,
                            white_shape=white.shape,
                            dark=dark,
                            dark_shape=dark.shape,
                            annotations=empty,
                            segmap=segmap_ids,
                            classnames=class_names,
                            temperature=np.array(cube_meta['temperature']),
                            wavelength=np.array(cube_meta['wavelength']),
                            fwhm=np.array(cube_meta['fwhm']),
                            origin=np.array([os.path.dirname(cube_fn)]),
                            white_meta=np.array([white_meta]),
                            dark_meta=np.array([dark_meta]),
                            cube_meta=np.array([cube_meta])
                            )


def hyperspectral_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a hyperspectral image to RGB using the average value of 1/3th of the HS cube for each RGB channel

    :param image: Image of size [h, w, c]
    :return: And RGB image with size [h, w, 3]
    """
    nc = image.shape[-1]
    img = np.empty((image.shape[0], image.shape[1], 3), dtype=image.dtype)
    img[..., 2] = np.average(image[..., 0:nc // 3], axis=-1)
    img[..., 1] = np.average(image[..., nc // 3:2 * (nc // 3)], axis=-1)
    img[..., 0] = np.average(image[..., 2 * (nc // 3):], axis=-1)
    return img
