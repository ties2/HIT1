import torch
import numpy as np
from common.elements.legacy.tiling import create_tiles, generate_fixed_tiles, generate_random_tiles, \
    generate_random_tiles_in_annot, get_padding
from collections import defaultdict

import torch.nn.functional

class RandomBoxesCollator():
    """
    Collates one tile per box (each annotation)
    Note: the tile_h and tile_w should be smaller than the smallest box
    """
    def __init__(self, tile_h, tile_w, num_tiles, transforms=None):
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.num_tiles = num_tiles
        self.transforms = transforms
        self.image_keys = ["img", "raw", "segmap"]
        self.image_keys_hline = ["white", "dark"]

    def __call__(self, data):

        result = defaultdict(list)
        for sample in data:
            tiles = None
            # Process images
            for key in [*self.image_keys, *self.image_keys_hline]:
                # Get data if exists
                if not key in sample.keys() or sample[key] is None:
                    continue
                img = sample[key]
                # Create tiles
                if tiles is None:
                    if not "annot" in sample.keys() or sample["annot"].shape[0] == 0:
                        raise Exception("Cannot generate tiles from bounding boxes if there are no bounding boxes present in the image. Please use the RandmTileCollator instead.")
                    tiles = generate_random_tiles_in_annot(img, self.tile_h, self.tile_w, self.num_tiles, sample["annot"])

                # Tile image (normal for images and whole columns for hline images)
                if key in self.image_keys_hline:
                    tiles_hline = [[0, x1, img.shape[0], x2] for _, x1, _, x2 in tiles]
                    img_tiles = create_tiles(tiles_hline, img)
                else:
                    img_tiles = create_tiles(tiles, img)
                # Add to result
                result[key].extend(img_tiles)

        #From defaultdict to regular dict
        result = {key: value for key, value in result.items()}

        # Transform
        if self.transforms:
            result = self.transforms(result)
        return result


class RandomTileCollator():
    """
    Collate random tiles for all image in sample
    """

    def __init__(self, tile_h, tile_w, num_tiles, transforms=None):
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.num_tiles = num_tiles
        self.transforms = transforms
        self.image_keys = ["img", "raw", "segmap"]
        self.image_keys_hline = ["white", "dark"]

    def __call__(self, data):

        result = defaultdict(list)
        for sample in data:
            tiles = None
            # Process images
            for key in [*self.image_keys, *self.image_keys_hline]:
                # Get data if exists
                if not key in sample.keys() or sample[key] is None:
                    continue
                img = sample[key]
                # Tile image (normal for images and whole columns for hline images)
                if tiles is None:
                    tiles = generate_random_tiles(img, self.tile_h, self.tile_w, self.num_tiles)  # Create random tile list
                if key in self.image_keys_hline:
                    tiles_hline = [[0, x1, img.shape[0], x2] for _, x1, _, x2 in tiles]
                    img_tiles = create_tiles(tiles_hline, img)
                else:
                    img_tiles = create_tiles(tiles, img)
                # Add to result
                result[key].extend(img_tiles)

        #From defaultdict to regular dict
        result = {key: value for key, value in result.items()}

        # Transform
        if self.transforms:
            result = self.transforms(result)
        return result


class FixedTileCollator():
    def __init__(self, tile_h, tile_w, margin_h, margin_w, transforms=None):
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.margin_h = margin_h
        self.margin_w = margin_w
        self.transforms = transforms
        self.image_keys = ["img", "raw", "segmap"]
        self.image_keys_hline = ["white", "dark"]

    def __call__(self, data):
        result = defaultdict(list)
        for sample in data:
            tiles = None
            # Process images
            img_w, img_h = 0, 0
            for key in list(sample.keys()):
                # Get data if exists
                if not key in [*self.image_keys, *self.image_keys_hline] or sample[key] is None:
                    result[key] = sample[key]
                    continue
                img = sample[key]
                if key in self.image_keys_hline:
                    ((_, _), (l, r), (_, _)) = get_padding(img.shape[0], img.shape[1], self.tile_h, self.tile_w, self.margin_h, self.margin_w)
                    padding = ((0, 0), (l, r), (0, 0)) # only use l, r padding for lines
                    img = torch.from_numpy(np.pad(img.numpy(), padding, mode='mean'))
                    if img_h == 0 or img_w == 0:
                        raise Exception(f"Cannot not determine size for key '{key}' because no full-size images have been processed.")
                    tiles = generate_fixed_tiles(None, self.tile_h, self.tile_w, self.margin_h, self.margin_w, img_w=img_w, img_h=img_h)
                    tiles = [[0, x1, img.shape[0], x2] for _, x1, _, x2 in tiles] # reset height of tiles
                    img_tiles = create_tiles(tiles, img)
                else:
                    padding = get_padding(img.shape[0], img.shape[1], self.tile_h, self.tile_w, self.margin_h, self.margin_w)
                    if key == "segmap":
                        img = torch.from_numpy(np.pad(img.numpy(), padding, mode='constant'))
                    else:
                        img = torch.from_numpy(np.pad(img.numpy(), padding, mode='mean'))
                    img_h, img_w, _ = img.shape
                    tiles = generate_fixed_tiles(img, self.tile_h, self.tile_w, self.margin_h, self.margin_w)
                    img_tiles = create_tiles(tiles, img)

                # Add to result
                result[key].extend(img_tiles)

        #From defaultdict to regular dict
        result = {key: value for key, value in result.items()}

        # Transform
        if self.transforms:
            result = self.transforms(result)
        return result


class FullCollator():
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, data):
        result = defaultdict(list)
        for sample in data:
            for key, value in sample.items():
                result[key].append(value)

        # Transform
        if self.transforms:
            result = self.transforms(result)
        return result