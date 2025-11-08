import torch
import numpy as np
from common.elements.legacy.tiling import create_tiles, generate_fixed_tiles, generate_random_tiles, \
    generate_random_tiles_annot, get_padding
from logging import getLogger

logger = getLogger("common.elements.legacy.detection.collator.py")
import torch.nn.functional


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)
    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class RandomPositivesTilesCollator:
    def __init__(self, tile_h, tile_w, num_tiles, transforms=None):
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.num_tiles = num_tiles
        self.transforms = transforms

    def __call__(self, data):

        # Tile all samples in data
        img_tiles_list = []
        annot_tiles_list = []
        for sample in data:
            img = sample["img"]
            annot = sample["annot"]
            if annot.shape[0] == 0:
                logger.warning("Empty annotation encountered for RandomPositivesTilesCollator")
                tiles = generate_random_tiles(img, self.tile_h, self.tile_w, self.num_tiles)
            else:
                tiles = generate_random_tiles_annot(img, self.tile_h, self.tile_w, self.num_tiles, annot)
            img_tiles, annot_tiles = create_tiles(tiles, img, annot)  # Tile images
            img_tiles_list.extend(img_tiles)
            annot_tiles_list.extend(annot_tiles)

        sample = {'img': img_tiles_list, 'annot': annot_tiles_list}

        # Transform
        if self.transforms:
            sample = self.transforms(sample)

        return sample


class RandomTileCollator:
    def __init__(self, tile_h, tile_w, num_tiles, transforms=None):
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.num_tiles = num_tiles
        self.transforms=transforms

    def __call__(self, data):

        # Tile all samples in data
        img_tiles_list = []
        annot_tiles_list = []
        for sample in data:
            img = sample["img"]
            annot = sample["annot"]
            tiles = generate_random_tiles(img, self.tile_h, self.tile_w, self.num_tiles)  # Create random tile list
            img_tiles, annot_tiles = create_tiles(tiles, img, annot)  # Tile images
            img_tiles_list.extend(img_tiles)
            annot_tiles_list.extend(annot_tiles)

        sample = {'img': img_tiles_list, 'annot': annot_tiles_list}

        # Transform
        if self.transforms:
            sample = self.transforms(sample)

        return sample


class FixedTileCollator:
    def __init__(self, tile_h, tile_w, margin_h, margin_w, transforms=None):
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.margin_h = margin_h
        self.margin_w = margin_w
        self.transforms = transforms

    def __call__(self, data):
        # Tile all samples in data
        img_tiles_list = []
        annot_tiles_list = []
        for sample in data:
            img = sample["img"]
            annot = sample["annot"]

            # Get padding
            img_h, img_w, _ = img.shape
            padding = get_padding(img_h, img_w, self.tile_h, self.tile_w, self.margin_h, self.margin_w)
            t = padding[0][0]
            l = padding[1][0]

            # Apply padding
            img = img.numpy()
            img = np.pad(img, padding)
            img = torch.from_numpy(img)
            annot = torch.tensor([[y1 + t, x1 + l, y2 + t, x2 + l, i] for (y1, x1, y2, x2, i) in annot])

            # Generate tiles
            tiles = generate_fixed_tiles(img, self.tile_h, self.tile_w, self.margin_h, self.margin_w)
            img_tiles, annot_tiles = create_tiles(tiles, img, annot)  # Tile images
            img_tiles_list.extend(img_tiles)
            annot_tiles_list.extend(annot_tiles)

        sample = {'img': img_tiles_list, 'annot': annot_tiles_list}

        # Transform
        if self.transforms:
            sample = self.transforms(sample)

        return sample


class FullCollator:
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, data):
        # Tile all samples in data
        img_list = []
        annot_list = []
        scale_list = []
        filename_list = []
        for sample in data:
            img_list.append(sample["img"])
            annot_list.append(sample["annot"])
            filename_list.append(sample["filename"])
            if "scale" in sample.keys():
                scale_list.append(sample["scale"])


        sample = {'img': img_list, 'annot': annot_list, 'filename': filename_list}
        if len(scale_list) > 0:
            sample['scale'] = scale_list

        # Transform
        if self.transforms:
            sample = self.transforms(sample)

        return sample