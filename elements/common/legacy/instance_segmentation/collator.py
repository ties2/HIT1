from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from common.elements.legacy.tiling import generate_fixed_tiles, generate_random_tiles, get_padding, create_tiles_with_mask
from logging import getLogger
import torch.nn.functional


logger = getLogger("common.elements.legacy.instance_segmentation.collator.py")


def collator(batch):
    """
    Default collator for MaskRCNN

    :param batch: Batch of containing a list of items from the dataset
    :return: a tuple of zipped items
    """
    return tuple(zip(*batch))


def create_new_samples(tiles: List[List[int]], image: torch.Tensor, annotation: Dict[str, Any], image_h=0, image_w=0) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Generate new samples by generating tiles from an original sample

    :param tiles: The tiles to generate from the annotation [[y1, x1, y2, x2] ... ]
    :param image: The image to take tiles from
    :param pad_t: Padding that has been added to the image (used to keep administration of the tile source)
    :param pad_b: Padding that has been added to the image (used to keep administration of the tile source)
    :param pad_l: Padding that has been added to the image (used to keep administration of the tile source)
    :param pad_r: Padding that has been added to the image (used to keep administration of the tile source)
    :param annotation: The annotation information as returned by the InstanceSegmentationDataset
    :return:
    """
    new_samples: List = []
    if "masks" in annotation:
        instance_mask = annotation["masks"]
    else:
        instance_mask = None
    if "class_masks" in annotation.keys():
        class_mask = annotation["class_masks"]
    else:
        class_mask = None
    if "boxes" in annotation.keys():
        boxes = annotation["boxes"]
    else:
        boxes = None
    images_and_masks = create_tiles_with_mask(tiles, image, boxes, instance_mask, class_mask, img_shape='chw', annot_format='xyxy')
    nr_of_tiles = len(tiles)
    for tile_idx, (image_item, boxes_item, instance_masks_item, class_mask_item, kept_indices) in enumerate(zip(*images_and_masks)):
        # Add boxes and masks to the new annotations
        tile_y1, tile_x1, tile_y2, tile_x2 = tiles[tile_idx]
        new_ann = {"tile_y1": tile_y1, "tile_x1": tile_x1, "tile_y2": tile_y2, "tile_x2": tile_x2,
                   "image_w": image_w, "image_h": image_h}
        # Add remaining keys to the annotations
        for key, value in annotation.items():
            if key in ["labels", "iscrowd", "keypoints"]:
                new_ann[key] = value[kept_indices]
            elif key in ["image_id"]:
                new_ann["image_id"] = (value * nr_of_tiles) + tile_idx
                new_ann["source_image_id"] = value.item()
            elif key in ["area"]:
                new_ann["area"] = np.array([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes_item[:, :4]])
            elif key in ["class_masks"]:
                new_ann["class_masks"] = class_mask_item
            elif key in ["masks"]:
                new_ann["masks"] = instance_masks_item
            elif key in ["boxes"]:
                new_ann["boxes"] = boxes_item
            else:
                new_ann[key] = value
        new_samples.append([image_item, new_ann])
    return new_samples


def add_padding(image: torch.Tensor, annotation: Dict[str, Any], top: int, left: int, bottom: int, right: int, value: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Adds a padding to the image and its annotations

    :param image: Image to be padded with shape [c,h,w] or a Dict of images with shape [c,h,w]
    :param annotation: The dictionary of annotation information from the InstanceSegmentationDataset
    :param top: padding to add to the top
    :param left: padding to add to the left
    :return: returns the padded image and its annotations
    """
    if isinstance(image, Dict):
        image = {key: torch.nn.functional.pad(img, (left, right, top, bottom), value=value) for key, img in image.items()}
    else:
        image = torch.nn.functional.pad(image, (left, right, top, bottom), value=value)
    for key in annotation.keys():
        if key == 'keypoints':
            annotation[key][:, :, 0] = annotation[key][:, :, 0] + left
            annotation[key][:, :, 1] = annotation[key][:, :, 1] + top
        elif key in ["masks", "class_masks"]:
            masks = annotation[key]
            annotation[key] = torch.nn.functional.pad(masks, (left, right, top, bottom))
        elif key == "boxes":
            annotation[key][:, 0] = annotation[key][:, 0] + left
            annotation[key][:, 1] = annotation[key][:, 1] + top
            annotation[key][:, 2] = annotation[key][:, 2] + left
            annotation[key][:, 3] = annotation[key][:, 3] + top

    return image, annotation


class RandomTileCollator:
    def __init__(self, tile_h, tile_w, num_tiles, transforms=None):
        """
        Collates the samples in the minibatch by generating random tiles of a predefined size

        :param tile_h: Tile height
        :param tile_w: Tile width
        :param num_tiles: Number of tiles
        :param transforms: Optional transform to apply on the tiles
        """
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.num_tiles = num_tiles
        self.transforms = transforms

    def __call__(self, batch):
        # Tile all samples in data
        new_batch = []
        for sample in batch:
            img, ann = sample
            c, h, w = img["image"].shape

            # Generate tiles and create new samples
            tiles = generate_random_tiles(None, tile_h=self.tile_h, tile_w=self.tile_w, num_tiles=self.num_tiles, image_h=h, image_w=w)

            new_samples = create_new_samples(tiles, img, ann, image_h=h, image_w=w)
            new_batch.extend(new_samples)

        # Transform batch
        if self.transforms:
             new_batch = self.transforms(new_batch)

        return collator(new_batch)


class FixedTileCollator:
    def __init__(self, tile_h, tile_w, margin_h, margin_w, transforms=None, pad_value=0):
        """
        Collates the samples in the minibatch by generating a grid of fixed tiles of a predefined size and overlap margin

        :param tile_h: Tile height
        :param tile_w: Tile width
        :param margin_h: Tile margin in the y direction
        :param margin_w: Tile margin in the x direction
        :param transforms: Optional transform to apply on the tiles
        """

        self.tile_h = tile_h
        self.tile_w = tile_w
        self.margin_h = margin_h
        self.margin_w = margin_w
        self.transforms = transforms
        self.pad_value = pad_value

    def __call__(self, batch):

        # Tile all samples in data
        new_batch = []
        for sample in batch:
            img, ann = sample

            # Get padding
            _, img_h, img_w = img["image"].shape
            padding = get_padding(img_h, img_w, self.tile_h, self.tile_w, self.margin_h, self.margin_w)

            # Apply padding
            ((t, b), (l, r), (_, _)) = padding
            img, ann = add_padding(img, ann, t, l, b, r, value=self.pad_value)

            # Generate tiles and create new samples
            tiles = generate_fixed_tiles(None, self.tile_h, self.tile_w, self.margin_h, self.margin_w, img_h=img_h + t + b, img_w=img_w + l + r)
            new_samples = create_new_samples(tiles, img, ann, img_h, img_w)
            new_batch.extend(new_samples)

        # Transform batch
        if self.transforms:
             new_batch = self.transforms(new_batch)

        return collator(new_batch)


class FullCollator:
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, data):
        return collator(data)
