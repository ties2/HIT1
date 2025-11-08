import copy

import numpy as np
import torch
from random import randint
from typing import List, Tuple, Union, Optional, Dict

from common.elements.legacy.loaders import Box


def clip_and_filter_annot(annot: Union[torch.Tensor, List[Box]], y1: int, x1: int, y2: int, x2: int, annot_format='yx', return_indices=False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]:
    """
    Moves the bounding boxes in annot to the tile defined by x1, y1, x2, y2.
    The annotations are automatically clipped or removed if does not fit in the tile.

    :param annot: Tensor with annotations [[y1, x1, y2, x2, class_id]] if annot_format == 'yxyx'
    :param y1: top of the tile
    :param x1: left of the tile
    :param y2: bottom of the tile
    :param x2: top of the tile
    :param annot_format: Format of the annotations ('yxyx', 'xyxy', etc)
    :param return_indices: If True the indices of the annotations that were kept are returned
    :return: the filtered annotations and optionally a list of kept indices

    >>> clip_and_filter_annot(torch.tensor([[10, 10, 25, 20, 0], [0, 0, 10, 15, 0]]), 0, 0, 25, 25)
    tensor([[10, 10, 25, 20,  0],
            [ 0,  0, 10, 15,  0]])
    >>> clip_and_filter_annot(torch.tensor([[10, 10, 25, 20, 0], [0, 0, 10, 15, 0]]), 15, 15, 18, 50)
    tensor([[0, 0, 3, 5, 0]])
    >>> clip_and_filter_annot(torch.tensor([[10, 10, 20, 20, 0]]), 20, 20, 25, 25)
    tensor([], size=(0, 5))
    """

    annot_tiles = []
    indices = []
    h, w = y2 - y1, x2 - x1
    for idx, a in enumerate(annot):
        if isinstance(a, Box):
            a2: Box = copy.copy(a)
            a2.y1 = min(max(a.y1 - y1, 0), h)
            a2.x1 = min(max(a.x1 - x1, 0), h)
            a2.y2 = min(max(a.y2 - y2, 0), h)
            a2.x2 = min(max(a.x2 - x2, 0), h)
            area = (a2.y2 - a2.y1) * (a2.x2 - a2.x1)
        else:
            if annot_format == 'yx' or annot_format == 'yxyx' or annot_format == 'yxi':
                a2: torch.Tensor = a.clone()
                a2[0] = min(max(a[0] - y1, 0), h)
                a2[1] = min(max(a[1] - x1, 0), w)
                a2[2] = min(max(a[2] - y1, 0), h)
                a2[3] = min(max(a[3] - x1, 0), w)
                area = (a2[2] - a2[0]) * (a2[3] - a2[1])
            elif annot_format == 'xy' or annot_format == 'xyi':
                a2: torch.Tensor = a.clone()
                a2[1] = min(max(a[1] - y1, 0), h)
                a2[0] = min(max(a[0] - x1, 0), w)
                a2[3] = min(max(a[3] - y1, 0), h)
                a2[2] = min(max(a[2] - x1, 0), w)
                area = (a2[2] - a2[0]) * (a2[3] - a2[1])
            else:
                raise RuntimeError(f"Invalid annotation format {annot_format}")
        if area > 0:
            annot_tiles.append(a2)
            indices.append(idx)
    if len(annot_tiles) == 0:
        if not isinstance(annot[0], Box):
            annot_tiles = torch.empty((0, annot.shape[1]))
    else:
        if not isinstance(annot[0], Box):
            annot_tiles = torch.stack(annot_tiles)

    if return_indices:
        return annot_tiles, indices
    else:
        return annot_tiles


def create_tiles(tiles: List[List[int]], img: torch.Tensor, annot: torch.Tensor = None, img_shape='hwc', annot_format='yxyx') -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]]:
    """
    Generate a tiled image and the corresponding annotations from the list of tiles

    :param tiles: the tiles to create (format is 'yxyx')
    :param img: image to take tiles from.
    :param annot: the annotations associated with the image in format [[y1, x1, y2, x2, class_id]] (if None, the boxes are ignored)
    :param img_shape: The shape of the input image and output tiles (either 'hwc' or 'chw')
    :param annot_format: The type of the annotation (either 'yxyx' or 'xyxy'
    :return: A list with image tiles and a list with the associated annotations

    >>> tiles = [[0, 0, 5, 10], [5, 10, 10, 15]]
    >>> img = torch.zeros([10, 20, 3])
    >>> annot = torch.tensor([[0, 0, 5, 10, 0], [5, 10, 10, 20, 0]])
    >>> imgs, annots = create_tiles(tiles, img, annot)
    >>> (imgs[0].shape, imgs[1].shape, annots[0], annots[1])
    (torch.Size([5, 10, 3]), torch.Size([5, 5, 3]), tensor([[ 0,  0,  5, 10,  0]]), tensor([[0, 0, 5, 5, 0]]))
    """
    img_tiles = []
    annot_tiles = []
    for y1, x1, y2, x2 in tiles:
        if len(img.shape) == 2:
            # height width only
            img_tile = img[y1:y2, x1:x2]
        elif img_shape.lower() == 'hwc':
            img_tile = img[y1:y2, x1:x2, :]
        elif img_shape.lower() == 'chw':
            img_tile = img[:, y1:y2, x1:x2]
        else:
            raise ValueError(f"Image shape {img_shape} is not supported.")
        img_tiles.append(img_tile)
        if annot is not None:
            annot_tile = clip_and_filter_annot(annot, y1, x1, y2, x2, annot_format=annot_format)
            annot_tiles.append(annot_tile)

    if annot is not None:
        return img_tiles, annot_tiles
    else:
        return img_tiles


def create_tiles_with_mask(tiles: List[List[int]], img: Union[torch.Tensor, Dict[str, torch.Tensor]], boxes: torch.Tensor, instance_mask: Optional[torch.Tensor], class_mask: Optional[torch.Tensor] = None, img_shape='hwc', annot_format='yxyx') -> Tuple[
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[int]]:
    """
    Generate a tiled image and the corresponding annotations and masks from the list of tiles

    :param tiles: the tiles to create (format is 'yxyx')
    :param img: image to take tiles from of shape [c, h, w] or [h, w, c] depending on img_shape, can also be a Dict of tensors for multiple input images.
    :param boxes: the boxes associated with the image in format [[y1, x1, y2, x2, class_id]]
    :param instance_mask: the mask with shape [#inst, h, w] or [h, w, #inst], depending on img_shape
    :param img_shape: The shape of the img and mask and output tiles (either 'hwc' or 'chw')
    :param annot_format: The type of the annotation (either 'yxyx' or 'xyxy'
    :return: A list with image tiles and a list with the associated annotations, masks (both instance and class) and which indices have been kept

    >>> tiles = [[0, 0, 5, 10], [5, 10, 10, 15]]
    >>> img = torch.zeros([10, 20, 3])
    >>> mask = torch.zeros([10, 20, 2])
    >>> annot = torch.tensor([[0, 0, 5, 10, 0], [5, 10, 10, 20, 0]])
    >>> imgs, annots, masks, kept_indices = create_tiles_with_mask(tiles, img, annot, mask)
    >>> (imgs[0].shape, imgs[1].shape, masks[0].shape, masks[1].shape, annots[0], annots[1], kept_indices[0], kept_indices[1])
    (torch.Size([5, 10, 3]), torch.Size([5, 5, 3]), torch.Size([5, 10, 1]), torch.Size([5, 5, 1]), tensor([[ 0,  0,  5, 10,  0]]), tensor([[0, 0, 5, 5, 0]]), [0], [1])
    """
    img_tiles = []
    instance_mask_tiles = []
    class_mask_tiles = []
    annot_tiles = []
    kept_indices = []
    for y1, x1, y2, x2 in tiles:
        if boxes is not None:
            annot_tile, indices = clip_and_filter_annot(boxes, y1, x1, y2, x2, annot_format=annot_format, return_indices=True)
            annot_tiles.append(annot_tile)
            kept_indices.append(indices)
        else:
            annot_tiles.append(None)
            kept_indices.append(None)
            indices = None

        if img_shape == 'hwc':
            if isinstance(img, Dict):
                img_tile = {key: single_image[y1:y2, x1:x2, :] for key, single_image in img.items()}
            else:
                img_tile = img[y1:y2, x1:x2, :]
            if instance_mask is not None:
                if indices is None:
                    raise NotImplemented("Cannot create tiles from instance masks if the boxes are None")
                instance_mask_tile = instance_mask[y1:y2, x1:x2, indices]
            else:
                instance_mask_tile = None
            if class_mask is not None:
                class_mask_tile = class_mask[y1:y2, x1:x2, :]
            else:
                class_mask_tile = None
        else:
            if isinstance(img, Dict):
                img_tile = {key: single_image[:, y1:y2, x1:x2] for key, single_image in img.items()}
            else:
                img_tile = img[:, y1:y2, x1:x2]
            if instance_mask is not None:
                if indices is None:
                    raise NotImplemented("Cannot create tiles from instance masks if the boxes are None")
                instance_mask_tile = instance_mask[indices, y1:y2, x1:x2]
            else:
                instance_mask_tile = None
            if class_mask is not None:
                class_mask_tile = class_mask[:, y1:y2, x1:x2]
            else:
                class_mask_tile = None

        class_mask_tiles.append(class_mask_tile)  # Can be none if class mask was not passed
        instance_mask_tiles.append(instance_mask_tile)  # Can be none if instances mask was not passed
        img_tiles.append(img_tile)

    return img_tiles, annot_tiles, instance_mask_tiles, class_mask_tiles, kept_indices


def generate_random_tiles(img: Optional[torch.Tensor], tile_h: int, tile_w: int, num_tiles: int,
                          image_h: Optional[int] = None, image_w: Optional[int] = None) -> List[List[int]]:
    """
    Generate a random set of tiles

    :param img: Input image for height and width
    :param tile_h: tile height
    :param tile_w: tile width
    :param num_tiles: number of tiles
    :param image_h: image height if img is None
    :param image_w: image width if img is None
    :return: A list of num_tiles random tiles in format [[y1, x1, y2, x2]]

    >>> img = torch.zeros([10, 20, 3])
    >>> tiles = generate_random_tiles(img, 3, 6, 2)
    >>> (tiles[0][2] - tiles[0][0], tiles[1][3] - tiles[1][1])
    (3, 6)
    """
    if img is None:
        h = image_h
        w = image_w
    else:
        h, w, _ = img.shape
    tiles = []
    for i in range(num_tiles):
        y = randint(0, h - tile_h)
        x = randint(0, w - tile_w)
        tiles.append([y, x, y + tile_h, x + tile_w])
    return tiles


def generate_random_tiles_hw(img: Optional[torch.Tensor], min_tile_h: int, min_tile_w: int,
                             max_tile_h: int, max_tile_w: int, num_tiles: int,
                             image_h: Optional[int] = None, image_w: Optional[int] = None) -> List[List[int]]:
    if img is None:
        h = image_h
        w = image_w
    else:
        h, w, _ = img.shape
    tiles = []
    for i in range(num_tiles):
        tile_h = randint(min_tile_h, max_tile_h)
        tile_w = randint(min_tile_w, max_tile_w)
        y = randint(0, h - tile_h)
        x = randint(0, w - tile_w)
        tiles.append([y, x, y + tile_h, x + tile_w])
    return tiles


def generate_random_tiles_annot(img: torch.Tensor, tile_h: int, tile_w: int, num_tiles: int, annot: torch.Tensor) -> List[List[int]]:
    """
    Generate a random set of tiles that are guaranteed to have a bounding box from annot in them (positives)

    :param img: Input image for height and width
    :param tile_h: tile height
    :param tile_w: tile width
    :param num_tiles: number of tiles
    :param annot: the annotations associated with img with shame
    :return: A list of num_tiles random tiles of shape [[y1, x1, y2, x2]]

    >>> annot = torch.tensor([[5, 5, 10, 15, 0]])
    >>> img = torch.zeros([20, 25, 3])
    >>> tile = generate_random_tiles_annot(img, 15, 15, 1, annot)[0]
    >>> tile[0] <= 5, tile[1] <= 5, tile[2] >= 15, tile[3] >= 10
    (True, True, True, True)
    >>> tile = generate_random_tiles_annot(img, 2, 2, 1, annot)[0]
    >>> (tile[0] >= 5, tile[1] >= 5, tile[2] <= 15, tile[3] <= 10)
    (True, True, True, True)
    """
    h, w, c = img.shape
    if tile_h > h or tile_w > w:
        raise RuntimeError("Cannot create tiles larger than the image")
    tiles = []
    for i in range(num_tiles):
        annot_y1, annot_x1, annot_y2, annot_x2, _ = annot[randint(0, annot.shape[0] - 1)]
        annot_h = annot_y2 - annot_y1
        annot_w = annot_x2 - annot_x1
        if annot_h > tile_h:
            # Take a random tile inside the annotation
            min_y = annot_y1
            max_y = annot_y2 - tile_h
        else:
            # Take a random tile around the annotation
            min_y = max(annot_y2 - tile_h, 0)
            max_y = min(annot_y1, h - tile_h)

        if annot_w > tile_w:
            # Take a random tile inside the annotation
            min_x = annot_x1
            max_x = annot_x2 - tile_w
        else:
            # Take a random tile around the annotation
            min_x = max(annot_x2 - tile_w, 0)
            max_x = min(annot_x1, w - tile_w)

        y = randint(min_y, max_y)
        x = randint(min_x, max_x)
        tiles.append([y, x, y + tile_h, x + tile_w])
    return tiles


def generate_random_tiles_in_annot(img: torch.Tensor, tile_h: int, tile_w: int, num_tiles: int, annot: torch.Tensor) -> List[List[int]]:
    """
    Generate a random set of tiles that are guaranteed to be within a bounding box from annot

    :param img: Input image for height and width
    :param tile_h: tile height
    :param tile_w: tile width
    :param num_tiles: number of tiles
    :param annot: the  annotations associated with img
    :return: A list of num_tiles random tiles of shape [[y1, x1, y2, x2]]

    >>> annot = torch.tensor([[5, 5, 10, 15, 0]])
    >>> img = torch.zeros([20, 25, 3])
    >>> tile = generate_random_tiles_in_annot(img, 5, 5, 1, annot)[0]
    >>> (tile[0] >= 5, tile[1] >= 5, tile[2] <= 10, tile[3] <= 15)
    (True, True, True, True)
    """
    h, w, c = img.shape
    tiles = []
    for i in range(num_tiles):
        annot_y1, annot_x1, annot_y2, annot_x2, _ = annot[randint(0, annot.shape[0] - 1)]
        annot_h = annot_y2 - annot_y1
        annot_w = annot_x2 - annot_x1
        if annot_h >= tile_h:
            # Take a random tile inside the annotation
            min_y = annot_y1
            max_y = annot_y2 - tile_h
        else:
            raise RuntimeError(f"Trying to take a random tile {(tile_h, tile_w)} from witin a smaller tile {(annot_h, annot_w)}.")

        if annot_w >= tile_w:
            # Take a random tile inside the annotation
            min_x = annot_x1
            max_x = annot_x2 - tile_w
        else:
            raise RuntimeError(f"Trying to take a random tile {(tile_h, tile_w)} from witin a smaller tile {(annot_h, annot_w)}.")

        y = randint(int(min_y), int(max_y))
        x = randint(int(min_x), int(max_x))
        tiles.append([y, x, y + tile_h, x + tile_w])
    return tiles


def generate_fixed_tiles(img, tile_h, tile_w, margin_h, margin_w, img_h=0, img_w=0, data_formats: str = "HWC") -> List[List[int]]:
    """
    Generate a fixed list of overlapping tiles. The tiles are oriented in a grid.
    To calculate the y and x index of a tile from its index :func:`tile_idx_to_coords`

    :param img: Input image for height and width
    :return: A list of num_tiles random tiles of shape [[y1, x1, y2, x2]]

    >>> img = torch.zeros([14, 14, 3])
    >>> generate_fixed_tiles(img, 12, 12, 2, 2)
    [[0, 0, 12, 12]]
    >>> img = torch.zeros([32, 32, 3])
    >>> generate_fixed_tiles(img, 20, 20, 5, 5)
    [[0, 0, 20, 20], [0, 10, 20, 30], [10, 0, 30, 20], [10, 10, 30, 30]]
    >>> img = torch.zeros([10, 10, 3])
    >>> generate_fixed_tiles(img, 8, 8, 4, 4)
    [[0, 0, 8, 8], [0, 1, 8, 9], [1, 0, 9, 8], [1, 1, 9, 9]]
    """

    if data_formats == "HWC":
        if img is None:
            h, w, c = img_h, img_w, 0
        else:
            if img.ndim == 2:
                img = np.expand_dims(img, -1)
            h, w, c = img.shape
    elif data_formats == "CHW":
        if img is None:
            c, h, w = 0, img_h, img_w
        else:
            if img.ndim == 2:
                img = np.expand_dims(img, 0)
            c, h, w = img.shape
    else:
        raise ValueError(f"Data format {data_formats} is not supported for generating fixed tiles.")

    inner_tile_h = tile_h - (2 * margin_h)
    inner_tile_w = tile_w - (2 * margin_w)
    if inner_tile_h == 0:
        num_tiles_y = (h - (2 * margin_h))
    else:
        num_tiles_y = (h - (2 * margin_h)) // inner_tile_h

    if inner_tile_h == 0:
        num_tiles_x = (w - (2 * margin_w))
    else:
        num_tiles_x = (w - (2 * margin_w)) // inner_tile_w

    tiles = []
    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            if inner_tile_h > 0:
                y1 = (tile_y * inner_tile_h)
            else:
                y1 = tile_y
            if inner_tile_w > 0:
                x1 = (tile_x * inner_tile_w)
            else:
                x1 = tile_x
            y2 = y1 + inner_tile_h + (2 * margin_h)
            x2 = x1 + inner_tile_w + (2 * margin_w)
            tiles.append([y1, x1, y2, x2])

    if len(tiles) == 0:
        raise RuntimeError(f"No tiles could be generated with tile height: {tile_h}, tile width: {tile_w},"
                           f" height margin: {margin_h}, width margin: {margin_w}, data format: {data_formats} "
                           f"and input data shape: {img.shape}")
    return tiles


def get_num_tiles(img_h, img_w, tile_h, tile_w, margin_h, margin_w) -> Tuple[int, int]:
    """
    Returns the number of tiles that can be generated from an image. All tiles should fit in the image.

    :param img_h: Height of the image
    :param img_w: Width of the image
    :param tile_h: Height of the tile
    :param tile_w: Width of the tile
    :param margin_h: Inner margin of the tile. This determined overlap
    :param margin_w: Inner margin of the tile. This determined overlap
    :return: The number of tiles in y direction and x direction

    >>> get_num_tiles(32, 40, 20, 20, 5, 5)
    (2, 3)
    """
    inner_tile_h = tile_h - (2 * margin_h)
    inner_tile_w = tile_w - (2 * margin_w)

    if inner_tile_h == 0:
        num_tiles_y = (img_h - (2 * margin_h))
    else:
        num_tiles_y = (img_h - (2 * margin_h)) // inner_tile_h

    if inner_tile_h == 0:
        num_tiles_x = (img_w - (2 * margin_w))
    else:
        num_tiles_x = (img_w - (2 * margin_w)) // inner_tile_w

    return num_tiles_y, num_tiles_x


def tile_idx_to_rect(idx: int, img_h: int, img_w: int, tile_h: int, tile_w: int, margin_h: int, margin_w: int) -> Tuple[int, int, int, int]:
    """
    Converts a tile index to a rectangle where the tile originated from.

    :param idx: Index of the tile
    :param img_h: Height of the image
    :param img_w: Width of the image
    :param tile_h: Height of the tile
    :param tile_w: Width of the tile
    :param stride_y: Stride of the tile. This determined overlap
    :param stride_x: Stride of the tile. This determined overlap
    :return: the y1, x1, y2, x2 coordinate of the tile

    >>> tile_idx_to_rect(0, 30, 40, 20, 20, 5, 5)
    (0, 0, 20, 20)
    >>> tile_idx_to_rect(1, 30, 40, 20, 20, 5, 5)
    (0, 10, 20, 30)
    >>> tile_idx_to_rect(3, 30, 40, 20, 20, 5, 5)
    (10, 0, 30, 20)
    """

    y1, x1 = tile_idx_to_coord(idx, img_h, img_w, tile_h, tile_w, margin_h, margin_w)

    inner_tile_h = tile_h - (2 * margin_h)
    inner_tile_w = tile_w - (2 * margin_w)

    y1 = (y1 * inner_tile_h)
    x1 = (x1 * inner_tile_w)
    y2 = y1 + inner_tile_h + (2 * margin_h)
    x2 = x1 + inner_tile_w + (2 * margin_w)
    return y1, x1, y2, x2


def tile_idx_to_inner_rect(idx: int, img_h: int, img_w: int, tile_h: int, tile_w: int, margin_h: int, margin_w: int) -> Tuple[int, int, int, int]:
    """
    Converts a tile index to a rectangle where the tile originated from (with the margins subtracted)

    :param idx: Index of the tile
    :param img_h: Height of the image
    :param img_w: Width of the image
    :param tile_h: Height of the tile
    :param tile_w: Width of the tile
    :param margin_h: Stride of the tile. This determined overlap
    :param margin_w: Stride of the tile. This determined overlap
    :return: the y1, x1, y2, x2 coordinate of the tile

    >>> tile_idx_to_rect(0, 30, 40, 20, 20, 5, 5)
    (5, 5, 15, 15)
    >>> tile_idx_to_rect(1, 30, 40, 20, 20, 5, 5)
    (5, 15, 15, 25)
    >>> tile_idx_to_rect(3, 30, 40, 20, 20, 5, 5)
    (15, 5, 25, 15)
    """

    y1, x1 = tile_idx_to_coord(idx, img_h, img_w, tile_h, tile_w, margin_h, margin_w)

    inner_tile_h = tile_h - (2 * margin_h)
    inner_tile_w = tile_w - (2 * margin_w)

    y1 = (y1 * inner_tile_h)
    x1 = (x1 * inner_tile_w)
    y2 = y1 + inner_tile_h + (2 * margin_h)
    x2 = x1 + inner_tile_w + (2 * margin_w)
    return y1 + margin_h, x1 + margin_w, y2 - margin_h, x2 - margin_w


def tile_idx_to_ctr(idx: int, img_h: int, img_w: int, tile_h: int, tile_w: int, margin_h: int, margin_w: int) -> Tuple[int, int]:
    """
    Converts a tile index to a rectangle where the tile originated from.

    :param idx: index of the tile
    :param img_h: Height of the image
    :param img_w: Width of the image
    :param tile_h: Height of the tile
    :param tile_w: Width of the tile
    :param margin_h: Stride of the tile. This determined overlap
    :param margin_w: Stride of the tile. This determined overlap
    :return: the y, x coordinate where the tile with index idx is

    >>> tile_idx_to_ctr(0, 30, 40, 20, 20, 5, 5)
    (10, 10)
    >>> tile_idx_to_ctr(1, 30, 40, 20, 20, 5, 5)
    (10, 20)
    >>> tile_idx_to_ctr(3, 30, 40, 20, 20, 5, 5)
    (20, 10)
    """

    y1, x1 = tile_idx_to_coord(idx, img_h, img_w, tile_h, tile_w, margin_h, margin_w)

    inner_tile_h = tile_h - (2 * margin_h)
    inner_tile_w = tile_w - (2 * margin_w)

    if inner_tile_h != 0:
        y1 = (y1 * inner_tile_h)
    if inner_tile_h != 0:
        x1 = (x1 * inner_tile_w)
    y2 = y1 + inner_tile_h + (2 * margin_h)
    x2 = x1 + inner_tile_w + (2 * margin_w)
    return (y1 + y2) // 2, (x1 + x2) // 2,


def tile_idx_to_coord(idx: int, img_h: int, img_w: int, tile_h: int, tile_w: int, margin_h: int, margin_w: int) -> Tuple[int, int]:
    """
    Get the tile y and x coordinate given an index of a tile.
    For example:
    y = idx // num_tiles_x
    x = idx % num_tiles_x,
    where num_tiles_x is the number of tiles in the x direction

    :param img_h: Height of the image
    :param img_w: Width of the image
    :param tile_h: Height of the tile
    :param tile_w: Width of the tile
    :param margin_y: Inner margin. This determined overlap
    :param margin_x: Inner margin. This determined overlap
    :return: the y, x coordinate where the tile with index idx is.

    :example:

    >>> idx = 0
    >>> tile_idx_to_coord(idx, 110, 210, 60, 60, 5, 5)
    (0, 0)
    >>> idx = 1
    >>> tile_idx_to_coord(idx, 110, 210, 60, 60, 5, 5)
    (0, 1)
    >>> idx = 4
    >>> tile_idx_to_coord(idx, 110, 210, 60, 60, 5, 5)
    (1, 0)
    >>> idx = 6
    >>> tile_idx_to_coord(idx, 110, 210, 60, 60, 5, 5)
    (1, 2)
    """
    _, num_tiles_x = get_num_tiles(img_h, img_w, tile_h, tile_w, margin_h, margin_w)

    y = idx // num_tiles_x
    x = idx % num_tiles_x
    return y, x


def pad_image(img: np.ndarray, h: int, w: int, tile_h: int, tile_w: int, margin_h: int, margin_w: int) -> np.ndarray:
    """
    Pad an image so that tiling has the optimal fit (this method is used by tiling collators)

    :param h: height of the image
    :param w: width of the image
    :param pad_modulo_y: image h should be dividable by
    :param pad_modulo_x: image w should be dividable by
    :return:

    :example:
    >>> img = np.zeros([20, 20, 3])
    >>> img = pad_image(img, 10, 10, 6, 6, 0, 0)
    >>> img.shape
    (22, 22, 3)
    """

    assert len(img.shape) == 3, "an image should have three dimensions (h, w, c)"
    padding = get_padding(h, w, tile_h, tile_w, margin_h, margin_w)
    img = np.pad(img, padding)
    return img


def pad_annotations(annot: np.ndarray, h: int, w: int, tile_h: int, tile_w: int, margin_h: int, margin_w: int) -> np.ndarray:
    """
    Pad an list of annotations so that they correspond to the padded image

    :param annot: Annotation in format [[y1, x1, y2, x2, class_id]]
    :param h: height of the image
    :param w: width of the image
    :return: padded annotations

    :example:
    >>> annot = np.array([[5, 5, 7, 7, 0]])
    >>> pad_annotations(annot, 10, 10, 6, 6, 0, 0)
    array([[6, 6, 8, 8, 0]])
    """

    ((y, _), (x, _), (_, _)) = get_padding(h, w, tile_h, tile_w, margin_h, margin_w)
    annot2 = [[y1 + y, x1 + x, y2 + y, x2 + x, class_id] for y1, x1, y2, x2, class_id in annot]
    annot2 = np.array(annot2)
    return annot2


def unpad_boxes(boxes: Union[List[List[int]], np.ndarray], h: int, w: int, tile_h: int, tile_w: int, margin_h: int, margin_w: int) -> np.ndarray:
    """
    Remove padding from a list of boxes. This is logically the inverse of :meth:`~elements.preprocess.tiling.pad_annotations`

    :param boxes: Annotation in format [[y1, x1, y2, x2, class_id, prob]]
    :param h: height of the image
    :param w: width of the image
    :return:

    :example:
    >>> annot = [[6, 6, 8, 8, 0, 1]]
    >>> unpad_boxes(annot, 10, 10, 6, 6, 0, 0)
    array([[5, 5, 7, 7, 0, 1]])
    """

    ((y, _), (x, _), (_, _)) = get_padding(h, w, tile_h, tile_w, margin_h, margin_w)

    boxes = np.array(boxes)
    boxes[:, :4] = boxes[:, :4] - [y, x, y, x]
    return boxes


def untile_boxes(boxes: Union[List[List[int]], np.ndarray], h: int, w: int, tile_h: int, tile_w: int, margin_h: int, margin_w: int, tile_y1, tile_x1) -> np.ndarray:
    """
    This method undoes tiling effects of the boxes and recalculates coordinates so that they are relative to the original image's coordinate system.
    It corrects padding, tiling and margin.

    :param boxes: Annotation in format [[y1, x1, y2, x2, class_id, prob]]
    :param h: height of the original image
    :param w: width of the original image
    :param tile_h: height of the tiles
    :param tile_w: width of the tiles
    :param margin_h: margin height of the tiles
    :param margin_w: margin width of the tiles
    :param tile_y1: the top coordinate of the tile this box belongs to
    :param tile_x1: the left coordinate of the tile this box belongs to
    :return: The untiled boxes

    >>> annot = [[6, 6, 8, 8, 0, 1]]
    >>> untile_boxes(annot, 10, 10, 6, 6, 0, 0, 100, 50)
    array([[105,  55, 107,  57,   0,   1]])
    """
    boxes = np.array(boxes)
    boxes = unpad_boxes(boxes, h, w, tile_h, tile_w, margin_h, margin_w)
    boxes[:, :4] = boxes[:, :4] + [tile_y1, tile_x1, tile_y1, tile_x1]
    return boxes


def get_padding_tl(h: int, w: int, tile_h: int, tile_w: int, margin_h: int, margin_w: int) -> Tuple[int, int]:
    """
    Return the added padding of the top and left

    :param h: height of the image
    :param w: width of the image
    :return: The number of pixels added to the top en left
    """

    ((t, _), (l, _), (_, _)) = get_padding(h, w, tile_h, tile_w, margin_h, margin_w)
    return t, l


def get_padded_hw(h: int, w: int, tile_h: int, tile_w: int, margin_h: int, margin_w: int) -> Tuple[int, int]:
    """
    Return the new height and width of a padded image

    :param h: Original image height
    :param w: Original image width
    :return: The height and width after adding padding
    """
    ((t, b), (l, r), (_, _)) = get_padding(h=h, w=w, tile_h=tile_h, tile_w=tile_w, margin_h=margin_h, margin_w=margin_w)
    return h + t + b, w + l + r


def get_padding(h: int, w: int, tile_h: int, tile_w: int, margin_h: int, margin_w: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Return three-dimensional image padding for np.pad() so that the tiling always fits

    :param h: height of the image
    :param w: width of the image
    :return:

    :example:
    >>> get_padding(10, 10, 5, 5, 0, 0)
    ((0, 0), (0, 0), (0, 0))
    """
    t, b, l, r = 0, 0, 0, 0

    inner_tile_h = tile_h - (2 * margin_h)
    inner_tile_w = tile_w - (2 * margin_w)

    if inner_tile_h <= 0 or inner_tile_w <= 0:
        raise RuntimeError("2 * margin is larger than the tile, please reduce the margins.")

    div_h = (h - (2 * margin_h)) // inner_tile_h
    mod_h = (h - (2 * margin_h)) % inner_tile_h
    if mod_h > 0:
        div_h += 1
    new_h = (div_h * inner_tile_h) + (2 * margin_h)

    pad_h = new_h - h
    t = pad_h // 2
    b = pad_h - t

    div_w = (w - (2 * margin_w)) // inner_tile_w
    mod_w = (w - (2 * margin_w)) % inner_tile_w
    if mod_w > 0:
        div_w += 1
    new_w = (div_w * inner_tile_w) + (2 * margin_w)

    pad_w = new_w - w
    l = pad_w // 2
    r = pad_w - l

    # Add an extra border to also detect object in the border of large images
    if l < margin_w or r < margin_w:
        l += inner_tile_w // 2
        r += inner_tile_w - (inner_tile_w // 2)
    if t < margin_h or b < margin_h:
        t += inner_tile_h // 2
        b += inner_tile_h - (inner_tile_h // 2)

    return (t, b), (l, r), (0, 0)
