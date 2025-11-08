from statistics import mean
from typing import Tuple, Optional
import numpy as np
import os
import torch.utils
import torch.utils.data
from enum import Enum

from common import deprecated


class CollatorType(Enum):
    RANDOM_TILES = 0
    RANDOM_BOX_TILES = 1
    FIXED_TILES = 2


class HSBoxesDataset(torch.utils.data.Dataset):
    @deprecated("This dataset is deprecated, please use the GenericDataset from common.data instead.")
    def __init__(self, path: str, transform=None):
        super().__init__()
        self.transform = transform
        self.image_paths = []
        self.image_shapes = {}
        self._load_data_dir(path)
        self._load_class_names()
        return

    def get_filename(self, idx):
        return self.image_paths[idx]

    def get_image_shape(self, idx):
        key = self.image_paths[idx]
        if key not in self.image_shapes.keys():
            self._load_image_and_boxes(idx)  # load image to store shape in self.image_shapes
        return self.image_shapes[key]

    def get_image(self, idx):
        npz = np.load(self.image_paths[idx])
        return npz['scan']

    def _load_class_names(self):
        if len(self.image_paths) > 0:
            npz = np.load(self.image_paths[0])
            self.class_names = npz['classnames']
        else:
            self.class_names = []

    def _load_image_and_boxes(self, idx):
        image = np.load(self.image_paths[idx])
        image, annot, segmap, raw, white, dark = self._get_npz(image)
        self.image_shapes[self.image_paths[idx]] = image.shape
        return image, annot, segmap, raw, white, dark

    def get_wavelengths(self, idx):
        zipped = np.load(self.image_paths[idx])
        if "wavelength" in zipped.files:
            return zipped["wavelength"]
        else:
            return []

    def get_class_name(self, idx):
        return self.class_names[int(idx)]

    def num_classes(self):
        return len(self.class_names)

    def get_min_box_size(self):
        """
        Return the minimum height and width of a box
        """

        width = []
        height = []
        for idx in range(len(self.image_paths)):
            _, boxes, _, _, _, _ = self._load_image_and_boxes(idx)
            for box in boxes:
                y1, x1, y2, x2, class_id = box
                width.append(x2 - x1)
                height.append(y2 - y1)
        min_width = min(width)
        min_height = min(height)

        return min_height, min_width

    def get_mean(self):
        """
        Returns the mean of the mean of all cubes
        """
        mn = []
        for idx in range(len(self.image_paths)):
            img, _, _, _, _, _ = self._load_image_and_boxes(idx)
            mn.append(float(np.mean(img)))
        mn = mean(mn)
        return mn

    def get_sd(self):
        """
        Returns the mean of the standard deviation of all cubes
        """
        sd = []
        for idx in range(len(self.image_paths)):
            img, _, _, _, _, _ = self._load_image_and_boxes(idx)
            sd.append(float(np.std(img)))
        sd = mean(sd)
        return sd

    def _load_data_dir(self, dir_path: str, recursive: bool = False):
        # get all items in directory
        items = os.listdir(dir_path)
        # get absolute paths
        items = [os.path.join(dir_path, item) for item in items]
        # get paths which are files
        files = [item for item in items if os.path.isfile(item) and item[-3:]=="npz"]
        self.image_paths.extend(files)
        # include items in subdirectories if recursive mode is enabled
        if recursive:
            [self._load_data_dir(item) for item in items if os.path.isdir(item)]

    def __len__(self):
        return len(self.image_paths)

    def _generate_segmap(self, img, boxes):
        # Just draw boxes. We let the collator handle the choice of tiles from these boxes.
        h, w, c = img.shape
        segmap = np.zeros([h, w, 1])
        for y1, x1, y2, x2, id in boxes:
            segmap[int(y1):int(y2), int(x1):int(x2), 0] = id
        return segmap

    def __getitem__(self, idx: int):
        img, boxes, segmap, raw, white, dark = self._load_image_and_boxes(idx)
        if segmap is None:
            if boxes is not None:
                segmap = self._generate_segmap(img, boxes)
            else:
                raise RuntimeError("No boxes and no segmentation map known.")
        sample = {'img': img, 'annot': boxes, 'segmap': segmap, 'raw': raw, 'white': white, 'dark': dark}
        return sample if self.transform is None else self.transform(sample)

    def _get_npz(self, npz: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        files = npz.files

        #Get raw
        raw = npz['raw'] if 'raw' in files else None

        #Get image data
        img = npz['scan'] if 'scan' in files else None

        #Get white ref
        white = npz['white'] if 'white' in files else None

        #Get dark ref
        dark = npz['dark'] if 'dark' in files else None

        #Get segmap
        segmap = None
        if 'segmap' in files:
            segmap = npz['segmap']
            segmap = np.expand_dims(segmap, 2) # from HW to HWC

        #Get boxes annotation
        if 'annotations' in files and len(npz["annotations"] > 0):
            annot = npz['annotations']
            #xyxy => yxyx
            annot2 = annot.copy()
            x1, y1, w, h = annot2[:, 0], annot2[:, 1], annot2[:, 2], annot2[:, 3]
            annot[:, 0] = y1
            annot[:, 1] = x1
            annot[:, 2] = y1 + h
            annot[:, 3] = x1 + w
        else:
            annot = np.empty([0, 5])

        return img, annot, segmap, raw, white, dark


def load_tests(loader, tests, ignore):
    import doctest
    tests.addTests(doctest.DocTestSuite())
    return tests


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
