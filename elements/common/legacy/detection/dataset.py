import copy
import glob
from typing import Dict, List, Callable

from torch.utils.data import Dataset
import os
import numpy as np

from common.elements.legacy.loaders import Box, LoadBoxes, LoadImage
from common import deprecated


class LabelImgDataset(Dataset):
    """
    Dataset for images annotated with LabelImg
    """

    @deprecated("This dataset is deprecated, please use the GenericDataset from common.data instead.")
    def __init__(self, images_folder, annot_folder, class_white_list=None, transforms=None, fixed_class_name=None, class_names = None, load_annot_func: Callable[[str], List[Box]]=LoadBoxes(), load_image_func: Callable[[str], np.ndarray]=LoadImage(), filetypes=LoadImage().get_types()):
        """

        :param images_folder:
        :param annot_folder:
        :param class_white_list:
        :param transforms:
        :param fixed_class_name:
        """
        if class_white_list is not None:
            fixed_class_name = None

        if annot_folder is not None:
            self.annot_folder = annot_folder
        else:  # Use the images folder if no seperate annotations folder is specified.
            self.annot_folder = images_folder
        self.class_names = class_names
        self.load_annot_func = load_annot_func
        self.load_image_func = load_image_func
        self.filetypes = filetypes
        self.images_folder = images_folder
        self.transforms = transforms
        self.class_white_list = class_white_list
        self.fixed_class_name = fixed_class_name
        self.samples = []  # List of dict with keys: ('filename', 'annot', 'boxes', 'hw')
        self.add_samples(self.images_folder)
        self.update_class_names()
        self.update_annot()
        return

    def get_annotations_per_filename(self) -> Dict[str, np.ndarray]:
        result = {}
        for sample in self.samples:
            result[os.path.basename(sample['filename'])] = sample['annot']
        return result

    def get_class_name(self, idx):
        return self.class_names[int(idx)]

    def num_classes(self):
        if len(self.class_names) == 0:
            raise RuntimeError("The number of classes is not known.")
        return len(self.class_names)

    def filter_boxes(self, boxes):

        #whitelist
        if self.class_white_list is not None:
            boxes = [box for box in boxes if box.class_name in self.class_white_list]

        #fixed names
        if self.fixed_class_name is not None:
            for i in range(len(boxes)):
                boxes[i].class_name = self.fixed_class_name

        return boxes

    def add_samples(self, dir):
        curdir = os.getcwd()
        os.chdir(dir)

        filenames = []
        for ext in self.filetypes:
            filenames.extend(list(glob.glob(f"*{ext}")))

        for filename in filenames:
            sample = {}
            sample['filename'] = os.path.abspath(filename)
            sample['hw'] = (0, 0)

            # load boxes
            boxes = self.load_annot_func(self.get_annotations_filename(filename))
            boxes = self.filter_boxes(boxes)
            sample['boxes'] = boxes

            self.samples.append(sample)
        os.chdir(curdir)

    def get_annotations(self, idx):
        return self.samples[idx]['annot']

    def get_image_filename(self, idx):
        return self.samples[idx]['filename']

    def get_annotations_filename(self, image_filename):
        filename = os.path.join(self.annot_folder, os.path.splitext(image_filename)[0])
        return filename

    def update_class_names(self):
        if self.class_names is None:
            self.class_names = sorted(list(set([box.class_name for sample in self.samples for box in sample['boxes']])), key=str.lower)
        self.class_ids = {name: id for id, name in enumerate(self.class_names)}
        for s in range(len(self.samples)):
            for b in range(len(self.samples[s]['boxes'])):
                box = self.samples[s]['boxes'][b]
                box.class_id = self.class_ids[box.class_name]

    def update_annot(self):
        for s in range(len(self.samples)):
            if len(self.samples[s]['boxes']) > 0:
                annot = np.vstack([box.numpy() for box in self.samples[s]['boxes']])
            else:
                annot = np.ndarray([0, 5])
            self.samples[s]['annot'] = annot

    def load_image(self, idx):
        filename = self.samples[idx]['filename']
        return self.load_image_func(filename)

    def load_annotations(self, idx):
        return self.samples[idx]['annot']

    def load_boxes(self, idx):
        return self.samples[idx]['boxes']

    def mean_stddev(self):
        mean_sum = 0
        sdev_sum = 0
        for sample in self:
            img = sample["img"].numpy()
            mean_sum = mean_sum + np.mean(img)
            sdev_sum = sdev_sum + np.std(img)
        return mean_sum / len(self), sdev_sum / len(self)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx).copy()
        boxes = copy.deepcopy(self.load_boxes(idx))

        # Transform sample and return
        sample = {'img': img, 'annot': annot, 'filename': self.samples[idx]['filename'], 'boxes': boxes}
        if self.transforms:
            sample = self.transforms(sample)
        self.samples[idx]['hw'] = (img.shape[0], img.shape[1])

        return sample

