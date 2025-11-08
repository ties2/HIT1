import os
import uuid
from typing import Any, Tuple, Optional

import torchvision
from torch.utils.data import Dataset
import numpy as np
import random

from .coders import encode
from common.elements.legacy.loaders import Box
from common.elements.legacy.detection import LabelImgDataset, Normalizer


class CentroidNetDataset(Dataset):

    def __init__(self, images_folder: str, annotations_folder: str, crop=(256, 256), max_dist=100, repeat=1, norm_sub=127, norm_div=256,
                 num_classes=2, transpose=np.array([[0, 1], [1, 0]]),
                 stride=np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]]), cache_folder=None):

        # Delegate image loading and bounding boxes loading to the LabeLImgDataset
        t = []
        t.append(Normalizer(mean=norm_sub, std=norm_div))
        self.od_dataset = LabelImgDataset(images_folder, annotations_folder,
                                          transforms=torchvision.transforms.Compose(t))

        self.count = 0
        self.crop = None
        self.transpose: Optional[Tuple[Any, Any]] = None
        self.stride: Optional[Tuple[Any, Any]] = None
        self.train_mode = None

        self.images = list()
        self.targets = list()

        self.set_crop(crop)
        self.set_repeat(repeat)
        self.set_transpose(transpose)
        self.set_stride(stride)
        self.num_classes = num_classes
        self.max_dist = max_dist

        if cache_folder is None:
            self.cache_path = os.path.join(os.path.sep, "tmp", "centroidnet_cache", str(uuid.uuid4()))
        else:
            self.cache_path = cache_folder
        os.makedirs(self.cache_path, exist_ok=True)
        self.train()

    def create_encoded_pair(self, index):
        # Get image from the od dataloader
        sample = self.od_dataset[index]
        boxes = []
        for box in sample['boxes']:
            b: Box = box
            boxes.append(np.array([b.y, b.x, b.y1, b.y2, b.x1, b.x2, b.class_id]))

        h, w, c = sample['img'].shape

        # Adjust crop is necessary
        if self.crop is not None:
            crop = min(h, w, self.crop[0], self.crop[1])
            if crop != self.crop[0]:
                print(f"Warning: random crop adjusted to {[crop, crop]}")
                self.set_crop([crop, crop])

        cache_file = os.path.join(self.cache_path, os.path.basename(sample['filename'] + f".{self.max_dist}" + f".{self.num_classes}" + ".npy"))
        if os.path.isfile(cache_file):
            target = np.load(cache_file)
        else:
            target = encode(boxes, h, w, self.max_dist, self.num_classes)
            np.save(cache_file, target)

        image = np.transpose(sample['img'].astype(np.float32), [2, 0, 1])
        target = target.astype(np.float32)

        return image, target

    def eval(self):
        self.train_mode = False

    def train(self):
        self.train_mode = True

    def set_repeat(self, repeat):
        if repeat < 0:
            self.repeat = 1
        self.repeat = repeat

    def set_crop(self, crop):
        if crop[0] is None or crop[1] is None or crop[0] == 0 or crop[1] == 0:
            self.crop = None
        else:
            self.crop = crop

    def set_transpose(self, transpose):
        if np.all(transpose == np.array([[0, 1]])):
            self.transpose = transpose

    def set_stride(self, stride):
        if np.all(stride == np.array([[1, 1]])):
            self.stride = stride

    @staticmethod
    def adjust_vectors(img, transpose, stride):
        if transpose is not None:
            img2 = img.copy()
            img[0] = img2[transpose[0]]
            img[1] = img2[transpose[1]]
        if stride is not None:
            img2 = img.copy()
            img[0] = img2[0] * stride[0]
            img[1] = img2[1] * stride[1]
        return img

    @staticmethod
    def adjust_image(img, transpose, slice, crop, stride):
        if transpose is not None:
            img = np.transpose(img, (0, transpose[0] + 1, transpose[1] + 1))
        if slice is not None:
            img = img[:, slice[0]:slice[0] + crop[0], slice[1]:slice[1] + crop[1]]
        if stride is not None:
            img = img[:, ::stride[0], ::stride[1]]
        return img

    def get_target(self, img: np.array, transpose, slice, crop, stride):
        img[0:2] = self.adjust_vectors(img[0:2], transpose, stride)
        img[2:4] = self.adjust_vectors(img[2:4], transpose, stride)
        img = self.adjust_image(img, transpose, slice, crop, stride)
        return img

    def get_input(self, img: np.array, transpose, slice, crop, stride):
        img = self.adjust_image(img, transpose, slice, crop, stride)
        return img

    def __getitem__(self, index):
        index = index // self.repeat
        input, target = self.create_encoded_pair(index)

        if self.stride is None and self.transpose is None and self.crop is None or not self.train_mode:
            return input.astype(np.float32), target.astype(np.float32)

        if self.transpose is not None:
            transpose = random.choice(self.transpose)
        else:
            transpose = None

        if self.stride is not None:
            stride = random.choice(self.stride)
        else:
            stride = None

        if self.crop is not None:
            min = np.array([0, 0])
            if transpose is not None:
                max = np.array([input.shape[transpose[0] + 1], input.shape[transpose[1] + 1]], dtype=int) - self.crop
            else:
                max = np.array([input.shape[1] - self.crop[0], input.shape[2] - self.crop[1]])
            slice = [random.randint(mn, mx) for mn, mx in zip(min, max)]
        else:
            slice = None

        input = self.get_input(input, transpose, slice, self.crop, stride).astype(np.float32)
        target = self.get_target(target, transpose, slice, self.crop, stride).astype(np.float32)

        return input, target

    def __len__(self):
        return len(self.od_dataset) * self.repeat
