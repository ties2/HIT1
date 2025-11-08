from collections import defaultdict
from typing import Optional

import albumentations
from torch.utils import data
from typing import List
import torch
from common import deprecated
from common.elements.legacy.loaders import CachedResource, Box
from glob import glob
import os
import numpy as np

class InstanceSegmentationDataset(data.Dataset):
    """
    Instance segmentation dataset
    """

    class SampleContainer:
        def __init__(self, image_file, inst_mask_file, class_mask_file):
            self.image_filename: Optional[str] = None
            self.image_file: Optional[str] = image_file
            self.inst_mask_file: Optional[str] = inst_mask_file
            self.class_mask_file: Optional[str] = class_mask_file
            self.loaded: bool = False
            self.boxes: Optional[List[Box]] = None
            self.inst_mask: Optional[CachedResource] = None
            self.class_mask: Optional[CachedResource] = None
            self.image: Optional[CachedResource] = None

    @deprecated("This dataset is deprecated, please use the GenericDataset from common.data instead.")
    def __init__(self, dataset_dir=None, images_dir=None, instance_masks_dir=None, class_masks_dir=None, instance_mask_loader_func=None, class_mask_loader_func=None, image_loader_func=None, pre_transforms=None, aug_transforms=None, post_transforms=None, num_classes=None, class_ids: List[int]=None, class_names: List[str]=None, class_colors: List[str]=None, verbose: bool = True):
        """
        Initialize the dataloader. Images are expected in: dataset_dir/<image id>/images_dir
            and corresponding masks in dataset_dir/<image id>/masks_dir if dataset_dir is not None, else the
            instance_masks_dir, class_masks_dir and the images_dir contain the masks and images (they will be matched based on their string-sorted order)

        :param dataset_dir: Root dir of the datasets
        :param images_dir: dir containing images
        :param instance_masks_dir: dir containing the instance masks
        :param class_masks_dir: dir containing the class masks
        :param instance_mask_loader_func: Function to load instance masks.
            Will get passed a folder or filename and needs to return the instance_id mask (unique instance id per pixel)
        :param class_mask_loader_func: Function to load class masks.
            Will get passed a folder or filename and needs to return the class_id mask (unique class_id per pixel)
        :param image_loader_func: Function to load the image.
            Will get passed a folder or filename and needs to return an image with shape [h,w,c].
        :param transforms: The transform object for transforming data (using Albumentations)
        :param verbose: Controls whether print() statements should be executed (e.g. when instance masks / class masks are not present)
        :param num_classes: The number of classes (including background)
        :param class_id_to_index: Callable that translates class ids to indices
        :param class_id_to_name:  Callable that translates class ids to names
        """
        self.verbose = verbose
        self.inst_mask_loader_func = instance_mask_loader_func
        self.class_mask_loader_func = class_mask_loader_func
        self.image_loader_func = image_loader_func
        self.dataset_dir = dataset_dir
        self.class_ids = class_ids if class_ids is not None else list(range(num_classes))
        self.class_names = class_names if class_names is not None else ["Background", *["Object"] * (num_classes-1)]
        self.class_colors = class_colors if class_colors is not None else [(0, 0, 0), *[(0, 0, 255)] * (num_classes-1)]
        self.class_indices = {class_id: class_idx for class_idx, class_id in enumerate(self.class_ids)}
        self.num_classes = num_classes
        self.clear_background = False
        self.include_raw = False
        self.mean = 0
        self.sd = 1

        if images_dir is None:
            self.images_dir = "images"
        else:
            self.images_dir = images_dir
        if instance_masks_dir is None:
            self.instance_masks_dir = "masks"
        else:
            self.instance_masks_dir = instance_masks_dir
        if class_masks_dir is None:
            self.class_masks_dir = "classmasks"
        else:
            self.class_masks_dir = class_masks_dir

        self.pre_transforms = pre_transforms
        self.aug_transforms = aug_transforms
        self.post_transforms = post_transforms

        self.samples: List[InstanceSegmentationDataset.SampleContainer] = []
        if dataset_dir is not None:
            self._load_data_from_folder()
        else:
            self._load_data_from_folders()

    def __len__(self):
        return len(self.samples)

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _load_data_from_folder(self):
        """
        Organized in dataset_dir/<image id>/images_dir and corresponding masks in dataset_dir/<image id>/masks_dir
        """
        folders = glob(os.path.join(self.dataset_dir, "*"))
        no_class_masks = False
        no_inst_masks = False
        for folder in folders:
            if os.path.isdir(folder):
                image_dir = os.path.join(folder, self.images_dir)
                inst_mask_dir = os.path.join(folder, self.instance_masks_dir)
                class_mask_dir = os.path.join(folder, self.class_masks_dir)
                if not os.path.isdir(inst_mask_dir) or no_inst_masks:
                    if not no_inst_masks:
                        self._print(f"No instance masks found in {inst_mask_dir}, skipping.")
                        no_inst_masks = True
                    inst_mask_dir = None
                if not os.path.isdir(class_mask_dir) or no_class_masks:
                    if not no_class_masks:
                        self._print(f"No class masks found in {class_mask_dir}, skipping.")
                        no_class_masks = True
                    class_mask_dir = None
                self.samples.append(InstanceSegmentationDataset.SampleContainer(image_dir, inst_mask_dir, class_mask_dir))

    def _load_data_from_folders(self):
        """
        Organized in string-sorted matching images_dir and masks_dir
        """
        images_list = list(sorted(glob(os.path.join(self.images_dir, "*"))))
        inst_masks_list = list(sorted(glob(os.path.join(self.instance_masks_dir, "*"))))
        class_masks_list = list(sorted(glob(os.path.join(self.class_masks_dir, "*"))))
        if len(class_masks_list) == 0:
            class_masks_list = [None] * len(images_list)
            self._print(f"No class masks found in {self.instance_masks_dir}, skipping.")
        if len(inst_masks_list) == 0:
            inst_masks_list = [None] * len(images_list)
            self._print(f"No instance masks found in {self.class_masks_dir}, skipping.")
        assert len(images_list) == len(inst_masks_list), "The number of images and instance masks do not match"
        assert len(images_list) == len(class_masks_list), "The number of images and class masks do not match"
        for image_fn, inst_mask_fn, class_mask_fn in zip(images_list, inst_masks_list, class_masks_list):
            self.samples.append(InstanceSegmentationDataset.SampleContainer(image_fn, inst_mask_fn, class_mask_fn))

    def get_sample(self, idx) -> 'InstanceSegmentationDataset.SampleContainer':
        sample = self.samples[idx]
        if not sample.loaded:
            image, filename = self.image_loader_func(sample.image_file)
            sample.image_filename = filename
            image_cache = CachedResource(data=image)
            sample.image = image_cache

            if sample.inst_mask_file is not None:
                inst_mask, boxes = self.inst_mask_loader_func(sample.inst_mask_file)
                inst_mask_cache = CachedResource(data=inst_mask)
                sample.inst_mask = inst_mask_cache
                sample.boxes = boxes

            if sample.class_mask_file is not None:
                class_mask = self.class_mask_loader_func(sample.class_mask_file)
                class_mask_cache = CachedResource(data=class_mask)
                sample.class_mask = class_mask_cache

            sample.loaded = True
        return sample

    def get_mean_and_sd(self, key: str="image"):
        sum = 0.
        sumsq = 0.
        n = 0.
        for item in self:
            image, _ = item
            image = image[key]
            if isinstance(image, torch.Tensor):
                n += float(image.numel())
                s = float(torch.sum(image).item())
                ss = float(torch.sum(image * image).item())
            elif isinstance(image, np.ndarray):
                n += int(image.size())
                s = float(np.sum(image))
                ss = float(np.sum(image * image))
            else:
                raise RuntimeError(f"Cannot get mean of type: {type(image)}")
            sum = sum + s
            sumsq = sumsq + ss

        mean = sum / n
        sd = (sumsq / n) - (mean * mean)
        sd = sd ** 0.5
        self.mean = mean
        self.sd = sd
        return mean, sd

    def get_mean_and_sd_per_class(self, image_key: str="image", class_mask_key: str="class_masks"):
        sum = defaultdict(lambda: 0.)
        sumsq = defaultdict(lambda: 0.)
        n = defaultdict(lambda: 0.)
        for item in self:
            image, annot = item
            image = image[image_key]
            mask = annot[class_mask_key]
            for class_id, class_mask in enumerate(mask):
                if isinstance(image, torch.Tensor):
                    bool_mask = class_mask > 0
                    masked_image = image.detach().clone() * bool_mask
                    n[class_id] += float(torch.sum(bool_mask))
                    masked_image = torch.reshape(masked_image, shape=(masked_image.shape[0], -1))
                    s = torch.sum(masked_image, dim=1)
                    ss = torch.sum(masked_image * masked_image, dim=1)
                elif isinstance(image, np.ndarray):
                    raise NotImplementedError(f"Cannot get mean of type: {type(image)}")
                else:
                    raise RuntimeError(f"Cannot get mean of type: {type(image)}")
                sum[class_id] = sum[class_id] + s
                sumsq[class_id] = sumsq[class_id] + ss
        mean = {}
        sd = {}
        for class_id in n.keys():
            mean[class_id] = sum[class_id] / n[class_id]
            sd[class_id] = (sumsq[class_id] / n[class_id]) - (mean[class_id] * mean[class_id])
            sd[class_id] = sd[class_id] ** 0.5
            mean[class_id] = [float(x) for x in mean[class_id]]
            sd[class_id] = [float(x) for x in sd[class_id]]
        self.mean = mean
        self.sd = sd
        return mean, sd

    @staticmethod
    def transform(image, target, transforms):
        """
        Performs Albumentations transforms in the image and the targets.

        :param transforms: the albumentations transform
        :param image: The input image
        :param target: Target dictionary
        :return: The transformed input and target.
        """
        if transforms is None:
            return image, target

        target_name = ""
        if target:
            # Change naming and set-up Albumentations
            if "boxes" in target.keys():
                target["bboxes"] = target["boxes"]
                del target["boxes"]
                transforms.processors["bboxes"] = albumentations.BboxProcessor(
                    albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))
            if "masks" in target.keys():
                target_name = "masks"
                target["mask"] = target["masks"]
                del target["masks"]
            if "class_masks" in target.keys():
                target_name = "class_masks"
                target["mask"] = target["class_masks"]
                del target["class_masks"]
            transformed = transforms(image=image, **target)

        image = transformed["image"]
        del transformed["image"]

        if target:
            # Restore naming
            if "bboxes" in transformed.keys():
                transformed["boxes"] = transformed["bboxes"]
                del transformed["bboxes"]
            if target_name == "masks":
                transformed["masks"] = transformed["mask"]
                del transformed["mask"]
            if target_name == "class_masks":
                transformed["class_masks"] = transformed["mask"]
                del transformed["mask"]

            return image, transformed
        else:
            return image

    def ids_to_one_hot(self, x):
        obj_ids = np.unique(x)
        obj_ids = obj_ids[1:]
        masks = x == obj_ids[:, None, None]
        masks = np.moveaxis(masks, 0, 2)
        return masks

    def class_ids_to_one_hot(self, x):
        h, w = x.shape
        masks = np.zeros([h, w, self.num_classes], dtype=np.bool_)
        class_ids = np.unique(x).astype(np.int32)
        for class_id in class_ids:
            class_idx = self.class_indices[class_id]
            if class_idx > self.num_classes:
                raise RuntimeError(f"The class index {class_idx} for class id {class_id} is out of range.")
            masks[:, :, class_idx] = x == class_id
        return masks

    def __getitem__(self, index: int):
        sample = self.get_sample(index)
        image = sample.image().astype(np.float32)
        image_id = np.array([index])
        filename = sample.image_filename
        target = {"image_id": image_id.astype(np.int64),
                  "filename": filename}
        if self.verbose:
            print(f"Providing item: {target} from dataloader.")

        try:
            # mask = np.ones(image.shape[:-1])

            if sample.boxes is not None:
                boxes = sample.boxes
                boxes_array = np.stack([box.numpy(box_format='xy') for box in boxes])
                labels_array = np.array([self.class_indices[box.class_id] for box in boxes])
                areas_array = np.array([box.area for box in boxes])
                iscrowd = np.array([False] * len(boxes))
                target["boxes"] = boxes_array.astype(np.float32)
                target["labels"] = labels_array.astype(np.int64)
                target["area"] = areas_array.astype(np.float32)
                target["iscrowd"] = iscrowd.astype(np.uint8)

            if sample.inst_mask is not None:
                inst_mask = sample.inst_mask()
                instance_ids = self.ids_to_one_hot(inst_mask)
                target["masks"] = instance_ids.astype(np.uint8)
                # mask[inst_mask == 0] = 0 if self.clear_background else None

            if sample.class_mask is not None:
                class_mask = sample.class_mask()
                class_ids = self.class_ids_to_one_hot(class_mask)
                target["class_masks"] = class_ids.astype(np.uint8)
                # mask[class_mask == 0] = 0 if self.clear_background else None

            # if self.clear_background:
            #    mask = np.expand_dims(mask, axis=2)
            #    image *= mask

            # Produce the pre-processed image
            image_input, target_input = self.transform(image, target, self.pre_transforms)
            image_input, target_input = self.transform(image_input, target_input, self.aug_transforms)
            image_input, target_input = self.transform(image_input, target_input, self.post_transforms)

            # Produce the raw image
            if self.include_raw:
                image_raw, _ = self.transform(image, target, self.post_transforms)

                input = {"image": image_input,
                         "raw": image_raw}
            else:
                input = {"image": image_input}
            return input, target_input
        except ValueError as e:
            print(f"Something went wrong while processing filename {filename}")
            raise e

    def get_image_filename(self, index: int) -> str:
        return self.samples[index].image_filename

    def get_boxes(self, index: int) -> np.ndarray:
        boxes = [box.numpy(box_format='yxi') for box in self.samples[index].boxes]
        boxes = np.stack(boxes)
        return boxes
