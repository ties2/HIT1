import random
from typing import Dict, Any, Tuple
from PIL import Image

import numpy as np
import torch
from albumentations import BasicTransform


class ToTensorAlb(BasicTransform):
    """
    Convert instance segmentation keys to `torch.Tensor`.
    """

    def __init__(self, always_apply=True, p=1.0, sub=0., div=1.):
        super(ToTensorAlb, self).__init__(always_apply=always_apply, p=p)
        self.sub = sub
        self.div = div

    @property
    def targets(self):
        return {"image": self.transpose_and_apply_subdiv,
                "bboxes": self.apply_and_tofloat32,
                "area": self.apply,
                "keypoints": self.apply,
                "labels": self.apply,
                "image_id": self.apply,
                "iscrowd": self.apply,
                "mask": self.transpose_and_apply,
                "masks": self.apply,
                "class_masks": self.transpose_and_apply}

    def transpose_and_apply_subdiv(self, array_or_list, **params):  # skipcq: PYL-W0613
        if isinstance(array_or_list, list):
            array_or_list = np.stack(array_or_list)

        if len(array_or_list.shape) not in [2, 3]:
            raise ValueError("Albumentations only supports images in HW or HWC format")

        if len(array_or_list.shape) == 2:
            array_or_list = np.expand_dims(array_or_list, 2)
        if self.sub is not None and self.div is not None:
            array_or_list = (array_or_list - self.sub) / self.div
        array_or_list = array_or_list.transpose((2, 0, 1))
        return torch.from_numpy(array_or_list)

    def transpose_and_apply(self, array_or_list, **params):  # skipcq: PYL-W0613
        if isinstance(array_or_list, list):
            array_or_list = np.stack(array_or_list)

        if len(array_or_list.shape) not in [2, 3]:
            raise ValueError("Albumentations only supports images in HW or HWC format")

        if len(array_or_list.shape) == 2:
            array_or_list = np.expand_dims(array_or_list, 2)
        array_or_list = array_or_list.transpose(2, 0, 1)
        return torch.from_numpy(array_or_list)

    def apply(self, array_or_list, **params):  # skipcq: PYL-W0613
        if isinstance(array_or_list, list):
            array_or_list = np.stack(array_or_list)
        return torch.from_numpy(array_or_list)

    def apply_and_tofloat32(self, array_or_list, **params):  # skipcq: PYL-W0613
        if isinstance(array_or_list, list):
            array_or_list = np.stack(array_or_list)
        return self.apply(array_or_list.astype(np.float32), **params)

    def get_transform_init_args_names(self):
        return ""

    def get_params_dependent_on_targets(self, params):
        return {}


class ToTensor(object):
    """
    Convert ndarrays to Tensors for the instance segmentation dataloader (The numpy array representing the image is divided by 255)
    """

    def __init__(self, sub=None, div=None, norm_boxes=True, is_8bit=True):
        self.sub = sub
        self.div = div
        self.norm_boxes = norm_boxes
        self.is_8bit = is_8bit

    def __call__(self, sample: Tuple[np.ndarray, Dict[str, Any]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image_arr, target_dict = sample
        h, w, c = image_arr.shape
        if self.sub is not None and self.div is not None:
            image_arr = (image_arr - self.sub) / self.div
        new_image = torch.from_numpy(np.moveaxis(image_arr, 2, 0)).type(torch.FloatTensor)
        if self.is_8bit:
            new_image /= 255

        new_target = {}
        for key, value in target_dict.items():
            if key in ["boxes"]:
                new_value = value.copy()
                if self.norm_boxes:
                    new_value[:, 0:4] /= [w, h, w, h]
                new_target[key] = torch.from_numpy(new_value).type(torch.FloatTensor)
            elif key in ["area", "keypoints"]:
                new_value = value
                new_target[key] = torch.from_numpy(new_value).type(torch.FloatTensor)
            elif key in ["labels", "image_id"]:
                new_target[key] = torch.from_numpy(value).type(torch.LongTensor)
            elif key in ["iscrowd", "masks"]:
                new_target[key] = torch.from_numpy(value).type(torch.ByteTensor)
            else:
                new_target[key] = value

        return new_image, new_target


class RandomCrop(object):
    """
    Take random crops from the input image
    """

    def __init__(self, height, width):
        self.crop_h = height
        self.crop_w = width

    def __call__(self, sample: Tuple[np.ndarray, Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        image_arr, target_dict = sample

        h, w, c = image_arr.shape
        if h < self.crop_h or w < self.crop_w:
            print("Warning: padded the image to fit the crop")
            diff_h = self.crop_h - h
            diff_w = self.crop_w - w
            pad_t = diff_h // 2
            pad_b = diff_h - pad_t
            pad_l = diff_w // 2
            pad_r = diff_w - pad_l
        else:
            pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0

        new_image = np.pad(image_arr, pad_width=((pad_t, pad_b), (pad_l, pad_r), (0, 0)))
        h, w, c = image_arr.shape
        crop_x1 = random.randint(0, w - self.crop_w)
        crop_y1 = random.randint(0, h - self.crop_h)
        crop_x2, crop_y2 = crop_x1 + self.crop_w, crop_y1 + self.crop_h
        new_image = new_image[crop_y1:crop_y2, crop_x1:crop_x2, :]

        new_target = {}
        for key, value in target_dict.items():

            if key in ["boxes"]:
                new_value = value
                new_value[:, 0:4] += [pad_l, pad_t, pad_l, pad_t]
                new_value[:, 0:4] -= [crop_x1, crop_y1, crop_x1, crop_y1]
                new_value[new_value[:, 0] < 0, 0] = 0
                new_value[new_value[:, 1] < 0, 1] = 0
                new_value[new_value[:, 2] < 0, 2] = 0
                new_value[new_value[:, 3] < 0, 3] = 0
                new_value[new_value[:, 0] >= self.crop_w, 0] = self.crop_w - 1
                new_value[new_value[:, 1] >= self.crop_h, 1] = self.crop_h - 1
                new_value[new_value[:, 2] >= self.crop_w, 2] = self.crop_w - 1
                new_value[new_value[:, 3] >= self.crop_h, 3] = self.crop_h - 1
                new_target[key] = new_value
            elif key in ["keypoints"]:
                new_value = value
                new_value[:, :, 0:2] += [pad_l, pad_t]
                new_value[:, :, 0:2] -= [crop_x1, crop_y1]
                new_target[key] = new_value
                assert False, "This part has not been tested"
            elif key in ["masks"]:
                new_value = value
                new_value = np.pad(new_value, pad_width=((0, 0), (pad_t, pad_b), (pad_l, pad_r)))
                new_value = new_value[:, crop_y1:crop_y2, crop_x1:crop_x2]
                new_target[key] = new_value
            else:
                new_target[key] = value
        new_target["area"] = np.array([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in new_target["boxes"]])

        # Select only instances that have an area > 0
        indices = np.where(new_target["area"] > 0)
        new_target2 = {}
        for key, value in new_target.items():
            if key in ["masks", "boxes", "area", "labels", "key_points", "iscrowd"]:
                new_target2[key] = new_target[key][indices]
            else:
                new_target2[key] = value
        return new_image, new_target2

