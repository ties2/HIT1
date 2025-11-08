from typing import Dict, Union, Any, List, Tuple
import numpy as np
import torch
import cv2


class Resizer(object):
    """
    Resize the input array and annotation. (Draft, Untested)
    """

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.resize_keys = ["img", "raw"]
        self.resize_keys_nearest = ["segmap"]
        self.resize_keys_hline = ["white", "dark"]

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resize the sample.

        :param sample: A dictionary containing images to be resized. The shape is HWC.
        :return: A dictionary containing images that are resized, aditionally the scale_y and scale_x are added to
            retain the original size.
        """
        assert True, "This method is currently untested"

        result = {}
        for key, value in sample.items():
            if key not in self.resize_keys and key not in self.resize_keys_nearest and key not in self.resize_keys_hline or value is None:
                result[key] = value
                continue
            # Determine scale on first image
            height, width, channels = value.shape
            if "scale" not in result.keys():
                result["scale"] = (self.height / height, self.width / width)

            if key in self.resize_keys:
                value_new = np.zeros([self.height, self.width, channels], dtype=value.dtype)
                for c in range(channels):
                    value_new[:, :, c] = cv2.resize(np.ascontiguousarray(value[:, :, c]), (self.width, self.height), interpolation=cv2.INTER_AREA)
                result[key] = value_new
            elif key in self.resize_keys_nearest:
                value_new = np.zeros([self.height, self.width, channels], dtype=value.dtype)
                for c in range(channels):
                    value_new[:, :, c] = cv2.resize(np.ascontiguousarray(value[:, :, 0]), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                result[key] = value_new
            elif key in self.resize_keys_hline:
                value_new = np.zeros([value.shape[0], self.width, channels], dtype=value.dtype)
                for c in range(channels):
                    value_new[:, :, c] = cv2.resize(np.ascontiguousarray(value[:, :, c]), (self.width, value_new.shape[0]))
                result[key] = value_new
            else:
                result[key] = value

        return result


class ToTensor(object):
    """
    Convert ndarrays to Tensors.
    """

    def __init__(self, dtype=np.float32):
        """
        :param dtype: Force the ndarray to this datatype before converting to a Tensor.
        """
        self.dtype = dtype
        self.convert_keys = ["img", "raw", "segmap", "white", "dark", "annot"]

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the sample to a PyTorch Tensor

        :param sample: A dictionary with numpy images as keys
        :return: A dictionary with torh
        """
        result = {}
        for key, value in sample.items():
            if value is not None:
                if key in self.convert_keys:
                    result[key] = torch.from_numpy(np.ascontiguousarray(value.astype(self.dtype)))
                else:
                    result[key] = value
        return result


class Normalizer(object):
    """
    Normalize the array by subtracting mean and dividing by std
    """

    def __init__(self, mean=0.5, std=0.5, clip_low: float = None, clip_high: float = None):
        """
        Initialize object

        :param mean: list of mean values (one for each channel)
        :param std: list of stddev values (one for each channel)
        :param clip_low: Clip values below this value.
        :param clip_high: Clip values above this value.
        """
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.image_keys = ["img"]

    def __call__(self, sample: Dict[str, Union[np.ndarray, np.ndarray, float]]) -> Dict[str, Union[np.ndarray, np.ndarray, float]]:
        """
        Normalize the input array

        :param sample: a dictionary with the 'img' array of shape [b, h, w, c]. The remaining keys are copied to the output.
        :return: A dictionary with normalized 'img'
        """
        result = {}
        for key, value in sample.items():
            if key in self.image_keys:
                value = (value.astype(np.float32) - self.mean) / self.std
                if self.clip_low is not None and self.clip_high is not None:
                    value = np.clip(value, self.clip_low, self.clip_high)
            result[key] = value
        return result


class Stacking(object):
    """
    Stacks a list of images a 4d tensors and and masks
    This is a collator augmenter expects HWC and converts it to CHW
    """

    def __init__(self):
        # Which dictionary keys should be stacked?
        self.stack_images = ["img", "raw", "segmap", "white", "dark"]

    def crop_tensors(self, tensors: List[torch.Tensor]) -> Tuple[List[torch.Tensor], bool, torch.Tensor]:
        min_shape = torch.tensor(tensors[0].shape[:2])
        crop = False
        for tensor in tensors:
            if not torch.all(torch.eq(min_shape, torch.tensor(tensor.shape[:2]))):
                crop = True
            min_shape = torch.min(torch.tensor(tensor.shape[:2]), min_shape)
        if crop:
            tensors = [tensor[0:min_shape[0], 0:min_shape[1], :] for tensor in tensors]

        return tensors, crop, min_shape

    def __call__(self, sample: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        """
        This stacking transformer should be used in the collator (for example, the tiling collators in :class:`common.segmentation.FixedTileCollator`

        :param sample:
        It takes a Dictionary with images to be stacked.
        1. It stacks images [h, w, c] into one image with a minibatch dimension [b, c, h, w] (note: so it also moves the channel to fit Torch convention)
        2. Returns a torch tensor in shape
        :return: A dictionary with the newly stacked Torch tensors and all other keys are copied.

        """
        result = {}
        for key, value in sample.items():
            if isinstance(value, list) and len(value) == 0:
                print(f"Empty image in '{key}'")
                continue
            if key in self.stack_images:
                value, cropped, min_shape = self.crop_tensors(value)
                value = torch.stack(value, axis=0).permute(0, 3, 1, 2)
                if cropped:
                    print(f"Tensors with key '{key}' have been cropped to {min_shape}")
            result[key] = value
        return result
