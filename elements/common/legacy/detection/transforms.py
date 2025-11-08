import numbers
import random
from enum import Enum
from logging import Logger
from math import radians
from typing import Dict, Union, Any, List, Tuple

from common import concat_np_arrays
from common.elements.math import *
from third_party.model_wrappers.efficientdet import pad_boxes

import numpy as np
import torch
import cv2


sample_type = Dict[str, Union[np.ndarray, np.ndarray, float]]


# Run the given transform on just an annotation rather than an entire sample.
def annotation_adapter(transform, annot):
    return transform({'img': None, 'annot': annot})['annot']


# Run the given transform on just an image rather than an entire sample.
def image_adapter(transform, image):
    return transform({'img': image, 'annot': np.array([]).reshape((0, 5))})['img']


class Resizer(object):
    """
    Resize the input array and annotation.
    """
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resize the sample.

        :param sample: a dictionary with the 'img' array of shape [b, h, w, c]
        and the 'annot' array of shape [b, N, 5] containing [y1, x1, y2, x2, class_id] for N bounding boxes.
        if the 'scale' key is present it is ignored.
        :param common_size: Resize to this
        :return: A dictionary with resized 'img', 'annot' and the scale factor 'scale'
        """
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        scale_y = self.height / height
        scale_x = self.width / width
        image = cv2.resize(np.ascontiguousarray(image), (self.width, self.height))

        annots = annots.copy()
        annots[:, 0] *= scale_y
        annots[:, 1] *= scale_x
        annots[:, 2] *= scale_y
        annots[:, 3] *= scale_x
        return {'img': image, 'annot': annots, 'scale': (scale_y, scale_x), 'filename': sample['filename']}


class ToTensor(object):
    """
    Convert ndarrays to Tensors.
    """

    def __init__(self, dtype=np.float32):
        """
        :param dtype: Force the ndarray to this datatype before converting to a Tensor.
        """
        self.dtype = dtype


    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the sample to a PyTorch Tensor

        :param sample: a dictionary with the 'img' array of shape [b, h, w, c]
        and the 'annot' array of shape [b, N, 5] containing [y1, x1, y2, x2, class_id] for N bounding boxes.
        :return: A dictionary with converted 'img', 'annot' and the scale factor 'scale'
        """
        image, annots = sample['img'], sample['annot']
        if 'scale' in sample.keys():
            scale = sample['scale']
        else:
            scale = 1
        return {'img': torch.from_numpy(np.ascontiguousarray(image).astype(self.dtype)), 'annot': torch.from_numpy(annots.astype(self.dtype)), 'scale': scale, 'filename': sample['filename']}


class FlipX(object):
    """Random flipping of ndarrays."""

    def __call__(self, sample: Dict[str, Any], flip_x: float=0.5) -> Dict[str, Any]:
        """
        Randomly flip the a numpy array in the x direction with flip_x probability

        :param sample: a dictionary with the 'img' array of shape [b, h, w, c]
        and the 'annot' array of shape [b, N, 5] containing [y1, x1, y2, x2, class_id] for N bounding boxes.
        :return: A dictionary with converted 'img', 'annot' and the scale factor 'scale'
        """
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            if 'scale' in sample.keys():
                scale = sample['scale']
            else:
                scale = 1

            sample = {'img': image, 'annot': annots, 'scale': scale, 'filename': sample['filename']}

        return sample


class Normalizer(object):
    """
    Normalize the array by subtracting mean and dividing by std
    """

    def __init__(self, mean, std):
        """
        Initialize object

        :param mean: list of mean values (one for each channel)
        :param std: list of stddev values (one for each channel)
        """
        # Default values for EficientDet backbone
        #if mean is None:
        #    mean = [[[0.485, 0.456, 0.406]]]
        #if std is None:
        #    std = [[[0.229, 0.224, 0.225]]]
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample: Dict[str, Union[np.ndarray, np.ndarray, float]]) -> Dict[str, Union[np.ndarray, np.ndarray, float]]:
        """
        Normalize the input array

        :param sample: a dictionary with the 'img' array of shape [b, h, w, c]
        and the 'annot' array of shape [b, N, 5] containing [y1, x1, y2, x2, class_id] for N bounding boxes
        optionally a 'scale' key.

        :return: A dictionary with converted 'img', 'annot', 'scale'
        """
        image, annots, filename = sample['img'], sample['annot'], sample['filename']
        d = {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots, 'filename': filename}

        if 'scale' in sample.keys():
            d['scale'] = sample['scale']
        else:
            d['scale'] = 1.
        if 'boxes' in sample.keys():
            d['boxes'] = sample['boxes']
        return d


class AnnotationFormat(Enum):
    XYXY = 0
    YXYX = 1
    XYWH = 2
    YXWH = 3
    XYHW = 4
    YXHW = 5


class AnnotationConverter(object):
    """
    Converts between different annotation formats.
    If the annotation contains any additional values, like classes or scores, these are left unchanged.
    """

    def __init__(self, from_fmt: AnnotationFormat, to_fmt: AnnotationFormat):
        self.from_fmt = from_fmt
        self.to_fmt = to_fmt

    @staticmethod
    def perform_conversion(annotation: np.ndarray, from_fmt: AnnotationFormat, to_fmt: AnnotationFormat) -> np.ndarray:
        def flip_first(annot):
            return concat_np_arrays(annot[[1, 0, 2, 3]], annot[4:])

        def flip_second(annot):
            return concat_np_arrays(annot[[0, 1, 3, 2]], annot[4:])

        def from_xywh(annot): return \
            AnnotationConverter.perform_conversion(annot, AnnotationFormat.XYWH, AnnotationFormat.XYXY)

        to_xyxy = {
            AnnotationFormat.XYXY: lambda annot: annot,
            AnnotationFormat.YXYX: lambda annot: flip_first(flip_second(annot)),
            AnnotationFormat.XYWH: lambda annot: concat_np_arrays(annot[0:2], annot[0:2] + annot[2:4], annot[4:]),
            AnnotationFormat.YXWH: lambda annot: flip_first(from_xywh(annot)),
            AnnotationFormat.XYHW: lambda annot: flip_second(from_xywh(annot)),
            AnnotationFormat.YXHW: lambda annot: flip_first(flip_second(from_xywh(annot))),
        }

        def to_xywh(annot):
            return AnnotationConverter.perform_conversion(annot, AnnotationFormat.XYXY, AnnotationFormat.XYWH)

        from_xyxy = {
            AnnotationFormat.XYXY: lambda annot: annot,
            AnnotationFormat.YXYX: lambda annot: flip_first(flip_second(annot)),
            AnnotationFormat.XYWH: lambda annot: concat_np_arrays(annot[0:2], annot[2:4] - annot[0:2], annot[4:]),
            AnnotationFormat.YXWH: lambda annot: flip_first(to_xywh(annot)),
            AnnotationFormat.XYHW: lambda annot: flip_second(to_xywh(annot)),
            AnnotationFormat.YXHW: lambda annot: flip_first(flip_second(to_xywh(annot)))
        }

        return from_xyxy[to_fmt](to_xyxy[from_fmt](annotation))

    def __call__(self, sample: sample_type) -> sample_type:
        input_shape = sample['annot'].shape
        result = { 'img': sample['img'], 'annot': [] }

        for i in range(len(sample['annot'])):
            result['annot'].append(AnnotationConverter.perform_conversion(
                sample['annot'][i],
                self.from_fmt,
                self.to_fmt
            ))

        result['annot'] = np.array(result['annot'])
        if len(result['annot']) == 0: result['annot'] = result['annot'].reshape(input_shape)

        return result


class RandomAffine(object):
    """
    Performs a random affine transform on the given sample. _Annotations are transformed as well.
    Author: Rob Klein Ikink
    """

    val_or_range = Union[float, Tuple[float, float]]

    class RotationPoint(Enum):
        IMAGE_CENTER = 0
        FIRST_ANNOTATION = 1
        RANDOM_ANNOTATION = 2

    class RetryBehaviour(Enum):
        RETRY  = 0,
        REMOVE = 1

    def __init__(
            self,
            rotation: val_or_range = 0,
            translation: val_or_range = 0,
            scaling: val_or_range = 1,
            shear_x: val_or_range = 0,
            shear_y: val_or_range = 0,
            flip_x_probability: float = 0,
            flip_y_probability: float = 0,
            fill_color: Tuple = (0,),
            annotation_fmt: AnnotationFormat = AnnotationFormat.YXYX,
            rotation_point: RotationPoint = RotationPoint.FIRST_ANNOTATION,
            retry_behaviour: RetryBehaviour = RetryBehaviour.REMOVE,
            max_retries: int = 5,
            min_remaining_area: float = 0.25,
            logger: Logger = None
    ):
        """
        Construct the object with the given transform parameters.
        Transform parameters may be given as a single number or a pair (min, max).
        If the former is used, a (min, max) pair is constructed with (-value, value),
        or (min(value, 1), max(value, 1)) in the case of scaling.
        Each given image is transformed with random values chosen uniformly from these (min, max) ranges.

        :param rotation:           Rotation in degrees.
        :param translation:        Translation in % of image size (e.g. 0.5 may translate up to (0.5w, 0.5h)
        :param scaling:            Scale factor. Values must be positive.
        :param shear_x:            Shear factor.
        :param shear_y:            Shear factor.
        :param flip_x_probability: Probability to flip around the horizontal axis. Value must be in the range [0, 1].
        :param flip_y_probability: Probability to flip around the vertical axis. Value must be in the range [0, 1].
        :param fill_color:         Value to fill empty pixels with. Must be a tuple of objects with the same type as image pixels.
                                   Tuple may either have the same number of channels as the image, or just one.
                                   If the tuple has one value, each channel of the pixel is filled with that value.
        :param annotation_fmt:     The format the provided annotations are in.
        :param rotation_point:     Rotate around the center of the image or the center of an annotation?
        :param retry_behaviour:    Retry if an annotation falls outside the image, or remove it from the list?
                                   This also affects what is done with annotations smaller than min_remaining_area.
        :param max_retries:        If retry_behaviour is RETRY, how often should we retry if an annotation falls outside the image
                                   before returning the sample unedited?
        :param min_remaining_area: How much of each bounding box area should minimally remain after the transform?
                                   Value must be in the range [0, 1].
                                   In REMOVE mode, bounding boxes smaller than this area are removed. In RETRY mode, the existence
                                   of bounding boxes below the size limit triggers a retry.
        :param logger:             Optional logger. Will warn if the transform failed for some reason.
        """

        def is_any(val, *types):
            return any(map(lambda t: isinstance(val, t), list(types)))

        def make_min_max(value, make_min=lambda x: -x, make_max = lambda x: x):
            if is_any(value, tuple, list, np.ndarray):
                assert len(value) == 2,      'Provided (min, max) pair does not have 2 values.'
                assert value[0] <= value[1], 'Provided (min, max) pair has a min greater than max.'

                return value

            if isinstance(value, numbers.Number):
                assert value >= 0, 'Please provide a positive number.'
                return make_min(value), make_max(value)

            assert False, 'Please provide a number or a pair of (min, max) values.'

        assert all(0 <= x <= 1 for x in [flip_x_probability, flip_y_probability]), 'Probability must be in range [0, 1].'

        self.rotation = make_min_max(rotation)
        self.translation = make_min_max(translation)
        self.scaling = make_min_max(scaling, lambda x: min(x, 1), lambda x: max(x, 1))
        self.shear_x = make_min_max(shear_x)
        self.shear_y = make_min_max(shear_y)
        self.flip_x = flip_x_probability
        self.flip_y = flip_y_probability
        self.fill_color = fill_color
        self.annot_fmt = annotation_fmt
        self.rotate_mode = rotation_point
        self.retry_mode = retry_behaviour
        self.max_retries = max_retries
        self.min_area = min_remaining_area
        self.logger = logger

    def __call__(self, sample: sample_type) -> sample_type:
        image = sample['img']
        annots = sample['annot']
        if annots.shape[0] > 0:
            annots = annotation_adapter(AnnotationConverter(self.annot_fmt, AnnotationFormat.XYXY), sample['annot'])

        h, w, c = image.shape
        img_center = np.array([w, h]) * 0.5 + 0.5

        # Split annotations into boxes and metadata like classes.
        boxes = annots[:, 0:4]
        metas = annots[:, 4:]

        def require_boxes_exist():
            if len(boxes) > 0: return True

            if self.logger is not None: self.logger.warning(
                'Cannot rotate around bounding box on sample without bounding boxes. '
                'The sample will be returned unmodified.'
            )

            return False

        if self.rotate_mode == RandomAffine.RotationPoint.IMAGE_CENTER:
            tf_center = img_center
        elif self.rotate_mode == RandomAffine.RotationPoint.FIRST_ANNOTATION:
            if not require_boxes_exist(): return sample
            tf_center = aabb_center(boxes[0])
        else:
            if not require_boxes_exist(): return sample
            tf_center = aabb_center(random.choice(boxes))

        transformed_boxes, matrix = None, None
        for n in range(1 if self.retry_mode == RandomAffine.RetryBehaviour.REMOVE else self.max_retries):
            tf_radians = radians(random.uniform(*self.rotation))
            tf_translate_x = random.uniform(*self.translation) * w
            tf_translate_y = random.uniform(*self.translation) * h
            tf_scaling = random.uniform(*self.scaling)
            tf_shear_x = random.uniform(*self.shear_x)
            tf_shear_y = random.uniform(*self.shear_y)
            tf_flip_x = random.uniform(0, 1) < self.flip_x
            tf_flip_y = random.uniform(0, 1) < self.flip_y

            matrix = repeated_dot_product(
                homogeneous_translation_matrix_2d((tf_translate_x, tf_translate_y)),
                homogeneous_translation_matrix_2d(tf_center),
                homogeneous_rotation_matrix_2d(tf_radians),
                homogeneous_scale_matrix_2d(tf_scaling),
                homogeneous_shear_x_matrix_2d(tf_shear_x),
                homogeneous_shear_y_matrix_2d(tf_shear_y),
                np.linalg.inv(homogeneous_translation_matrix_2d(tf_center)),
                homogeneous_translation_matrix_2d(img_center),
                homogeneous_flip_x_matrix_2d() if tf_flip_x else np.identity(3),
                homogeneous_flip_y_matrix_2d() if tf_flip_y else np.identity(3),
                np.linalg.inv(homogeneous_translation_matrix_2d(img_center)),
            )

            # check if there are boxes
            if annots.shape[0] > 0:
                # Get the corners of each box and apply the given transform to each of them.
                box_corners = np.array(list(map(
                    lambda box: aabb_to_corners(box),
                    boxes
                ))).reshape((-1, 2))

                box_corners = np.append(box_corners, [[1]] * len(box_corners), axis = 1).transpose()
                box_corners = matrix.dot(box_corners).transpose()[:, 0:2].reshape((-1, 4, 2))

                transformed_boxes = np.array(list(map(
                    lambda corners: corners_to_aabb(corners),
                    box_corners
                )))

                # If we are in REMOVE mode, remove any annotations outside the image bounds.
                # Otherwise, retry if some annotations went outside the image,
                # or return the untransformed sample if we're out of retries.
                image_aabb = np.array([0, 0, w, h])

                def is_valid(pair):
                    before, after = pair
                    return aabb_inside(after, image_aabb) and aabb_relative_area(before, after) >= self.min_area

                if self.retry_mode == RandomAffine.RetryBehaviour.REMOVE:
                    bad_indices = [i for i, box_pair in enumerate(zip(boxes, transformed_boxes)) if not is_valid(box_pair)]

                    transformed_boxes = np.delete(transformed_boxes, bad_indices, axis = 0)
                    metas = np.delete(metas, bad_indices, axis = 0)
                else:
                    if not all(map(is_valid, zip(boxes, transformed_boxes))):
                        if n + 1 == self.max_retries:
                            if self.logger is not None:
                                self.logger.warning(
                                    'Failed to find a random transform that keeps all annotations visible and inside the image. '
                                    'The sample will be returned unmodified.'
                                )

                            return sample

        effective_fill_color = list(self.fill_color) * c if len(self.fill_color) == 1 else self.fill_color
        assert len(effective_fill_color) == c, 'Provided fill color has incorrect number of channels.'

        image = cv2.warpAffine(
            src=image,
            M=np.delete(matrix, 2, axis = 0),
            dsize=(w, h),
            borderValue=effective_fill_color
        )

        # OpenCV will remove the channel dimension on grayscale images.
        if len(image.shape) == 2:
            image = image.reshape((h, w, c))

        sample['img'] = image
        if annots.shape[0] > 0:
            sample['annot'] = np.append(transformed_boxes, metas, axis=1)
            sample['annot'] = annotation_adapter(AnnotationConverter(AnnotationFormat.XYXY, self.annot_fmt), sample['annot'])
        return sample


class PaddingAndStacking(object):
    """
    Pads and stacks a list of images and annotations into two 4d tensors.
    This is a collator augmenter expects HWC and converts it to CHW
    """
    def __init__(self):
        pass

    def __call__(self, sample: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        This padding and stacking transformer should be used in the collator (for example, the tiling collators in :class:`common.elements.legacy.detection.FixedTileCollator`

        :param sample:
        It takes a Dictionary with two lists: {'img': List and 'annot': List} from the collater and
        1. It stacks images in key 'img' [h, w, c] into one image with a minibatch dimension [b, c, h, w] (so it also moves the channel to fit Torch convention
        2. It stacks the annotations in key 'annot' [M, 5], but they need to be padded with [-1,] * 5 to be stackable.
        3. Returns a torch tensor.
        :return: A dictionary with the newly stacked Torch tensors in 'img' and 'annot'.

        """
        # Create minibatch of images
        img = torch.stack(sample['img'], axis=0).permute(0, 3, 1, 2).contiguous()
        sample['img'] = img
        sample['annot'] = pad_boxes(sample['annot'])

        return sample


class Stacking(object):
    """
    Stacks a list of images a 4d tensors and leaves the annotations untouched.
    This is a collater augmenter expects HWC and converts it to CHW
    """
    def __init__(self):
        pass

    def __call__(self, sample: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        """
        This stacking transformer should be used in the collator (for example, the tiling collators in :class:`common.detection.FixedTileCollator`

        :param sample:
        It takes a Dictionary with two lists: {'img': List and 'annot': List} from the collater and
        1. It stacks images in key 'img' [h, w, c] into one image with a minibatch dimension [b, c, h, w] (so it also moves the channel to fit Torch convention
        2. Returns a torch tensor.
        :return: A dictionary with the newly stacked Torch tensors in 'img'.

        """
        # Create minibatch of images
        img = torch.stack(sample['img'], axis=0).permute(0, 3, 1, 2).contiguous()
        sample['img'] = img
        return sample
