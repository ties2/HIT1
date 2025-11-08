import os
import gc
import json
import uuid
import pickle
import numpy as np
from abc import ABC
from glob import glob
from skimage import io
from skimage.draw import polygon
import xml.etree.ElementTree as et
from typing import List, Dict, Callable, Tuple, Any, Union

from common.elements.utils import get_cache_dir
from common.elements.encryption import decrypt_file
from common.data.loaders import load_npy, load_npz, load_jl, load_mono_image, load_avif_heif
from common.data.loaders.annotation import load_svly_as_inst_ids, load_svly_as_class_ids


class AnnotationType(ABC):
    pass


class Box(AnnotationType):
    """
    Generic container for a bounding box with metadata
    """

    def __init__(self, y1, x1, y2, x2, box_format="yxi", class_name="", class_id=0, filename="", patch_id=0, num_patches=0, image_height=1, image_width=1):
        self._y1 = y1
        self._x1 = x1
        self._y2 = y2
        self._x2 = x2
        self._box_format = box_format
        self._relative = False
        self.class_name = class_name
        self.class_id = class_id
        self.filename = filename
        self.patch_id = patch_id
        self.num_patches = num_patches
        self.image_height = image_height
        self.image_width = image_width

    @property
    def relative(self):
        return self._relative

    @relative.setter
    def relative(self, relative: bool):
        if self._relative == relative:
            return

        self._relative = relative
        if relative:
            self._y1 /= self.image_height
            self._x1 /= self.image_width
            self._y2 /= self.image_height
            self._x2 /= self.image_width
        else:
            self._y1 *= self.image_height
            self._x1 *= self.image_width
            self._y2 *= self.image_height
            self._x2 *= self.image_width

    @property
    def y1(self):
        return self._y1

    @y1.setter
    def y1(self, y1: Union[int, float]):
        self._y1 = y1

    @property
    def x1(self):
        return self._x1

    @x1.setter
    def x1(self, x1: Union[int, float]):
        self._x1 = x1

    @property
    def y2(self):
        return self._y2

    @y2.setter
    def y2(self, y2: Union[int, float]):
        self._y2 = y2

    @property
    def x2(self):
        return self._x2

    @x2.setter
    def x2(self, x2: Union[int, float]):
        self._x2 = x2

    @property
    def area(self):
        return self.width * self.height

    @property
    def width(self):
        return abs(self.x2 - self.x1)

    @property
    def height(self):
        return abs(self.y2 - self.y1)

    @property
    def y(self):
        return (self.y1 + self.y2) // 2

    @property
    def x(self):
        return (self.x1 + self.x2) // 2

    def clip(self, t=0., l=0., b=1., r=1.) -> 'Box':
        self.x1 = max(self.x1, l)
        self.y1 = max(self.y1, t)
        self.x2 = max(self.x2, l)
        self.y2 = max(self.y2, t)
        self.x1 = min(self.x1, r)
        self.y1 = min(self.y1, b)
        self.x2 = min(self.x2, r)
        self.y2 = min(self.y2, b)
        return self

    def set_xy_coords(self, relative: bool = False, *args, **kwargs):
        if all(coord in kwargs for coord in ("x1", "x2", "y1", "y2")):
            self.x1 = kwargs['x1']
            self.x2 = kwargs['x2']
            self.y1 = kwargs['y1']
            self.y2 = kwargs['y2']
        elif all(coord in kwargs for coord in ("x1", "y1", "w", "h")):
            self.x1 = kwargs['x1']
            self.y1 = kwargs['y1']
            self.x2 = kwargs['x1'] + kwargs['w']
            self.y2 = kwargs['y1'] + kwargs['h']
        elif all(coord in kwargs for coord in ("x", "y", "w", "h")):
            self.x1 = kwargs['x'] - (kwargs['w'] // 2)
            self.y1 = kwargs['y'] - (kwargs['h'] // 2)
            self.x2 = kwargs['x'] + (kwargs['w'] // 2)
            self.y2 = kwargs['y'] + (kwargs['h'] // 2)
        else:
            raise KeyError(f'Not all coordinate keys are set as keyword arguments: {kwargs}')

    def __str__(self):
        return f"{self.y1}, {self.x1}, {self.y2}, {self.x2}, {self.class_name}"

    @property
    def box_format(self):
        return self._box_format

    @box_format.setter
    def box_format(self, box_format: str):
        """
        :param box_format: Formats to use:
            yxi -> y1, x1, y2, x2, class_id,
            xy -> x1, y1, x2, y2
            xywh -> x, y, width, height (yolo)
            minxywh -> x1, y1, width, height (coco)
            nxy -> x1/img_w, y1/img_h, x2/img_w, y2/img_h
        :param box_format:
        :return:
        """
        self._box_format = box_format

    def numpy(self):
        """
        Return a numpy representation of the Box object
        :return: The Box information as a numpy array
        """
        if self.box_format == 'yxi':
            return np.array([self.y1, self.x1, self.y2, self.x2, self.class_id], dtype=np.float32)
        elif self.box_format == 'yx':
            return np.array([self.y1, self.x1, self.y2, self.x2], dtype=np.float32)
        elif self.box_format == 'xy':
            return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)
        elif self.box_format == 'xyi':
            return np.array([self.x1, self.y1, self.x2, self.y2, self.class_id], dtype=np.float32)
        elif self.box_format == 'xywh':  # yolo
            return np.array([self.x, self.y, self.width, self.height], dtype=np.float32)
        elif self.box_format == 'minxywh':  # coco
            return np.array([self.x1, self.y1, self.width, self.height], dtype=np.float32)
        elif self.box_format == 'nxy':
            return np.array([self.x1 / self.image_width, self.y1 / self.image_height, self.x2 / self.image_width, self.y2 / self.image_height], dtype=np.float32)
        else:
            raise NotImplementedError(f"Unknown format {self.box_format}")

    def as_filename(self):
        return f"{os.path.splitext(self.filename)[0]}_patchid_{self.patch_id}_patch_{self.y1}_{self.x1}_{self.y2}_{self.x2}"


def mask_to_boxes(mask: np.ndarray, class_ids: List[int] = None, class_names: List[str] = None, box_format: str = "yxi") -> List[Box]:
    """
    Extract boxes from mask with shape [h,w,c] or [h,w]

    :param class_ids: A list of class_ids corresponding to the instance_ids
    :param mask: The mask with unique instance ids
    :return: a list of Box
    """
    if len(mask.shape) == 3:
        mask = mask[:, :, 2]
    result = []
    instance_ids = [id for id in np.unique(mask) if id != 0]
    if class_ids is None:
        class_ids = [1] * len(instance_ids)
    if class_names is None:
        class_names = ["Object"] * len(instance_ids)
    h, w = mask.shape
    for class_id, class_name, instance_id in zip(class_ids, class_names, instance_ids):
        yy, xx = np.where(mask == instance_id)
        y1, y2, x1, x2 = np.min(yy), np.max(yy) + 1, np.min(xx), np.max(xx) + 1
        b = Box(y1=y1, x1=x1, y2=y2, x2=x2, box_format=box_format, class_id=class_id, class_name=class_name, image_height=h, image_width=w)
        result.append(b)
    return result


class LoadInstanceMasksFolder:
    """
    Handler to load masks of the Kaggle Data Science Bowl 2018
    """

    loaders = {'png': load_mono_image,
               'npy': load_npy}

    def get_types(self):
        return self.loaders.keys()

    def __call__(self, folder: str) -> Tuple[np.ndarray, List[Box]]:
        i = 1
        instance_ids = np.array([])
        for ext in self.loaders:
            filter = os.path.join(folder, "*." + ext)
            for filename in glob(filter):
                mask = self.loaders[ext](filename)
                mask[mask > 0] = i
                if instance_ids.size == 0:
                    instance_ids = mask
                else:
                    instance_ids += mask
                i += 1
        if instance_ids.size == 0:
            raise RuntimeError(f"Could not find any mask in {os.path.abspath(folder)}")
        boxes = mask_to_boxes(instance_ids)
        return instance_ids, boxes


class LoadClassMasksFolder:
    """
    Handler to load masks
    """

    def __call__(self, folder: str) -> Tuple[np.ndarray, List[Box]]:
        raise NotImplementedError()


class LoadInstanceMasksFile:
    """
    Handler to load masks of the PennFudan-type dataset (each pixel represents an instance id)
    """

    loaders = {'png': lambda x: [load_mono_image(x), None, None],
               'npy': lambda x: [load_npy(x), None, None],
               'json': load_svly_as_inst_ids}

    def get_types(self):
        return self.loaders.keys()

    def __call__(self, filename: str) -> Tuple[np.ndarray, List[Box]]:
        ext = os.path.splitext(filename)[1][1:]
        instance_ids, class_ids, class_names = self.loaders[ext](filename)
        boxes = mask_to_boxes(instance_ids, class_ids, class_names)
        return instance_ids, boxes


class LoadClassMasksFile:
    """
    Handler to load masks containing class ids (each pixel represents an class id) and a dictionary to translate class_ids to class_names
    """

    loaders = {'json': load_svly_as_class_ids}

    def get_types(self):
        return self.loaders.keys()

    def __call__(self, filename: str) -> [np.ndarray, Dict[int, str]]:
        ext = os.path.splitext(filename)[1][1:]
        class_ids, class_ids_names = self.loaders[ext](filename)
        return class_ids


class LoadBinMasksFile:
    """
    Handler to load binary masks
    """
    loaders = {'png': load_mono_image}

    def __init__(self, scale=1, invert=False, dtype=np.int16):
        self.scale = scale
        self.invert = invert
        self.dtype = dtype

    def get_types(self):
        return self.loaders.keys()

    def __call__(self, filename: str) -> Tuple[np.ndarray, List[Box]]:
        ext = os.path.splitext(filename)[1][1:]
        class_ids = self.loaders[ext](filename)
        dtype = class_ids.dtype
        class_ids = class_ids / self.scale
        class_ids = class_ids.astype(np.int64)
        if self.invert:
            class_ids = (class_ids - np.max(class_ids)) * -1
        if self.dtype is None:
            return class_ids.astype(dtype)
        else:
            return class_ids.astype(self.dtype)


def xml_to_boxes(xml_str: str) -> List[Box]:
    tree = et.fromstring(xml_str)
    bbs = []
    if tree is not None:
        for o in tree.iter("object"):
            class_name = o.find("name").text
            b = o.find("bndbox")
            y1 = int(b.find("ymin").text)
            x1 = int(b.find("xmin").text)
            y2 = int(b.find("ymax").text)
            x2 = int(b.find("xmax").text)
            bbs.append(Box(y1, x1, y2, x2, class_name=class_name))
    return bbs


def json_to_boxes(json_str: str) -> List[Box]:
    bbs = []
    content = json.loads(json_str)
    for shape in content['shapes']:
        p = shape['points']
        if shape['shape_type'] == 'circle':
            cx, cy, ex, ey = int(p[0][0]), int(p[0][1]), int(p[1][0]), int(p[1][1])
            r = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
            x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
        elif shape['shape_type'] == 'rectangle':
            x1 = shape['points'][0][0]
            y1 = shape['points'][0][1]
            x2 = shape['points'][1][0]
            y2 = shape['points'][1][1]
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            x1, y1, x2, y2 = xmin, ymin, xmax, ymax
        else:
            raise RuntimeError(f"Unsupported shape type: {shape['shape_type']}")

        class_name = shape['label']
        bbs.append(Box(y1, x1, y2, x2, class_name=class_name))
    return bbs


class LoadBoxes:
    loaders: Dict[str, Callable[[str], List[Box]]] = {
        '.xml': xml_to_boxes,
        '.json': json_to_boxes}
    """
    loaders = {'.png': load_image,
               '.jpg': load_image,
               '.npy': load_npy}
    """

    def get_types(self):
        return self.loaders.keys()

    def __call__(self, filename: str) -> List[Box]:
        for ext, func in self.loaders.items():
            fn = os.path.splitext(filename)[0] + ext
            if os.path.isfile(fn):
                with open(fn, "r") as f:
                    return self.loaders[ext]("".join(f.readlines()))
        return []


class LoadCryptBoxes(LoadBoxes):
    """
    Handler to load bounding box annotations
    """

    def __init__(self, key: bytes):
        self.key = key

    def __call__(self, filename: str) -> List[Box]:
        os.path.splitext(filename)
        for ext, func in self.loaders.items():
            fn = filename + ext + ".crypt"
            if os.path.isfile(fn):
                data = decrypt_file(fn, self.key)
                return self.loaders[ext](data)

        print("[Warning] LoadBoxes could not find any encrypted annotations")
        return []


def load_image(filename: str):
    img = io.imread(filename)
    if len(img.shape) == 2 or img.shape[2] == 1:  # Convert to RGB
        img_rgb = np.empty([img.shape[0], img.shape[1], 3], dtype=img.dtype)
        img_rgb[:, :, 0] = img
        img_rgb[:, :, 1] = img
        img_rgb[:, :, 2] = img
    elif len(img.shape) == 3 and img.shape[2] > 3:  # Remove alpha version
        img_rgb = img[:, :, :3]
    else:
        img_rgb = img
    return img_rgb


def load_binary_image(filename: str):
    img = io.imread(filename, as_gray=True)
    if len(img.shape) > 2 and img.shape[2] == 1:
        raise RuntimeError(f"The image is not monochrome: {img.shape}")
    return (img > 0.5) * 255


class LoadSingleImageFolder:
    """
    Handler to load a single image with a random filename from a folder (Kaggle DSB2018 format)
    """
    loaders = {'.png': load_image,
               '.jpg': load_image,
               '.npy': load_npy}

    def __call__(self, folder: str):
        filter = os.path.join(folder, "*.*")
        for filename in glob(filter):
            ext = os.path.splitext(filename)[1]
            if ext in self.loaders.keys():
                return self.loaders[ext](filename), filename

        raise RuntimeError(f"Could not find any image in {os.path.abspath(folder)} with extension {self.loaders.keys()}")


class LoadImage:
    """
    Handler to load images
    """

    loaders = {'.png': load_image,
               '.jpg': load_image,
               '.JPG': load_image,
               '.npy': load_npy,
               '.npz': load_npz,
               '.jl': load_jl,
               '.avif': load_avif_heif,
               '.heif': load_avif_heif,
               }

    def get_types(self):
        return self.loaders.keys()

    def __call__(self, filename: str):
        ext = os.path.splitext(filename)[1]
        return self.loaders[ext](filename)


class LoadImageFn(LoadImage):
    """
    Handler to load images (this version also returns the filename)
    """

    def __call__(self, filename: str):
        return super().__call__(filename), filename


class CachedResource:
    def __init__(self, data: Any, cache_file: str = None, ext="",
                 persist=False, save_func=lambda data, fname: pickle.dump(data, open(fname, "wb")),
                 load_func=lambda fname: pickle.load(open(fname, "rb"))):
        """
        This is a convenient wrapper around a cached data resource. In the constructor pass either the filename or the data.

        :param data: The data the object represents
        :param cache_file: The filename of the item. If this file exists it wil not be deleted if this object goes out of scope
        :param ext: The extension to add.
        :param persist: Determines if the object is persistent (if this value is false, the memory is automatically deleted)
        :param save_func: The function to use to save the resource (by default Pickle is used, but can be overridden to store images in known formats)
        :param load_func: The function to use to load the resource (by default Pickle is used, but can be overridden to load images in known formats)
        """
        self.load_func = load_func
        self.save_func = save_func
        if cache_file is None:
            self.cache_file = os.path.join(get_cache_dir(), str(uuid.uuid4())) + ext
        else:
            self.cache_file = cache_file
        self.own_file = False
        if not os.path.isfile(self.cache_file):
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            self.own_file = True
        self.data = data
        self.persist = persist
        self._save()

    def get(self) -> Any:
        if self.data is None:
            self._load()
        result = self.data
        if not self.persist:
            self._free()
        return result

    def _free(self):
        del self.data
        self.data = None
        gc.collect()

    def _load(self):
        self.data = self.load_func(self.cache_file)

    def _save(self):
        if not os.path.isfile(self.cache_file):
            self.save_func(self.data, self.cache_file)
        if not self.persist:
            self._free()

    def __del__(self):
        if os.path.isfile(self.cache_file) and self.own_file:
            os.remove(self.cache_file)

    def __call__(self):
        return self.get()
