import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
import os
import cv2

from common import deprecated
from common.elements.legacy.dataset import get_dataset_info, DatasetInfo
import numpy as np


def np_one_hot(image, num_classes):            # for numpy
    assert image.ndim == 2, f'tensor has {image.ndim} dimensions'
    one_hot = np.empty([num_classes, image.shape[0], image.shape[1]], np.float32)
    one_hot[0] = ~(image != 0)
    for i in range(1, num_classes): one_hot[i] = image == i
    return one_hot


def torch_one_hot(image, num_classes):          # for pytorch
    assert image.ndim == 2, f'tensor has {image.ndim} dimensions'
    one_hot = torch.empty([num_classes, image.shape[0], image.shape[1]], dtype=torch.float)
    one_hot[0] = ~(image != 0)
    for i in range(1, num_classes): one_hot[i] = image == i
    return one_hot


def one_hot_tensor(t, num_classes):              # for pytorch
    assert t.ndim == 4, f'tensor has {t.ndim} dimensions'
    assert t.shape[1] == 1, f'tensor.t[1] is {t.image.shape[1]}'
    one_hot = torch.empty(t.shape[0], num_classes, t.shape[2], t.shape[3])
    for n in range(t.shape[0]):
        one_hot[n] = torch_one_hot(t[n,0], num_classes)
    return one_hot


class SegmentationDataset(torch.utils.data.Dataset):
    @deprecated("This dataset is deprecated, please use the GenericDataset from common.data instead.")
    def __init__(self, images_dir, masks_dir, image_names, num_classes, shape, sub, div):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        with open(image_names) as f:
            self.files = f.read().splitlines()
        self.num_classes = num_classes
        self.shape = shape
        self.sub = sub
        self.div = div

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_fn = os.path.join(self.images_dir, self.files[index])
        image = cv2.imread(image_fn, cv2.IMREAD_ANYDEPTH)
        assert image is not None, f'{image_fn} not found'
        mask_fn = os.path.join(self.masks_dir, self.files[index])
        mask = cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f'{os.path.join(self.masks_dir, self.files[index])} not found'
        image = cv2.resize(image, self.shape)
        mask = cv2.resize(mask, self.shape, interpolation=cv2.INTER_NEAREST)
        image = 2 * ((image.astype(np.float32) - self.sub) / self.div).astype(np.float32)
        input = image.reshape(1, image.shape[0], image.shape[1])
        targets = np_one_hot(mask, self.num_classes)
        return input, targets


def create_segmentation_data_loaders_generic(images_dir, masks_dir, image_names, num_classes,
                                            shape, sub, div,
                                            batch_size, valid_size, num_workers):
    dataset = SegmentationDataset(images_dir, masks_dir, image_names, num_classes, shape, sub, div)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    if valid_size > 0:
        split = int(np.floor(valid_size * len(dataset)))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=num_workers, pin_memory=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,
                                                   num_workers=num_workers, pin_memory=True, drop_last=True)
        assert len(train_loader) > 0, 'nr batches for train_loader == 0'
        assert len(valid_loader) > 0, 'nr batches for valid_loader == 0'
        return [train_loader, valid_loader]
    else:
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        sampler = SubsetRandomSampler(indices)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                             num_workers=num_workers, pin_memory=True, drop_last=True)
        assert len(loader) > 0, 'nr batches for loader == 0'
        return [loader]


def get_segmentation_data_loaders_generic(dataset, shape, batch_size, valid_size):
    data_set_dir = get_dataset_info(dataset, DatasetInfo.DATA)
    images_dir = os.path.join(data_set_dir, get_dataset_info(dataset, DatasetInfo.IMAGE_DIR))
    masks_dir = os.path.join(data_set_dir, get_dataset_info(dataset, DatasetInfo.MASK_DIR))
    max_pixel_value = get_dataset_info(dataset, DatasetInfo.MAX_PIXEL_VALUE)
    num_classes = get_dataset_info(dataset, DatasetInfo.NUM_CLASSES)
    image_names = os.path.join(data_set_dir, get_dataset_info(dataset, DatasetInfo.IMAGE_NAMES))
    num_workers = max(batch_size, 4)
    return create_segmentation_data_loaders_generic(images_dir, masks_dir, image_names, num_classes, shape, max_pixel_value/2,
                                                    max_pixel_value, batch_size, valid_size, num_workers)


def create_segmentation_data_loader(images_dir, masks_dir, image_names, num_classes,
                                    shape, sub, div,
                                    batch_size, num_workers):
    loaders = create_segmentation_data_loaders_generic(images_dir, masks_dir, image_names, num_classes,
                                                       shape, sub, div, batch_size, valid_size=0, num_workers=num_workers)
    return loaders[0]


def create_segmentation_data_loaders(images_dir, masks_dir, image_names, num_classes,
                                     shape, sub, div,
                                     batch_size, valid_size, num_workers):
    assert valid_size > 0
    loaders = create_segmentation_data_loaders_generic(images_dir, masks_dir, image_names, num_classes,
                                                       shape, sub, div, batch_size, valid_size, num_workers)
    return loaders[0], loaders[1]

