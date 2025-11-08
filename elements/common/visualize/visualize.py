from typing import Optional, Tuple
import matplotlib.pyplot as plt
import cv2
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def hyperspectral_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a hyperspectral image to RGB using the average value of 1/3th of the HS cube for each RGB channel

    :param image: Image of size [h, w, c]
    :return: And RGB image with size [h, w, 3]
    """
    nc = image.shape[-1]
    img = np.empty((image.shape[0], image.shape[1], 3), dtype=image.dtype)
    img[..., 2] = np.average(image[..., 0:nc // 3], axis=-1)
    img[..., 1] = np.average(image[..., nc // 3:2 * (nc // 3)], axis=-1)
    img[..., 0] = np.average(image[..., 2 * (nc // 3):], axis=-1)
    return img


def add_image_name(img: np.ndarray, w: int, dark_mode: bool, image_name: str):
    if w < 100:
        print("Image has not enough width for the use of image_name above the image")
        return img
    mult_factor_name_banner = 48 if dark_mode else 255
    img = np.concatenate((np.ones(shape=(80, w, 3)).astype(np.uint8) * mult_factor_name_banner, img), axis=0)
    cv2.putText(img, image_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    return img


def show_image_tb(img: np.ndarray, writer: SummaryWriter, epoch=0, name: str = "image", dataformats: str = "HWC",
                  image_name: str = None, dark_mode: bool = True, invert: bool = False,
                  normalize=False, clip_values: Optional[Tuple[float, float]] = None):
    """
    Show an image TensorBoard.

    :param img: Image to display. The shape of this image is equal to dataformats
    :param epoch: The tensorboard step
    :param invert: Inverts the values for each channel (255 - values), can be useful for grayscale images
    :param name: The name of the image
    :param image_name: Text to display above the image
    :param dark_mode: image_name banner above the image blends into tensorboard given dark_mode or not
    :param dataformats: the dataformat: HWC or CHW
    :param normalize: Flag indicating if images should be rescaled between 0 an 255.
    :param writer: A SummaryWriter to write to
    :param clip_values: Tuple of [a, b], clip values based on the statistics in the image [-a * sigma .. +b * sigma]

    >>> import os
    >>> wr = SummaryWriter()
    >>> image = np.zeros((10, 10, 3))
    >>> show_image_tb(img=image, epoch=0, writer=wr)
    """
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        dataformats = "HWC"
    if dataformats == "CHW":
        img = np.moveaxis(img, 0, 2)
        dataformats = "HWC"
    # Take the mean of the channels if it's a hyperspectral cube
    h, w, c = img.shape
    if c > 3:
        img = hyperspectral_to_rgb(img)
    if clip_values is not None:
        mn = np.mean(img)
        sd = np.std(img)
        img = np.clip(img, mn - clip_values[0] * sd, mn + clip_values[1] * sd)
    if normalize:
        img = (((img - np.min(img)) / (np.max(img) - np.min(img))) * 255)
    img = np.ascontiguousarray(img.astype(np.uint8))

    if invert:
        img = 255 - img

    if image_name:
        add_image_name(img=img, w=w, dark_mode=dark_mode, image_name=image_name)

    writer.add_image(name, img, epoch, dataformats=dataformats)
    writer.flush()

