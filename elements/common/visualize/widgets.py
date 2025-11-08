""" This utils file allows the creation of interactive plots with pywidgets """
import numpy as np
import cv2
import os

from ipywidgets import interact, interact_manual
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython.display import display


def change_pixels_digits(img: np.ndarray) -> np.ndarray:
    """
    This method changes the pixels of the letters/digits/minus signs to 255, while leaving the rest of the image
    unchanged.

    :param img: Input image
    :return: image with the pixels changed
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    bin_mask = thresh1 == 0
    img_zeros = img.copy()
    img_zeros[bin_mask] = 255

    return img_zeros


def change_pixels_background(img: np.ndarray) -> np.ndarray:
    """
    This method adds the value 80 to all the pixels from the NL background (on the left) of the plate image, while leaving the rest
    of the image unchanged. * The stars and the letters should remain unchanged.


    :param img: Input image
    :return: image with the pixels changed
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh1 = cv2.inRange(img, 90, 120)
    thresh1 = thresh1[:, 0:50]
    bin_mask = thresh1 == 255
    img_short = img[:, 0:50].copy()
    img_short[bin_mask] = img_short[bin_mask] + 80
    img_change = img.copy()
    img_change[:, 0:50] = img_short
    return img_change


def correct_cheque_background(img: np.ndarray) -> np.ndarray:
    """
    This method corrects the cheque image using the division and subtraction operations so the image
    text can become a bit more visible.

    :param img: Input image
    :return: divided image and the subtracted image (in this specific order)
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    back = np.tile(img[:, 390:395], 114)

    img_div = cv2.divide(img, back)
    img_subt = cv2.subtract(img, back)

    img_div = cv2.normalize(img_div, None, 0, 255, cv2.NORM_MINMAX)
    img_subt = cv2.normalize(img_subt, None, 0, 255, cv2.NORM_MINMAX)

    return img_div, img_subt


def nb_show_image(name: str = "circles.bmp"):
    """
        Plots an image
    :param name: string name of the file to be plotted
    """

    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fig = plt.figure(name[:-4], figsize=(9, 5))
    fig = plt.imshow(img, cmap="gray")


def nb_show_multiple(img_names: [], color: str = "gray"):
    """
        Plots multiple images in a grid format
    param name: list of string names of the files to be plotted
    param color: color code of the image to be plotted, default is gray
    """

    img_list = list()
    for name in img_names:
        img = cv2.imread(name)
        if color == "gray":
            img_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            cmap = "gray"
        else:
            img_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cmap = "jet"

    fig, axes = plt.subplots(nrows=len(img_names), ncols=1, figsize=(9, 9))

    for i in range(0, len(img_names)):
        axes[i].imshow(img_list[i], cmap=cmap)
        axes[i].title.set_text(os.path.split(img_names[i])[1][:-4])


class widget_image:
    """
        This class creates plots that perform certain tasks depending on the choices the user selects in the interactive
        plot
    """

    def change_value_image(self, operation: str, value: int):
        """
            Performs the mathematical operation defined by the user

        param operation: name of the operation to be performed
        param value: value to be used in the operation
        """

        img = cv2.imread(self.name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_image = img.copy()
        if operation == "Add":
            new_image = cv2.add(img, value)
        elif operation == "Subtract":
            new_image = cv2.subtract(img, value)
        elif operation == "Multiply":
            new_image = cv2.multiply(img, value)
        elif operation == "Divide":
            new_image = cv2.divide(img, value)

        # subplots to show the image before and after transforms
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
        axes[0].imshow(img, cmap='gray')
        axes[1].imshow(new_image, cmap='gray')
        fig.tight_layout()

        clear_output(wait=True)
        plt.pause(0.1)

    def threshold_value_image(self, value: int):
        """
            Performs thresholding with the value selected by the user

        param value: value to be used in the operation
        """

        img = cv2.imread(self.name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_image = img.copy()
        res, new_image = (cv2.threshold(new_image, value, 255, cv2.THRESH_BINARY))

        # subplots to show the image before and after transforms
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
        axes[0].imshow(img, cmap='gray')
        axes[1].imshow(new_image * 255, cmap='gray')
        fig.tight_layout()
        clear_output(wait=True)
        plt.pause(0.1)

    def show_images_math(self, operation=["Add", "Subtract", "Multiply", "Divide"], value=(0, 254, 1)):
        self.change_value_image(operation, value)

    def show_images_threshold(self, value=(0, 254, 1)):
        self.threshold_value_image(value)

    def __init__(self, name: str = "circles.bmp", operation_type="math"):
        self.name = name
        if operation_type == "math":
            my_interact_manual = interact_manual.options(manual_name="Apply")
            im = my_interact_manual(self.show_images_math)
        else:
            if operation_type == "threshold":
                my_interact_manual = interact_manual.options(manual_name="Apply")
                im = my_interact_manual(self.show_images_threshold)
            else:
                im = None
        display(im)


def show_image_operation():
    img1 = cv2.imread("/media/public_data/datasets/internal/other/landscape.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img2 = cv2.imread("/media/public_data/datasets/internal/other/background.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    imgsum = cv2.add(img1, img2)
    imgsub = cv2.subtract(imgsum, img2)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title("Background")
    axes[1, 0].imshow(imgsum)
    axes[1, 0].set_title("Addition")
    axes[1, 1].imshow(imgsub)
    axes[1, 1].set_title("Addition -> Subtraction")
    fig.tight_layout()
    clear_output(wait=True)
    plt.pause(0.1)


def show_image_circles():
    """
        This Example shows the image with different grayscale circles, run this function in a new cell below to plot the image
    """
    nb_show_image("/media/public_data/datasets/internal/other/image_math/circles.bmp")

def show_image_plate20():
    """
        This Example shows the car plate image necessary for Exercise 1, run this function in a new cell below to plot the image
    """
    nb_show_image("/media/public_data/datasets/internal/other/image_math/plate20.bmp")


def create_widget_circle():
    """
        This Example creates an interactive widget for the circle image
    """
    widget_image(name="/media/public_data/datasets/internal/other/image_math/circles.bmp", operation_type="math")


def create_widget_dice():
    """
        This Example creates an interactive widget for the dice image
    """
    widget_image(name="/media/public_data/datasets/internal/other/image_math/dice.png", operation_type="threshold")


def show_multiple():
    path = "/media/public_data/datasets/internal/other/image_math"
    nb_show_multiple([os.path.join(path,"cheque.bmp"), os.path.join(path,"cheque_div.png"), os.path.join(path,"cheque_subt.png")])