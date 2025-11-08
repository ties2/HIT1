import matplotlib.pyplot as plt
import numpy as np


def plt_image(img, title=None, show=False, binary=False):
    if img.ndim == 2:
        if binary:
            plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(img, cmap='gray')
    else:
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    if title is not None: plt.title(title)
    if show: plt.show()


def plt_images(nrows, ncols, images, captions, figsize=None, show=False):
    if figsize is None:
        figsize = [12, 8]  # use default figure size
    nimages = min(nrows * ncols, len(images), len(captions))
    images = [i.squeeze() for i in images]
    fig = plt.figure(figsize=figsize)
    for i in range(0, nimages):
        ax = fig.add_subplot(nrows, ncols, i + 1, xticks=[], yticks=[])
        binary = ((images[i] == 0) | (images[i] == 1)).all()
        plt_image(images[i], captions[i], show=False, binary=binary)
    if show: plt.show()
    return fig
