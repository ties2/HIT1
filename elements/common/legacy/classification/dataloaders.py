import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, List

from common.elements.utils import DevNull
from common.elements.legacy.dataset import get_dataset_info, DatasetInfo, Dataset


def get_mnist_loader_pt(folder=get_dataset_info(Dataset.MNIST_TORCHVISION, DatasetInfo.INPUT_DATA), batch_size: int = 32, train: bool = False, shuffle: bool = True, reduced_train: int = 0, num_workers: int = 0) -> Tuple[torch.utils.data.DataLoader, List[str]]:
    """
    Returns a dataloader containing samples from the MNIST dataset

    :param folder: location of the dataset
    :param batch_size: The amount of MNIST samples to return to the iterator
    :param train: if True, the training set is returned, otherwise the testing set is returned.
    :param shuffle: if True, the samples are shuffled, otherwise not.
    :param reduced_train: size of the reduced dataset, in case not all samples should be used for training, =  zero takes the whole set
    :return: a dataloader

    >>> from common.elements.legacy.dataset import Dataset, DatasetInfo, get_dataset_info
    >>> loader, classes = get_mnist_loader_pt(folder = get_dataset_info(Dataset.MNIST_TORCHVISION, DatasetInfo.INPUT_DATA))
    >>> isinstance(loader, torch.utils.data.DataLoader)
    True
    """
    import sys
    out = sys.stdout
    sys.stdout = DevNull()
    try:

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        dataset = torchvision.datasets.MNIST(root=folder,
                                             train=train,
                                             download=True,
                                             transform=transform)
        if reduced_train > 0:
            dataset = torch.utils.data.random_split(dataset, [reduced_train, len(dataset) - reduced_train])[0]

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    finally:
        sys.stdout = out
    return loader, labels


def get_cifar10_dataloader_pt(folder: str = "cifar10_download", train: bool = True, batch_size=4, shuffle=True, suppress_stdout=False) -> Tuple[torch.utils.data.DataLoader, List[str]]:
    """
    Get the dataloader and the corresponding classnames for the CIFAR10 dataset

    :param folder: Folder where to find the dataset
    :param train: Boolean indicating if the training set is required (set to False when testing)
    :param batch_size: The batch size
    :param: shuffle: If the testing set is processed set shuffle to False.
    :param: suppress_stdout: If enabled, sys.stdout will be temporarily set to None to prevent the underlying torchvision dataset from printing to stdout.
    :return: The dataloader and the list of classes for converting class ids to classnames

    :example: Load CIFAR10 dataset.
    >>> from common.data.datasets_info import SupervisedClassificationDatasets, ABCDatasetInfo
    >>> from common.elements.legacy.dataset import Dataset, DatasetInfo, get_dataset_info
    >>> from elements.load_data import get_dataloader_classification
    >>> train_ds_info = SupervisedClassificationDatasets.cifar10_train
    >>> train_loader = get_dataloader_classification(ds_info=train_ds_info, preprocessing=[], batch_size=12, shuffle=True)
    >>> isinstance(loader, torch.utils.data.DataLoader)
    True
    >>> train_ds_info.class_names
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """
    import sys
    out = sys.stdout
    if suppress_stdout:
        sys.stdout = DevNull()
    try:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        set = torchvision.datasets.CIFAR10(root=folder, train=train, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    finally:
        sys.stdout = out
    return loader, set.classes
