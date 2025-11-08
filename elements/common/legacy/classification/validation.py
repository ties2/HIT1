import torch
import torch.nn
import torch.utils
import torch.utils.data


def calc_class_accuracy_pt(model: torch.nn.Module, loader: torch.utils.data.DataLoader, dev: str = "cuda:0"):
    """
    This method runs a model over the inputs and targets in the dataloader and returns the accuracy.

    :param model: the classification model to run
    :param loader: the dataloader is assumed to produce a Tuple of inputs and targets with shape [b,c,h,w] and [b,c]
    respectively, where b, c, h, w are the batch size, number of classes, height and width respectively.
    :param dev: the device to run on.
    :return: the accuracy (number of correct classifications divided by the

    >>> from common.elements.legacy.classification import calc_class_accuracy_pt
    >>> from common.elements.model.torch_models.lenet import LeNet
    >>> from common.elements.legacy.classification import get_cifar10_dataloader_pt

    >>> l, _ = get_cifar10_dataloader_pt(batch_size=5000)
    >>> round(calc_class_accuracy_pt(LeNet(), l), 1)
    0.1
    """
    model.to(dev)
    with torch.no_grad():
        tp = 0
        n = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(dev), targets.to(dev)
            outputs = model(inputs)
            output_ids = torch.argmax(outputs, dim=1)
            tp += torch.sum(output_ids == targets).item()
            n += len(output_ids)
        return tp / n
