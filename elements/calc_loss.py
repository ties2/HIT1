import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
NoneType = type(None)
import torch.nn.functional as F
from typing import Callable

class MaskedKLDivergenceLoss(nn.Module):
    def __init__(self, ignore_indices=None):
        """
        Custom KL divergence loss function that ignores the specified classes.

        :param ignore_indices: List of indices of classes to ignore in loss computation.
        """
        super(MaskedKLDivergenceLoss, self).__init__()
        self.ignore_indices = ignore_indices if ignore_indices is not None else []

    def forward(self, outputs, targets):
        # create a mask to exclude ignored indices
        mask = torch.ones_like(targets, dtype=torch.bool)
        for index in self.ignore_indices:
            mask[:, index] = False

        # apply the mask
        masked_outputs = outputs[mask].view(outputs.size(0), -1)
        masked_targets = targets[mask].view(targets.size(0), -1)

        # compute the KLD loss
        loss = F.kl_div(masked_outputs, masked_targets, reduction='mean')
        return loss

def calc_divergence_loss(ignore_indices=None) -> Callable:
    """
    Initialize the MaskedKLDivergenceLoss with an optional ignore_indices parameter.

    :param ignore_indices: Indices to ignore when calculating the loss.
    :return: An instance of MaskedKLDivergenceLoss configured with the given parameters.
    """
    return MaskedKLDivergenceLoss(ignore_indices=ignore_indices)




