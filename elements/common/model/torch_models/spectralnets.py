import csv
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn
import torch.nn.functional


class SpectralNetBase(torch.nn.Module, ABC):
    """
    Wrapper for Hyperspectral Analyser Software
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_rf(self) -> int:
        """
        Get the size of this network's receptive field
        """
        pass


class SAMNet(torch.nn.Module):
    """
    A Spectral Angle Mapping Neural Network
    """
    def __init__(self, channels, classes, map_to_angles=False, kernel_size=1):
        super(SAMNet, self).__init__()
        if kernel_size == 1:
            self.conv = torch.nn.Conv2d(in_channels=channels, out_channels=classes, bias=False, kernel_size=1)
        elif kernel_size == 3:
            self.conv = torch.nn.Conv2d(in_channels=channels, out_channels=classes, bias=False, kernel_size=3, padding=1)
        else:
            raise Exception(f"Unsupported kernel size {kernel_size}")
        self.channels = channels
        self.classes = classes
        self.map_to_angles = map_to_angles
        self.kernel_size = kernel_size

    def get_normalized_input(self, x):
        x = x.clone()
        ndims = x.shape[1]
        norm = torch.linalg.norm(x, dim=1, keepdim=True)
        # make zero sum spectral ranges a perfect white
        mask = torch.squeeze(norm, dim=1) == 0
        x = torch.movedim(x, 1, 0)  # Move spectral dim to front
        x[:, mask] = 1 / ndims
        x = torch.movedim(x, 0, 1)  # Move spectral dim back

        norm[norm == 0] = 1
        x = x / norm
        return x

    def normalize_weights(self):
        self.conv.weight = torch.nn.Parameter(self.conv.weight / torch.linalg.norm(self.conv.weight, dim=1, keepdim=True), requires_grad=True)

    def set_vectors_per_class(self, vectors: Dict[int, List[float]]):
        for class_id, vector in vectors.items():
            t = torch.tensor(vector, dtype=self.conv.weight.dtype)
            t2 = torch.empty(t.shape[0], self.kernel_size, self.kernel_size)
            # TODO: replace with resize or broadcast
            for y in range(self.kernel_size):
                for x in range(self.kernel_size):
                    t2[:, y, x] = t
            self.conv.weight.data[class_id, :, 0:self.kernel_size, 0:self.kernel_size] = t2

    def save_weights_to_csv(self, filename, row_names: Optional[List[str]]=None):
        nclasses, nchannels = self.conv.weight.data.shape[:2]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if row_names is not None:
                writer.writerow(row_names)
            else:
                writer.writerow(list(range(nclasses)))
            data = self.conv.weight.data[:, :, 0, 0].transpose(0, 1)
            for row in data:
                writer.writerow([float(item) for item in row])

    def forward(self, x):
        # Spectral Angle Mapping:
        # cos(y_c) = (w_c * x) / (norm(w_c) x norm(x)), where c is the filter index and * is the inner product
        self.normalize_weights()  # norm(w)
        x_norm = self.get_normalized_input(x)  # norm(x)
        y = self.conv(x_norm)  # w * x
        # nr_invalid = torch.sum(x > 1.) + torch.sum(x < -1.)
        # if nr_invalid > 0:
        #     print(f"[Warning] {nr_invalid} value(s) with an invalid range encountered in SAMNet")
        if self.map_to_angles:
            # x = torch.tanh(x)  # Alternative for clamp
            y = torch.clamp(y, -1, 1)
            y = torch.acos(y)
        return y


class MLPNet(torch.nn.Module):
    """
    A per-pixel MLP network
    """
    def __init__(self, channels, classes, hidden_units, activation=torch.nn.Tanh()):
        super(MLPNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, bias=True, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=channels, out_channels=hidden_units, bias=True, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(in_channels=hidden_units, out_channels=classes, bias=True, kernel_size=1)
        self.channels = channels
        self.classes = classes
        self.act = activation

    def forward(self, x):
        c1 = self.act(self.conv1(x))
        c2 = self.act(self.conv2(c1))
        c3 = self.conv3(c2)
        return c3


class PlasticNet(torch.nn.Module):
    """
    MRDNet adapted by KD with padding and removed tanh (like Unet)
    """
    def __init__(self, num_channels, num_classes: int, softmax=True, layers=(3, 7, 13)):
        super(PlasticNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channels, out_channels=112, kernel_size=layers[0], padding=layers[0] // 2)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(112, out_channels=56, kernel_size=layers[1], padding=layers[1] // 2)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(56, out_channels=28, kernel_size=layers[2], padding=layers[2] // 2)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(28, out_channels=num_classes, kernel_size=1)

        self.final = None
        if softmax:
            self.final = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        # x = F.max_pool2d(x, 2)
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)

        if self.final is not None:
            x = self.final(x)

        return x


class PlasticNetGeneral(SpectralNetBase):
    """
    More generalised form of PlasticNet, allowing for differing layer counts and different final activation
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layers: List[int] = None,
                 padding: bool = True,
                 final: Optional[torch.nn.Module] = torch.nn.Softmax(dim=1)):
        super().__init__()

        # handle default arguments
        if layers is None:
            # default layers
            layers = [3, 7, 13, 1]

        # check arguments
        assert in_channels > 0
        assert out_channels > 0
        assert len(layers) > 0
        assert all([kernel_size > 0 and (not padding or kernel_size % 2) for kernel_size in layers])

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._layers = layers
        self._padding = padding
        self._final = final

        # generate convolution layers
        conv_layers = self._generate_conv_layers(low_channels_mode=self._in_channels < (self._out_channels * 2))
        self._conv_block = torch.nn.Sequential(*conv_layers)

    def _generate_conv_layers(self, low_channels_mode: bool = False):
        channel_steps = [self._out_channels * 2, self._out_channels * 4, self._out_channels * 2]

        layers = []
        channels_div = (self._in_channels / self._out_channels) ** (
                    1 / len(self._layers))  # division factor for channel count for each layer
        in_channels = self._in_channels
        out_channels_float = float(self._in_channels)
        pad_size = 0
        for i, kernel_size in enumerate(self._layers):
            if i > 0:
                layers.append(torch.nn.ReLU())
            if self._padding:
                pad_size = kernel_size // 2

            if i == len(self._layers) - 1:
                out_channels = self._out_channels
            elif not low_channels_mode:
                out_channels_float = out_channels_float / channels_div
                out_channels = round(out_channels_float)
            elif i < len(channel_steps):
                out_channels = channel_steps[i]
            else:
                out_channels = channel_steps[-1]

            layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad_size))

            in_channels = out_channels
        return layers

    def get_rf(self) -> int:
        """
        Get the size of this network's receptive field
        """
        rf = 1
        for kernel_size in self._layers:
            rf += kernel_size - 1
        return rf

    def get_in_channels(self) -> int:
        """
        Get the number of input channels this network has been trained on
        """
        return self._in_channels

    def get_out_channels(self):
        """
        Get the number of output channels this network has been trained for
        """
        return self._out_channels

    def forward(self, x):
        x = self._conv_block(x)
        if self._final:
            x = self._final(x)
        return x


class CascadeSegLoss(torch.nn.Module):
    def __init__(self, alpha=1.):
        super(CascadeSegLoss, self).__init__()
        self.alpha = alpha

    def forward(self, first, second, target):
        first = torch.squeeze(first, dim=1)
        target_data_first = torch.stack([t["class_masks"][0] for t in target]).type(first.dtype)
        loss_first = torch.nn.BCEWithLogitsLoss()(first, target_data_first)

        target_data_second = torch.stack([t["class_masks"] for t in target])
        target_data_second = torch.argmax(target_data_second, dim=1, keepdim=False)
        target_data_second = target_data_second.long()
        loss_second = torch.nn.CrossEntropyLoss()(second, target_data_second)
        return self.alpha * loss_first + (1 - self.alpha) * loss_second


class CascadeSegNet(torch.nn.Module):
    """
    A spectral net that has a separate head for detecting background. During training pass a Dict of two Tensors:
    1. key "raw" containing the raw input image
    2. key "image" containing the preprocessed image (normalized spectral pixels)
    """
    def __init__(self, first_model, second_model, threshold=0.5, alpha=0.5):
        super(CascadeSegNet, self).__init__()
        self.first_model = first_model
        self.second_model = second_model
        self.threshold = threshold
        self.loss = CascadeSegLoss(alpha=alpha)
        self.sigmoid = torch.nn.Sigmoid()

    def process(self, x, y):
        input_first = torch.stack([images["raw"] for images in x])
        input_second = torch.stack([images["image"] for images in x])
        first = self.sigmoid(self.first_model(input_first))
        second = self.second_model(input_second)
        # Create mask for all probabilities above than threshold
        bg_mask = first[:, 0, :, :] >= self.threshold
        # Assign all pixels classified as background to background in the second model
        bg_mask = torch.unsqueeze(bg_mask, dim=0)
        second = torch.movedim(second, 1, 0)
        second *= ~bg_mask  # Make logits for all classes zero for background coordinates
        second[0][bg_mask[0]] = 1  # Force the background logit to one for background coordinates
        second[0][~bg_mask[0]] = 0  # Force the background logit to zero for background coordinates
        #  Logits for other classes which are not background are left untouched and will be determined by the second stage.
        second = torch.movedim(second, 0, 1)
        return first, second

    def forward(self, x, y=None):
        if y is None:
            _, second = self.process(x, y)
            result = [{"class_masks": t} for t in second]
            return result
        else:
            first, second = self.process(x, y)
            ls = self.loss(first, second, y)
            return ls


class CascadeInstSegLoss(torch.nn.Module):
    def __init__(self, alpha=1.):
        super(CascadeInstSegLoss, self).__init__()
        self.alpha = alpha

    def forward(self, first, loss_second, target):
        first = torch.squeeze(first, dim=1)
        target_data_first = torch.stack([t["class_masks"][0] for t in target]).type(first.dtype)
        loss_first = torch.nn.BCEWithLogitsLoss()(first, target_data_first)

        return self.alpha * loss_first + (1 - self.alpha) * loss_second


class CascadeInstSegNet(torch.nn.Module):
    """
    A spectral net that has a separate segmentation head for detecting background. During training pass a Dict of two Tensors:
    1. key "raw" containing the raw input image
    2. key "image" containing the preprocessed image (normalized spectral pixels)
    """
    def __init__(self, first_model, second_model, second_forward_func=None, threshold=0.5, alpha=0.5):
        """
        Initialized the class

        :param first_model: A segmentation model
        :param second_model: A model that returns a loss when in training mode (like MaskRCNN)
        :param threshold: Threshold value for background
        :param alpha: The weight between the first en de second model's loss.
        """
        super(CascadeInstSegNet, self).__init__()
        self.first_model = first_model
        self.second_model = second_model
        self.threshold = threshold
        self.second_forward_func = second_forward_func
        self.loss = CascadeInstSegLoss(alpha=alpha)

    def process(self, x, y):
        input_first = torch.stack([images["raw"] for images in x])
        input_second = torch.stack([images["image"] for images in x])
        first = self.first_model(input_first)
        # Create mask for all probabilities above than threshold
        bg_mask = first[:, 0, :, :] >= self.threshold

        # Make all background pixels 0 so that they easily found by the second model
        input_second = torch.movedim(input_second, 1, 0)
        input_second *= ~bg_mask
        input_second = torch.movedim(input_second, 0, 1)

        # Run second model
        if self.second_forward_func is None:
            second = self.second_model(input_second, y)  # Just run normally
        else:
            second = self.second_forward_func(self.second_model, input_second, y)
        return first, second

    def forward(self, x, y=None):
        if y is None:
            assert not self.second_model.training, "If second model is in training mode a target should be passed"
            _, second = self.process(x, y)
            return second
        else:
            assert self.second_model.training, "Unexpected target passed (the second model is not in training mode)"
            result_first, loss_second = self.process(x, y)
            ls = self.loss(result_first, loss_second, y)
            return ls

