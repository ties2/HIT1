import torch
import torch.nn as nn


class ChannelConv(nn.Module):

    def __init__(self, backbone: torch.nn.Module, in_channels, out_channels=3, kernel_size=1):
        """
        This model encapsulates any model and adds an additional layer to convert from in_channels to 3 channels.

        :param backbone: The backbone to wrap
        :param in_channels: The number of features of the input
        :param out_channels: The number of features for the input of model
        """
        super(ChannelConv, self).__init__()
        self.cv_backbone = backbone
        self.cv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size))

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        return self.cv_backbone.__getattr__(name)

    def forward(self, x):
        if isinstance(x, list):
            x = [torch.squeeze(self.cv_layer(torch.unsqueeze(xx, dim=0)), dim=0) for xx in x]
        else:
            x = self.cv_layer(x)
        y = self.cv_backbone(x)
        return y
