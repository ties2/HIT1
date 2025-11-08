import torch
import torch.utils
import torch.utils.data
NoneType = type(None)
from elements.model_wrappers import UNet, DynamicCNN1D, DynamicCNN3D

# configure logging
from elements.utils import LoggerSingleton
logger = LoggerSingleton.get_logger()

def load_weights_cnn1d_to_cnn3d(cnn1d_model_path, cnn3d_model):
    """
    Load weights from a trained DynamicCNN1D model into a DynamicCNN3D model.

    :param cnn1d_model_path: Path to the trained DynamicCNN1D state dictionary.
    :param cnn3d_model: Instance of DynamicCNN3D to which the weights will be loaded.
    :return: The updated DynamicCNN3D model.
    """
    # load cnn1d state
    cnn1d_state_dict = torch.load(cnn1d_model_path)

    # get cnn3d state
    cnn3d_state_dict = cnn3d_model.state_dict()

    for name, param in cnn1d_state_dict.items():
        if name in cnn3d_state_dict:
            if "bias" in name:
                cnn3d_state_dict[name] = param # bis dimenstions are the same
            else:
                cnn3d_state_dict[name] = param.unsqueeze(-1).unsqueeze(-1)  # add two dimensions for weights

    # load the revised cnn1d state to cnn3d
    cnn3d_model.load_state_dict(cnn3d_state_dict, strict=False)

    return cnn3d_model

def initialize_model(model_type, in_channels, out_classes,start_filters, cnn_input_length, cnn_conv_block_type, cnn_conv_layers=2,
                     cnn_fc_layers=5,cnn_final_activation='logsoftmax', cnn_dropout=0.3, unet_depth=4,  best_state_path=None):
    """
    Initialize and return a model based on the specified model type.

    Args:
        model_type (str): Type of model to create ('cnn1d' or 'unet' or 'cnn3d).
        in_channels (int): Number of input channels.
        out_classes (int): Number of output classes.
        cnn_input_length (int): Length of the input vector (for 'cnn1d' model).
        cnn_conv_block_type (str): Type of convolutional block to use. Choices: 'A' for maxpool downsizing, 'B' for strided conv downsizing , 'C' for both.
        cnn_conv_layers (int, optional): Number of convolutional layers (for 'cnn1d' model). Defaults to 2.
        cnn_fc_layers (int, optional): Number of fully connected layers (for 'cnn1d' model). Defaults to 3.
        cnn_final_activation (str): The final activation function.
        cnn_dropout (float, optional): Dropout rate for the first fully connected layer (for 'cnn1d' model). Defaults to 0.3.
        unet_depth (int, optional): Depth parameter for U-Net (for 'unet' model). Defaults to 4.
        start_filters (int, optional): Starting number of filters for U-Net (for 'unet' model). Defaults to 64.
        best_state_path (str, optional): Path to load a saved model state. If provided, loads the saved state.

    Returns:
        nn.Module: Initialized model loaded to the specified device.
    """

    if model_type == 'cnn1d':
        model = DynamicCNN1D(in_channels=in_channels,num_classes=out_classes,input_length=cnn_input_length, conv_block_type=cnn_conv_block_type,
            num_conv_layers=cnn_conv_layers,num_fc_layers=cnn_fc_layers,final_activation=cnn_final_activation,
            start_filters=start_filters,dropout=cnn_dropout)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))

    elif model_type == 'cnn3d':
        model = DynamicCNN3D(in_channels=in_channels,num_classes=out_classes,num_conv_layers=cnn_conv_layers,
            num_fc_layers=cnn_fc_layers,final_activation=cnn_final_activation,start_filters=start_filters,
            dropout=cnn_dropout)
        if best_state_path is not None:
            try:
                model.load_state_dict(torch.load(best_state_path))
            except RuntimeError:
                load_weights_cnn1d_to_cnn3d(best_state_path, model)

    elif model_type == 'unet':
        model = UNet(in_channels=in_channels, num_classes=out_classes, depth=unet_depth, start_filters=start_filters)
        if best_state_path is not None:
            model.load_state_dict(torch.load(best_state_path))
    else:
        raise ValueError("Invalid model type. Choose 'cnn1d' or 'unet' or 'cnn3d'.")

    return model
