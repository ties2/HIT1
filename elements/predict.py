import torch
import torch.utils
import torch.utils.data
import time
import numpy as np
from elements.model_wrappers import UNet, DynamicCNN1D, DynamicCNN3D

# configure logging
from elements.utils import LoggerSingleton
logger = LoggerSingleton.get_logger()


def process_model_output(model, inputs, tile_height=100, inference_mode='patched', patch_size=(5, 5), stride=5,
                         pixel_batch_size=80000):
    """
    Process model outputs depending on whether the model is UNet, DynamicCNN3D, or DynamicCNN1D.
    - if UNet or DynamicCNN3D, applies torch.exp(model(inputs)).
    - if DynamicCNN1D:
        - 'patched' mode performs a grid search over the input, calculates the mean for each tile, passes it to the model,
          and fills the tile region with the model's prediction.
        - 'pixel-wise' mode processes each pixel independently.

    :param model: trained model (either UNet, DynamicCNN3D, or DynamicCNN1D)
    :param inputs: input tensor
    :param tile_height: tile height in pixels, used in UNet and DynamicCNN3D models
    :param inference_mode: 'patched' or 'pixel-wise' processing for DynamicCNN1D
    :param patch_size: size of the patches (height, width)
    :param stride: stride of the patches
    :param pixel_batch_size: batch size for processing in pixel-wise mode
    :return: final assembled output
    """
    torch.cuda.empty_cache()

    if isinstance(model, (DynamicCNN3D, UNet)):
        height = inputs.size(2)
        outputs_list = []
        for start in range(0, height, tile_height):
            end = min(start + tile_height, height)
            tile = inputs[:, :, start:end, :]
            tile_output = torch.exp(model(tile))
            outputs_list.append(tile_output)
        outputs = torch.cat(outputs_list, dim=2)

    elif isinstance(model, DynamicCNN1D):
        if inference_mode == 'patched':
            batch_size, _, input_height, input_width = inputs.shape
            num_classes = model.fc_layers[-1].out_features
            outputs = torch.zeros((batch_size, num_classes, input_height, input_width), device=inputs.device)
            patch_means = []
            patch_locations = []
            for y in range(0, input_height - patch_size[0] + 1, stride):
                for x in range(0, input_width - patch_size[1] + 1, stride):
                    patch = inputs[:, :, y:y + patch_size[0], x:x + patch_size[1]]
                    patch_mean = patch.mean(dim=[2, 3]).unsqueeze(1)
                    patch_means.append(patch_mean)
                    patch_locations.append((y, x))
            patch_means = torch.cat(patch_means, dim=0)
            patch_outputs = torch.exp(model(patch_means))
            patch_outputs = patch_outputs.view(-1, num_classes, 1, 1)
            for i, (y, x) in enumerate(patch_locations):
                tiled_output = patch_outputs[i].repeat(1, 1, patch_size[0], patch_size[1])
                outputs[:, :, y:y + patch_size[0], x:x + patch_size[1]] = tiled_output
        elif inference_mode == 'pixel-wise':
            batch_size, channels, height, width = inputs.shape
            num_classes = model.fc_layers[-1].out_features
            inputs = inputs.permute(0, 2, 3, 1).contiguous()
            inputs = inputs.view(-1, channels).unsqueeze(1)
            outputs = []
            for i in range(0, inputs.size(0), pixel_batch_size):
                sub_batch = inputs[i:i + pixel_batch_size]
                sub_output = torch.exp(model(sub_batch))
                outputs.append(sub_output)
            outputs = torch.cat(outputs, dim=0)
            outputs = outputs.view(batch_size, height, width, num_classes)
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            raise ValueError("Invalid inference_mode. It should be either 'patched' or 'pixel-wise'.")
    else:
        raise ValueError("Model type not recognized. It should be either UNet, DynamicCNN3D, or DynamicCNN1D.")

    return outputs

@torch.no_grad()
def collect_predictions(test_loader, model, results_dict, config):
    """
        Collects predictions from a model using a specified test loader, accumulates runtime statistics,
        and organizes predictions, labels, and other related data into a results dictionary structured
        by file name.

        This function iterates over batches provided by the test_loader, processes each batch through the model,
        and aggregates the results along with the corresponding inputs and labels into the results_dict.
        It also tracks the inference time for performance analysis.

        Parameters:
            test_loader (DataLoader): DataLoader containing the test dataset with batched inputs, labels, coordinates, and file names.
            model (torch.nn.Module): The model to evaluate.
            results_dict (dict): Dictionary to store inputs, predictions, labels, and coordinates grouped by file name.
            config (object): Configuration object containing device, inference parameters, and model-specific settings.

        Key Config Attributes:
            device (torch.device): Device to perform computations on (CPU or GPU).
            inference_tile_height (int): The tile height used for model inference.
            inference_mode (str): Mode of inference, e.g., 'patched' or 'pixel-wise'.
            inference_patch_size (tuple): Size of patches used in inference.
            inference_stride (int): Stride used in inference for patching.
            pixel_batch_size (int): Batch size for processing in pixel-wise mode.
            cascading_classifier (bool): Flag indicating whether cascading classifier adjustments are needed.
            background_threshold (float): Threshold value used for background classification in cascading classifier.
            background_index (int): Index representing the background class in the classifier's output.

        Effects:
            Updates the results_dict with predictions, corresponding labels, inputs, and coordinates for each file processed.
            Logs the average inference time per sample once processing is complete.

        Returns:
            dict: Updated results_dict containing structured prediction data.

        """
    total_runtime = 0
    num_samples = 0

    for inputs, labels, coords, file_names in test_loader:
        inputs, labels = inputs.to(config.device), labels.to(config.device)

        # process outputs depending on the model type, also calculating inference time
        start_time = time.time()
        outputs = process_model_output(model=model, inputs=inputs,tile_height=config.inference_tile_height,inference_mode=config.inference_mode,
                                       patch_size=config.inference_patch_size,stride=config.inference_stride,pixel_batch_size=config.pixel_batch_size)
        end_time = time.time()
        runtime = end_time - start_time
        total_runtime += runtime
        num_samples += 1

        if config.cascading_classifier:
            outputs = cascading_classifier_adjustment(outputs=outputs, threshold = config.background_threshold, background_index = config.background_index)

        # store predictions and labels by file name
        for i, file_name in enumerate(file_names):
            if file_name not in results_dict['data']:
                results_dict['data'][file_name] = {
                    'inputs': [],
                    'preds': [],
                    'labels': [],
                    'coords': []
                }
            results_dict['data'][file_name]['inputs'].append(inputs[i].cpu().numpy())
            results_dict['data'][file_name]['labels'].append(labels[i].cpu().numpy())
            results_dict['data'][file_name]['preds'].append(outputs[i].cpu().numpy())
            results_dict['data'][file_name]['coords'].append(coords[i].cpu().numpy())

    # convert collected data to numpy arrays
    for file_name, data in results_dict['data'].items():
        data['inputs'] = np.array(data['inputs'])
        data['preds'] = np.array(data['preds'])
        data['labels'] = np.array(data['labels'])
        data['coords'] = np.array(data['coords'])

        if data['inputs'].ndim == 4:
            data['inputs'] = np.moveaxis(data['inputs'], [0, 1, 2, 3], [0, 3, 1, 2])
            data['preds'] = np.moveaxis(data['preds'], [0, 1, 2, 3], [0, 3, 1, 2])
            data['labels'] = np.moveaxis(data['labels'], [0, 1, 2, 3], [0, 3, 1, 2])

    average_runtime = total_runtime / num_samples
    logger.info(f"Average inference time per sample: {average_runtime:.4f} seconds")
    return results_dict

def cascading_classifier_adjustment(outputs, threshold, background_index):
    """
    Adjust predictions using cascading classifier logic. If the background probability is below the threshold,
    normalize the probabilities of the remaining classes, excluding the background.

    :param outputs: Tensor of shape (bs, c, h, w) containing class probabilities.
    :param threshold: Threshold for background probability.
    :param background_index: Index of the background class.
    :return: Adjusted outputs tensor with normalized probabilities.
    """
    adjusted_outputs = outputs.clone()

    # extract background
    background_probs = adjusted_outputs[:, background_index, :, :]

    # mask for pixels where the background probability is below the threshold
    low_background_mask = background_probs < threshold

    # for these pixels, set the background probability to zero and normalize the rest
    adjusted_outputs[:, background_index, :, :][low_background_mask] = 0

    # calculate the sum of remaining probabilities for normalization
    normalization_factors = adjusted_outputs.sum(dim=1, keepdim=True)

    # avoid division by zero (happens when all probabilities are zero after adjustment)
    normalization_factors[normalization_factors == 0] = 1

    # normalize the adjusted outputs
    adjusted_outputs = adjusted_outputs / normalization_factors

    return adjusted_outputs
