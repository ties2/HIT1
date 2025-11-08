"""
Hyperspectral Image Processing Pipeline

This script handles the end-to-end processing of hyperspectral image data,
from loading and preprocessing through to training, validating, and testing deep learning models.
It focuses on managing datasets for different types of neural network architectures,
including CNN1D, CNN3D, and UNet models. It includes dynamic configuration based on model type,
training, and inference settings.

Provides capabilities for:
1. Loading and splitting hyperspectral image datasets.
2. Configuring and initializing models with specific parameters tailored to the network type.
3. Training models with real-time visualization of loss metrics and model performance.
4. Validating and saving the best-performing model snapshots based on validation metrics.
5. Testing models with loaded datasets to evaluate performance and generate predictions.
6. Visualizing output mappings from the testing phase and saving results.

Environment setup:
- Adds necessary directories to the system path for module imports.
- Sets environment variables to configure library behaviors.

Configuration:
- Uses a `ParameterConfig` class to dynamically adjust settings based on the chosen model type,
  facilitating easy switches and experimentation with different model configurations.
- Handles device setup for GPU acceleration if available.

Logging:
- Integrates logging throughout the pipeline to provide insights into model training and evaluation processes.

Dependencies:
- torch: For model training, data handling, and GPU acceleration.
- elements: A collection of modules tailored for processing hyperspectral imaging data.

Usage:
- This script is intended to be run directly with Python after configuring paths and dependencies.
- Adjust the `ParameterConfig` class to switch between different operational modes (training, testing, visualization).

Examples of functionality include training a 3D CNN on tiled hyperspectral data, testing a trained UNet for semantic segmentation,
and visualizing the output masks compared to ground truth.

Author:
- [Milad Isakhani Zakaria]

Notes:
- Ensure all dependencies are installed and paths in `ParameterConfig` are correctly set relative to the dataset locations.
- Review the usage of environment variables and logging setups to align with your deployment environment or debugging preferences.
"""

import os
import time
import torch
import warnings
# from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader,random_split
from torch.amp import GradScaler, autocast
from elements.common.model.torch_models.ganomaly import weights_init



# working dir
def _get_working_dir():
    return  '/home/student/myprojects/HIT1/'

# configure logging
from elements.utils import LoggerSingleton
LoggerSingleton.setup_logger(_get_working_dir())
logger = LoggerSingleton.get_logger()

# pipeline imports
from elements.common.utils import static_var
from elements.load_data import load_hsi_dataset, stratified_split_by_composition, load_prediction_dict
from elements.load_model import initialize_model
from elements.optimize import get_adam_optimizer_pt
from elements.predict import collect_predictions
from elements.tune_params import get_step_lr_pt
from elements.calc_loss import calc_divergence_loss
from elements.calc_metrics import calculate_metrics
from elements.visualize import create_tb, show_loss_tb, display_output_mapping, create_output_mapping
from elements.save_model import save_model
from elements.save_results import create_prediction_dict, create_results_dict, save_prediction_dict

# disable specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

@static_var(model_saved=False)
@static_var(best_val_loss=9999)
def run_training(train_loader, val_loader, model, optimizer, loss_func, scheduler,config):
    """
    Executes the training process using specified data loaders, optimizer, loss function,
    and learning rate scheduler. Handles training over a set number of epochs, defined in a global configuration.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        model (torch.nn.Module): Neural network model to train.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        loss_func (callable): Loss function to evaluate model performance.
        scheduler (torch.optim.lr_scheduler): Scheduler to adjust learning rate based on performance.
        config (ParameterConfig): A ParameterConfig object.

    Returns:
        None
    """
    train_losses, val_losses = [], []
    scaler = GradScaler()
    start_time = time.time()

    for epoch in range(config.num_epochs):
        model.train()
        running_train_loss = 0.0
        running_loss_count = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=config.device):
                outputs = model(inputs)
                loss = loss_func(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            running_loss_count += 1

        train_losses.append(running_train_loss / running_loss_count)

        if (epoch + 1) % config.val_frequency == 0 or epoch == config.num_epochs - 1 or epoch == 0:
            val_loss = run_validation(model, val_loader, loss_func, config.device)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < run_training.best_val_loss:
                run_training.best_val_loss = val_loss
                save_model(model, os.path.join(_get_working_dir(), 'log', 'experiment'))
                run_training.model_saved = True
            else:
                run_training.model_saved = False

            # visualization & logs
            show_loss_tb(val_loss, epoch, writer=config.writer, name='Valid Loss')
            show_loss_tb(train_losses[epoch], epoch, writer=config.writer, name='Train Loss')
            time_passed = time.time() - start_time
            logger.info(f"Epoch {epoch + 1:<3}/{config.num_epochs:<3} | "
                        f"TrainLoss: {train_losses[epoch]:<8.6f} | "
                        f"ValidLoss: {val_losses[epoch] if val_losses[epoch] is not None else 'N/A':<8.6f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:<8.6f} | "
                        f"ModelSaved: {str(run_training.model_saved):<5} | "
                        f"Relative Time: {time_passed:<6.2f}s")
        else:
            val_losses.append(None)

@torch.no_grad()
def run_validation(model, data_loader, loss_func, device):
    """
    Evaluate the model's loss on a given dataset.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        data_loader (DataLoader): DataLoader for the dataset to evaluate on.
        loss_func (callable): Loss function.
        device (str): Device to run evaluation on ('cuda' or 'cpu').

    Returns:
        float: Average loss over the entire dataset.
    """
    model.eval()
    running_val_loss = 0.0
    running_loss_count = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        running_val_loss += loss.item()
        running_loss_count += 1

    return running_val_loss / running_loss_count

@torch.no_grad()
def run_testing(test_loader,model,results_dict, config):
    """
    Executes the testing process for a given model using a specified data loader, accumulating results,
    mapping outputs, calculating metrics, and compiling them into a final prediction dictionary.

    This function coordinates several steps in the evaluation of the model:
    1. Collecting predictions from the model using test data.
    2. Mapping these predictions to their respective outputs.
    3. Calculating evaluation metrics based on these predictions.
    4. Creating a structured prediction dictionary that summarizes the testing results.

    Parameters:
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        model (torch.nn.Module): The trained model to be evaluated.
        results_dict (dict): A dictionary where intermediate results are stored.
        config (ParameterConfig): A ParameterConfig object.

    Returns:
        dict: A dictionary containing the consolidated results and metrics from the model testing.

    """

    # step 1: collect predictions and save to results dict
    results_dict = collect_predictions(test_loader=test_loader, model=model, results_dict=results_dict,config=config)

    # step 2: create output mappings and save to results dict
    results_dict = create_output_mapping(results_dict=results_dict)

    # step 3: calculate metrics and save to results dict
    results_dict = calculate_metrics(results_dict=results_dict,saved_dir=_get_working_dir())

    # step 4: create and return the final prediction dictionary
    prediction_dict = create_prediction_dict(results_dict)

    return prediction_dict

def run_pipeline(config):
    """
    Runs the training, validation, and testing pipeline on the dataset.

        Parameters:
        config (ParameterConfig): A ParameterConfig object.
    """

    logger.info("Starting pipeline...")
    for attr, value in vars(config).items():
        logger.info(f"{attr}: {value}")

    if config.do_train:
        logger.info("Starting training pipeline ...")

        dataset = load_hsi_dataset(dataset_path=os.path.join(_get_working_dir(), 'dataset', config.train_dataset_name))
        #old method
        # train_dataset, val_dataset = stratified_split_by_composition(dataset=dataset,train_ratio=config.train_ratio,
        #     model_type=config.model_type,check_stratification=config.check_stratification)

        #update by me
        train_size = int(config.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # for reproducibility
        )


        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)

        model = initialize_model(model_type=config.model_type,in_channels=config.in_channel,out_classes=len(dataset.get_class_names()),
            cnn_conv_block_type=config.cnn_conv_block_type,start_filters=config.start_filters,cnn_input_length=config.cnn_input_length,
            cnn_conv_layers=config.cnn_conv_layers,cnn_fc_layers=config.cnn_fc_layers,
            cnn_dropout=config.cnn_dropout_rate,unet_depth=config.unet_depth).to(config.device)
        loss_func = calc_divergence_loss(ignore_indices=config.ignore_indices)
        optimizer = get_adam_optimizer_pt(model=model,learning_rate=config.learning_rate)
        scheduler = get_step_lr_pt(optimizer, mode='min', patience=config.patience, factor=config.factor)

        run_training(train_loader=train_loader,val_loader=val_loader,model=model,optimizer=optimizer,loss_func=loss_func,scheduler=scheduler,config=config)

    if config.do_test:
        logger.info("Starting inference pipeline ...")
        torch.cuda.empty_cache()
        dataset = load_hsi_dataset(dataset_path=os.path.join(_get_working_dir(), 'dataset', config.test_dataset_name))
        #update by me
        dataset.training_mode = False  # Ensure dataset returns all 4 items

        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

        model = initialize_model(model_type=config.model_type,in_channels=config.in_channel,out_classes=len(dataset.get_class_names()),
            cnn_conv_block_type=config.cnn_conv_block_type, start_filters=config.start_filters,cnn_input_length=config.cnn_input_length,cnn_conv_layers=config.cnn_conv_layers,
            cnn_fc_layers=config.cnn_fc_layers,cnn_dropout=config.cnn_dropout_rate,unet_depth=config.unet_depth,
            best_state_path=os.path.join(_get_working_dir(),'log',config.experiment if config.do_test else 'experiment','model.npz')).to(config.device)

        results_dict = create_results_dict(dataset=dataset)
        prediction_dict = run_testing(test_loader=test_loader, model=model, results_dict=results_dict, config=config)
        save_prediction_dict(prediction_dict, os.path.join(_get_working_dir(), 'log', 'experiment', 'prediction_dict.npz'))

    if config.display_mapping:
        prediction_dict = load_prediction_dict(os.path.join(_get_working_dir(), 'log', 'experiment', 'prediction_dict.npz'))
        results_dir = os.path.join(_get_working_dir(), 'log', 'experiment', 'results')

        # check if the results directory exists, if not create it
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        display_output_mapping(prediction_dict,results_dir)
        logger.info(f"Results successfully saved to {results_dir}")

# execute
# -------------------------------------------------
# cnn1d      → PATCHES       → .npz   (1D spectra)
# cnn1d      → PATCHES       → .npz   (1D spectra)
# cnn3d      → TILES         → .npz   (3D cubes)
# cnn3d      → TILES         → .npz   (3D cubes)
# unet       → TILES         → .pt    (full image + coordinates)
# unet       → TILES         → .pt    (full image + coordinates)
# -------------------------------------------------
# hitfusion  → TILES         → .npz   (fusion-ready cubes)
#

if __name__ == '__main__':
    class ParameterConfig:
        def __init__(self):
            self.model_type = 'hitfusion'                          # cnn1d or cnn3d or unet or hitfusion

            # specific parameters of cnn1d
            if self.model_type == 'cnn1d':
                self.train_dataset_name = 'train_patched.npz'  # dataset name for training
                self.test_dataset_name = 'test_patched.npz'     #update by me
                self.in_channel = 1                             # input channel of cnn1d is set to 1
                self.start_filters = 32                         # start filters of cnn1d
                self.cnn_conv_block_type = 'A'                  # Choices: 'A' for maxpool downsizing, 'B' for strided conv downsizing , 'C' for both.
                self.cnn_input_length = 224                     # default is 224, choose 223 when using HH preprocessing
                self.cnn_conv_layers = 3                        # number of conv layers of cnn1d
                self.cnn_fc_layers = 2                          # number of dense layers of cnn1d
                self.cnn_dropout_rate = 0.3                     # dropout rate after the first fc layer
                self.batch_size = 1600                          # batch size for cnn1d
                self.num_workers = 4                            # number of dataloader workers, best setting is 4
                self.patience = 7                               # scheduler patience
                self.factor = 0.7                               # scheduler factor
                self.learning_rate = 0.001                      # learning rate for cnn1d
                self.unet_depth = None                          # unet depth is set to None using cnn1d model

            # specific parameters of cnn3d
            if self.model_type == 'cnn3d':
                self.train_dataset_name = 'train_tiled.npz'     # dataset name for training
                self.test_dataset_name = 'test_tiled.npz'       # update by me
                self.in_channel = 224                           # default is 224, choose 223 when using HH preprocessing
                self.start_filters = 32                          # start filters of cnn3d, best to choose 4 due to memory limitations
                self.cnn_conv_layers = 3                        # number of conv layers of cnn3d
                self.cnn_fc_layers = 2                          # number of dense layers of cnn3d
                self.cnn_dropout_rate = 0.3                     # dropout rate after the first fc layer
                self.batch_size = 64                            # batch size for cnn3d
                self.num_workers = 4                            # number of dataloader workers, best setting is 4
                self.patience = 7                               # scheduler patience
                self.factor = 0.7                               # scheduler factor
                self.learning_rate = 0.0007                     # learning rate for cnn3d
                self.cnn_conv_block_type = None                 # None for cnn3d
                self.unet_depth = None                          # unet depth is set to None using cnn3d model
                self.cnn_input_length = None                    # cnn input length is set to None using cnn3d model
                self.cnn_final_activation = 'none'              # cnn input length is set to 'none' using cnn3d model

            # specific parameters of unet
            if self.model_type == 'unet':
                self.train_dataset_name = 'train_tiled.npz'     # dataset name for training
                self.test_dataset_name = 'test_tiled.pt'        # TILES → .pt  (MUST BE .pt!)
                self.in_channel = 224                           # default is 224, choose 223 when using HH preprocessing
                self.start_filters = 48                         # start filters of unet
                self.unet_depth = 5                             # depth of the encoder-decoder layers in unet
                self.batch_size = 640                           # batch size for unet
                self.num_workers = 6                            # number of dataloader workers, best setting is 6 using unet
                self.patience = 7                               # scheduler patience
                self.factor = 0.7                               # scheduler factor
                self.learning_rate = 0.0007                     # learning rate for unet
                self.cnn_conv_block_type = None                 # None when using unet
                self.cnn_input_length = None                    # cnn input length is set to None using the unet model
                self.cnn_conv_layers = None                     # cnn number of conv layers is set to None using the unet model
                self.cnn_fc_layers = None                       # cnn number of fc layers is set to None using the unet model
                self.cnn_dropout_rate = None                    # cnn dropout rate is set to None using the unet model
                self.cnn_final_activation = 'none'              # cnn final activation layer is set to 'none' using the unet model

            if self.model_type == 'hitfusion':
                self.ignore_indices = [0]  # choose indices to be ignored by the model 0 for void
                self.train_dataset_name = 'train_tiled_new.npz'
                self.test_dataset_name = 'test_tiled.npz'
                self.in_channel = 224
                self.start_filters = 32
                self.cnn_conv_block_type = None
                self.cnn_input_length = None
                self.cnn_conv_layers = 3  # reuse as number of context layers
                self.cnn_fc_layers = None
                self.cnn_dropout_rate = 0.3
                self.unet_depth = None
                self.batch_size = 128
                self.num_workers = 4
                self.patience = 7
                self.factor = 0.7
                self.learning_rate = 0.0007

            # choose general parameters
            self.ignore_indices = [0]                           # choose indices to be ignored by the model 0 for void
            self.train_ratio = 0.7                              # set train-validation ratio
            self.check_stratification = False                   # check tiles or patches stratification
            self.pin_memory = True                              # dataloader pin memory
            self.num_epochs = 400                               # number of epochs
            self.val_frequency = 1                              # set validation frequency
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.writer = create_tb(os.path.join(_get_working_dir(),'log','experiment'))

            # inference parameters
            self.test_dataset_name = 'test.npz'                 # dataset name for inference
            self.cascading_classifier = False                   # used when data is preprocessed using HyperHue
            self.background_threshold = 0                       # used during cascading_classifier = True
            self.background_index = 1                           # used during cascading_classifier = True
            self.inference_tile_height = 100                    # used during inference using 'unet' or 'cnn3d'
            self.inference_mode = 'pixel-wise'                  # 'patched' is more accurate or 'pixel-wise' is faster
            self.inference_patch_size = (5,5)                   # used during inference_mode = 'patched'
            self.inference_stride = 5                           # used during inference_mode = 'patched'
            self.pixel_batch_size = 80000                       # used during inference_mode = 'pixel-wise'

            self.do_train = True                              # True to run training pipeline
            self.do_test = False                             # True to run test pipeline
            self.display_mapping = False                      # True to display output mapping
            self.experiment = 'experiment'                      # my update


    run_pipeline(config = ParameterConfig())