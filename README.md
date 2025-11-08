
Project HIT
---
## Overview
This repository contains a comprehensive hyperspectral image processing pipeline capable of handling tasks from dataset creation to training, validating, and testing deep learning models. The primary focus is on managing datasets specifically formatted for neural network architectures such as CNN1D, CNN3D, and UNet. The repository includes various scripts and utilities that automate the processing of hyperspectral data, model training, performance evaluation, and result visualization.

---
## Repository Structure
```plaintext
/
├── hit_pipeline.py                 # Main pipeline script for model training and evaluation
├── create_hs_dataset.py            # Script to create hyperspectral image datasets from CSV descriptions
├── process_hsi_to_rgb.py           # Script to convert hyperspectral images to RGB format
├── process_labelme_to_mask.py      # Script to generate segmentation masks from LabelMe JSON annotations
├── spectral_visualization.py       # Script to analyze and visualize spectral data
├── dataset/
│   ├── data/                       # Contains raw .hsimage, JSON, converted RGB, and corresponding masks
│   │   ├── sample001.hsimage
│   │   ├── sample001.json
│   │   ├── sample001.png
│   │   └── sample001_mask.hsimage
│   ├── train_patched.npz           # Dataset suitable for CNN1D
│   ├── train_tiled.npz             # Dataset suitable for U-Net and CNN3D
│   ├── test.npz                    # Dataset created with 'raw' extraction mode for inference
│   ├── train.csv                   # Describes training samples and class distribution
│   └── test.csv                    # Describes test samples and class distribution
├── elements/                       # Custom modules and utilities for data handling and model operations
└── log/
    └── experiment/                 # Location for logs, TensorBoard, results, model, prediction_dict, and results
        ├── tensorboard/
        ├── results/
        ├── model.npz
        ├── prediction_dict.npz
        ├── results.txt
        └── pipeline_logs.txt
```
---
## Installation and Setup
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. **Create a Conda environment:**
   ```bash
    conda create --name hit python=3.11.9
   ```
3. **Activate the Conda environment:**
   ```bash
    conda activate hit
   ```
4. **Install dependencies:**
     ```bash
     pip install -r requirements.txt
     ```
5. **Environment Setup:**
   - Adjust the python interpreter and virtual environment directory
   - Adjust working directory path of _get_working_dir in `hit_pipeline.py` if necessary to align with your system configuration.

6. **Ready for Pipelines:**
   - You are now ready to choose your appropriate pipeline from the pipelines section below and run the usage code in the console.

---
## Pipelines
### Overview
The usage section is divided into several pipelines, each responsible for a specific task in the hyperspectral image processing workflow. Each pipeline has configurable parameters that can be adjusted according to your dataset and requirements. Below, detailed information about each pipeline, including descriptions and parameter configurations, is provided.

---
### 1. Processing Hyperspectral Images to RGB (process_hsi_to_rgb.py)
**Description:** This pipeline converts hyperspectral image files (.hsimage) to RGB format. The RGB images are used for annotation purposes in the LabelMe software.

**Usage:**
  ```bash
  python process_hsi_to_rgb.py
  ```
**Parameters:**
- directory_path (str): Path to the directory containing .hsimage files.
**Additional Steps:**
- Use the generated RGB files to annotate fabric and background in LabelMe software.
- Label fabric as 'fabric' and background as 'background,' using polygons for annotations.
- Copy the resulting JSON files to the dataset/data folder.
---
### 2. Generating Masks from LabelMe Annotations (process_labelme_to_mask.py)
**Description:** This pipeline generates segmentation masks from JSON annotation files created using LabelMe. The masks are used to collect tiles and patches in the create_hs_dataset.py pipeline from the segmented fabric and background and mark the remaining areas as the void class.

**Usage:**
  ```bash
  python process_labelme_to_mask.py
  ```

**Parameters:**
- directory_path (str): Path to the directory containing .hsimage files and JSON files.

**Additional Information:**
- Using this pipeline, masks are generated using the JSON files created by LabelMe earlier
- Ensure that JSON files are correctly formatted and correspond to the RGB images generated in the previous step.
---
### 3. Visualizing Spectral Data (spectral_visualization.py)
**Description:** This pipeline analyzes and visualizes spectral statistics of the hyperspectral datasets. It helps in understanding the spectral characteristics and distribution of the data.

**Usage:**
  ```bash
  python spectral_visualization.py
  ```
**Parameters:**
- directory_path (str): Path to the directory containing .hsimage files. 
- sample_list (list): List of samples to display the spectral
- patch_size (tuple): Size of the extracted random patch to aggregate the spectral

**Additional Information:**
- The generated plot will help identify the differences between spectral features of different samples
---
### 4. Creating Datasets (create_hs_dataset.py)
**Description:** This pipeline creates Hyperspectral datasets from hyperspectral images based on the descriptions provided in train.csv and test.csv. It supports different extraction modes such as 'smart_patching' suitable for CNN1D, 'smart_tiling' suitable for U-Net and CNN3D models, and also 'raw' extraction mode suitable for inferencing all architectures.

**Usage:**
  ```bash
  python create_hs_dataset.py
  ```
**Parameters:**
- csv_file: Path to the CSV file listing HSI images and metadata.
- hsimage_folder: Directory containing the HSI files referenced in the csv_file.
- dataset_path: Output path for the created dataset file.
- extraction_mode: Method of data extraction ('smart_tiling,' 'smart_patching,' or 'raw'). 'smart_patching' mode is suitable for CNN1D models, and 'smart_tiling' mode is suitable for U-Net and CNN3D models, and 'raw' extraction mode is suitable for inferencing all models.
- tile_size: Dimensions of the tiles or patches to be extracted (applicable in tiling and patching modes).
- stride: Overlap between consecutive tiles or patches (applicable in smart_tiling and smart_patching modes).
- use_aggregation: Whether to aggregate spectral data (applicable in smart_patching mode).
- preprocessing: List of preprocessing operations to apply to each image before extraction.

**Additional Information:**
- train.csv: Contains information about training samples and the distribution of classes. Ensure it is properly formatted with necessary columns such as sample_id, class_label, etc.
- test.csv: Contains similar information for test samples.


### 5. Running the Main Pipeline
**Description:** This is the core pipeline that handles the training, validating, and testing of deep learning models. It integrates all preprocessing steps and model operations to provide end-to-end functionality.

**Usage:**
- This script is intended to be run directly with Python after configuring paths and dependencies.
- Adjust the `ParameterConfig` class to switch between different operational modes (training, testing, visualization).
  
```bash
  python hit_pipeline.py
  ```

**Parameters:**
- Uses a `ParameterConfig` class to dynamically adjust settings based on the chosen model type,
  facilitating easy switches and experimentation with different model configurations.

**Class ParameterConfig Attributes:**

- model_type (str): Determines which type of model is initialized. Valid choices are `'cnn1d'`, `'cnn3d'`, or `'unet'`. Based on this choice, relevant parameters are automatically configured in the constructor.

- train_dataset_name (str): The filename (relative to the dataset directory) used for loading the training dataset. This is set to different .npz files based on the chosen `model_type`.

- test_dataset_name (str): The filename (relative to the dataset directory) used for loading the testing dataset. Defaults to `'test.npz'`.

- in_channel (int): The number of channels in the input data. For a 1D CNN, this is typically set to 1. For a 3D CNN or U-Net, it corresponds to the spectral dimension of the hyperspectral data (e.g., 224).

- start_filters (int): The number of filters (or feature maps) in the first convolutional layer for CNNs or the base number of feature maps for U-Net. This value can be smaller for 1D CNN models (e.g., 4) and larger for U-Nets (e.g., 48).

- cnn_conv_block_type (str): The convolutional block types based on different downsizing techniques. # Choices: 'A' for maxpool downsizing, 'B' for strided conv downsizing , 'C' for both.

- cnn_input_length (int or None): The length of the input vector for a 1D CNN (e.g., 224). If `model_type` is `cnn3d` or `unet`, this parameter is set to `None` because those models do not rely on a single spectral dimension length as 1D CNNs do.

- cnn_conv_layers (int or None): The number of convolutional layers in a CNN. Used for `cnn1d` and `cnn3d` models. Set to `None` if `model_type` is `unet`.

- cnn_fc_layers (int or None): The number of fully connected layers following the convolutional layers in CNN architectures. Set to `None` for `unet`.

- cnn_dropout_rate (float or None): The dropout rate for CNN architectures. If the `model_type` is `unet`, this is set to `None`. For CNNs, this rate is applied to fully connected layers or selected dropout layers within the network.

- unet_depth (int or None): The number of down-sampling/encoder stages in a U-Net model. For example, a value of 5 corresponds to 5 levels of encoders/decoders in the U-Net. For `cnn1d` and `cnn3d`, this is set to `None`.

- batch_size (int): The number of samples per batch of training. Adjust this based on GPU memory capacity and dataset size. For 1D CNNs, larger batch sizes (e.g., 1600) can be used; for 3D CNNs or U-Net, usually smaller batch sizes (e.g., 64 or 640) are used.

- num_workers (int): The number of subprocesses to use for data loading. A higher value can speed up data loading but also increase the load on the CPU.

- pin_memory (bool): If `True`, DataLoader will copy Tensors into CUDA pinned memory before returning them. It can improve performance when transferring data to the GPU.

- patience (int): The number of epochs with no improvement, after which the learning rate will be reduced (when using a scheduler in `torch.optim.lr_scheduler`).

- factor (float): Factor by which the learning rate will be reduced when patience is reached. New learning rate = `learning_rate * factor`.

- learning_rate (float): The initial learning rate used by the optimizer.

- ignore_indices (list of int): List of class indices to be ignored during loss calculation. Typically, this is used to ignore background or void classes.

- train_ratio (float): Specifies the split ratio between training and validation when performing a stratified split of the dataset (e.g., 0.7 for 70% training and 30% validation).

- check_stratification (bool): When `True`, the dataset splitting process will check class distributions to ensure a stratified split. If `False`, no distribution check is performed.

- num_epochs (int): The total number of training epochs.

- val_frequency (int): The frequency (in epochs) at which the model is validated. For example, a value of 1 means the model is validated every epoch.

- cascading_classifier (bool): Indicates if a cascading classifier approach is being used. This is relevant when data is preprocessed using a custom pipeline that first detects background vs. target materials.

- background_threshold (float): Threshold used to classify background when `cascading_classifier=True`.

- background_index (int): The label index representing the background class when `cascading_classifier=True`.

- inference_tile_height (int): Used during inference with 'unet' or 'cnn3d'. This indicates the tile or slice size along one dimension for patch-based inference on larger images.

- inference_mode (str): Determines the style of inference used for generating predictions using 1D CNN. It can be `'pixel-wise'` or `'patched'`. `'pixel-wise'` runs inference on every pixel. `'patched'` divides the input into patches and processes each patch.

- inference_patch_size (tuple of int): The patch size (height, width) used when `inference_mode='patched'`.

- inference_stride (int): The stride or overlap when sliding a patch window across the image in `patched` inference mode.

- pixel_batch_size (int): When `inference_mode='pixel-wise'`, this parameter defines how many pixels are processed in each mini-batch.

- device (str): The device on which the model will run. Automatically set to `'cuda'` if a GPU is available; otherwise `'cpu'`.

- writer (torch.utils.tensorboard.SummaryWriter): A TensorBoard writer object for logging and visualizing metrics during training. Created in the constructor and stored for global usage.

- do_train (bool): Flag to determine whether the training process should be executed.

- do_test (bool): Flag to determine whether the testing (inference) process should be executed.

- display_mapping (bool): Flag to determine whether the output mapping (e.g., segmentation maps) should be visualized after testing.

- experiment (str): The experiment name used to save logging, models and the results. By default, set to `'experiment'`.

**Logging:**
- Integrates logging throughout the pipeline to provide insights into model training and evaluation processes. Pipeline logs are saved in the log/experiment folder.

**The pipeline provides capabilities for:**
1. Loading and splitting hyperspectral image datasets.
2. Configuring and initializing models with specific parameters tailored to the type of neural network.
3. Training models with tensorboard visualization of loss metrics and model performance.
4. Validating and saving the best-performing model snapshots based on the lowest validation loss.
5. Testing models with loaded datasets to evaluate performance and generate predictions.
6. Visualizing output mappings from the testing phase and saving results.

---

## Additional Information
### Logging and Monitoring
-   All logs, including TensorBoard logs, model weights, prediction dictionaries, and results, are stored in the `log/experiment/` directory.


### Custom Modules
-   The `elements/` directory contains custom modules and utilities that handle data preprocessing, model definitions, and other operations.
-   Ensure that any custom changes in these modules are reflected in the main pipeline configurations.

### CSV Files

-   **train.csv:**
    
    -   Contains information about the training samples.
    -   Includes columns such as `sample_id`, `class_label`, `file_path`, and other relevant metadata.
    -   Provides details on the distribution of classes within the training dataset to ensure balanced training.
-   **test.csv:**
    
    -   Contains information about the test samples.
    -   Includes similar columns as `train.csv` for consistency.
    -   Ensures that the test dataset is well-represented for accurate evaluation of model performance.



---
## Acknowledgments

-   We would like to express gratitude to our partner organizations for their valuable support and contributions to this research. We thank Sympany, CuRe Circular Textile B.V., Arapaha, House of Design, Batenburg Beenen, de Tijdelijke Expert, Havep, Modint, and NRK for providing essential resources, and industry insights that significantly enhanced the relevance and impact of this work.
---
## Author

**Milad Isakhani Zakaria**  
_Supervisors:_ **Ben Wolf**, **Klaas Dijkstra**

