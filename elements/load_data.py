import torch.utils
import torch.utils.data
import os
import numpy as np
import pandas as pd
import cv2
import torch

from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from elements.common.data.datatypes.scandata import ScanData
from elements.preprocess import apply_flatfield_correction
from elements.visualize import visualize_extracted_tiles

# configure logging
from elements.utils import LoggerSingleton
logger = LoggerSingleton.get_logger()

class HSITileExtractor:
    """
    A class to extract tiles from a binary mask using 'smart_tiling' or 'smart_patching' or 'raw'.
    The class validates parameters, provides optimized region extraction methods, and supports
    settings for stride and tile size.
    """

    def __init__(self, mask, tile_size, extraction_mode, stride):
        """
        Initialize the extractor with necessary parameters.

        :param mask: Mask image with void (0,0,0), background (255,0,0), and fabric (255,255,255) regions
        :param tile_size: Tuple specifying (tile_width, tile_height)
        :param extraction_mode: 'smart_tiling' or 'smart_patching'
        :param stride: Stride for the grid search method
        """
        self.mask = mask
        self.tile_width, self.tile_height = tile_size
        self.mask_height, self.mask_width = mask.shape[:2]
        self.extraction_mode = extraction_mode
        self.stride = stride

        self._validate_parameters()

    def _validate_parameters(self):
        """Validates the initialization parameters."""
        valid_modes = ['smart_tiling', 'smart_patching']
        if self.extraction_mode not in valid_modes:
            raise ValueError(f"Invalid extraction_mode: {self.extraction_mode}. Must be one of: {valid_modes}.")

        if self.stride <= 0:
            raise ValueError(f"Stride must be a positive integer. Received: {self.stride}")

    def extract_tiles(self):
        """Main method to extract regions based on the selected extraction_mode."""
        if self.extraction_mode == 'smart_tiling':
            return self._extract_tiles_smart_tiling()
        elif self.extraction_mode == 'smart_patching':
            return self._extract_tiles_smart_patching()

    def _extract_tiles_smart_tiling(self):
        """
        Perform a grid search over the entire image from (0,0), discarding tiles that are fully void.
        If any part of the tile is non-void, the tile and coordinates are returned.

        :return: List of coordinates for extracted tiles (x1, y1, x2, y2)
        """
        tiles = []
        for y in range(0, self.mask_height - self.tile_height + 1, self.stride):
            for x in range(0, self.mask_width - self.tile_width + 1, self.stride):
                tile = self.mask[y:y + self.tile_height, x:x + self.tile_width]

                # discard tile if it is completely void
                if np.all(tile == [0, 0, 0]):
                    continue

                # return coordinates in (x1, y1, x2, y2) format
                tiles.append((x, y, x + self.tile_width, y + self.tile_height))

        return tiles

    def _extract_tiles_smart_patching(self):
        """
        Start grid search from the first non-void pixel; extract tiles covering only fabric or background
        regions, with no void areas present.

        :return: List of coordinates for extracted tiles (x1, y1, x2, y2)
        """
        tiles = []
        start_y, start_x = np.where(np.any(self.mask != [0, 0, 0], axis=-1))[0][0], 0

        # start grid search over non-void areas
        for y in range(start_y, self.mask_height - self.tile_height + 1, self.stride):
            for x in range(start_x, self.mask_width - self.tile_width + 1, self.stride):
                tile = self.mask[y:y + self.tile_height, x:x + self.tile_width]

                # skip tiles that contain any void pixels, but allow background ([255, 0, 0]) or fabric ([255, 255, 255])
                if np.any((tile == [0, 0, 0]).all(axis=-1)):
                    continue

                # ensure tile is either entirely fabric or entirely background
                if np.all(tile == [255, 255, 255]) or np.all(tile == [255, 0, 0]):
                    tiles.append((x, y, x + self.tile_width, y + self.tile_height))

        return tiles

def stratified_split_by_composition(dataset, train_ratio, model_type,check_stratification=False):
    """
    Perform a stratified split on the dataset based on unique composition labels.

    :param dataset: Instance of HSITileDataset
    :param train_ratio: Ratio of the dataset to allocate to training (default 0.8)
    :param model_type: value indicating if the dataset is for U-Net (with 4D labels) or Cnn (with 2D labels)
    :param check_stratification: Whether to check the results of the stratified split
    :return: train_dataset, val_dataset
    """
    file_name_hashes = [hash(f) for f in dataset.file_names]
    file_name_tensor = torch.tensor(file_name_hashes).view(-1, 1).float()

    # process labels based on the model structure
    if model_type == 'unet' or model_type == 'cnn3d':
        # for unet dataset, reshape labels to (samples, composition, pixels) and find the mode composition
        num_samples, num_classes, width, height = dataset.labels.shape
        labels_reshaped = dataset.labels.view(num_samples, num_classes, -1)

        # find the most repeated composition for each tile
        labels_reduced = []
        for i in range(num_samples):
            unique_compositions, counts = torch.unique(labels_reshaped[i].T, dim=0, return_counts=True)
            most_common_composition = unique_compositions[counts.argmax()]
            labels_reduced.append(most_common_composition)
        labels_reduced = torch.stack(labels_reduced)
    else:
        labels_reduced = dataset.labels

    # combine file names and labels for unique composition checking
    combined = torch.cat((file_name_tensor, labels_reduced), dim=1)
    unique_combinations, unique_ids = torch.unique(combined, dim=0, return_inverse=True)

    data_df = pd.DataFrame({
        'tile_index': range(len(dataset)),
        'composition_label': unique_ids
    })

    # split based on composition label
    train_indices, val_indices = train_test_split(
        data_df['tile_index'].values,
        stratify=data_df['composition_label'],
        test_size=1 - train_ratio,
        random_state=42,
    )

    train_indices, val_indices = list(train_indices), list(val_indices)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    if check_stratification:
        check_stratified_split(train_dataset, val_dataset, dataset.file_names, unique_ids)

    return train_dataset, val_dataset

def check_stratified_split(train_dataset, val_dataset, file_names, unique_ids):
    """
    Count and print the number of samples for each unique label per file name in both train and validation sets,
    along with the percentage of samples in the validation set.

    :param train_dataset: Subset of HSITileDataset for the training set
    :param val_dataset: Subset of HSITileDataset for the validation set
    :param file_names: Original list of file names from the complete dataset
    :param unique_ids: Unique IDs for each label combination
    """

    def count_samples(dataset):
        """
        Helper function to count samples for each unique label in a dataset subset.
        """
        label_counts = defaultdict(lambda: defaultdict(int))

        for idx in dataset.indices:
            file_name = file_names[idx]
            label_id = unique_ids[idx].item()  # Retrieve unique ID for the label
            label_counts[file_name][label_id] += 1  # Increment count for this label within the file name

        return label_counts

    # get sample counts for train and validation sets
    train_label_counts = count_samples(train_dataset)
    val_label_counts = count_samples(val_dataset)

    # print results side-by-side with percentage calculation
    print("Sample counts per unique label per file name (Train vs Validation):\n")
    for file_name in set(train_label_counts.keys()).union(val_label_counts.keys()):
        print(f"File Name: {file_name}")
        unique_labels = set(train_label_counts[file_name].keys()).union(val_label_counts[file_name].keys())

        for label_id in unique_labels:
            train_count = train_label_counts[file_name].get(label_id, 0)
            val_count = val_label_counts[file_name].get(label_id, 0)
            total_count = train_count + val_count
            val_percentage = (val_count / total_count * 100) if total_count > 0 else 0
            print(
                f"  Label ID: {label_id}, Train Count: {train_count}, Validation Count: {val_count}, Validation %: {val_percentage:.2f}%")
        print()

class HSITileDataset(Dataset):
    def __init__(self, tiles, labels, coords, file_names, rgb_images,masks, tiled_images, class_names, training_mode=False, preprocessing=None):
        """
        Custom dataset for HSI tile data with optional preprocessing.

        :param tiles: Tensor containing tile data (num_tiles, channels, tile_size, tile_size)
        :param labels: Tensor containing labels for each tile (num_tiles, num_classes)
        :param coords: Tensor containing tile coordinates (num_tiles, 4)
        :param file_names: Array containing file names for each tile (num_tiles,)
        :param rgb_images: Dictionary mapping file names to their corresponding RGB image arrays
        :param masks: Dictionary mapping file names to their corresponding mask arrays
        :param tiled_images: Dictionary mapping file names to their corresponding tiled images
        :param class_names: List of class names corresponding to label compositions
        :param training_mode: Boolean indicating if the dataset is used for training; if True, skips returning rgb/tiled images
        :param preprocessing: Optional preprocessing transformations to apply to the tiles
        """
        if preprocessing:
            print("Applying preprocessing during dataset initialization...")
            tiles = torch.stack([torch.tensor(preprocessing(image=tile.numpy())['image']) for tile in tiles])

        self.tiles = tiles
        self.labels = labels
        self.coords = coords
        self.file_names = file_names
        self.rgb_images = rgb_images
        self.tiled_images = tiled_images
        self.masks = masks
        self.class_names = class_names
        self.training_mode = training_mode

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        """
        Retrieve the tile, label, and coordinates for a given index, optionally excluding RGB and tiled images.

        :param idx: Index of the tile to retrieve
        :return: Tuple containing (tile, label, coordinates) if training_mode=True,
                 otherwise (tile, label, coordinates, file_name, RGB image, tiled image)
        """
        tile = self.tiles[idx]
        label = self.labels[idx]

        if self.training_mode:
            return tile, label
        else:
            coord = self.coords[idx]
            file_name = self.file_names[idx]

            return tile, label, coord, file_name

    def get_rgb_images(self):
        return self.rgb_images

    def get_tiled_images(self):
        return self.tiled_images

    def get_class_names(self):
        return self.class_names

    def get_mask(self):
        return self.masks

def create_hsi_dataset_from_csv(csv_file: str, hsimage_folder: str, dataset_path: str, extraction_mode: str = 'raw',
                                tile_size: tuple = (64, 64), stride: int = 64, use_aggregation: bool = False, preprocessing: list = None):
    """
    Creates and saves an HSI dataset from annotation masks, either using extraction modes of 'smart_tiling', 'smart_patching' or 'raw'.

    Prerequisites:
      - Each `.hsimage` file should have an associated `.png` RGB image and `_mask.png` segmentation mask
        in the same directory, with matching filenames.
      - Mask image must have void (0,0,0), background (255,0,0), and fabric (255,255,255) regions

    Parameters:
        csv_file (str): Path to a CSV file with filenames and class compositions.
        hsimage_folder (str): Directory containing `.hsimage` files, `.png` RGB images, and `_mask.png` segmentation masks.
        dataset_path (str): Path where the processed dataset will be saved.
        extraction_mode (str): Mode for tile extraction (e.g., 'smart_tiling') or 'raw' for non-tiled.
        tile_size (tuple): Size of each tile (height, width) when tiling is used.
        stride (int): Stride for tile extraction, applicable in tiling mode.
        use_aggregation (bool): If True, aggregates patches for CNN1D models.
        preprocessing (list): List of preprocessing transformations to apply to the hyperspectral image.

    Returns:
        HSITileDataset: The processed dataset with HSI tiles or raw samples, labels, and coordinates.
    """
    df = pd.read_csv(csv_file)
    tiles_list, coords_list, labels_list, file_names_list = [], [], [], []
    rgb_images_mapping = {}
    tiled_images_mapping = {}
    rgb_images_mask = {}
    class_names = df.columns[1:].tolist()
    num_classes = len(class_names)
    scan_data = ScanData()

    for _, row in df.iterrows():
        file_name = row['file_name']
        class_composition = row[class_names].values.astype(np.float32)

        if extraction_mode != 'raw':
            print(f"Processing: {file_name} (extraction_mode={extraction_mode}, tile_size={tile_size}, stride={stride}, aggregation={use_aggregation})")
        else:
            print(f"Processing: {file_name} (extraction_mode={extraction_mode}, aggregation={use_aggregation})")

        # load and apply flat field correction
        scan_data.load(os.path.join(hsimage_folder, f'{file_name}.hsimage'))
        ffc_image = apply_flatfield_correction(scan_data.get_raw(), scan_data.get_whiteref(), scan_data.get_darkref())

        # apply preprocessing steps
        if preprocessing:
            for transform in preprocessing:
                # ensure ffc_image shape matches expected input for transformations
                ffc_tensor = torch.tensor(ffc_image, dtype=torch.float32).permute(2, 0, 1)  # shape (C, H, W)
                try:
                    ffc_tensor = transform(ffc_tensor)
                except Exception as e:
                    print(f"Direct transformation failed: {e}. Trying named argument approach.")
                    ffc_tensor = transform(image=ffc_tensor.numpy())['image']
                    ffc_tensor = torch.tensor(ffc_tensor, dtype=torch.float32)
                ffc_image = ffc_tensor.permute(1, 2, 0).numpy()  # back to (H, W, C)

        # load RGB image and mask
        rgb_image = cv2.imread(os.path.join(hsimage_folder, f'{file_name}.png'), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(hsimage_folder, f'{file_name}_mask.png'), cv2.IMREAD_COLOR)

        # crop the railing edge from width, crop the height by 64 multiplier
        crop_height = (ffc_image.shape[0] // 64) * 64
        ffc_image_cropped = ffc_image[:crop_height, :-64]
        rgb_image_cropped = rgb_image[:crop_height, :-64]
        mask_cropped = mask[:crop_height, :-64]

        # create label matrix
        label_matrix = np.zeros((mask_cropped.shape[0], mask_cropped.shape[1], num_classes), dtype=np.float32)
        label_matrix[(mask_cropped == [0, 0, 0]).all(axis=-1), 0] = 1  # mark black as void
        label_matrix[(mask_cropped == [255, 0, 0]).all(axis=-1), 1] = 1  # mark red as background
        label_matrix[(mask_cropped == [255, 255, 255]).all(axis=-1)] = class_composition  # mark white as fabric

        # tiling (used for training) or raw (used for inference) extraction mode
        if extraction_mode != 'raw':
            tile_extractor = HSITileExtractor(mask_cropped, tile_size, extraction_mode, stride)
            tile_coords = tile_extractor.extract_tiles()
            tiles_ffc = [ffc_image_cropped[c[1]:c[3], c[0]:c[2]] for c in tile_coords]
            tiles_labels = [label_matrix[c[1]:c[3], c[0]:c[2]] for c in tile_coords]
            tiled_image = visualize_extracted_tiles(rgb_image_cropped, tile_coords, file_name, extraction_mode)

            if use_aggregation:
                # specific structure for 1D cnn ==> shape is (bc,c,vector_length) (bc,1,224) spectral is transformed into input vector
                tiles_ffc = [np.mean(tile, axis=(0, 1)) for tile in tiles_ffc]
                tiles_labels = [np.mean(tile, axis=(0, 1)) for tile in tiles_labels]
                tiles_tensor = torch.tensor(np.array(tiles_ffc), dtype=torch.float32).unsqueeze(1)
                labels_tensor = torch.tensor(np.array(tiles_labels), dtype=torch.float32)
            else:
                # retain spatial structure for UNet shape is (bs,c,h,w)
                tiles_tensor = torch.tensor(np.array(tiles_ffc), dtype=torch.float32).permute(0, 3, 1, 2)
                labels_tensor = torch.tensor(np.array(tiles_labels), dtype=torch.float32).permute(0, 3, 1, 2)

            coords_tensor = torch.tensor(np.array(tile_coords), dtype=torch.int32)
            tiles_list.append(tiles_tensor)
            labels_list.append(labels_tensor)
            coords_list.append(coords_tensor)
            file_names_list.extend([file_name] * len(tiles_tensor))

        else:
            # raw mode (no tiling) used for inference
            tile_coords = [[0, 0, ffc_image_cropped.shape[1], ffc_image_cropped.shape[0]]]
            tiled_image = visualize_extracted_tiles(rgb_image_cropped, tile_coords, file_name, 'raw')

            ffc_tensor = torch.tensor(ffc_image_cropped, dtype=torch.float32).permute(2, 0, 1)
            label_tensor = torch.tensor(label_matrix, dtype=torch.float32).permute(2, 0, 1)
            coord_tensor = torch.tensor(tile_coords[0], dtype=torch.int32)

            tiles_list.append(ffc_tensor)
            labels_list.append(label_tensor)
            coords_list.append(coord_tensor)
            file_names_list.append(file_name)

        # store tiled and original rgb images
        rgb_images_mapping[file_name] = rgb_image_cropped
        tiled_images_mapping[file_name] = tiled_image
        rgb_images_mask[file_name] = mask_cropped

    # concatenate lists into tensors (used list to avoid memory issues)
    if extraction_mode != 'raw':
        tiles_tensor = torch.cat(tiles_list, dim=0)
        labels_tensor = torch.cat(labels_list, dim=0)
        coords_tensor = torch.cat(coords_list, dim=0)

    # used lists instead of torch.stack to include images from multiple sizes
    else:
        tiles_tensor = tiles_list
        labels_tensor = labels_list
        coords_tensor = coords_list

    file_names_array = np.array(file_names_list)

    # create and save the dataset
    dataset = HSITileDataset(tiles_tensor, labels_tensor, coords_tensor, file_names_array,
                             rgb_images_mapping,rgb_images_mask, tiled_images_mapping, class_names,training_mode=False if extraction_mode == 'raw' else True)
    torch.save(dataset, dataset_path)
    print(f"HSITileDataset saved to {dataset_path} with {len(tiles_tensor)} samples.")

    return dataset

def load_hsi_dataset(dataset_path):
    """
    Load an HSI tiled dataset and apply preprocessing.

    :param dataset_path: Path to the saved dataset (.pt file)
    :return: An instance of HSITileDataset
    """
    dataset = torch.load(dataset_path,weights_only=False)
    if not isinstance(dataset, HSITileDataset):
        raise TypeError(f"The loaded object is not of type 'HSITileDataset', but {type(dataset)}")
    logger.info(f"HSITileDataset loaded from {dataset_path} with {len(dataset)} samples.")
    return dataset


def load_prediction_dict(path):
    """
    Load the prediction dictionary from the specified file path.

    :param path: The file path from which to load the dictionary.
    :return: The loaded prediction dictionary.
    """
    try:
        prediction_dict = torch.load(path,weights_only=False)
        logger.info(f"Prediction cube loaded from {path}")
        return prediction_dict
    except Exception as e:
        logger.error(f"Failed to load prediction dictionary from {path}: {e}")
        raise
