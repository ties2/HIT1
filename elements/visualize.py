import numpy as np
NoneType = type(None)
from elements.common.visualize import launch_tb
import os
from torch.utils.tensorboard import SummaryWriter
import cv2
import matplotlib.pyplot as plt
from matplotlib.table import Table


def plot_spectral_data(ffc_data, title):
    """
    plot flat-field correction data across wavelengths for one or more samples.

    :param ffc_data: ffc data array with shape (num_samples, 1, 224) or (224,)
    :param title: title for the plot
    """
    wavelengths = np.linspace(935.61, 1720.23, 224)
    plt.figure(figsize=(10, 6))

    # plot multiple samples or a single sample
    if ffc_data.ndim == 3:
        for i, sample in enumerate(ffc_data[:, 0]):
            plt.plot(wavelengths, sample, label=f'patch {i}')
        plt.legend()
    else:
        plt.plot(wavelengths, ffc_data)

    plt.title(title)
    plt.xlabel('wavelength (nm)')
    plt.ylabel('ffc reflectance')
    plt.xlim(950, 1700)
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.show()

def visualize_image_and_mask(image, mask, title):
    """
    Visualize the original image and its corresponding mask side by side.

    Args:
        image (numpy.ndarray): The original image loaded using OpenCV.
        mask (numpy.ndarray): The generated mask image.
        title (str): A title for the visualization, typically the base name of the processed file.
    """
    plt.figure(figsize=(12, 6))

    # display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Original Image: {title}')
    plt.axis('off')

    # display the mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title(f'Mask: {title}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def create_tb(experiment_dir: str, tb_name: str = "tensorboard", delete_previous: bool = True, start_tb: bool = True) -> SummaryWriter:
    """
    Create a TensorBoard SummaryWriter object, saving logs in the specified experiment directory.

    :param experiment_dir: Directory where TensorBoard logs will be saved.
    :param tb_name: Subdirectory for this specific TensorBoard run.
    :param delete_previous: Should the previous writer with the same name be deleted or should this value be appended?
    :param start_tb: Should a TensorBoard be started with this SummaryWriter.
    :return: The SummaryWriter object

    :example:

    True
    """
    # full path for TensorBoard logs
    tb_path = os.path.join(experiment_dir, tb_name)

    # optionally delete the previous log directory
    if delete_previous and os.path.exists(tb_path):
        import shutil
        shutil.rmtree(tb_path)

    # create the SummaryWriter at the specified path
    writer = SummaryWriter(log_dir=tb_path)

    # optionally start TensorBoard
    if start_tb:
        launch_tb(tb_path)

    return writer

def show_loss_tb(loss, epoch, writer: SummaryWriter, name: str = "loss"):
    """
    Show loss on a TensorBoard.

    :param loss: Current loss
    :param epoch: Tensorboard step
    :param name: Name of the metric
    :param writer: Writer to use

    """
    writer.add_scalar(name, loss, epoch)
    writer.flush()

def display_output_mapping(output_dict,output_dir):
    """
    Display the segmentation output for each file in the test dataset using specified classes for RGB channels.

    :param output_dict: dictionary containing predictions, labels, and output mappings
    :param output_dir: Directory where output will be saved
    """
    class_names = output_dict['class_names']
    file_names = output_dict['file_names']

    for file_name in file_names:
        all_true = output_dict[file_name]['labels_composition']
        top_indexes = np.argsort(all_true)[::-1]  # Sort indices by descending values
        red_idx = top_indexes[0]
        green_idx = top_indexes[1]
        blue_idx = top_indexes[2]

        converted_rgb_image = output_dict[file_name]['input_image_mapping']
        predicted_segmentation_mask = (output_dict[file_name]['output_image_mapping'][:, :, [red_idx, green_idx, blue_idx]]*255).astype(np.uint8)
        ground_truth_mask = output_dict[file_name]['input_image_mask']

        fabrics = [class_names[red_idx], class_names[green_idx], class_names[blue_idx]]
        true = [all_true[red_idx], all_true[green_idx], all_true[blue_idx]]

        # assign true values to fabrics on the mask for visualization purposes
        red_mask = (ground_truth_mask[:, :, 0] == 255) & (ground_truth_mask[:, :, 1] == 0) & (ground_truth_mask[:, :, 2] == 0)
        ground_truth_mask[red_mask] = [0, 0, 0]

        # set (255, 255, 255) pixels to scaled true values
        white_mask = (ground_truth_mask[:, :, 0] == 255) & (ground_truth_mask[:, :, 1] == 255) & (ground_truth_mask[:, :, 2] == 255)
        true_scaled = (np.array([all_true[red_idx], all_true[green_idx], all_true[blue_idx]]) * 255).astype(np.int32)
        ground_truth_mask[white_mask] = true_scaled

        all_pred = output_dict[file_name]['predicted_composition']
        pred = np.round([all_pred[red_idx], all_pred[green_idx], all_pred[blue_idx]],2)
        red_info = (fabrics[0],true[0],pred[0])
        green_info = (fabrics[1],true[1],pred[1])
        blue_info = (fabrics[2],true[2],pred[2])
        fig = visualize_segmentation_results(converted_rgb_image=converted_rgb_image,ground_truth_segmentation=ground_truth_mask,
                                     predicted_segmentation=predicted_segmentation_mask,main_title=f'{file_name}',
                                     red_info=red_info,green_info=green_info,blue_info=blue_info)
        # Save the plot
        save_path = os.path.join(output_dir, f"{file_name}_results.png")
        fig.savefig(save_path)
        plt.close(fig)  # Close the plot to free up memory


def create_output_mapping(results_dict):
    """
    create segmentation output for each rgb image in the test dataset based on model predictions.

    :param results_dict: dictionary containing predictions, labels, and coordinates
    :return: updated prediction dictionary with segmentation output mappings
    """
    input_image_mapping = results_dict['input_image_mapping']
    class_names = results_dict['class_names']
    num_classes = len(class_names)

    for file_name in results_dict['data'].keys():
        rgb_img = input_image_mapping[file_name]
        height, width, _ = rgb_img.shape
        results_dict['output_image_mapping'][file_name] = np.zeros((height, width, num_classes), dtype=np.float32)
        results_dict['label_image_mapping'][file_name] = np.zeros((height, width, num_classes), dtype=np.float32)

    # fill the output mapping based on predictions and coordinates
    for file_name, data in results_dict['data'].items():
        for i, coords in enumerate(data['coords']):
            x1, y1, x2, y2 = coords
            results_dict['output_image_mapping'][file_name][y1:y2, x1:x2, :] += data['preds'][i]
            results_dict['label_image_mapping'][file_name][y1:y2, x1:x2, :] += data['labels'][i]


        # fill all [zeros] with the background vector [1, 0, ..., 0]
        background_vector = np.zeros(num_classes, dtype=int)
        background_vector[1] = 1
        zero_mask = np.all(results_dict['output_image_mapping'][file_name] == 0, axis=-1)
        results_dict['output_image_mapping'][file_name][zero_mask] = background_vector
        results_dict['label_image_mapping'][file_name][zero_mask] = background_vector

    return results_dict

def visualize_segmentation_results(converted_rgb_image, ground_truth_segmentation, predicted_segmentation, main_title, red_info, green_info, blue_info):
    """
    Visualize the converted RGB image, ground truth segmentation, and predicted segmentation side by side
    with a main title, image titles, and a structured table legend showing RGB fabric names,
    true labels, and predictions. Also returns the matplotlib figure object for further use.

    :param converted_rgb_image: The RGB-converted HSI image to display.
    :param ground_truth_segmentation: The ground truth segmentation to display.
    :param predicted_segmentation: The predicted segmentation to display.
    :param main_title: Main title displayed above the images.
    :param red_info: Tuple containing (fabric name, true label, prediction) for the red channel.
    :param green_info: Tuple containing (fabric name, true label, prediction) for the green channel.
    :param blue_info: Tuple containing (fabric name, true label, prediction) for the blue channel.

    :return: matplotlib figure object
    """
    # create subplots to visualize the input, true mask, and output image
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(main_title, fontsize=16)  # Main title

    # display the converted RGB image
    axes[0].imshow(converted_rgb_image)
    axes[0].set_title('Converted RGB Image')
    axes[0].axis('off')

    # display the ground truth segmentation
    axes[1].imshow(ground_truth_segmentation)
    axes[1].set_title('Ground Truth Segmentation')
    axes[1].axis('off')

    # display the predicted segmentation
    axes[2].imshow(predicted_segmentation)
    axes[2].set_title('Predicted Segmentation')
    axes[2].axis('off')

    # create table data for the legend
    cell_text = [
        ['    ', 'Red: ' + red_info[0], 'Green: ' + green_info[0], 'Blue: ' + blue_info[0]],
        ['True', f"{red_info[1]:.2f}", f"{green_info[1]:.2f}", f"{blue_info[1]:.2f}"],
        ['Pred', f"{red_info[2]:.2f}", f"{green_info[2]:.2f}", f"{blue_info[2]:.2f}"]
    ]

    # create a table for the RGB legend
    ax_table = fig.add_subplot(111)
    ax_table.axis('off')  # Hide axes for the table
    table = Table(ax_table, bbox=[0.1, -0.3, 0.8, 0.2])  # Define the table's position (bbox)

    # define uniform column width and row height
    col_width = 0.05
    row_height = 0.05

    # add each cell to the table
    for i, row in enumerate(cell_text):
        for j, cell in enumerate(row):
            table.add_cell(i, j, col_width, row_height, text=cell, loc='center', facecolor='white')

    # set font size for the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    ax_table.add_table(table)

    plt.tight_layout()
    plt.show()

    return fig

def visualize_extracted_tiles(image, coords, file_name, tile_extraction_mode):
    """
    visualize the extracted patches on a copy of the input image with a dynamically generated title.
    :param image: input fabric image
    :param coords: list of coordinates of patches
    :param file_name: name of the image file for logging
    :param tile_extraction_mode: 'fabric_grid' or 'fabric_random' search method used to extract patches
    """
    tiled_image = image.copy()

    for i, (x1, y1, x2, y2) in enumerate(coords):
        cv2.rectangle(tiled_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # display the image with the patches
    plt.figure(figsize=(10, 10))
    plt.imshow(tiled_image)
    plt.title(f'{len(coords)} tiles extracted using {tile_extraction_mode} from {file_name}')
    plt.axis('off')
    plt.show()
    return tiled_image

