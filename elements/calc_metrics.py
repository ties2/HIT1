import numpy as np
import os
NoneType = type(None)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# configure logging
from elements.utils import LoggerSingleton
logger = LoggerSingleton.get_logger()

def calculate_metrics(results_dict,saved_dir):
    """
    calculate metrics for each file in the prediction dictionary.

    :param results_dict: dictionary containing predictions, labels, and metadata
    :param saved_dir: directory to save the predictions

    :return: updated dictionary with calculated metrics
    """
    class_names = results_dict['class_names']
    num_classes = len(class_names)
    for file_name, data in results_dict['data'].items():
        output_image = results_dict['output_image_mapping'][file_name]
        label_image = results_dict['label_image_mapping'][file_name]

        # exclude background from output
        background_vector = np.zeros(num_classes, dtype=int)
        background_vector[1] = 1
        void_vector = np.zeros(num_classes, dtype=int)
        void_vector[0] = 1

        background_mask = np.all(label_image == background_vector, axis=-1)
        void_mask = np.all(label_image == void_vector, axis=-1)
        both_mask = void_mask + background_mask
        no_background_void_mask =~ both_mask
        data['predicted_composition'] = np.mean(output_image[no_background_void_mask], axis=0)
        data['labels_composition'] = np.median(label_image[no_background_void_mask], axis=0)

        cosine_sim = cosine_similarity([data['labels_composition']], [data['predicted_composition']])[0][0]
        mse = mean_squared_error(data['labels_composition'], data['predicted_composition'])
        mae = mean_absolute_error(data['labels_composition'], data['predicted_composition'])

        # add metrics to the dictionary
        data['cosine_sim'] = cosine_sim
        data['mse'] = mse
        data['mae'] = mae

    # display metrics for each file
    logger.info(f"{'file name':<15} {'cosine similarity':<20} {'mae':<10} {'mse':<10}")
    logger.info("-" * 60)
    for file_name, data in results_dict['data'].items():
        logger.info(f"{file_name:<15} {data['cosine_sim']:<20.4f} {data['mae']:<10.4f} {data['mse']:<10.4f}")

    # display the average predicted composition for each sample (file name)
    class_names = results_dict['class_names']
    logger.info("\naverage predicted and true composition per sample:")
    header = f"{'file name':<20} {' '.join([f'{name:<15}' for name in class_names])}"
    logger.info(header)
    logger.info("-" * len(header))

    for file_name, data in results_dict['data'].items():
        # predicted composition
        avg_comp_pred = data['predicted_composition']
        comp_str_pred = " ".join([f"{value * 100:.1f}%".ljust(15) for value in avg_comp_pred])
        logger.info(f"{file_name}-pred".ljust(20) + comp_str_pred)

        # true composition
        avg_comp_true = data['labels_composition']  # Assuming labels_composition is the true composition
        comp_str_true = " ".join([f"{value * 100:.1f}%".ljust(15) for value in avg_comp_true])
        logger.info(f"{file_name}-true".ljust(20) + comp_str_true)


    # write results
    results_file_path = os.path.join(saved_dir,'log','experiment', 'results.txt')
    with open(results_file_path, 'w') as f:
        f.write(f"{'file name':<15} {'cosine similarity':<20} {'mae':<10} {'mse':<10}\n")
        f.write("-" * 60 + "\n")
        for file_name, data in results_dict['data'].items():
            f.write(f"{file_name:<15} {data['cosine_sim']:<20.4f} {data['mae']:<10.4f} {data['mse']:<10.4f}\n")

        f.write("\naverage predicted and true composition per sample:\n")
        header = f"{'file name':<20} {' '.join([f'{name:<15}' for name in results_dict['class_names']])}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for file_name, data in results_dict['data'].items():
            avg_comp_pred = data['predicted_composition']
            comp_str_pred = " ".join([f"{value * 100:.1f}%".ljust(15) for value in avg_comp_pred])
            f.write(f"{file_name}-pred".ljust(20) + comp_str_pred + "\n")

            avg_comp_true = data['labels_composition']
            comp_str_true = " ".join([f"{value * 100:.1f}%".ljust(15) for value in avg_comp_true])
            f.write(f"{file_name}-true".ljust(20) + comp_str_true + "\n")

    return results_dict


