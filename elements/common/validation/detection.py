from typing import Iterable, Dict, Callable, List, Sized, Union, Tuple
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.elements.validation.geometry import get_area_and_overlap
from common.elements.validation.lamdas import MetricLambda, MatchLambda


def calc_confusion_metrics(result_boxes_per_image: Dict[str, Union[Sized, Iterable]], target_boxes_per_image: Dict[str, Union[Sized, Iterable]], num_classes: int, match_func: Callable[[int, int, int], float]=MatchLambda.IoU, match_t=0.5, prob_t=0.5) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Efficient and generic function to calculate metrics. It calculates the metric over all images and over all classes at one point of match_t and prob_t.

    :param result_boxes_per_image: Boxes per image returned by the model, with keys as image identifiers.
        For example: {'img_0.png': [[box1_y1, box1_x1, box1_y2, box1_x2, class_id, probability], ... , [boxn_y1, boxn_x1, boxn_y2, boxn_x2, class_id, probability]]}
    :param target_boxes_per_image: Boxes per image for the ground truth, with keys as image identifiers.
        For example: {'img_0.png': [[box1_y1, box1_x1, box1_y2, box1_x2, class_id], ... , [boxn_y1, boxn_x1, boxn_y2, boxn_x2, class_id]]}
    :param num_classes: Number of classes the model is trained on.
    :param match_t: The overlap threshold above which to accept boxes as a match
    :param prob_t: The probability threshold above which to accept boxes for matching
    :return: Dictionary with class_id as key and List of metric values as value
    """

    # Calculate the metric
    tp = dict.fromkeys(range(num_classes), 0)
    fp = dict.fromkeys(range(num_classes), 0)
    fn = dict.fromkeys(range(num_classes), 0)
    tn = dict.fromkeys(range(num_classes), 0)

    image_ids = set(target_boxes_per_image.keys()).union(set(result_boxes_per_image.keys()))
    for image_id in image_ids:
        result_boxes = result_boxes_per_image[image_id] if image_id in result_boxes_per_image.keys() else []
        target_boxes = target_boxes_per_image[image_id] if image_id in target_boxes_per_image.keys() else []

        # Create a datastructure revolving around class_id
        result_boxes_per_class = defaultdict(list)
        [result_boxes_per_class[int(class_id)].append([int(y1), int(x1), int(y2), int(x2)]) for y1, x1, y2, x2, class_id, prob in result_boxes if prob >= prob_t]
        target_boxes_per_class = defaultdict(list)
        [target_boxes_per_class[int(class_id)].append([int(y1), int(x1), int(y2), int(x2)]) for y1, x1, y2, x2, class_id in target_boxes]
        class_ids = set(result_boxes_per_class.keys()).union(set(target_boxes_per_class.keys()))

        # Perform metric for each class individually
        for class_id in class_ids:
            num_result_boxes = len(result_boxes_per_class[class_id])
            num_target_boxes = len(target_boxes_per_class[class_id])
            num_matches = 0
            if num_result_boxes > 0 and num_target_boxes > 0:

                # Create a cost matrix
                cost_matrix = np.empty([num_result_boxes, num_target_boxes])
                for i, (result_y1, result_x1, result_y2, result_x2) in enumerate(result_boxes_per_class[class_id]):
                    for j, (target_y1, target_x1, target_y2, target_x2) in enumerate(target_boxes_per_class[class_id]):
                        target_area, result_area, overlap = get_area_and_overlap(target_y1, target_x1, target_y2,
                                                                                 target_x2, result_y1, result_x1,
                                                                                 result_y2, result_x2)
                        match = match_func(target_area, result_area, overlap)
                        cost_matrix[i, j] = 1. - match

                # Perform Hungrian matching
                result_ind, target_ind = linear_sum_assignment(cost_matrix)
                assert len(result_ind) == len(target_ind)
                num_matches = len([cost for cost in cost_matrix[result_ind, target_ind] if cost < 1. - match_t])

            # Aggregate metrics over all images and all classes
            tp[class_id] += num_matches
            fp[class_id] += num_result_boxes - num_matches
            fn[class_id] += num_target_boxes - num_matches

    return tp, tn, fp, fn


def calc_metrics_overall(tp: Dict[int, int], tn: Dict[int, int], fp: Dict[int, int], fn: Dict[int, int], metrics_funcs: Iterable[Callable[[int, int, int, int], int]]=(MetricLambda.F1,)) -> List[float]:
    tp = sum([value for key, value in tp.items()])
    tn = sum([value for key, value in tn.items()])
    fp = sum([value for key, value in fp.items()])
    fn = sum([value for key, value in fn.items()])
    return [metric(tp, tn, fp, fn) for metric in metrics_funcs]
