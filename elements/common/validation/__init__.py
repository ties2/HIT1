"""
---------------------
**validation** module
---------------------

This module contains the public validation functionality.

>>> from common.elements.validation import *

"""


from .lamdas import (
    MetricLambda,
    MatchLambda,
    precision, recall, f1score
)

from .ssim import SSIM

from .map import (
    calc_iou_individual,
    calc_precision_recall,
    get_model_scores_map,
    get_single_image_results,
    get_avg_precision_at_iou
)

from .detection import (
    calc_metrics_overall,
    calc_confusion_metrics,
)

from .geometry import (
    get_overlap,
    get_area_and_overlap
)

from .mahal_dist import mahalanobis_distance