
def f1score(tp: int, fp: int, fn: int) -> float:
    try:
        return (2 * tp) / (2 * tp + fp + fn)
    except ZeroDivisionError:
        return float('NaN')


def precision(tp: int, fp: int) -> float:
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return float('NaN')


def recall(tp: int, fn: int) -> float:
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return float('NaN')


class MetricLambda:
    """
    Lambdas for calculating metrics
    """
    F1 = lambda tp, tn, fp, fn: f1score(tp=tp, fp=fp, fn=fn)
    Precision = lambda tp, tn, fp, fn: precision(tp=tp, fp=fp)
    Recall = lambda tp, tn, fp, fn: recall(tp=tp, fn=fn)


class MatchLambda:
    """
    Lambdas for determining overlap
    """
    IoU = lambda pred_area, gt_area, overlap: overlap / (pred_area + gt_area - overlap)
    Dice = lambda pred_area, gt_area, overlap: (2 * overlap) / (pred_area + gt_area)