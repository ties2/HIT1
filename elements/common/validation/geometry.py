
def get_overlap(box1_y1, box1_x1, box1_y2, box1_x2, box2_y1, box2_x1, box2_y2, box2_x2):
    # https://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    w = max(0, min(box1_x2, box2_x2) - max(box1_x1, box2_x1))
    h = max(0, min(box1_y2, box2_y2) - max(box1_y1, box2_y1))
    return w * h


def get_area_and_overlap(box1_y1, box1_x1, box1_y2, box1_x2, box2_y1, box2_x1, box2_y2, box2_x2):
    a1 = (max(box1_y1, box1_y2) - min(box1_y1, box1_y2)) * (max(box1_x1, box1_x2) - min(box1_x1, box1_x2))
    a2 = (max(box2_y1, box2_y2) - min(box2_y1, box2_y2)) * (max(box2_x1, box2_x2) - min(box2_x1, box2_x2))
    o = get_overlap(box1_y1, box1_x1, box1_y2, box1_x2, box2_y1, box2_x1, box2_y2, box2_x2)
    return a1, a2, o
