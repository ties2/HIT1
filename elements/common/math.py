from functools import reduce
from math import cos, sin

import numpy as np

infinity = float('inf')
nan = float('nan')


def homogeneous_translation_matrix_2d(translation):
    return np.array([
        1, 0, translation[0],
        0, 1, translation[1],
        0, 0, 1
    ]).reshape((3, 3))


def homogeneous_rotation_matrix_2d(radians):
    return np.array([
        cos(radians), -sin(radians), 0,
        sin(radians), cos(radians), 0,
        0, 0, 1
    ]).reshape((3, 3))


def homogeneous_scale_matrix_2d(scale_factor):
    return np.array([
        scale_factor, 0, 0,
        0, scale_factor, 0,
        0, 0, 1
    ]).reshape((3, 3))


def homogeneous_shear_x_matrix_2d(shear_factor):
    return np.array([
        1, shear_factor, 0,
        0, 1, 0,
        0, 0, 1
    ]).reshape((3, 3))


def homogeneous_shear_y_matrix_2d(shear_factor):
    return np.array([
        1, 0, 0,
        shear_factor, 1, 0,
        0, 0, 1
    ]).reshape((3, 3))


def homogeneous_flip_x_matrix_2d():
    return np.array([
        -1, 0, 0,
        0, 1, 0,
        0, 0, 1
    ]).reshape((3, 3))


def homogeneous_flip_y_matrix_2d():
    return np.array([
        1, 0, 0,
        0, -1, 0,
        0, 0, 1
    ]).reshape((3, 3))


# Performs a left fold of np.dot over the given matrices or vectors.
# i.e. calculates dot(arg1, dot(arg2, dot(arg3, ...)))
def repeated_dot_product(*matrices):
    return reduce(np.dot, list(matrices))


# Gives the center of an xyxy bounding box.
def aabb_center(box):
    return np.divide(box[0:2] + box[2:4], 2)


# Gets the four corner points of an xyxy bounding box.
def aabb_to_corners(box):
    x, y = box[0:2]
    w, h = box[2:4] - box[0:2]

    return np.array([
        [x + 0, y + 0],
        [x + w, y + 0],
        [x + 0, y + h],
        [x + w, y + h]
    ])


# Gets the smallest xyxy-aabb that contains the given four corners.
def corners_to_aabb(corners):
    xs = corners[:, 0]
    ys = corners[:, 1]

    return np.array([min(xs), min(ys), max(xs), max(ys)])


# Checks whether the given xyxy-aabb is completely inside another one.
def aabb_inside(contained, container):
    xs = contained[0:4:2]
    ys = contained[1:4:2]

    return (
            all([container[0] <= x < container[2] for x in xs]) and
            all([container[1] <= y < container[3] for y in ys])
    )


# Returns the relative area of the given xyxy-aabbs.
# e.g. if first = [0, 0, 1000, 1000] and second = [0, 0, 500, 500], this method will return 0.25.
def aabb_relative_area(first, second):
    first_pixels = np.multiply(*(first[2:4] - first[0:2]))
    second_pixels = np.multiply(*(second[2:4] - second[0:2]))

    return second_pixels / first_pixels
