import torch
import torch.nn
import torch.autograd
import torch.nn.modules.loss
import torch.nn.functional as functional
import numpy as np
from typing import Tuple, Dict, List

from skimage.feature import peak_local_max


def variance_filter(tensor: torch.tensor, structuring_element: torch.tensor) -> torch.tensor:
    element_sum = torch.sum(structuring_element)
    tensor = functional.pad(tensor, [1, 1, 1, 1])  # t, b, l, r
    mu = functional.conv2d(tensor, structuring_element) / element_sum
    mu2 = functional.conv2d(tensor ** 2, structuring_element) / element_sum
    mu_mu = mu ** 2
    sigma = mu2 - mu_mu
    return sigma


def binary_edge_detector(instance_ids: np.ndarray) -> np.array:
    assert len(instance_ids.shape) == 2
    filter_a = np.array([[[0., 0., 0.],
                          [0., 1., 1.],
                          [0., 1., 1.]]], dtype=np.float32)
    filter_b = np.array([[[1., 1., 0.],
                          [1., 1., 0.],
                          [0., 0., 0.]]], dtype=np.float32)
    filter_a = torch.from_numpy(filter_a[np.newaxis])
    filter_b = torch.from_numpy(filter_b[np.newaxis])
    img = torch.from_numpy(instance_ids[np.newaxis, np.newaxis])
    a = variance_filter(img, filter_a)
    b = variance_filter(img, filter_b)
    c = a + b
    c[c > 0] = 1
    c *= img
    return c.squeeze().cpu().numpy()


def vector_distance_point(y, x, image_coords: np.ndarray = None, shape=None) -> Tuple[np.ndarray, np.ndarray]:
    assert (image_coords is not None or shape is not None)
    if shape is not None:
        assert len(shape) == 2
    image_coords = image_coords if image_coords is not None else np.transpose(np.indices(shape), (1, 2, 0))
    vec = np.array([y, x], dtype=np.float32) - image_coords
    dist = vec ** 2
    dist = np.sum(dist, axis=2)
    dist = np.sqrt(dist)
    return dist, vec


def calc_border_and_centroid_coords(instance_ids: np.ndarray, border_ids: np.ndarray) -> \
        Dict[int, Tuple[np.ndarray, np.ndarray]]:
    # Get unique labels
    uniq = np.unique(instance_ids)
    uniq = uniq[uniq != 0]
    # Get object coordinates
    object_coords = {int(label): np.transpose(np.argwhere(instance_ids == label)) for label in uniq}
    # Get object centroids
    centroid_coords = {int(label): [np.mean(y), np.mean(x)] for label, (y, x) in object_coords.items()}
    # Get border coordinates
    coords = {label: [centroid_coords[label], np.transpose(np.argwhere(border_ids == label))]
              for label in centroid_coords.keys()}
    # Return centroid and border coords
    return coords


def calc_vector_map(coords: Dict[int, Tuple[np.ndarray, np.ndarray]], shape: Tuple[int, int]) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the vector map from the instance centroids and border coordinates

    :param coords: A dict connecting the instance id (key) to it centroid and border coordinates
    :param shape: The shape of the output image [h,w]
    :return: An array with vector coords and border coords

    >>> centroid1 = np.array([1, 1])
    >>> border1 = np.array([[1-1, 1-1, 1+1, 1+1],
    ...                     [1-1, 1+1, 1-1, 1+1]])
    >>> centroid2 = np.array([2, 2])
    >>> border2 = np.array([[2-1, 2-1, 2+1, 2+1],
    ...                     [2-1, 2+1, 2-1, 2+1]])
    >>> coords = {1: [centroid1, border1],
    ...           2: [centroid2, border2]}
    >>> c, b = calc_vector_map(coords, (4, 4))
    >>> c
    array([[[ 1.,  1.,  1.,  1.],
            [ 0.,  0.,  0.,  1.],
            [-1., -1.,  0.,  0.],
            [-2., -1., -1., -1.]],
    <BLANKLINE>
           [[ 1.,  0., -1., -2.],
            [ 1.,  0., -1., -1.],
            [ 1.,  0.,  0., -1.],
            [ 1.,  1.,  0., -1.]]])
    >>> b
    array([[[ 0.,  0.,  0.,  0.],
            [-1., -1., -1.,  0.],
            [ 0.,  0., -1., -1.],
            [-1.,  0.,  0.,  0.]],
    <BLANKLINE>
           [[ 0., -1.,  0., -1.],
            [ 0., -1.,  0.,  0.],
            [ 0., -1., -1.,  0.],
            [ 0.,  0., -1.,  0.]]])
    """
    assert len(shape) == 2
    image_coords_planar = np.indices(shape)
    image_coords = np.transpose(image_coords_planar, (1, 2, 0))
    if len(coords) == 0:
        vec_ctr = np.zeros((2, *shape))
        return vec_ctr, vec_ctr

    # Start of centroids
    dist_ctr = np.ndarray([])
    vec_ctr = np.ndarray([])
    inst_ctr = np.zeros(shape)

    # Calculate centroid with minimal distance
    for i, (instance_id, ((y, x), (_, _))) in enumerate(coords.items()):
        if i > 0:
            # Generate distance map
            dist_ctr_new, vec_ctr_new = vector_distance_point(y, x, image_coords)
            # Check if smaller distances have been found
            dist_ctr_is_smaller = dist_ctr_new < dist_ctr
            # Assign smaller distances
            dist_ctr[dist_ctr_is_smaller] = dist_ctr_new[dist_ctr_is_smaller]
            # Assign vector of smaller distances
            vec_ctr[dist_ctr_is_smaller] = vec_ctr_new[dist_ctr_is_smaller]
            # Keep track of which instance the vector is pointing to
            inst_ctr[dist_ctr_is_smaller] = i
        else:
            # Init distances and vectors with first distance map
            dist_ctr, vec_ctr = vector_distance_point(y, x, image_coords)

    # Start of borders
    vec_brd = np.zeros((*shape, 2), dtype=np.float32)
    for i, (instance_id, (_, yx)) in enumerate(coords.items()):
        yx = np.transpose(yx)  # convert to (#, yx)
        # Get all coordinates of pixels closest to centroid i.
        image_coords = np.transpose(np.stack(np.where(inst_ctr == i)))
        # Get distance to border pixels of object i for all image coordinates
        dists = [np.sqrt(np.sum((yx - coord) ** 2, axis=1)) for coord in image_coords]
        # Get index of minimum distance border
        min_vec = np.stack([yx[np.argmin(dist)] for dist in dists])
        # Make coordinates relative
        min_vec -= image_coords
        # Write relative coordinates to result image
        vec_brd[image_coords[:, 0], image_coords[:, 1]] = min_vec

    return np.transpose(vec_ctr, (2, 0, 1)), np.transpose(vec_brd, (2, 0, 1))


def calc_centroid_and_border_vectors(instance_ids: np.ndarray, max_dist=0) -> Tuple[np.ndarray, np.ndarray]:
    instance_ids = instance_ids.astype(np.float32)
    border_ids = binary_edge_detector(instance_ids=instance_ids)
    coords = calc_border_and_centroid_coords(instance_ids=instance_ids, border_ids=border_ids)
    centroid_vectors, border_vectors = calc_vector_map(coords=coords, shape=instance_ids.shape)
    if max_dist != 0:
        active = np.sqrt(np.sum(border_vectors ** 2, axis=0)) > max_dist
        centroid_vectors[:, active] = 0
        border_vectors[:, active] = 0
    return centroid_vectors, border_vectors


def calc_vote_image(centroid_vectors: np.array) -> np.ndarray:
    # Convert to relative to absolute voting vectors
    channels, height, width = centroid_vectors.shape
    indices = np.indices((height, width), dtype=centroid_vectors.dtype)
    vectors = np.round(centroid_vectors + indices).astype(np.int32)

    # Prevent votes outside of the image
    logic = np.logical_and(
        np.logical_and(vectors[0] >= 0, vectors[1] >= 0),
        np.logical_and(vectors[0] < height, vectors[1] < width))
    coords = vectors[:, logic]

    # Cast votes
    votes = np.zeros([height, width])
    np.add.at(votes, (coords[0], coords[1]), 1)
    return np.expand_dims(votes, axis=0)


def calc_centroids(centroid_votes: np.array, centroid_threshold, nm_window) -> Tuple[np.ndarray, np.ndarray]:
    vote_image = centroid_votes[0]
    vote_image_rnd = vote_image * 100
    noise = np.random.rand(*vote_image_rnd.shape) * 100
    vote_image_rnd += noise

    vote_mask_indices = peak_local_max(image=vote_image_rnd, min_distance=nm_window // 2, threshold_abs=centroid_threshold)
    # indices to mask: see https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max
    vote_mask = np.zeros_like(vote_image_rnd, dtype=bool)
    vote_mask[tuple(vote_mask_indices.T)] = True

    vote_image *= vote_mask
    coords = np.argwhere(centroid_votes[0] > centroid_threshold)
    votes = centroid_votes[0, coords[:, 0], coords[:, 1]]
    return coords, votes


def calc_borders(centroid_list: np.ndarray, centroid_vectors: np.ndarray, border_vectors: np.ndarray,
                 border_threshold: int) -> Tuple[List[np.ndarray], np.ndarray]:
    channels, height, width = centroid_vectors.shape
    indices = np.indices((height, width), dtype=centroid_vectors.dtype)

    # Convert to absolute vectors
    image_ctr = (centroid_vectors + indices).astype(np.int32)
    image_brd = (border_vectors + indices).astype(np.int32)

    # Calculate border vectors
    all_border_coords = []
    border_votes = np.zeros([height, width])
    for centroid_coord in centroid_list:
        y, x = centroid_coord

        # Get all locations of contributing centroid_vectors
        contrib = np.logical_and(image_ctr[0] == y, image_ctr[1] == x)

        # Get border vector contribution
        border_coords = image_brd[:, contrib]

        # Filter border vectors outside of the image
        logic = np.logical_and(np.logical_and(border_coords[0] >= 0, border_coords[1] >= 0),
                               np.logical_and(border_coords[0] < height, border_coords[1] < width))
        border_coords = border_coords[:, logic]

        # Filter based on the votes
        if border_threshold > 0:
            vote_image = np.zeros((height, width))
            np.add.at(vote_image, (border_coords[0], border_coords[1]), 1)
            border_coords = np.argwhere(vote_image >= border_threshold)
            border_coords = np.transpose(border_coords)

            # Update the border votes
            border_votes = np.maximum(vote_image, border_votes)
        else:
            border_votes[border_coords[0], border_coords[1]] = 1

        all_border_coords.append(border_coords)

    return all_border_coords, border_votes


def nms(boxes: List[np.ndarray], iou_t: float) -> List[np.ndarray]:
    # If no bounding boxes, return empty list
    if len(boxes) == 0:
        return []

    # Get coordinates
    boxes = np.array(boxes)
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    votes = boxes[:, 4]

    # Result bounding boxes
    result_boxes = []
    result_score = []

    # Compute areas of bounding boxes
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(votes)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Get the bounding box with largest confidence score
        result_boxes.append(boxes[index])
        result_score.append(votes[index])

        # Compute intersection
        x1 = np.maximum(x1[index], x1[order[:-1]])
        x2 = np.minimum(x2[index], x2[order[:-1]])
        y1 = np.maximum(y1[index], y1[order[:-1]])
        y2 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute union
        union = areas[index] + areas[order[:-1]]

        # Compute iou
        iou = intersection / (union - intersection)

        left = np.where(iou < iou_t)
        order = order[left]

    return result_boxes