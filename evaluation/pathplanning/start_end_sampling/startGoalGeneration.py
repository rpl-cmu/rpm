import matplotlib.pyplot as plt
from typing import Tuple, List, Callable
import numpy as np
from enum import Enum
from functools import partial
import warnings

from .roomSampling import sample_from_polygons, sample_points_from_different_polygons


class StartSamplingMethod(Enum):
    FREESPACE = 0
    ROOM = 1
    FREESPACE_MUTUAL = 2


def startSamplingFreeSpace(
    map: np.ndarray,
    num_starts: int,
    validation_map: np.ndarray,
    map_validation_tf: np.ndarray,
) -> np.ndarray:
    valid_indices = np.argwhere(map < 0.2)

    if len(valid_indices) < num_starts:
        warnings.warn(f"Not enough free space to sample {num_starts} points")

    selected_indices = np.random.choice(
        len(valid_indices), size=num_starts, replace=False
    )
    return valid_indices[selected_indices]


def startSamplingInRooms(
    map: np.ndarray,
    num_starts: int,
    validation_map: np.ndarray,
    map_validation_tf: np.ndarray,
    obboxes: List[np.ndarray],
) -> np.ndarray:
    """
    obboxes: bounding boxes
    We assumes bboxes are already within the map
    """
    return sample_from_polygons(obboxes, num_starts)


def startSamplingInMutualFreeSpace(
    map: np.ndarray,
    num_starts: int,
    validation_map: np.ndarray,
    map_validation_tf: np.ndarray,
) -> np.ndarray:
    rounded_int_tf = np.flip(np.round(map_validation_tf).astype(np.int32))
    map_free_indices = np.argwhere(map < 0.2)

    validation_indices = map_free_indices + rounded_int_tf

    (max_x, max_y) = validation_map.shape
    valid_x = np.bitwise_and(
        validation_indices[:, 0] >= 0, validation_indices[:, 0] < max_x
    )
    valid_y = np.bitwise_and(
        validation_indices[:, 1] < max_y, validation_indices[:, 1] >= 0
    )
    valid_validation_map_indices = np.bitwise_and(valid_x, valid_y)
    validation_indices = validation_indices[valid_validation_map_indices]

    new_matrix = np.zeros_like(validation_map)
    new_matrix[validation_indices[:, 0], validation_indices[:, 1]] = 1

    validation_valid_mask = (
        validation_map[validation_indices[:, 0], validation_indices[:, 1]] < 0.2
    )
    
    valid_indices = validation_indices[validation_valid_mask] - rounded_int_tf

    if len(valid_indices) < num_starts:
        warnings.warn(
            f"Not enough free space shared between map and validation map for {num_starts} points, only {len(valid_indices)} available"
        )

    selected_mask = np.random.choice(len(valid_indices), size=num_starts, replace=False)
    return valid_indices[selected_mask]


def startSamplingFactory(
    sampling_type: StartSamplingMethod, obboxes: List[np.ndarray]
) -> Callable[[np.ndarray, int, np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a function that takes in a map, returns a np.ndarray (n, 2) representing all of the starts
    """
    if sampling_type == StartSamplingMethod.FREESPACE:
        return startSamplingFreeSpace
    elif sampling_type == StartSamplingMethod.ROOM:
        return partial(startSamplingInRooms, obboxes=obboxes)
    elif sampling_type == StartSamplingMethod.FREESPACE_MUTUAL:
        return startSamplingInMutualFreeSpace
    else:
        raise NotImplemented("Sampling function not implemented for start")


class GoalSamplingMethod(Enum):
    FREESPACE = 0
    MINSEPARTION = 1
    ROOM = 2  # A different room from Start
    FREESPACE_MUTUAL = 3


def goalSamplingFreeSpace(
    map: np.ndarray,
    start: np.ndarray,
    validation_map: np.ndarray,
    map_validation_tf: np.ndarray,
) -> np.ndarray:
    """ """

    return startSamplingFreeSpace(map, len(start))


def goalSamplingMutualFreeSpace(
    map: np.ndarray,
    start: np.ndarray,
    validation_map: np.ndarray,
    map_validation_tf: np.ndarray,
):
    return startSamplingInMutualFreeSpace(
        map, len(start), validation_map, map_validation_tf
    )


def goalSamplingMinSeparation(
    map: np.ndarray,
    start: np.ndarray,
    min_separation: float,
    validation_map: np.ndarray,
    map_validation_tf: np.ndarray,
) -> np.ndarray:
    # TODO
    pass


def goalSamplingDifferentRoom(
    map: np.ndarray,
    start: np.ndarray,
    validation_map: np.ndarray,
    map_validation_tf: np.ndarray,
    obboxes: List[np.ndarray],
) -> Tuple[np.ndarray]:
    """
    Generate randomly sampled start and end goals on map,
    """
    return sample_points_from_different_polygons(start, obboxes)


def goalSamplingFactory(
    sampling_type: GoalSamplingMethod, min_separation: float, obboxes: List[np.ndarray]
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a function that takes in a map and the list of starting points to generate

    map, num, validation map, map -> validation_map tf
    """
    if sampling_type == GoalSamplingMethod.FREESPACE:
        return goalSamplingFreeSpace
    elif sampling_type == GoalSamplingMethod.MINSEPARTION:
        return partial(goalSamplingMinSeparation, min_separation=min_separation)
    elif sampling_type == GoalSamplingMethod.ROOM:
        return partial(goalSamplingDifferentRoom, obboxes=obboxes)
    elif sampling_type == GoalSamplingMethod.FREESPACE_MUTUAL:
        return goalSamplingMutualFreeSpace
    else:
        raise NotImplemented("Sampling function not implemented for goal")
