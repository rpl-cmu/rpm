from typing import Tuple, List, Callable
import numpy as np
from enum import Enum
from functools import partial
import warnings


class StartSamplingMethod(Enum):
    FREESPACE = 0
    ROOM = 1


def startSamplingFreeSpace(map: np.ndarray, num_starts: int) -> np.ndarray:
    valid_indices = np.argwhere(map < 0.5)

    if len(valid_indices) < num_starts:
        warnings.warn(f"Not enough free space to sample {num_starts} points")

    selected_indices = np.random.choice(
        len(valid_indices), size=num_starts, replace=False
    )
    return valid_indices[selected_indices]


def startSamplingInRooms(
    map: np.ndarray, num_starts: int, obboxes: List[np.ndarray]
) -> np.ndarray:
    """
    obboxes: oriented bounding box - assuming we'll need some
    """
    # TODO
    pass


def startSamplingFactory(
    sampling_type: StartSamplingMethod, obboxes: List[np.ndarray]
) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    Returns a function that takes in a map, returns a np.ndarray (n, 2) representing all of the starts
    """
    if sampling_type == StartSamplingMethod.FREESPACE:
        return startSamplingFreeSpace
    elif sampling_type == StartSamplingMethod.ROOM:
        return partial(startSamplingInRooms, obboxes=obboxes)
    else:
        raise NotImplemented("Sampling function not implemented for start")


class GoalSamplingMethod(Enum):
    FREESPACE = 0
    MINSEPARTION = 1
    ROOM = 2  # A different room from Start


def goalSamplingFreeSpace(map: np.ndarray, start: np.ndarray) -> np.ndarray:
    """ """
    valid_indices = np.argwhere(map < 0.5)
    num_goals = len(start)

    if len(valid_indices) < num_goals:
        warnings.warn(f"Not enough free space to sample {num_goals} points")

    selected_indices = np.random.choice(
        len(valid_indices), size=num_goals, replace=False
    )
    return valid_indices[selected_indices]


def goalSamplingMinSeparation(
    map: np.ndarray, start: np.ndarray, min_separation: float
) -> np.ndarray:
    # TODO
    pass


def goalSamplingDifferentRoom(
    map: np.ndarray, start: np.ndarray, obboxes: List[np.ndarray]
) -> Tuple[np.ndarray]:
    """
    Generate randomly sampled start and end goals on map,
    """
    # TODO
    pass


def goalSamplingFactory(
    sampling_type: GoalSamplingMethod, min_separation: float, obboxes: List[np.ndarray]
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a function that takes in a map and the list of starting points to generate
    """
    if sampling_type == GoalSamplingMethod.FREESPACE:
        return goalSamplingFreeSpace
    elif sampling_type == GoalSamplingMethod.MINSEPARTION:
        return partial(goalSamplingMinSeparation, min_separation=min_separation)
    elif sampling_type == GoalSamplingMethod.ROOM:
        return partial(goalSamplingDifferentRoom, obboxes=obboxes)
    else:
        raise NotImplemented("Sampling function not implemented for goal")
