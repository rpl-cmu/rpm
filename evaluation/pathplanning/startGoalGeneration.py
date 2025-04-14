from typing import Tuple, List, Callable
import numpy as np
from enum import Enum
from functools import partial


class StartSamplingMethod(Enum):
    FREESPACE = 0
    ROOM = 1


def startSamplingFreeSpace(map: np.ndarray) -> np.ndarray:
    # TODO
    pass


def startSamplingInRooms(map: np.ndarray, obboxes: List[np.ndarray]) -> np.ndarray:
    """
    obboxes: oriented bounding box - assuming we'll need some
    """
    # TODO
    pass


def startSamplingFactory(
    sampling_type: StartSamplingMethod, obboxes: List[np.ndarray]
) -> Callable[[np.ndarray], np.ndarray]:
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


def goalSamplingFreeSpace(map: np.ndarray, _: np.ndarray) -> np.ndarray:
    """ """
    pass


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
