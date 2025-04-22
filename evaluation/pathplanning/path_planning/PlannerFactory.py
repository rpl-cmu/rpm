from typing import Callable, Any, Tuple
import numpy as np
from enum import Enum
from .astar import aStar
from .voronoi import voronoi, generate_voronoi_from_probability_map
from .rrt import rrt_path_planning
from .apf import potential_field_planning

class PlannerType(Enum):
    ASTAR = 0
    VORONOI = 1
    RRT = 2
    APF = 3

def cacheGeneratorPlaceHolder(map: np.ndarray):
    return None

def plannerFactory(
    planner_type: PlannerType,
) -> Tuple[Callable[[np.ndarray, np.ndarray, np.ndarray, Any], np.ndarray], Callable]:
    """
    Factory method for planner to be used in benchmark
    """
    if planner_type == PlannerType.ASTAR:
        return aStar, cacheGeneratorPlaceHolder
    elif planner_type == PlannerType.VORONOI:
        return voronoi, generate_voronoi_from_probability_map
    elif planner_type == PlannerType.RRT:
        return rrt_path_planning, cacheGeneratorPlaceHolder
    elif planner_type == PlannerType.APF:
        return potential_field_planning, cacheGeneratorPlaceHolder
    else:
        raise NotImplementedError()
