from typing import Tuple
import numpy as np
from dataclasses import dataclass

@dataclass 
class PathEvaluationResult:
    percentageInvalid: float

def evaluatePath(singleTrajectory: np.ndarray, validationMap: np.ndarray) -> float:
    """
    returns percentage of path that's valid
    """

    return sum(validationMap[singleTrajectory[:, 1], singleTrajectory[:, 0]] < 0.3) / len(singleTrajectory)
