from ExperimentRunner import PathPlanningTaskConfig, PathPlanningTaskParams
from typing import List
from pathlib import Path
import numpy as np
import json

def loadOrientedBoundingBox(obboxes: Path) -> np.ndarray:
    """
    obboxes are stored in a 3d array, 

    result[0] returns a (4, 2) matrix -> 4 points of bbox
    """
    with open(obboxes, 'rb') as file:
        obbox_json = json.load(file)

    result = np.zeros((0, 4, 2))
    for pts in obbox_json['boxes']:
        result = np.append(result, np.asarray(pts['points']).reshape((1, 4, 2)), axis=0)
    
    return result
