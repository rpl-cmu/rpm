import numpy as np
from typing import List

def highLightFailureRates(
    paths: List[np.ndarray],
    map: np.ndarray,
    map_map_tf: np.ndarray
) -> np.ndarray:
    heat_map = np.zeros_like(map)
    rounded_tf = np.round(map_map_tf).astype(np.int32)
    for path in paths:
        path = path + rounded_tf
        invalid = np.argwhere(
            map[path[:, 1], path[:, 0]] > 0.5
        )
        
        heat_map[path[invalid, 1], path[invalid, 0]] = 1
    
    return heat_map
