from typing import List, Tuple
import tyro
from pathlib import Path
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import imageio.v3 as iio
from dataclasses import dataclass

from pathplanning.json_utils import loadOrientedBoundingBox
from pathplanning.ExperimentRunner import (
    ExperimentRunner,
    PathPlanningTaskConfig,
    PathPlanningTaskParams,
)
from pathplanning.start_end_sampling.startGoalGeneration import StartSamplingMethod, GoalSamplingMethod
from pathplanning.path_planning import PlannerType
from pathplanning.pathEvaluation import highLightFailureRates

@dataclass 
class Result:
    path_length_ratio: float
    success_rate_rol: float
    severity_of_failure_rol: float
    success_rate_lor: float
    severity_of_failure_lor: float


def visualizePaths(
    map: np.ndarray, validation_map: np.ndarray, paths: List[np.ndarray], tf: np.ndarray
):
    lc = LineCollection(paths)
    lc.set_array(np.arange(len(paths)))

    # Because reuse in different subplots is not allowed
    validation_paths = list()
    for p in paths:
        validation_paths.append(p.copy() + tf)
    lc2 = LineCollection(validation_paths)
    lc2.set_array(np.arange(len(validation_paths)))

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.imshow(map, cmap="gray")
    ax1.add_collection(lc)

    ax2.imshow(validation_map, cmap="gray")
    ax2.add_collection(lc2)

    ax1.set_title("Trajectories over default map")
    ax2.set_title("Trajectories over validation map")
    plt.tight_layout()
    plt.show()

def visualizeStartEnds(starts: np.ndarray, goals: np.ndarray, map: np.ndarray):
    plt.plot(starts[:, 1], starts[:, 0], 'ro')
    plt.plot(goals[:, 1], goals[:, 0], 'wx')
    # Draw lines connecting original points to corresponding points
    for i in range(len(starts)):
        plt.plot([starts[i, 1], goals[i, 1]], [starts[i, 0], goals[i, 0]], 'k--', alpha=0.5)
    if len(map) != 0:
        plt.imshow(map)
    plt.show()

def numbers(results: np.ndarray) -> Tuple[float, float]:
    PERCENTILE = 1
    full_success = np.sum(results >= PERCENTILE) / len(results)
    severity = np.mean(1 - results[np.bitwise_and(results < PERCENTILE, results > 0)])
    return full_success, severity

def path_ratios(
    radar_path: List[np.ndarray],
    lidar_path: List[np.ndarray],
    results: np.ndarray
):
    ratio_results = list()
    for idx in range(len(radar_path)):
        if (results[idx] < 1):
            continue

        if len(radar_path[idx]) > 0 and len(lidar_path[idx]) > 0:
            ratio_results.append(len(radar_path[idx]) / len(lidar_path[idx]))

    return np.mean(np.asarray(ratio_results))

def experiment(
    config: PathPlanningTaskConfig,
    params: PathPlanningTaskParams,
    debug: bool = False,
) -> Result:
    runner = ExperimentRunner()
    runner.setupTask(config, params)
    
    if debug:
        visualizeStartEnds(runner.starts, runner.goals, params.map)
        visualizeStartEnds(runner.starts + np.flip(params.map_map_tf), runner.goals + np.flip(params.map_map_tf), params.validation_map)

    paths, lidar_paths = runner.runTask(debug)

    results, _ = runner.runValidation(
        paths,
        params.validation_map,
        params.map_map_tf
    )

    sr_rol, sof_rol = numbers(results)
    path_length_ratio = path_ratios(paths, lidar_paths, results)
    
    lidar_on_radar, _ = runner.runValidation(
        lidar_paths,
        params.map,
        -params.map_map_tf
    )
    sr_lor, sof_lor = numbers(lidar_on_radar)
    exp_result = Result(
        path_length_ratio=path_length_ratio,
        success_rate_rol=sr_rol,
        severity_of_failure_rol=sof_rol,
        success_rate_lor=sr_lor,
        severity_of_failure_lor=sof_lor
    )
    
    if debug:
        print(results)
        heat_map = highLightFailureRates(paths, params.validation_map, params.map_map_tf)
        plt.imshow(heat_map, alpha=0.9, cmap="Reds")
        plt.imshow(params.validation_map, alpha=0.1, cmap='gray')
        plt.show(block=False)
        result = results[results > 0]
        plt.hist(result, rwidth=0.5)
        visualizePaths(params.map, params.validation_map, paths, params.map_map_tf)

        radar_failed_hm = highLightFailureRates(lidar_paths, params.map, -np.round(params.map_map_tf))
        plt.imshow(radar_failed_hm, alpha=0.9, cmap="Reds")
        plt.imshow(params.map, alpha=0.1, cmap='gray')
        plt.show(block=False)
        visualizePaths(params.validation_map, params.map, lidar_paths, -np.round(params.map_map_tf))

    return exp_result

def main(
    gt_file: Path,
    pred_file: Path,
    obbox: Path = Path(),
    num_pairs: int = 5,
    robot_radius = 0.2,
    planner_type: PlannerType = PlannerType.ASTAR,
    debug: bool = False
) -> Result:
    gt_map = pkl.load(open(gt_file, "rb"))
    gt_data = gt_map["data"].astype(np.float32)

    if (np.max(gt_data) > 250):
        gt_data = gt_data / 255
    gt_t = np.asarray(gt_map["t"][:-1])

    pred_map = pkl.load(open(pred_file, "rb"))
    pred_data = pred_map["data"].astype(np.float32)
    resoln = pred_map["resolution"]
    pred_t = np.asarray(pred_map["t"][:-1])

    tf = (pred_t - gt_t) / (gt_map["resolution"])
    tf = np.array([tf[-1], tf[0]])

    bboxes = list()
    if len(obbox.name) != 0:
        bboxes = loadOrientedBoundingBox(obbox)

    config = PathPlanningTaskConfig(
        StartSamplingMethod.FREESPACE_MUTUAL, GoalSamplingMethod.FREESPACE_MUTUAL, planner_type
    )

    robot_radius_m = robot_radius
    param = PathPlanningTaskParams(
        map=pred_data,
        validation_map=gt_data,
        map_map_tf=tf,
        num_start_end_pairs=num_pairs,
        obboxes=bboxes,
        min_separation=10 / resoln,
        map_inflation_radius=robot_radius_m / resoln,
    )

    return experiment(config, param, debug)


if __name__ == "__main__":
    tyro.cli(main)
