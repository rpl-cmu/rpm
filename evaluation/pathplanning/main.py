from typing import List
import tyro
from pathlib import Path
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from json_utils import loadOrientedBoundingBox
from ExperimentRunner import (
    ExperimentRunner,
    PathPlanningTaskConfig,
    PathPlanningTaskParams,
)
from start_end_sampling.startGoalGeneration import StartSamplingMethod, GoalSamplingMethod
from pathPlanner import PlannerType


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

def experiment(
    config: PathPlanningTaskConfig,
    params: PathPlanningTaskParams,
):
    runner = ExperimentRunner()
    runner.setupTask(config, params)

    visualizeStartEnds(runner.starts, runner.goals, params.map)

    paths = runner.runTask()

    results, failed = runner.runValidation(paths)
    print(f"Failed {failed}")

    result = results[results > 0]
    plt.hist(result, rwidth=0.5)
    visualizePaths(params.map, params.validation_map, paths, params.map_map_tf)


def main(gt_file: Path, pred_file: Path, obbox: Path = Path(), num_pairs: int = 5, robot_radius = 0.3):
    gt_map = pkl.load(open(gt_file, "rb"))
    gt_data = gt_map["data"].astype(np.float32)
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
        StartSamplingMethod.ROOM, GoalSamplingMethod.ROOM, PlannerType.ASTAR
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

    experiment(config, param)


if __name__ == "__main__":
    tyro.cli(main)
