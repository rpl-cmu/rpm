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
from startGoalGeneration import StartSamplingMethod, GoalSamplingMethod
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


def experiment(
    config: PathPlanningTaskConfig,
    params: PathPlanningTaskParams,
):
    runner = ExperimentRunner()
    # Maybe make this more explicit
    runner.setupTask(config, params)
    paths = runner.runTask()

    visualizePaths(params.map, params.validation_map, paths, params.map_map_tf)


def main(gt_file: Path, pred_file: Path, obbox: Path = Path()):
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
        StartSamplingMethod.FREESPACE,
        GoalSamplingMethod.FREESPACE,
        PlannerType.ASTAR
    )

    param = PathPlanningTaskParams(
        map=pred_data,
        validation_map=gt_data,
        map_map_tf=tf,
        num_start_end_pairs=20,
        obboxes=bboxes,
        min_separation=10 / resoln,
    )

    experiment(config, param)


if __name__ == "__main__":
    tyro.cli(main)
