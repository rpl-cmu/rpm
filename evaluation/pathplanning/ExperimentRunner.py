import concurrent.futures
from dataclasses import dataclass
from start_end_sampling.startGoalGeneration import (
    StartSamplingMethod,
    GoalSamplingMethod,
)
from start_end_sampling.startGoalGeneration import (
    startSamplingFactory,
    goalSamplingFactory,
)
from path_planning import PlannerType, plannerFactory
from pathEval import evaluatePath
import concurrent
import numpy as np
from typing import List, Tuple
from scipy import ndimage
from tqdm import tqdm
import os


# "Type" config required to launch the evaluations
@dataclass
class PathPlanningTaskConfig:
    start_sampling: StartSamplingMethod
    goal_sampling: GoalSamplingMethod
    planner_type: PlannerType


# Numerical values required to launch the evaluations
@dataclass
class PathPlanningTaskParams:
    map: np.ndarray
    validation_map: np.ndarray
    map_map_tf: np.ndarray
    num_start_end_pairs: int
    obboxes: List[np.ndarray]
    min_separation: float
    map_inflation_radius: int


def process_path(idx, map_to_use, start, goal, cache, planner):
    path = planner(map_to_use, start, goal, cache)
    return idx, path


class ExperimentRunner:
    def __init__(self):
        self.startSamplingFn = None
        self.goalSamplingFn = None
        self.planner = None
        self.starts = None
        self.goals = None
        self.map_to_use = np.zeros((0, 0))
        return

    def setupTask(
        self,
        config: PathPlanningTaskConfig,
        params: PathPlanningTaskParams,
    ) -> List[np.ndarray]:
        self.params = params
        self.config = config

        self.startSamplingFn = startSamplingFactory(
            config.start_sampling, params.obboxes
        )
        self.goalSamplingFn = goalSamplingFactory(
            config.goal_sampling, params.min_separation, params.obboxes
        )

        (self.planner, cacheGenerator) = plannerFactory(config.planner_type)

        self.map_to_use = np.zeros_like(self.params.map)
        self.map_to_use[self.params.map < 0.4] = 0
        self.map_to_use[self.params.map >= 0.4] = 1
        if not np.isclose(params.map_inflation_radius, 0):
            print("Inflating map")
            dilation_struct = ndimage.generate_binary_structure(2, 1)
            self.map_to_use = ndimage.binary_dilation(
                self.map_to_use,
                structure=dilation_struct,
                iterations=int(params.map_inflation_radius),
            )

        self.starts = self.startSamplingFn(
            self.map_to_use,
            params.num_start_end_pairs,
            params.validation_map,
            params.map_map_tf,
        )
        self.goals = self.goalSamplingFn(
            self.map_to_use, self.starts, params.validation_map, params.map_map_tf
        )
        self.cache = cacheGenerator(self.map_to_use)

    def runTask(self) -> List[np.ndarray]:

        results = [None] * len(self.starts)

        if self.config.planner_type == PlannerType.VORONOI:
            for idx in range(len(self.starts)):
                results[idx] = self.planner(
                    self.map_to_use, self.starts[idx], self.goals[idx], self.cache
                )
        else:
            # Use ProcessPoolExecutor for CPU-bound tasks
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=int(os.cpu_count() - 2)
            ) as executor:
                futures = []
                for idx in range(len(self.starts)):
                    future = executor.submit(
                        process_path,
                        idx,
                        self.map_to_use,
                        self.starts[idx],
                        self.goals[idx],
                        self.cache,
                        self.planner,
                    )
                    futures.append(future)

                # Process results as they complete with a progress bar
                for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures)
                ):
                    idx, path = future.result()
                    results[idx] = path
        return results

    def runValidation(self, paths: List[np.ndarray]) -> Tuple[np.ndarray, int]:
        # Run validation -> data perhaps along the lines of percentage of path invalid
        results = np.zeros(len(paths))
        failed_path_count = 0
        rounded_tf = np.round(self.params.map_map_tf).astype(np.int32)
        for idx, path in enumerate(paths):
            if (len(path)) == 0:
                failed_path_count += 1
                continue
            results[idx] = evaluatePath(path + rounded_tf, self.params.validation_map)

        return results, failed_path_count
