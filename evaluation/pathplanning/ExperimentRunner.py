from dataclasses import dataclass
from startGoalGeneration import StartSamplingMethod, GoalSamplingMethod
from startGoalGeneration import startSamplingFactory, goalSamplingFactory
from pathPlanner import plannerFactory, PlannerType
import numpy as np
from typing import List
import time


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


class ExperimentRunner:
    def __init__(self):
        self.startSamplingFn = None
        self.goalSamplingFn = None
        self.planner = None
        self.starts = None
        self.goals = None
        return

    def setupTask(
        self,
        config: PathPlanningTaskConfig,
        params: PathPlanningTaskParams,
    ) -> List[np.ndarray]:
        self.startSamplingFn = startSamplingFactory(
            config.start_sampling, params.obboxes
        )
        self.goalSamplingFn = goalSamplingFactory(
            config.goal_sampling, params.min_separation, params.obboxes
        )

        self.planner = plannerFactory(config.planner_type)
        self.starts = self.startSamplingFn(params.map, params.num_start_end_pairs)
        self.goals = self.goalSamplingFn(params.map, self.starts)

        self.params = params
        self.config = config

    def runTask(self) -> List[np.ndarray]:
        results = list()
        # TODO: speed this up / multithreading
        for idx in range(len(self.starts)):
            start_time = time.perf_counter()
            path = self.planner(self.params.map, self.starts[idx], self.goals[idx])
            end_time = time.perf_counter()
            runtime = end_time - start_time
            print(f"Function '{self.planner.__name__}' runtime: {runtime:.4f} seconds")

            if (len(path) == 0):
                print(f"Couldn't find start/end")
                continue

            results.append(path)
        return results
    
    def runValidation(self, paths: List[np.ndarray]) -> np.ndarray:
        # Run validation -> data perhaps along the lines of percentage of path invalid
        pass
