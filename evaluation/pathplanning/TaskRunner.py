from dataclasses import dataclass
from startGoalGeneration import StartSamplingMethod, GoalSamplingMethod
from startGoalGeneration import startSamplingFactory, goalSamplingFactory
from pathPlanner import plannerFactory, PlannerType
import numpy as np
from typing import List

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
    num_start_end_pairs: int
    obboxes: List[np.ndarray]
    min_separation: float

class TaskRunner:
    def __init__(self):
        self.startSamplingFn = None
        self.goalSamplingFn = None
        self.planner = None
        return

    def setupTask(
        self,
        map: np.ndarray,
        config: PathPlanningTaskConfig,
        params: PathPlanningTaskParams,
    ):
        self.startSamplingFn = startSamplingFactory(
            config.start_sampling, params.obboxes
        )
        self.goalSamplingFn = goalSamplingFactory(
            config.goal_sampling, params.min_separation, params.obboxes
        )

        self.planner = plannerFactory(config.planner_type)

        starts = self.startSamplingFn(map)
        goals = self.goalSamplingFn(map, starts)

        # Dispatch to run path planning
        # collect path outcomes -> since we potentially would like to see path visualized
        # Run validation -> data perhaps along the lines of percentage of path invalid 
