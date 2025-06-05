import tyro
from typing import List
from pathlib import Path
from dataclasses import dataclass

from main import Result, main

@dataclass
class ExperimentResult:
    method: str
    scenario: str
    outcome: Result

def printExperimentResult(res: ExperimentResult):
    print(f"Method: {res.method}, scenario: {res.scenario}")
    print(f"{res.outcome.path_length_ratio:.3f} & {res.outcome.success_rate_rol:.3f} & {res.outcome.severity_of_failure_rol:.3f} & {res.outcome.success_rate_lor:.3f} & {res.outcome.severity_of_failure_lor:.3f}")

def navExperiments(
        lidar_map_path: str,
        radar_map_path: List[str],
        debug_output: str = ""
    ):
    lidar_pkls = list(Path(lidar_map_path).rglob("*.pkl"))
    lidar_map_set = dict((n.stem, idx) for idx, n in enumerate(lidar_pkls))

    for method in radar_map_path:
        radar_map_dirs = [subf for subf in Path(method).iterdir() if subf.is_dir()]

        method_name = (Path(method).stem).removeprefix("map_")

        for r_map_dir in radar_map_dirs:
            exp_name = r_map_dir.stem.removesuffix("_test")
            if (exp_name in lidar_map_set):
                validation_file = lidar_pkls[lidar_map_set[exp_name]]
                radar_map = r_map_dir.joinpath("prob.pkl")
                outcome = main(validation_file, radar_map, num_pairs = 100, robot_radius = 0.3)
                result = ExperimentResult(method=method_name, scenario=exp_name, outcome=outcome)
                printExperimentResult(result)


if __name__ == "__main__":
    tyro.cli(navExperiments)
