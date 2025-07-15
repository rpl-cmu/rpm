import tyro
from typing import List
from pathlib import Path
from dataclasses import dataclass
import pickle as pkl

from main import Result, main

@dataclass
class ExperimentResult:
    method: str
    scenario: str
    outcome: Result

def printExperimentResult(res: ExperimentResult):
    print(f"Method: {res.method}, scenario: {res.scenario}")
    print(f"{res.outcome.path_length_ratio:.3f} & {res.outcome.success_rate_rol:.3f} & {res.outcome.severity_of_failure_rol:.3f} & {res.outcome.success_rate_lor:.3f} & {res.outcome.severity_of_failure_lor:.3f}")

def formatForTable(res: List[ExperimentResult]):
    row1 = ["nsh", "nsh_short", "nsh_a1", "nsh_b"]
    row2 = ["c_corridor", "square1", "square2", "w_corridor1"]
    row3 = ["w_corridor2", "wean", "n/a", "n/a"]

    methods = ["cfar", "ra", "sar"]

    cfar_results = dict([(r.scenario, r) for r in res if r.method == "cfar"])
    ra_results = dict([(r.scenario, r) for r in res if r.method == "ra"])
    sar_results = dict([(r.scenario, r) for r in res if r.method == "sar"])

    results = [cfar_results, ra_results, sar_results]

    rows = [row1, row2, row3]

    for row in rows:
        for result in results:
            output_str = f"{result['nsh'].method}: "
            for scenario in row:
                if (scenario in result):
                    numbers = result[scenario]
                    output_str = output_str + f" {numbers.outcome.path_length_ratio:.3f} & {numbers.outcome.success_rate_rol:.3f} & {numbers.outcome.severity_of_failure_rol:.3f} & {numbers.outcome.success_rate_lor:.3f} & "
                else:
                    output_str = output_str + f" & & & &"
            print(output_str)


def printExistingResult(path: str):
    output = list()
    with open(path, 'rb') as file:
        output = pkl.load(file)

    formatForTable(output)

def navExperiments(
        lidar_map_path: str,
        radar_map_path: List[str],
        existing_output: str = "",
        write_to_output: bool = False
    ):

    if (len(existing_output) != 0):
        printExistingResult(existing_output)
        return

    lidar_pkls = list(Path(lidar_map_path).rglob("*.pkl"))
    lidar_map_set = dict((n.stem, idx) for idx, n in enumerate(lidar_pkls))
    
    total_results = list()

    for method in radar_map_path:
        radar_map_dirs = [subf for subf in Path(method).iterdir() if subf.is_dir()]

        method_name = (Path(method).stem).removeprefix("map_")

        pkl_name = "cfar.pkl" if method_name == "cfar" else "prob.pkl"

        for r_map_dir in radar_map_dirs:
            exp_name = r_map_dir.stem.removesuffix("_test")
            if (exp_name in lidar_map_set):
                validation_file = lidar_pkls[lidar_map_set[exp_name]]
                radar_map = r_map_dir.joinpath(pkl_name)
                outcome = main(validation_file, radar_map, num_pairs = 200, robot_radius = 0.2)
                result = ExperimentResult(method=method_name, scenario=exp_name, outcome=outcome)
                total_results.append(result)
                printExperimentResult(result)
    
    if write_to_output:
        with open("output.pkl", 'wb') as file:
            pkl.dump(total_results, file)

if __name__ == "__main__":
    tyro.cli(navExperiments)
