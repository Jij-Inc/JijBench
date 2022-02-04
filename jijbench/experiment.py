import datetime
import os
import json
import jijmodeling as jm
from dataclasses import dataclass, asdict
from typing import Dict, Callable, Any
from jijbench import problems


@dataclass
class ExperimentSetting:
    updater: str = ""
    sampler: str = ""
    problem_name: str = ""
    mathmatical_model: dict = None
    instance_file: str = ""
    ph_value: dict = None
    opt_value: float = -1.0
    multipliers: dict = None


@dataclass
class ExperimentResult:
    penalties: dict = None
    raw_response: dict = None
    result_file: str = ""
    log_dir: str = ""
    img_dir: str = ""
    table_dir: str = ""


class Experiment:
    log_filename = "experiment.json"

    def __init__(
        self,
        updater: Callable[[jm.Problem, jm.DecodedSamples, Dict], Dict],
        sampler: Any,
        result_dir="./Results",
        optional_args=None,
    ) -> None:
        self.updater = updater
        self.sampler = sampler
        if optional_args:
            self.optional_args = optional_args
        else:
            self.optional_args = {}

        self.setting = ExperimentSetting(
            updater=self.updater.__name__, sampler=self.sampler.__name__
        )
        self.results = ExperimentResult()

        self.datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.result_dir = result_dir
        benchmark_number = len([d for d in os.listdir(result_dir) if "benchmark" in d])
        self.log_dir = f"{result_dir}/benchmark_{benchmark_number}/logs"
        self.img_dir = f"{result_dir}/benchmark_{benchmark_number}/imgs"
        self.table_dir = f"{result_dir}/benchmark_{benchmark_number}/tables"

    def run(self, problem: jm.Problem, ph_value: Dict, max_iters=10):
        def _initialize_multipliers():
            _multipliers = {}
            for _key in problem.constraints.keys():
                _multipliers[_key] = 1
            return _multipliers

        self.setting.problem_name = problem.name
        self.setting.mathmatical_model = jm.expression.serializable.to_serializable(
            problem
        )
        self.setting.ph_value = ph_value
        self.setting.opt_value = ph_value.pop("opt_value", None)
        self.setting.multipliers = {}

        self.results.penalties = {}
        self.results.raw_response = {}

        multipliers = _initialize_multipliers()
        # 一旦デフォルトのnum_reads, num_sweepsでupdaterを動かす
        for step in range(max_iters):
            self.setting.multipliers[step] = multipliers
            multipliers, self.optional_args, response = self.updater(
                self.sampler,
                problem,
                ph_value,
                multipliers,
                self.optional_args,
                step=step,
                experiment=self,
            )

            decoded = problem.decode(response, ph_value, {})
            penalties = []
            for violations in decoded.constraint_violations:
                penalties.append(sum(value for value in violations.values()))
            self.results.penalties[step] = penalties
            self.results.raw_response[step] = response.to_serializable()

    def load(self, filename: str) -> "Experiment":
        """load date
        saveで保存した結果をそのままloadする.
        Returns:
            Experiment: loaded Experiment object.
        """
        with open(filename, "r") as f:
            data = json.load(f)

        date = data["date"]
        setting = data["setting"]
        results = data["results"]

        self.datetime = date
        self.setting = ExperimentSetting(**setting)
        self.results = ExperimentResult(**results)

    def save(self, savename: str = None):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)
        self.results.log_dir = self.log_dir
        self.results.img_dir = self.img_dir
        self.results.table_dir = self.table_dir

        if savename is None:
            savename = self.log_filename

        filename = f"{self.log_dir}/{savename}"
        print(self.log_dir)
        self.results.result_file = filename

        save_obj = {
            "date": str(self.datetime),
            "setting": asdict(self.setting),
            "results": asdict(self.results),
        }
        with open(filename, "w") as f:
            json.dump(save_obj, f)


if __name__ == "__main__":
    from users.makino.updater import update_simple
    from users.makino.solver import sample_model

    problem = problems.knapsack()
    instance_file = "/home/azureuser/JijBenchmark/jijbench/Instances/small/knapsack/f1_l-d_kp_10_269.json"
    with open(instance_file, "r") as f:
        ph_value = json.load(f)

    experiment = Experiment(
        updater=update_simple, sampler=sample_model, result_dir="./"
    )
    experiment.run(problem, ph_value, max_iters=0)
    experiment.save()

    experiment = experiment.load(
        "/home/azureuser/JijBenchmark/jijbench/results_3/logs/experiment.json"
    )
