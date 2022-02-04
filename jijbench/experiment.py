from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Callable, Any

from jijbench import problems
import datetime
import os
import json
import jijmodeling as jm


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


class Experiment:
    log_filename = "log.json"

    def __init__(
        self,
        updater: Callable[[jm.Problem, jm.DecodedSamples, Dict], Dict],
        sampler: Any,
        result_dir="./Results",
        optional_args=None,
    ) -> None:
        self.updater = updater
        self.sampler = sampler
        self.result_dir = result_dir
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
        self.log_dir = f"{result_dir}/{self.datetime}/logs"
        self.img_dir = f"{result_dir}/{self.datetime}/imgs"
        self.table_dir = f"{result_dir}/{self.datetime}/tables"

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

        updater = self.updater
        sampler = self.sampler
        optional_args = self.optional_args
        result_dir = self.result_dir

        obj = Experiment(updater, sampler, result_dir, optional_args)
        obj.datetime = date
        obj.setting = ExperimentSetting(**setting)
        obj.results = ExperimentResult(**results)
        return obj

    def save(self):
        instance_name = self.setting.instance_name
        save_dir = f"{self.log_dir}/{instance_name}"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{save_dir}/{self.log_filename}"
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
    experiment.run(problem, ph_value, max_iters=1)
    experiment.save()

    experiment = experiment.load(
        "/home/azureuser/JijBenchmark/jijbench/20220203_213241/logs/log.json"
    )
    print(experiment.setting)
