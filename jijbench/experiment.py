from dataclasses import dataclass, asdict
from typing import Dict, Callable, Any
from parasearch import problems
import datetime
import os
import json
import jijmodeling as jm


@dataclass
class ExperimentSetting:
    updater: str = ""
    sampler: str = ""
    problem_name: str = ""
    instance_name: str = ""
    mathmatical_model: dict = None
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
        self.setting = ExperimentSetting(
            updater=updater.__name__, sampler=sampler.__name__
        )
        self.results = ExperimentResult()
        self.datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = result_dir
        self.log_dir = f"{result_dir}/{self.datetime}/logs"
        self.img_dir = f"{result_dir}/{self.datetime}/imgs"
        self.table_dir = f"{result_dir}/{self.datetime}/tables"

    def run(self, problem: jm.Problem, ph_value: Dict, max_iters=10):
        def _initialize_multipliers(problem: jm.Problem):
            _multipliers = {}
            for _key in problem.constraints.keys():
                _multipliers[_key] = 1
            return _multipliers
        
        problem = self.setting.problem_name
        ph_value = self.setting.ph_value
        multipliers = _initialize_multipliers(problem)
        
        self.setting.multipliers = {}
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
            self.results["penalties"][step] = penalties
            self.results["raw_response"][step] = response.to_serializable()

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

    @classmethod
    def load(cls, filename: str) -> "Experiment":
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

        obj = Experiment()
        obj.datetime = date
        obj.setting = ExperimentSetting(**setting)
        obj.results = ExperimentResult(**results)

        return obj


if __name__ == "__main__":
    setting = ExperimentSetting()
    print(setting)
