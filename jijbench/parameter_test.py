from collections import UserDict
import glob
import pickle
from sys import settrace
import problems
from visualize import make_step_per_violation
from alm import alm_transpile, alm_pyqubo_compile
from update import parameter_update
import jijzept as jz
import jijmodeling as jm
from jijmodeling.expression.serializable import to_serializable
import openjij as oj
import datetime
import json
from typing import List, Dict, Callable, Any


class BaseBenchDict(dict):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        for k in self.default_keys():
            self[k] = None
            setattr(self, k, None)

    def default_keys(self):
        return []

    def keys(self):
        return tuple(super().keys())

    def values(self):
        return tuple(super().values())

    def items(self):
        return tuple(super().items())


class BenchSetting(BaseBenchDict):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def default_keys(self):
        return ("problem_name", "mathmatical_model", "ph_value", "multipliers")


class BenchResult(BaseBenchDict):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def default_keys(self):
        return ("penalties", "raw_response")


class Experiment:
    def __init__(self) -> None:
        self.setting = BenchSetting()
        self.results = BenchResult()
        self.datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def save(self, path: str = ""):
        filename = path + "log_" + self.datetime + ".json"
        save_obj = {
            "date": str(self.datetime),
            "setting": self.setting,
            "results": self.results,
        }
        with open(filename, "w") as f:
            json.dump(save_obj, f)


class PSBench:
    instance_dir = "./Instance"

    def __init__(
        self,
        updaters: List[
            Callable[
                [jm.Problem, jm.DecodedSamples, Dict[str, float]], Dict[str, float]
            ]
        ],
        sampler: Any,
        target_instances="all",
    ) -> None:
        self.updaters = updaters
        self.sampler = sampler
        self.target_instances = target_instances

        self._problems = {}
        self._experiments = []

    @property
    def problems(self):
        return self._problems

    @problems.setter
    def problems(self, problems: Dict[str, jm.Problem]):
        self._problems = problems

    @property
    def experiments(self):
        return self._experiments

    @experiments.setter
    def experiments(self, experiments: List[Experiment]):
        self._experiments = experiments

    def prepare_instance_data(self):
        pass

    def setup(self):
        if self.target_instances == "all":
            for name in problems.__all__:
                self.problems[name] = getattr(problems, name)()
        else:
            for name in self.target_instances:
                self.problems[name] = getattr(problems, name)()

        for name, problem in self.problems.items():
            instance_files = glob.glob(f"{self.instance_dir}/{name}/*.pickle")
            for instance_file in instance_files:
                experiment = Experiment()
                experiment.setting["problem_name"] = name
                experiment.setting[
                    "mathmatical_model"
                ] = jm.expression.serializable.to_serializable(problem)
                with open(instance_file, "rb") as f:
                    experiment.setting["ph_value"] = pickle.load(f)
                experiment.setting["multipliers"] = {}

                experiment.results["penalties"] = {}
                experiment.results["raw_response"] = {}
                self._experiments.append(experiment)

    def initialize_multipliers(self, problem: jm.Problem):
        multipliers = {}
        for key in problem.constraints.keys():
            multipliers[key] = 1
        return multipliers

    def update_multipliers(self):
        pass

    def run_for_onecase(
        self,
        updater,
        experiment: DataSaver,
        sampling_params={},
        max_iters=10,
    ):
        problem = self.problems[experiment.setting["problem_name"]]
        ph_value = experiment.setting["ph_value"]
        multipliers = self.initialize_multipliers(problem=problem)

        for step in range(max_iters):
            experiment.setting["multipliers"][step] = multipliers
            response = self.sampler.sample_model(
                problem, ph_value, multipliers, **sampling_params
            )
            decoded = problem.decode(response, ph_value, {})

            penalties = []
            for violations in decoded.constraint_violations:
                penalties.append(sum(value for value in violations.values()))
            experiment.results["penalties"][step] = penalties
            experiment.results["raw_response"][step] = response.to_serializable()

            multipliers = updater(problem, decoded, multipliers)

        experiment.save(path="Results/")
        make_step_per_violation(experiment.datetime)

    def run(self, sampling_params={}, max_iters=10) -> None:
        self.setup()
        for updater in self.updaters:
            for experiment in self.experiments[0:1]:
                experiment.setting["num_iterations"] = max_iters
                self.run_for_onecase(
                    updater=updater,
                    experiment=experiment,
                    sampling_params=sampling_params,
                    max_iters=max_iters,
                )


def main():
    target_instances = "all"
    sampler = jz.JijSASampler(config="../../../config/config.toml")

    bench = PSBench([parameter_update], sampler, target_instances=target_instances)
    sampling_params = {"num_sweeps": 5, "num_reads": 5}
    bench.run(sampling_params=sampling_params, max_iters=2)


if __name__ == "__main__":
    main()