import glob
import pickle
import problems
from visualize import make_step_per_violation
from update import parameter_update
import openjij as oj
import jijzept as jz
import jijmodeling as jm
import datetime
import json
from typing import List, Dict, Callable, Any


class BaseBenchDict(dict):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        for k, v in self.default_dict().items():
            self[k] = v
            setattr(self, k, v)

    def default_dict(self):
        return {}

    def keys(self):
        return tuple(super().keys())

    def values(self):
        return tuple(super().values())

    def items(self):
        return tuple(super().items())


class BenchSetting(BaseBenchDict):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def default_dict(self):
        return {
            "updater": "",
            "problem_name": "",
            "mathmatical_model": {},
            "ph_value": {},
            "optimal_solution": [],
            "multipliers": {},
        }


class BenchResult(BaseBenchDict):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def default_dict(self):
        return {"penalties": {}, "raw_response": {}}


class Experiment:
    def __init__(self, updater) -> None:
        self.updater = updater
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

    def plot_penalty_by_step(self):
        pass

    def tts(self, num_sweeps_list=[30, 50, 80, 100, 150, 200], pr=0.99):
        pass


class PSBench:
    instance_dir = "./Instances"

    def __init__(
        self,
        updaters: List[Callable[[jm.Problem, jm.DecodedSamples, Dict], Dict]],
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

        for updater in self.updaters:
            for name, problem in self.problems.items():
                instance_files = glob.glob(f"{self.instance_dir}/{name}/*.pickle")
                for instance_file in instance_files:
                    experiment = Experiment(updater)
                    experiment.setting["updater"] = updater.__name__
                    experiment.setting["problem_name"] = name
                    experiment.setting[
                        "mathmatical_model"
                    ] = jm.expression.serializable.to_serializable(problem)
                    with open(instance_file, "rb") as f:
                        experiment.setting["ph_value"] = pickle.load(f)

                    # 最適解を設定（とりあえずdummy）
                    experiment.setting["optimal_solution"] = [0 for _ in range(2700)]

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
        experiment: Experiment,
        sampling_params={},
        max_iters=10,
    ):
        problem = self.problems[experiment.setting["problem_name"]]
        ph_value = experiment.setting["ph_value"]
        multipliers = self.initialize_multipliers(problem=problem)
        
        import numpy as np
        x1 = jm.BinaryArray("x", shape=(5, 3, 5, 18, 2))
        solution = np.zeros((5, 3, 5, 18, 2), dtype=int)
        energy = jm.evaluate_solutions(problem, [{x1: solution}], ph_value)
        print(energy)
        fafafa

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

            multipliers = experiment.updater(problem, decoded, multipliers)

        experiment.save(path="Results/")
        """ oj.solver_benchmark(
            solver=lambda time, **args: self.sampler.sample_model(
                problem, ph_value, multipliers, num_sweeps=time
            ),
            time_list=[30, 50, 80, 100, 150, 200],
            solutions=experiment.setting["optimal_solution"],
            p_r=0.99,
        ) """
        make_step_per_violation(experiment.datetime)

    def run(self, sampling_params={}, max_iters=10) -> None:
        self.setup()

        for experiment in self.experiments[0:1]:
            experiment.setting["num_iterations"] = max_iters
            self.run_for_onecase(
                experiment=experiment,
                sampling_params=sampling_params,
                max_iters=max_iters,
            )


def main():
    # target_instances = "all"
    target_instances = ["strip_packing"]
    sampler = jz.JijSASampler(config="../../../config/config.toml")

    bench = PSBench([parameter_update], sampler, target_instances=target_instances)
    sampling_params = {"num_sweeps": 5, "num_reads": 5}
    bench.run(sampling_params=sampling_params, max_iters=2)


if __name__ == "__main__":
    main()
