import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import problems
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
            "instance_name": "",
            "mathmatical_model": {},
            "ph_value": {},
            "opt_value": [],
            "multipliers": {},
        }


class BenchResult(BaseBenchDict):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def default_dict(self):
        return {"penalties": {}, "raw_response": {}}


class Experiment:
    def __init__(self, updater, result_dir="./Results") -> None:
        self.updater = updater
        self.setting = BenchSetting()
        self.results = BenchResult()
        self.evaluation_metrics = pd.DataFrame()
        self.datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = result_dir
        self.log_dir = f"{result_dir}/logs"
        self.img_dir = f"{result_dir}/imgs"

    def save(self):
        filename = f"{self.log_dir}/{self.datetime}.json"
        save_obj = {
            "date": str(self.datetime),
            "setting": self.setting,
            "results": self.results,
        }
        with open(filename, "w") as f:
            json.dump(save_obj, f)

    def plot_evaluation_metrics(self):
        result_file = f"{self.log_dir}/{self.datetime}.json"
        with open(result_file, "r") as f:
            experiment = json.load(f)

        steps = range(experiment["setting"]["num_iterations"])
        penalties = experiment["results"]["penalties"]
        best_penalties = [min(value) for value in penalties.values()]

        instance_name = experiment["setting"]["instance_name"]
        save_dir = f"{self.img_dir}/{instance_name}"
        os.makedirs(save_dir, exist_ok=True)

        plt.plot(steps, best_penalties, marker="o")
        plt.title("step - sum of penalties")
        plt.xlabel("step")
        plt.ylabel("sum of penalties")
        plt.savefig(f"{save_dir}/sum_of_penalties.jpg")

        self.evaluation_metrics.plot(x="annealing_time", y="tts")
        plt.savefig(f"{save_dir}/tts.jpg")
        self.evaluation_metrics.plot(x="annealing_time", y="success_probability")
        plt.savefig(f"{save_dir}/success_probability.jpg")
        self.evaluation_metrics.plot(x="annealing_time", y="residual_energy")
        plt.savefig(f"{save_dir}/residual_energy.jpg")

    def evaluate(
        self,
        sampler,
        problem,
        ph_value,
        num_reads=1,
        num_sweeps_list=[30, 50, 80, 100, 150, 200],
        pr=0.99,
    ):
        result_file = f"{self.log_dir}/{self.datetime}.json"
        with open(result_file, "r") as f:
            experiment = json.load(f)

        steps = experiment["setting"]["num_iterations"]
        init_multipliers = experiment["setting"]["multipliers"]["0"]
        updated_multipliers = experiment["setting"]["multipliers"][str(steps - 1)]

        tts_list = []
        ps_list = []
        tau_list = []
        min_energy_list = []
        mean_eneagy_list = []
        for num_sweeps in num_sweeps_list[:2]:
            baseline = sampler.sample_model(
                problem,
                ph_value,
                init_multipliers,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                search=True,
            )
            baseline_decoded = problem.decode(baseline, ph_value, {})
            min_energy = baseline_decoded.feasibles().energy.min()

            response = sampler.sample_model(
                problem,
                ph_value,
                updated_multipliers,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                search=True,
            )
            tau = response.info["sampling_time"]

            decoded = problem.decode(response, ph_value, {})
            energies = decoded.feasibles().energy
            ps = (energies <= min_energy).sum() / len(decoded.solutions) + 1e-16

            tts_list.append(np.log(1 - pr) / np.log(1 - ps) * tau if ps < pr else tau)
            ps_list.append(ps)
            tau_list.append(tau)
            min_energy_list.append(min_energy)
            mean_eneagy_list.append(energies.mean())

        self.evaluation_metrics["annealing_time"] = tau_list
        self.evaluation_metrics["tts"] = tts_list
        self.evaluation_metrics["success_probability"] = ps_list
        self.evaluation_metrics["residual_energy"] = np.array(
            mean_eneagy_list
        ) - np.array(min_energy)


class PSBench:
    def __init__(
        self,
        updaters: List[Callable[[jm.Problem, jm.DecodedSamples, Dict], Dict]],
        sampler: Any,
        target_instances="all",
        n_instances_per_problem="all",
        instance_dir="./Instances",
    ) -> None:
        self.updaters = updaters
        self.sampler = sampler
        self.target_instances = target_instances
        self.n_instances_per_problem = n_instances_per_problem
        self.instance_dir = instance_dir

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
                instance_files = glob.glob(
                    f"{self.instance_dir}/{name}/**/*.json", recursive=True
                )
                if isinstance(self.n_instances_per_problem, int):
                    instance_files = instance_files[: self.n_instances_per_problem]
                print(instance_files)

                for instance_file in instance_files:
                    experiment = Experiment(updater)
                    experiment.setting["updater"] = updater.__name__
                    experiment.setting["problem_name"] = name
                    instance_name = instance_file.lstrip(self.instance_dir).rstrip(".json")
                    experiment.setting["instance_name"] = instance_name
                    experiment.setting[
                        "mathmatical_model"
                    ] = jm.expression.serializable.to_serializable(problem)
                    with open(instance_file, "rb") as f:
                        experiment.setting["ph_value"] = json.load(f)

                    experiment.setting["opt_value"] = experiment.setting[
                        "ph_value"
                    ].pop("opt_value", None)

                    self._experiments.append(experiment)

    def initialize_multipliers(self, problem: jm.Problem):
        multipliers = {}
        for key in problem.constraints.keys():
            multipliers[key] = 1
        return multipliers

    def update_multipliers(self):
        pass

    def run_for_one_experiment(
        self,
        experiment: Experiment,
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

            multipliers = experiment.updater(problem, decoded, multipliers)

        experiment.save()

        experiment.evaluate(self.sampler, problem, ph_value)
        experiment.plot_evaluation_metrics()

    def run(self, sampling_params={}, max_iters=10) -> None:
        self.setup()

        for experiment in self.experiments:
            experiment.setting["num_iterations"] = max_iters
            self.run_for_one_experiment(
                experiment=experiment,
                sampling_params=sampling_params,
                max_iters=max_iters,
            )


def main():
    # target_instances = "all"
    target_instances = ["knapsack"]
    sampler = jz.JijSASampler(config="../../../config/config.toml")

    bench = PSBench(
        [parameter_update],
        sampler,
        target_instances=target_instances,
        n_instances_per_problem=1,
    )
    sampling_params = {"num_sweeps": 5, "num_reads": 5}
    bench.run(sampling_params=sampling_params, max_iters=2)


if __name__ == "__main__":
    main()
