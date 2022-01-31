import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jijzept as jz
import jijmodeling as jm
import datetime
import json
from parasearch import problems
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
            "opt_value": -1,
            "multipliers": {},
        }


class BenchResult(BaseBenchDict):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def default_dict(self):
        return {"penalties": {}, "raw_response": {}}


class Experiment:
    jijzept_config_file = "/home/azureuser/.config/jijzept/config.toml"
    baseline_sampler = jz.JijSASampler(config=jijzept_config_file)
    log_filename = "log.json"

    def __init__(self, updater, result_dir="./Results") -> None:
        self.updater = updater
        self.setting = BenchSetting()
        self.results = BenchResult()
        self.evaluation_metrics = None
        self.datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = result_dir
        self.log_dir = f"{result_dir}/{self.datetime}/logs"
        self.img_dir = f"{result_dir}/{self.datetime}/imgs"
        self.table_dir = f"{result_dir}/{self.datetime}/tables"

    def save(self):
        instance_name = self.setting["instance_name"]
        save_dir = f"{self.log_dir}/{instance_name}"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{save_dir}/{self.log_filename}"
        save_obj = {
            "date": str(self.datetime),
            "setting": self.setting,
            "results": self.results,
        }
        with open(filename, "w") as f:
            json.dump(save_obj, f)

    def plot_evaluation_metrics(self):
        instance_name = self.setting["instance_name"]
        result_file = f"{self.log_dir}/{instance_name}/{self.log_filename}"
        with open(result_file, "r") as f:
            experiment = json.load(f)

        steps = range(experiment["setting"]["num_iterations"])
        penalties = experiment["results"]["penalties"]
        best_penalties = [min(value) for value in penalties.values()]

        save_dir = f"{self.img_dir}/{instance_name}"
        os.makedirs(save_dir, exist_ok=True)
        if best_penalties:
            plt.plot(steps, best_penalties, marker="o")
            plt.title("step - sum of penalties")
            plt.xlabel("step")
            plt.ylabel("sum of penalties")
            plt.savefig(f"{save_dir}/sum_of_penalties.jpg")

        self.evaluation_metrics.plot(x="annealing_time", y="time_to_solution")
        plt.savefig(f"{save_dir}/time_to_solution.jpg")
        self.evaluation_metrics.plot(x="annealing_time", y="success_probability")
        plt.savefig(f"{save_dir}/success_probability.jpg")
        self.evaluation_metrics.plot(x="annealing_time", y="residual_energy")
        plt.savefig(f"{save_dir}/residual_energy.jpg")

    def evaluate(
        self,
        sampler,
        problem,
        ph_value,
        multipliers,
        optional_args,
        num_reads=1,
        num_sweeps_list=[25, 50, 100, 150, 200, 300],
        pr=0.99,
    ):
        instance_name = self.setting["instance_name"]

        """ result_file = f"{self.log_dir}/{instance_name}/{self.log_filename}"
        with open(result_file, "r") as f:
            experiment = json.load(f)

        steps = experiment["setting"]["num_iterations"]
        updated_multipliers = experiment["setting"]["multipliers"][str(steps - 1)] """

        evaluation_metrics = {
            "annealing_time": [],
            "feasible_rate_for_baseline": [],
            "feasible_rate_for_new_updater": [],
            "time_to_solution": [],
            "success_probability": [],
            "min_energy": [],
            "mean_eneagy": [],
            "residual_energy": [],
        }
        init_multipliers = initialize_multipliers(problem)
        for num_sweeps in num_sweeps_list:
            baseline = self.baseline_sampler.sample_model(
                problem,
                ph_value,
                init_multipliers,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                search=True,
            )

            baseline_decoded = problem.decode(baseline, ph_value, {})
            min_energy = baseline_decoded.feasibles().energy.min()

            response = sampler(
                problem,
                ph_value,
                multipliers,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                **optional_args,
            )
            tau = response.info["sampling_time"]

            decoded = problem.decode(response, ph_value, {})
            energies = decoded.feasibles().energy
            ps = (energies <= min_energy).sum() / len(decoded.solutions) + 1e-16

            evaluation_metrics["annealing_time"].append(tau)
            evaluation_metrics["feasible_rate_for_baseline"].append(
                len(baseline_decoded.feasibles()) / len(baseline_decoded)
            )
            evaluation_metrics["feasible_rate_for_new_updater"].append(
                len(decoded.feasibles()) / len(decoded)
            )
            evaluation_metrics["time_to_solution"].append(
                np.log(1 - pr) / np.log(1 - ps) * tau if ps < pr else tau
            )
            evaluation_metrics["success_probability"].append(ps)
            evaluation_metrics["min_energy"].append(min_energy)
            evaluation_metrics["mean_eneagy"].append(energies.mean())
            evaluation_metrics["residual_energy"].append(energies.mean() - min_energy)

        self.evaluation_metrics = pd.DataFrame(evaluation_metrics)

        save_dir = f"{self.table_dir}/{instance_name}"
        os.makedirs(save_dir, exist_ok=True)

        self.evaluation_metrics.to_csv(f"{save_dir}/metrics.csv")


class PSBench:
    def __init__(
        self,
        updater: Callable[[jm.Problem, jm.DecodedSamples, Dict], Dict],
        sampler: Any,
        target_instances="all",
        n_instances_per_problem="all",
        optional_args={},
        instance_dir="./Instances",
        result_dir="./Results",
    ) -> None:
        self.updater = updater
        self.sampler = sampler
        self.target_instances = target_instances
        self.n_instances_per_problem = n_instances_per_problem
        self.instance_dir = instance_dir
        self.result_dir = result_dir
        self.optional_args = optional_args

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
            instance_files = glob.glob(
                f"{self.instance_dir}/{name}/**/*.json", recursive=True
            )
            if isinstance(self.n_instances_per_problem, int):
                instance_files = instance_files[: self.n_instances_per_problem]
            for instance_file in instance_files:
                experiment = Experiment(self.updater, result_dir=self.result_dir)
                experiment.setting["updater"] = self.updater.__name__
                experiment.setting["problem_name"] = name
                instance_name = instance_file.lstrip(self.instance_dir).rstrip(".json")
                experiment.setting["instance_name"] = instance_name
                experiment.setting[
                    "mathmatical_model"
                ] = jm.expression.serializable.to_serializable(problem)
                with open(instance_file, "rb") as f:
                    experiment.setting["ph_value"] = json.load(f)
                experiment.setting["opt_value"] = experiment.setting["ph_value"].pop(
                    "opt_value", None
                )
                self._experiments.append(experiment)

    def run_for_one_experiment(
        self,
        experiment: Experiment,
        max_iters=10,
    ):
        problem = self.problems[experiment.setting["problem_name"]]
        ph_value = experiment.setting["ph_value"]
        multipliers = initialize_multipliers(problem=problem)

        # 一旦デフォルトのnum_reads, num_sweepsでupdaterを動かす
        for step in range(max_iters):
            experiment.setting["multipliers"][step] = multipliers
            multipliers, self.optional_args, response = experiment.updater(
                self.sampler,
                problem,
                ph_value,
                multipliers,
                self.optional_args,
                step=step,
                experiment=experiment,
            )

            decoded = problem.decode(response, ph_value, {})
            penalties = []
            for violations in decoded.constraint_violations:
                penalties.append(sum(value for value in violations.values()))
            experiment.results["penalties"][step] = penalties
            experiment.results["raw_response"][step] = response.to_serializable()

            if step == max_iters - 1:
                experiment.save()
                experiment.evaluate(
                    self.sampler, problem, ph_value, multipliers, self.optional_args
                )
                experiment.plot_evaluation_metrics()

    def run(self, sampling_params={}, max_iters=10):
        self.setup()

        for experiment in self.experiments:
            instance_name = experiment.setting["instance_name"]
            print(f">> instance_name = {instance_name}")
            experiment.setting["num_iterations"] = max_iters
            self.run_for_one_experiment(
                experiment=experiment,
                max_iters=max_iters,
            )
            print()


def initialize_multipliers(problem: jm.Problem):
    multipliers = {}
    for key in problem.constraints.keys():
        multipliers[key] = 1
    return multipliers


def main():
    from users.makino.updater import update_simple
    from users.makino.solver import sample_model

    # target_instances = "all"
    target_instances = ["knapsack"]

    instance_size = "small"
    instance_dir = f"./Instances/{instance_size}"
    result_dir = f"./Results/makino/{instance_size}"

    bench = PSBench(
        update_simple,
        sample_model,
        target_instances=target_instances,
        n_instances_per_problem=1,
        instance_dir=instance_dir,
        result_dir=result_dir,
    )
    bench.run(max_iters=2)


if __name__ == "__main__":
    main()
