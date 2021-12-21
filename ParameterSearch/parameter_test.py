import pickle
from Problem.strip_packing import make_problem
from visualize import make_step_per_violation
from user_script import transpile_problem, make_initial_multipliers, parameter_update
from jijzept import JijSASampler
from jijmodeling.expression.serializable import to_serializable
import datetime
import json


class DataSaver:
    def __init__(self) -> None:
        self.setting = {}
        self.results = {}
        self.data_time = None

    def save(self, path: str = ""):
        now = datetime.datetime.now()
        self.data_time = now.strftime('%Y%m%d_%H%M%S')
        filename = path + 'log_' + now.strftime('%Y%m%d_%H%M%S') + '.json'
        save_obj = {
            'date': str(now),
            'setting': self.setting,
            'results': self.results
          }
        with open(filename, "w") as f:
            json.dump(save_obj, f)


def parameter_test():
    # parameter
    num_sweeps = 5
    num_reads = 5
    num_iterations = 10

    # 問題の作成
    problem = make_problem()
    transpiled_problem = transpile_problem(problem)

    # ph_valueの読み込み
    with open("Instances/strip_packing/instance_data2.pickle", "rb") as f:
        ph_value = pickle.load(f)

    experiment = DataSaver()
    experiment.setting["num_sweeps"] = num_sweeps
    experiment.setting["num_reads"] = num_reads
    experiment.setting["num_iterations"] = num_iterations
    experiment.setting["mathmatical model"] = to_serializable(transpiled_problem)
    experiment.setting["ph_value"] = ph_value

    # 初期パラメータの設定
    multipliers = make_initial_multipliers(transpiled_problem)

    # solverの設定
    sampler = JijSASampler(config="config.toml")

    experiment.results["raw_response"] = {}
    experiment.results["penalties"] = {}
    experiment.setting["multipliers"] = {}
    for step in range(num_iterations):
        # 問題を解く
        response = sampler.sample_model(transpiled_problem, ph_value, multipliers, num_sweeps=num_sweeps, num_reads=num_reads)

        # 解のデコード
        decoded = transpiled_problem.decode(response, ph_value, {})

        penalties = []
        for violations in decoded.constraint_violations:
            penalties.append(sum(value for key, value in violations.items()))
        experiment.results["penalties"][step] = penalties

        # 結果の保存
        experiment.results["raw_response"][step] = response.to_serializable()
        experiment.setting["multipliers"][step] = multipliers

        # パラメータの更新
        multipliers = parameter_update(transpiled_problem, decoded, multipliers)
    experiment.save(path="Results/")
    make_step_per_violation(experiment.data_time)


parameter_test()
