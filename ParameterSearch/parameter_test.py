import pickle
from Problem.strip_packing import make_problem
from update import parameter_update, make_initial_multipliers
from jijzept import JijSASampler


def parameter_test():
    # parameter
    num_sweeps = 5
    num_reads = 5
    num_iterations = 5

    # 問題の作成
    problem = make_problem()

    # ph_valueの読み込み
    with open("Instances/strip_packing/instance_data2.pickle", "rb") as f:
        ph_value = pickle.load(f)

    # 初期パラメータの設定
    multipliers = make_initial_multipliers(problem)

    sampler = JijSASampler(config="config.toml")

    result = {"num_sweeps": num_sweeps, "num_reads": num_reads, "num_iterations": num_iterations}
    parameters, answers = [], []
    for _ in range(num_iterations):
        parameters.append(multipliers)
        # 問題を解く
        response = sampler.sample_model(problem, ph_value, multipliers, num_sweeps=num_sweeps, num_reads=num_reads)
        print(response)

        # 解のデコード
        decoded = problem.decode(response, ph_value, {})
        if len(decoded.feasibles()):
            break

        # パラメータの更新
        multipliers = parameter_update(problem, decoded, multipliers)

#     ans[mul]

parameter_test()
