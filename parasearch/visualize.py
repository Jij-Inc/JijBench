import openjij as oj
import matplotlib.pyplot as plt

# from jijmodeling.expression.serializable import from_serializable
import json


def make_step_per_violation(data_time):
    path = "Results/log_" + data_time + ".json"
    with open(path, "r") as f:
        experiments = json.load(f)
    print(experiments["setting"]["updater"])

    steps = range(experiments["setting"]["num_iterations"])
    penalties = experiments["results"]["penalties"]
    best_penalties = [min(value) for value in penalties.values()]

    plt.plot(steps, best_penalties, marker="o")
    plt.title("step - sum of penalties")
    plt.xlabel("step")
    plt.ylabel("sum of penalties")
    plt.savefig("image.png")


def tts(data_time):
    path = "Results/log_" + data_time + ".json"
    with open(path, "r") as f:
        experiments = json.load(f)


def tts_sample():
    # 最適解を作成します。
    correct_state = [(-1) ** i for i in range(N)]
    # 最適値を計算しておきます。
    bqm = oj.BinaryQuadraticModel.from_ising(h, J)
    minimum_energy = bqm.energy(correct_state)

    # TTS を計算するのに必要なpRを定義します。
    pR = 0.99

    # Samplerの引数の というパラメータに渡すリスト: num_sweeps_listを定義します。
    # num_sweepsはアニーリング中のパラメータ(温度, 横磁場)を下げていくときの分割数です。
    # よって増やすほどゆっくりアニーリングをすることに相当し、アニーリング時間が伸びます。
    num_sweeps_list = [30, 50, 80, 100, 150, 200]

    TTS_list = []  # 各計算時間に対するTTSを格納しておくリストを定義します。
    tau_list = []  # 計算時間を格納しておくリストを定義します。
    e_mean_list = []  # エネルギーの平均値
    e_min_list = []  # 最小エネルギー

    # 計算の過程で成功確率が求まるので、それを格納しておくリストも定義します。
    ps_list = []

    # 確率を計算するために1回のアニーリングを行う回数を定義します。
    num_reads = 1000

    for num_sweeps in num_sweeps_list:
        sampler = oj.SASampler(num_sweeps=num_sweeps, num_reads=num_reads)
        response = sampler.sample_ising(h, J)

        # 計算結果のうち、最適解の数を数えて最適解を得た確率を計算します。
        tau = response.info["execution_time"]

        # psを計算します。最適値以下のエネルギーの個数をカウントします。
        energies = response.energies
        ps = len(energies[energies <= minimum_energy]) / num_reads

        # ps = 0のときTTSが無限大に発散してしまうため、それを回避します。
        if ps == 0.0:
            continue

        # TTSを計算します。
        TTS_list.append(np.log(1 - pR) / np.log(1 - ps) * tau if ps < pR else tau)
        tau_list.append(tau)

        # 成功確率を計算します。
        ps_list.append(ps)

        e_mean_list.append(np.mean(energies))
        e_min_list.append(np.min(energies))
