import matplotlib.pyplot as plt
# from jijmodeling.expression.serializable import from_serializable
import json


def make_step_per_violation(data_time):
    path = "Results/log_" + data_time + ".json"
    with open(path, "r") as f:
        experiments = json.load(f)

    steps = range(experiments["setting"]["num_iterations"])
    penalties = experiments["results"]["penalties"]
    best_penalties = [min(value) for key, value in penalties.items()]

    plt.plot(steps, best_penalties, marker="o")
    plt.title("step - sum of penalties")
    plt.xlabel("step")
    plt.ylabel("sum of penalties")
    plt.savefig("image.png")
