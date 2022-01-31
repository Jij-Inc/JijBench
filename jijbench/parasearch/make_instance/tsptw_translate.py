from abc import abstractmethod
import json
import glob
import os


def make_dist_matrix(coordinates):
    n = len(coordinates)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        x1, y1 = coordinates[i]
        for j in range(n):
            x2, y2 = coordinates[j]
            d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
            dist[i][j] = int(d2**0.5)
    return dist


files = glob.glob("Langevin/*")
for file in files:
    with open(file, "r") as f:
        problem = f.read()

    problem = problem.split("\n")
    N = int(float(problem[0]))

    e = []
    l = []
    for i in problem[1:N+1]:
        data = i.split()
        e.append(int(float(data[0])))
        l.append(int(float(data[1])))


    dist = []
    for i in problem[N+1:-1]:
        data = list(map(float, i.split()))
        dist.append(data)

    ph_value = {"N": N, "e": e, "l": l, "dist": dist}

    problem_dir = "/".join(file.split("/")[:-1])
    save_dir = "Instances/TSPTW/" + problem_dir

    file_name = file.split("/")[-1]
    file_name = file_name.split(".")[:-1]
    file_name = "_".join(file_name)
    file_name = file_name + ".json"

    if not os.path.exists("/app/ParameterSearch/" + save_dir):
        os.makedirs("/app/ParameterSearch/" + save_dir)

    with open(save_dir + "/" + file_name, "w") as f:
        json.dump(ph_value, f)

